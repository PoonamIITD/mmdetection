# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import torch
from mmengine import ConfigDict
from mmengine.structures import InstanceData
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from mmdet.registry import TASK_UTILS
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@TASK_UTILS.register_module()
class HungarianAssigner(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of some components.
    For DETR the costs are weighted sum of classification cost, regression L1
    cost and regression iou cost. The targets don't include the no_object, so
    generally there are more predictions than targets. After the one-to-one
    matching, the un-matched are treated as backgrounds. Thus each query
    prediction will be assigned with `0` or a positive integer indicating the
    ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        match_costs (:obj:`ConfigDict` or dict or \
            List[Union[:obj:`ConfigDict`, dict]]): Match cost configs.
    """

    def __init__(
        self, match_costs: Union[List[Union[dict, ConfigDict]], dict,
                                 ConfigDict], 
                                 topk=2,iou_thr=0.5, small_obj_thr=32**2,
                                 visibility_thr = 0.7, enable_topk=True
    ) -> None:
        if isinstance(match_costs, dict):
            match_costs = [match_costs]
        elif isinstance(match_costs, list):
            assert len(match_costs) > 0, \
                'match_costs must not be a empty list.'

        self.match_costs = [
            TASK_UTILS.build(match_cost) for match_cost in match_costs
        ]
        self.topk = topk
        self.iou_thr = iou_thr
        self.small_obj_thr = small_obj_thr
        self.enable_topk = enable_topk
        self.visibility_thr = visibility_thr

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               img_meta: Optional[dict] = None,
               **kwargs) -> AssignResult:
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the ``InstanceData``
                in other places. It may includes ``masks``, with shape
                (n, h, w) or (n, l).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                ``labels``, with shape (k, ) and ``masks``, with shape
                (k, h, w) or (k, l).
            img_meta (dict): Image information.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert isinstance(gt_instances.labels, Tensor)
        num_gts, num_preds = len(gt_instances), len(pred_instances)
        gt_labels = gt_instances.labels
        device = gt_labels.device

        # 1. assign -1 by default
        assigned_gt_inds = torch.full((num_preds, ),
                                      -1,
                                      dtype=torch.long,
                                      device=device)
        assigned_labels = torch.full((num_preds, ),
                                     -1,
                                     dtype=torch.long,
                                     device=device)

        if num_gts == 0 or num_preds == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=None,
                labels=assigned_labels)

        # 2. compute weighted cost
        cost_list = []
        for match_cost in self.match_costs:
            cost = match_cost(
                pred_instances=pred_instances,
                gt_instances=gt_instances,
                img_meta=img_meta)
            cost_list.append(cost)
        cost = torch.stack(cost_list).sum(dim=0)

        cost_device = cost.detach()  # for GPU ops
        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost_cpu = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')

        matched_row_inds, matched_col_inds = linear_sum_assignment(cost_cpu)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

        # ============================
        # 🔥 Top-k augmentation block
        # ============================
        if (self.enable_topk and
            hasattr(gt_instances, 'visibility') and
            hasattr(pred_instances, 'bboxes')):

            vis = gt_instances.visibility.to(device)  # (num_gts,)
            gt_bboxes = gt_instances.bboxes
            pred_bboxes = pred_instances.bboxes

            # compute area (xyxy format)
            areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * \
                    (gt_bboxes[:, 3] - gt_bboxes[:, 1])

            # 🎯 target GTs: small OR occluded
            small_mask = areas < self.small_obj_thr
            occluded_mask = vis < self.visibility_thr
            target_mask = small_mask | occluded_mask

            target_gt_inds = torch.nonzero(target_mask).squeeze(1)

            if len(target_gt_inds) > 0:
                from mmdet.structures.bbox import bbox_overlaps

                # IoU matrix (num_preds, num_gts)
                ious = bbox_overlaps(pred_bboxes, gt_bboxes)

                for gt_idx in target_gt_inds:

                    gt_cost = cost_device[:, gt_idx].clone()

                    already_matched = (assigned_gt_inds == (gt_idx + 1))
                    gt_cost[already_matched] = float('inf')

                    valid_mask = (ious[:, gt_idx] > self.iou_thr) & (~already_matched)
                    valid_inds = torch.nonzero(valid_mask).squeeze(1)

                    if len(valid_inds) == 0:
                        continue

                    valid_ious = ious[valid_inds, gt_idx]

                    # 🔥 best version (less noisy)
                    topk_iou_vals = valid_ious.topk(min(3, len(valid_ious))).values
                    score = topk_iou_vals.mean() * (1 - vis[gt_idx])

                    k = int(self.topk * (1 + 2 * score))
                    k = min(k, self.topk * 3)  # optional safety
                    k = max(1, min(k, len(valid_inds)))

                    valid_cost = gt_cost[valid_inds]
                    topk_local = torch.topk(valid_cost, k=k, largest=False).indices
                    topk_inds = valid_inds[topk_local]

                    for pred_idx in topk_inds:
                        if assigned_gt_inds[pred_idx] == 0:
                            assigned_gt_inds[pred_idx] = gt_idx + 1
                            assigned_labels[pred_idx] = gt_labels[gt_idx]

            # Count matches per GT (ignore background 0)
            gt_match_count = torch.bincount(
                assigned_gt_inds.clamp(min=0),
                minlength=num_gts + 1
            )[1:]  # remove background

            # Extra positives beyond Hungarian (which gives 1)
            extra_pos = gt_match_count - 1

            # ===== Overall stats =====
            pos_mask = extra_pos > 0
            num_extra = pos_mask.sum().item()
            avg_extra = extra_pos[pos_mask].float().mean().item() if num_extra > 0 else 0.0

            print(f"[TopK] GTs with extra matches: {num_extra}/{num_gts}")
            print(f"[TopK] Avg extra per GT: {avg_extra:.2f}")

            # ===== Target (small/occluded) stats =====
            target_gt_inds = torch.nonzero(target_mask, as_tuple=False).flatten()

            if len(target_gt_inds) > 0:
                target_extra = extra_pos[target_gt_inds]

                target_mask_pos = target_extra > 0
                num_target_extra = target_mask_pos.sum().item()
                avg_target_extra = (
                    target_extra[target_mask_pos].float().mean().item()
                    if num_target_extra > 0 else 0.0
                )

                print(f"[TopK] Target GTs: {len(target_gt_inds)}")
                print(f"[TopK] Target GTs with extra: {num_target_extra}")
                print(f"[TopK] Avg extra (target): {avg_target_extra:.2f}")
        
        return AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=None,
            labels=assigned_labels)
