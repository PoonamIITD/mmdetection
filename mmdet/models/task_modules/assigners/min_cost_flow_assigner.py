from typing import List, Optional, Union

import torch
from torch import Tensor
from mmengine.structures import InstanceData
from mmdet.registry import TASK_UTILS
from .assign_result import AssignResult
from .base_assigner import BaseAssigner
from torchvision.ops.boxes import box_area
from ortools.graph.python import min_cost_flow
from functools import reduce
from operator import add
import numpy as np
# ------------------------------------------------------------
#  Helper functions (copied from DETR-style utils)
# ------------------------------------------------------------

def box_iou(boxes1, boxes2):
    """IoU between two sets of boxes."""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    return iou, union


# ================================================================
# 1️⃣ Helper: Flow-based matching (Ajay Shastry WACV 2024)
# ================================================================
def match_with_dummy_gt(cost_matrix: torch.Tensor,
                        max_matches: int = 6,
                        dummy_weight: float = 1.8):
    """
    Flow-based many-to-one matcher using OR-Tools.
    Args:
        cost_matrix: [num_preds, num_gts] cost matrix (torch.Tensor)
        max_matches: max allowed matches per GT (repetition)
        dummy_weight: cost for dummy assignment
    Returns:
        pred_inds, gt_inds (torch.Tensor, torch.Tensor)
    """
    C = cost_matrix
    smcf = min_cost_flow.SimpleMinCostFlow()
    num_queries, num_gt = C.shape

    if num_gt == 0:
        return torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64)

    # Node indices
    num_gt_new = num_gt + 1  # extra dummy node
    sink_ind = num_queries + num_gt_new + 1
    gt_start = num_queries + 1

    # Start nodes
    R = reduce(add, [[1 + i] * num_gt for i in range(num_queries)])
    S = R + [0] * num_queries + [i + 1 for i in range(num_queries)] + [gt_start + i for i in range(num_gt_new)]
    # End nodes
    E = [gt_start + i for i in range(num_gt)] * num_queries + [i + 1 for i in range(num_queries)] + [sink_ind - 1] * num_queries + [sink_ind] * num_gt_new
    # Costs
    # Costs = list((C[:, :num_gt].numpy().ravel()) * 1e4) + [0] * num_queries + [dummy_weight] * num_queries + [0] * num_gt_new
    Costs = list((C[:, :num_gt].detach().cpu().numpy().ravel()) * 1e4) + [0] * num_queries + [dummy_weight] * num_queries + [0] * num_gt_new

    start_nodes = np.array(S)
    end_nodes = np.array(E)
    capacities = np.array([1] * num_queries + [1] * (num_queries * num_gt_new) + [max_matches - 1] * num_gt + [(num_gt) * (max_matches - 1)] * 1)
    unit_costs = np.array(Costs)

    # Node supplies
    supplies = [max_matches * num_gt] + [0] * num_queries + [-1] * num_gt + [0] + [(1 - max_matches) * num_gt]

    # Add arcs and supplies
    all_arcs = smcf.add_arcs_with_capacity_and_unit_cost(start_nodes, end_nodes, capacities, unit_costs)
    smcf.set_nodes_supplies(np.arange(len(supplies)), supplies)

    status = smcf.solve()
    if status != smcf.OPTIMAL:
        raise RuntimeError(f"MinCostFlow failed with status {status}")

    solution_flows = smcf.flows(all_arcs)

    # Extract matched indices
    Q_ind, T_ind = [], []
    indices = np.where(solution_flows != 0)
    arcs = all_arcs[indices]
    for arc in arcs:
        h = smcf.head(arc)
        t = smcf.tail(arc)
        if t != 0 and h != sink_ind:
            Q_ind.append(t - 1)
            T_ind.append(h - 1 - num_queries)

    return torch.tensor(Q_ind, dtype=torch.int64), torch.tensor(T_ind, dtype=torch.int64)


# -------------------------
# MinCostFlowAssigner
# -------------------------
@TASK_UTILS.register_module()
class MinCostFlowAssigner(BaseAssigner):
    """Many-to-one flow-based assigner with weighted cost computation."""

    def __init__(self,
                 match_costs: Union[List[dict], dict],
                 repetition: int = 6,
                 dummy_weight: float = 1.8):
        """
        Args:
            match_costs: dict or list of dicts for computing cost (classification, bbox, iou, etc.)
            repetition: maximum allowed matches per GT
            dummy_weight: weight for dummy assignment
        """
        super().__init__()
        if isinstance(match_costs, dict):
            match_costs = [match_costs]
        elif isinstance(match_costs, list):
            assert len(match_costs) > 0, "match_costs must not be empty"
        self.match_costs = [TASK_UTILS.build(mc) for mc in match_costs]
        self.repetition = repetition
        self.dummy_weight = dummy_weight

    # def assign(self,
    #        pred_instances: InstanceData,
    #        gt_instances: InstanceData,
    #        img_meta: Optional[dict] = None,
    #        **kwargs) -> AssignResult:
    #     """
    #     Args:
    #         pred_instances: InstanceData for predictions (bboxes, labels, scores)
    #         gt_instances: InstanceData for GT (bboxes, labels)
    #         img_meta: optional image info
    #     Returns:
    #         AssignResult
    #     """
    #     assert isinstance(gt_instances.labels, Tensor)
    #     num_gts, num_preds = len(gt_instances), len(pred_instances)
    #     pred_boxes = pred_instances.bboxes  # [num_preds, 4]
    #     gt_boxes = gt_instances.bboxes
    #     gt_labels = gt_instances.labels
    #     device = gt_labels.device

    #     assigned_gt_inds = torch.full((num_preds,), -1, dtype=torch.long, device=device)
    #     assigned_labels = torch.full((num_preds,), -1, dtype=torch.long, device=device)

    #     # ------------------ Handle no-GT or no-pred cases ------------------
    #     if num_gts == 0 or num_preds == 0:
    #         if num_gts == 0:
    #             assigned_gt_inds[:] = 0
    #         return AssignResult(num_gts, assigned_gt_inds, None, assigned_labels)

    #     # ------------------ 1️⃣ Compute weighted & normalized cost ------------------
    #     normed_costs = []
    #     print(f"\n[MinCostFlowMatcher] --- Computing Costs for {num_preds} preds × {num_gts} GTs ---")
    #     for i, match_cost in enumerate(self.match_costs):
    #         raw_cost = match_cost(pred_instances=pred_instances,
    #                             gt_instances=gt_instances,
    #                             img_meta=img_meta)
    #         # normalize within each term, then re-weight
    #         c = (raw_cost - raw_cost.min()) / (raw_cost.max() - raw_cost.min() + 1e-6)
    #         c = c * match_cost.weight
    #         normed_costs.append(c)

    #         # diagnostic print
    #         print(f"[Cost {i}] type={match_cost.__class__.__name__:<25} "
    #             f"raw(min={raw_cost.min():.3f}, max={raw_cost.max():.3f}, mean={raw_cost.mean():.3f}) "
    #             f"→ norm_mean={c.mean():.3f}, weight={match_cost.weight}")

    #     cost_matrix = torch.stack(normed_costs).sum(dim=0)

    #     # ------------------ 2️⃣ Run many-to-one flow matching ------------------
    #     pred_inds, gt_inds = match_with_dummy_gt(cost_matrix,
    #                                             max_matches=self.repetition,
    #                                             dummy_weight=self.dummy_weight)

    #     # move to correct device
    #     pred_inds = torch.as_tensor(pred_inds, dtype=torch.long, device=device)
    #     gt_inds = torch.as_tensor(gt_inds, dtype=torch.long, device=device)

    #     # ------------------ 3️⃣ Filter dummy GTs ------------------
    #     valid = gt_inds < num_gts
    #     pred_inds, gt_inds = pred_inds[valid], gt_inds[valid]

    #     # ------------------ 4️⃣ Assign foreground/background labels ------------------
    #     assigned_gt_inds[:] = 0  # background by default
    #     assigned_gt_inds[pred_inds] = gt_inds + 1
    #     assigned_labels[pred_inds] = gt_labels[gt_inds]

    #     print('assigned_gt_inds')
        
    #     return AssignResult(
    #         num_gts=num_gts,
    #         gt_inds=assigned_gt_inds,
    #         max_overlaps=None,  # can be IoU if needed later
    #         labels=assigned_labels)


    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               img_meta: Optional[dict] = None,
               **kwargs) -> AssignResult:
        """
        Args:
            pred_instances: InstanceData for predictions (bboxes, labels, scores)
            gt_instances: InstanceData for GT (bboxes, labels)
            img_meta: optional image info
        Returns:
            AssignResult
        """
        assert isinstance(gt_instances.labels, Tensor)
        num_gts, num_preds = len(gt_instances), len(pred_instances)
        pred_boxes = pred_instances.bboxes  # [num_preds, 4]
        gt_boxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        device = gt_labels.device

        # print("pred score shape:",pred_instances.scores.shape)
        # print(gt_labels)
        # print('max gt_label:', gt_labels.max().item())

        assigned_gt_inds = torch.full((num_preds,), -1, dtype=torch.long, device=device)
        assigned_labels = torch.full((num_preds,), -1, dtype=torch.long, device=device)

        if num_gts == 0 or num_preds == 0:
            if num_gts == 0:
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts, assigned_gt_inds, None, assigned_labels)

        # 1️. Compute weighted cost
        cost_list = [] 
        for match_cost in self.match_costs: 
            cost = match_cost(pred_instances=pred_instances, gt_instances=gt_instances, img_meta=img_meta) 
            cost_list.append(cost) 
        cost_matrix = torch.stack(cost_list).sum(dim=0) # shape [num_preds, num_gts]

        # 2️. Run many-to-one flow matching
        pred_inds, gt_inds = match_with_dummy_gt(cost_matrix,
                                                 max_matches=self.repetition,
                                                 dummy_weight=self.dummy_weight)
        
        # move to correct device
        pred_inds = torch.as_tensor(pred_inds, dtype=torch.long, device=device)
        gt_inds = torch.as_tensor(gt_inds, dtype=torch.long, device=device)

        # 3. Filtering the dummy Ground Truth indices
        valid = gt_inds < num_gts
        pred_inds, gt_inds = pred_inds[valid], gt_inds[valid]

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        assigned_gt_inds = torch.zeros(num_preds, dtype=torch.long, device=cost_matrix.device)

        assigned_gt_inds[pred_inds] = gt_inds + 1
        assigned_labels[pred_inds] = gt_labels[gt_inds]

        # print('Matched pred indices:', pred_inds)
        # print('Matched gt indices:', gt_inds)
        # print('Pred boxes:', pred_instances.bboxes[pred_inds])
        # print('GT boxes:', gt_instances.bboxes[gt_inds])

        num_real = valid.sum().item()
        num_dummy = (~valid).sum().item()

        # This gives how many *assignments* are real (not dummy)
        num_real_assignments = num_real

        # To compute how many *unique* predictions* were assigned to any GT:
        num_pred_assigned = pred_inds.unique().numel()

        # To compute how many *unique GTs* got at least one prediction:
        num_gt_covered = gt_inds.unique().numel()

        # print(f"[Flow] Unique matched GTs: {num_gt_covered}/{num_gts}")
        # print(f"[Flow] Total matched preds: {num_pred_assigned}/{num_preds}")
        # print(f"[Flow] Total dummy preds: {num_preds - num_pred_assigned}")
        # print(f"[Flow] Ratio of real to dummy preds: "
        #     f"{100 * num_pred_assigned / num_preds:.2f}% real, "
        #     f"{100 * (1 - num_pred_assigned / num_preds):.2f}% dummy")

        return AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=None,   # or IoU tensor if you want
            labels=assigned_labels)