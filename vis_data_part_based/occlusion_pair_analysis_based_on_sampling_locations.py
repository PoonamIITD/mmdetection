import os
import json
import math
import argparse
from pathlib import Path

import torch
import numpy as np
from collections import defaultdict


# ============================================================
# CONFIG
# ============================================================

RSUD_CLASSES = [
    'person','rickshaw','rickshaw van','auto rickshaw',
    'truck','pickup truck','private car','motorcycle',
    'bicycle','bus','micro bus','covered van','human hauler'
]


# ============================================================
# COCO GT LOADER
# ============================================================

def load_coco_gt(gt_json_path):

    with open(gt_json_path, "r") as f:
        coco = json.load(f)

    images = {}
    anns_per_image = defaultdict(list)

    for img in coco["images"]:
        images[img["id"]] = img

    for ann in coco["annotations"]:
        anns_per_image[ann["image_id"]].append(ann)

    return images, anns_per_image


# ============================================================
# BOX UTILS
# ============================================================

def coco_xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter   = inter_w * inter_h

    area1 = max(0, box1[2]-box1[0]) * max(0, box1[3]-box1[1])
    area2 = max(0, box2[2]-box2[0]) * max(0, box2[3]-box2[1])

    union = area1 + area2 - inter + 1e-6
    return inter / union


def compute_occlusion_ratio(box_occ, box_occder):
    """
    What fraction of the occluded object's area is covered
    by the occluder's bounding box.
    """
    xA = max(box_occ[0], box_occder[0])
    yA = max(box_occ[1], box_occder[1])
    xB = min(box_occ[2], box_occder[2])
    yB = min(box_occ[3], box_occder[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter   = inter_w * inter_h

    occ_area = (
        max(0, box_occ[2]-box_occ[0]) *
        max(0, box_occ[3]-box_occ[1])
    ) + 1e-6

    return inter / occ_area


# ============================================================
# QUERY MATCHING
# ============================================================

def normalized_cxcywh_to_xyxy(box_norm, W, H):
    cx, cy, bw, bh = box_norm
    x1 = (cx - bw/2) * W
    y1 = (cy - bh/2) * H
    x2 = (cx + bw/2) * W
    y2 = (cy + bh/2) * H
    return [x1, y1, x2, y2]


def get_query_predictions(data, layer=-1):
    cls_scores  = data["cls_scores"]       # [L, Q, C]
    refs        = data["reference_points"] # [L+1, Q, 4]

    cls_l = cls_scores[layer]              # [Q, C]
    pred_scores, pred_classes = cls_l.max(dim=1)
    pred_boxes  = refs[layer][:, :4]

    return pred_boxes, pred_classes, pred_scores


def match_gt_to_queries(
    data,
    gt_annotations,
    img_w,
    img_h,
    iou_threshold=0.3,
    layer=-1,
    topk_iou_candidates=5,
    iou_eps=0.03
):
    pred_boxes_norm, pred_classes, pred_scores = \
        get_query_predictions(data, layer)

    pred_boxes = [
        normalized_cxcywh_to_xyxy(b.tolist(), img_w, img_h)
        for b in pred_boxes_norm
    ]

    matched = []

    for ann in gt_annotations:

        gt_box = coco_xywh_to_xyxy(ann["bbox"])
        gt_cls = ann["category_id"] - 1

        candidates = []

        for q_idx in range(len(pred_boxes)):

            if pred_classes[q_idx].item() != gt_cls:
                continue

            iou = compute_iou(gt_box, pred_boxes[q_idx])

            if iou < iou_threshold:
                continue

            candidates.append({
                "query_id": q_idx,
                "iou": iou,
                "score": pred_scores[q_idx].item()
            })

        if len(candidates) == 0:
            continue

        candidates = sorted(candidates, key=lambda x: x["iou"], reverse=True)
        candidates = candidates[:topk_iou_candidates]

        best_iou  = candidates[0]["iou"]
        near_best = [c for c in candidates if c["iou"] >= (best_iou - iou_eps)]

        best_candidate = max(near_best, key=lambda x: x["score"])
        best_q         = best_candidate["query_id"]

        matched.append({
            "query_id":   best_q,
            "gt_box":     gt_box,
            "gt_class":   gt_cls,
            "gt_ann_id":  ann["id"],
            "iou":        best_candidate["iou"],
            "score":      best_candidate["score"]
        })

    return matched


# ============================================================
# SAMPLING LOCATION OVERLAP
# ============================================================

def get_absolute_sampling_locations(data, q_idx):
    """
    Convert normalised sampling locations to absolute (x, y) pixel
    coordinates for one query across all decoder layers.

    sampling_locations : [L, Q, H, Lv, P, 2]   — values in [0, 1]
    spatial_shapes     : [Lv, 2]                — (H_lv, W_lv) per level
    valid_ratios       : [Lv, 2]                — (ratio_h, ratio_w) per level

    The stored normalised coordinates are relative to the padded canvas
    scaled by valid_ratio, so:
        abs_x = loc_x / valid_ratio_w  * img_W   (clamped to [0,1] first)
        abs_y = loc_y / valid_ratio_h  * img_H

    We flatten over (H, Lv, P) heads/levels/points and return
        shape: [L, N_pts, 2]   where N_pts = H * Lv * P
    """

    sl          = data["sampling_locations"]   # [L, Q, H, Lv, P, 2]
    valid_ratios = data["valid_ratios"]         # [Lv, 2]   (ratio_h, ratio_w)

    L, Q, H, Lv, P, _ = sl.shape

    # --- query slice: [L, H, Lv, P, 2] ---------------------------------
    q_locs = sl[:, q_idx]                      # [L, H, Lv, P, 2]

    # broadcast valid_ratios [Lv, 2] → [1, 1, Lv, 1, 2]
    vr = valid_ratios.view(1, 1, Lv, 1, 2)     # ratio order: (h, w) → (y, x)

    # normalise to true [0,1] image space
    # loc[..., 0] = x  → divide by valid_ratio_w  (index 1)
    # loc[..., 1] = y  → divide by valid_ratio_h  (index 0)
    vr_xy = vr[..., [1, 0]]                     # swap to (w, h) = (x, y) order

    abs_locs = q_locs / vr_xy.clamp(min=1e-6)  # [L, H, Lv, P, 2]
    abs_locs = abs_locs.clamp(0.0, 1.0)

    # flatten heads / levels / points → [L, N_pts, 2]
    abs_locs = abs_locs.reshape(L, H * Lv * P, 2)

    return abs_locs   # normalised absolute coords in [0,1]


def sampling_overlap_per_layer(
    locs_i,          # [L, N_pts, 2]  — occluded query
    locs_j,          # [L, N_pts, 2]  — occluder query
    radius=0.02,     # neighbourhood radius in normalised [0,1] space
    attn_weights_i=None,   # [L, H, Lv, P]  optional — weight by attention
    attn_weights_j=None,
    q_idx_i=None,
    q_idx_j=None,
    data=None
):
    """
    For each decoder layer compute a soft overlap score between the
    sampling point clouds of two queries.

    Method
    ------
    For each point p in query_i, count how many points in query_j
    fall within `radius` (in normalised image space).  Normalise by
    total possible cross-pairs.  Average over all points in query_i.

    If attention weights are supplied the counts are weighted by the
    product of the two attention weights, giving a semantically
    meaningful overlap that emphasises actually-attended locations.

    Returns
    -------
    overlap_per_layer : list[float]  length L
    mean_overlap      : float        scalar summary
    """

    L = locs_i.shape[0]
    overlap_per_layer = []

    # optionally flatten attention weights to [L, N_pts]
    if attn_weights_i is not None and q_idx_i is not None:
        # attn shape: [L, Q, H, Lv, P]
        aw_i = data["attention_weights"][:, q_idx_i]   # [L, H, Lv, P]
        aw_j = data["attention_weights"][:, q_idx_j]   # [L, H, Lv, P]
        H, Lv, P = aw_i.shape[1], aw_i.shape[2], aw_i.shape[3]
        aw_i = aw_i.reshape(L, -1)   # [L, N_pts]
        aw_j = aw_j.reshape(L, -1)   # [L, N_pts]

        # normalise per layer so weights sum to 1
        aw_i = aw_i / (aw_i.sum(dim=1, keepdim=True) + 1e-9)
        aw_j = aw_j / (aw_j.sum(dim=1, keepdim=True) + 1e-9)
    else:
        aw_i = None
        aw_j = None

    for l in range(L):
        pts_i = locs_i[l]   # [N_pts, 2]
        pts_j = locs_j[l]   # [N_pts, 2]

        # pairwise L2 distances  [N_i, N_j]
        diff = pts_i.unsqueeze(1) - pts_j.unsqueeze(0)   # [Ni, Nj, 2]
        dist = diff.norm(dim=-1)                          # [Ni, Nj]

        within = (dist < radius).float()                  # [Ni, Nj]  binary mask

        if aw_i is not None:
            # weighted overlap: sum_i sum_j  w_i * w_j * within(i,j)
            wi = aw_i[l].unsqueeze(1)   # [Ni, 1]
            wj = aw_j[l].unsqueeze(0)   # [1,  Nj]
            score = (wi * wj * within).sum().item()
        else:
            # unweighted: fraction of query_i points that have ≥1 neighbour in j
            score = (within.sum(dim=1) > 0).float().mean().item()

        overlap_per_layer.append(float(score))

    mean_overlap = float(np.mean(overlap_per_layer))
    return overlap_per_layer, mean_overlap


# ============================================================
# OCCLUSION ANALYSIS
# ============================================================

def analyze_occlusion_pairs(
    matched_queries,
    occ_ratio_threshold=0.7,
    occ_score_low=0.6,
    occ_score_high=0.8,
    occder_score_low=0.88,
    occder_score_high=1.01,
    same_class_only=None
):
    pairs = []
    N = len(matched_queries)

    for i in range(N):
        for j in range(N):

            if i == j:
                continue

            obj_i = matched_queries[i]
            obj_j = matched_queries[j]

            if obj_i["gt_ann_id"] == obj_j["gt_ann_id"]:
                continue

            cls_i = obj_i["gt_class"]
            cls_j = obj_j["gt_class"]

            if same_class_only is True  and cls_i != cls_j:
                continue
            if same_class_only is False and cls_i == cls_j:
                continue

            box_i = obj_i["gt_box"]
            box_j = obj_j["gt_box"]

            occ_ratio = compute_occlusion_ratio(box_i, box_j)
            score_i   = obj_i["score"]
            score_j   = obj_j["score"]

            if occ_ratio < occ_ratio_threshold:
                continue

            if not (occ_score_low <= score_i <= occ_score_high):
                continue

            if not (occder_score_low <= score_j <= occder_score_high):
                continue

            pairs.append({
                "occluded_query":     obj_i["query_id"],
                "occluder_query":     obj_j["query_id"],
                "occluded_gt_ann_id": obj_i["gt_ann_id"],
                "occluder_gt_ann_id": obj_j["gt_ann_id"],
                "occluded_class":     cls_i,
                "occluder_class":     cls_j,
                "occluded_score":     score_i,
                "occluder_score":     score_j,
                "occlusion_ratio":    occ_ratio,
                "same_class":         cls_i == cls_j,
            })

    return pairs


# ============================================================
# DECODER TRAJECTORY
# ============================================================

def confidence_trajectory(cls_scores, q_idx):
    scores = cls_scores[:, q_idx]
    max_scores, _ = scores.max(dim=1)
    return max_scores.numpy()


def analyze_decoder_drop(data, q_idx):
    traj = confidence_trajectory(data["cls_scores"], q_idx).tolist()

    return {
        "trajectory":   traj,
        "start_conf":   float(traj[0]),
        "end_conf":     float(traj[-1]),
        "max_conf":     float(max(traj)),
        "min_conf":     float(min(traj)),
        "delta":        float(traj[-1] - traj[0]),
        "argmax_layer": int(np.argmax(traj)),
        "argmin_layer": int(np.argmin(traj)),
    }


# ============================================================
# MAIN FILE PROCESSOR
# ============================================================

def process_pt_file(pt_file, coco_images, coco_anns, args):

    data = torch.load(pt_file, map_location="cpu")

    img_name = Path(data["img_path"]).name

    image_id = next(
        (k for k, v in coco_images.items() if v["file_name"] == img_name),
        None
    )
    if image_id is None:
        return []

    gt_annotations = coco_anns[image_id]
    img_w = coco_images[image_id]["width"]
    img_h = coco_images[image_id]["height"]

    matched_queries = match_gt_to_queries(
        data, gt_annotations, img_w, img_h,
        iou_threshold=args.match_iou,
        layer=args.layer
    )

    if len(matched_queries) < 2:
        return []

    same_class_only = {"same": True, "different": False}.get(args.mode, None)

    pairs = analyze_occlusion_pairs(
        matched_queries,
        occ_ratio_threshold=args.occlusion_ratio,
        occ_score_low=args.occ_score_low,
        occ_score_high=args.occ_score_high,
        occder_score_low=args.occder_score_low,
        occder_score_high=args.occder_score_high,
        same_class_only=same_class_only
    )

    # check if sampling_locations is available in this pt file
    has_sampling = "sampling_locations" in data

    for p in pairs:

        qi = p["occluded_query"]
        qj = p["occluder_query"]

        # ── confidence trajectories ──────────────────────
        occ_stats  = analyze_decoder_drop(data, qi)
        occr_stats = analyze_decoder_drop(data, qj)

        p["occluded_scores_per_layer"]  = occ_stats["trajectory"]
        p["occluded_start_conf"]        = occ_stats["start_conf"]
        p["occluded_end_conf"]          = occ_stats["end_conf"]
        p["occluded_max_conf"]          = occ_stats["max_conf"]
        p["occluded_min_conf"]          = occ_stats["min_conf"]
        p["occluded_delta"]             = occ_stats["delta"]
        p["occluded_peak_layer"]        = occ_stats["argmax_layer"]

        p["occluder_scores_per_layer"]  = occr_stats["trajectory"]
        p["occluder_start_conf"]        = occr_stats["start_conf"]
        p["occluder_end_conf"]          = occr_stats["end_conf"]
        p["occluder_max_conf"]          = occr_stats["max_conf"]
        p["occluder_min_conf"]          = occr_stats["min_conf"]
        p["occluder_delta"]             = occr_stats["delta"]
        p["occluder_peak_layer"]        = occr_stats["argmax_layer"]

        # ── sampling location overlap ────────────────────
        if has_sampling:
            locs_i = get_absolute_sampling_locations(data, qi)  # [L, N_pts, 2]
            locs_j = get_absolute_sampling_locations(data, qj)  # [L, N_pts, 2]

            # --- unweighted overlap (proximity-based) ----
            overlap_unweighted, mean_unweighted = sampling_overlap_per_layer(
                locs_i, locs_j,
                radius=args.overlap_radius
            )

            # --- attention-weighted overlap ---------------
            overlap_weighted, mean_weighted = sampling_overlap_per_layer(
                locs_i, locs_j,
                radius=args.overlap_radius,
                attn_weights_i=True,   # flag to trigger weighting
                attn_weights_j=True,
                q_idx_i=qi,
                q_idx_j=qj,
                data=data
            )

            p["sampling_overlap_per_layer"]          = [round(v, 6) for v in overlap_unweighted]
            p["mean_sampling_overlap"]               = round(mean_unweighted, 6)

            p["attn_weighted_overlap_per_layer"]     = [round(v, 6) for v in overlap_weighted]
            p["mean_attn_weighted_overlap"]          = round(mean_weighted, 6)

            # peak overlap layer — useful for cross-referencing with
            # confidence drop layers
            p["peak_overlap_layer"] = int(np.argmax(overlap_unweighted))
            p["peak_weighted_overlap_layer"] = int(np.argmax(overlap_weighted))

        else:
            # pt file predates sampling_locations being saved
            p["sampling_overlap_per_layer"]      = None
            p["mean_sampling_overlap"]           = None
            p["attn_weighted_overlap_per_layer"] = None
            p["mean_attn_weighted_overlap"]      = None
            p["peak_overlap_layer"]              = None
            p["peak_weighted_overlap_layer"]     = None

        p["image_name"] = img_name

    return pairs


# ============================================================
# ENTRY
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Occlusion pair analysis with sampling overlap for DETR-family models."
    )

    parser.add_argument("--pt_dir",  type=str, required=True)
    parser.add_argument("--gt_json", type=str, required=True)
    parser.add_argument("--output",  type=str, default="occlusion_results.json")

    # geometry
    parser.add_argument("--occlusion_ratio",  type=float, default=0.5)
    parser.add_argument("--match_iou",        type=float, default=0.3)

    # occluded confidence range
    parser.add_argument("--occ_score_low",    type=float, default=0.6)
    parser.add_argument("--occ_score_high",   type=float, default=0.8)

    # occluder confidence range
    parser.add_argument("--occder_score_low", type=float, default=0.75)
    parser.add_argument("--occder_score_high",type=float, default=1.01)

    # class pair mode
    parser.add_argument("--mode", type=str, default="same",
                        choices=["same", "different", "all"])

    parser.add_argument("--layer", type=int, default=-1,
                        help="Decoder layer for GT→query matching (-1 = last)")

    # ── NEW: sampling overlap radius ──────────────────────
    parser.add_argument(
        "--overlap_radius", type=float, default=0.02,
        help=(
            "Neighbourhood radius in normalised [0,1] image space for "
            "counting two sampling points as 'overlapping'. "
            "Default 0.02 ≈ 2%% of image width/height. "
            "Decrease for stricter overlap (e.g. 0.01), "
            "increase for looser (e.g. 0.05)."
        )
    )

    args = parser.parse_args()

    coco_images, coco_anns = load_coco_gt(args.gt_json)

    pt_files  = sorted(Path(args.pt_dir).glob("*.pt"))
    all_pairs = []

    for pt_file in pt_files:
        print(f"[INFO] processing {pt_file.name}")
        pairs = process_pt_file(pt_file, coco_images, coco_anns, args)
        all_pairs.extend(pairs)

    # ── summary ───────────────────────────────────────────
    same_count = sum(p["same_class"] for p in all_pairs)
    diff_count = len(all_pairs) - same_count

    # sampling overlap summary (only pairs that have it)
    pairs_with_overlap = [
        p for p in all_pairs if p["mean_sampling_overlap"] is not None
    ]

    print("\n================================================")
    print("SUMMARY")
    print("================================================")
    print(f"Total pairs              : {len(all_pairs)}")
    print(f"Same-class pairs         : {same_count}")
    print(f"Diff-class pairs         : {diff_count}")
    print(f"Pairs with overlap data  : {len(pairs_with_overlap)}")

    if pairs_with_overlap:
        mean_ov    = np.mean([p["mean_sampling_overlap"]        for p in pairs_with_overlap])
        mean_ov_w  = np.mean([p["mean_attn_weighted_overlap"]   for p in pairs_with_overlap])

        # per-layer mean overlap across all pairs
        n_layers = len(pairs_with_overlap[0]["sampling_overlap_per_layer"])
        layer_means = np.mean(
            [p["sampling_overlap_per_layer"] for p in pairs_with_overlap],
            axis=0
        )
        layer_means_w = np.mean(
            [p["attn_weighted_overlap_per_layer"] for p in pairs_with_overlap],
            axis=0
        )

        print(f"\nSampling overlap (unweighted)  mean: {mean_ov:.4f}")
        print(f"Sampling overlap (attn-weighted) mean: {mean_ov_w:.6f}")
        print(f"\nPer-layer unweighted overlap:")
        for l, v in enumerate(layer_means):
            print(f"  L{l}: {v:.4f}")
        print(f"\nPer-layer attn-weighted overlap:")
        for l, v in enumerate(layer_means_w):
            print(f"  L{l}: {v:.6f}")

        # peak overlap layer distribution
        peak_layer_counts = defaultdict(int)
        for p in pairs_with_overlap:
            peak_layer_counts[p["peak_overlap_layer"]] += 1
        print(f"\nPeak overlap layer distribution:")
        for l in sorted(peak_layer_counts):
            print(f"  L{l}: {peak_layer_counts[l]} pairs")

    # per-class breakdown
    class_pair_counts = defaultdict(int)
    for p in all_pairs:
        key = (
            RSUD_CLASSES[p["occluded_class"]],
            RSUD_CLASSES[p["occluder_class"]]
        )
        class_pair_counts[key] += 1

    print("\nPer-class-pair breakdown (occluded → occluder):")
    for (occ, occder), cnt in sorted(
        class_pair_counts.items(), key=lambda x: -x[1]
    ):
        print(f"  {occ:20s} → {occder:20s} : {cnt}")

    with open(args.output, "w") as f:
        json.dump(all_pairs, f, indent=2)

    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()