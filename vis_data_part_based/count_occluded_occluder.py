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
    """
    Match GT objects to query predictions.

    Strategy:
      1. Collect all same-class queries above IoU threshold
      2. Keep top-K IoU candidates
      3. Among near-best-IoU candidates, select highest confidence

    This avoids selecting:
      - high-IoU but low-confidence noisy queries
      - unstable assignments
    """

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

        # --------------------------------------------------
        # collect all valid candidate queries
        # --------------------------------------------------
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

        # --------------------------------------------------
        # sort by IoU descending
        # --------------------------------------------------
        candidates = sorted(
            candidates,
            key=lambda x: x["iou"],
            reverse=True
        )

        # keep top-K IoU candidates
        candidates = candidates[:topk_iou_candidates]

        # --------------------------------------------------
        # keep candidates close to best IoU
        # --------------------------------------------------
        best_iou = candidates[0]["iou"]

        near_best = [
            c for c in candidates
            if c["iou"] >= (best_iou - iou_eps)
        ]

        # --------------------------------------------------
        # among near-best IoU → pick highest confidence
        # --------------------------------------------------
        best_candidate = max(
            near_best,
            key=lambda x: x["score"]
        )

        best_q = best_candidate["query_id"]

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
# OCCLUSION ANALYSIS  (fixed)
# ============================================================

def analyze_occlusion_pairs(
    matched_queries,
    occ_ratio_threshold=0.7,
    # occluded object confidence range
    occ_score_low=0.6,
    occ_score_high=0.8,
    # occluder object confidence range
    occder_score_low=0.88,
    occder_score_high=1.01,       # 1.01 so >= 0.88 is inclusive of 1.0
    same_class_only=None
):
    """
    Find (occluded, occluder) pairs where:

      1.  They correspond to DIFFERENT GT annotation IDs.
      2.  The occluder's box covers >= occ_ratio_threshold of the
          occluded object's area.
      3.  occluded  confidence ∈ [occ_score_low,   occ_score_high]
      4.  occluder  confidence ∈ [occder_score_low, occder_score_high]
      5.  same_class_only controls class-pair filtering.

    Note: i is occluded, j is occluder  (ordered, so i≠j is not symmetric).
    """

    pairs = []
    N = len(matched_queries)

    for i in range(N):
        for j in range(N):

            if i == j:
                continue

            obj_i = matched_queries[i]   # candidate occluded
            obj_j = matched_queries[j]   # candidate occluder

            # --------------------------------------------------
            # CRITICAL: must be different GT objects
            # --------------------------------------------------
            if obj_i["gt_ann_id"] == obj_j["gt_ann_id"]:
                continue

            cls_i = obj_i["gt_class"]
            cls_j = obj_j["gt_class"]

            # --------------------------------------------------
            # class-pair filter
            # --------------------------------------------------
            if same_class_only is True  and cls_i != cls_j:
                continue
            if same_class_only is False and cls_i == cls_j:
                continue

            box_i = obj_i["gt_box"]
            box_j = obj_j["gt_box"]

            occ_ratio = compute_occlusion_ratio(box_i, box_j)

            score_i = obj_i["score"]   # occluded confidence
            score_j = obj_j["score"]   # occluder confidence

            # --------------------------------------------------
            # occlusion geometry check
            # --------------------------------------------------
            if occ_ratio < occ_ratio_threshold:
                continue

            # --------------------------------------------------
            # FIX: range-based confidence checks
            # occluded  ∈ [0.60, 0.80]
            # occluder  ∈ [0.88, 1.00]
            # --------------------------------------------------
            if not (occ_score_low <= score_i <= occ_score_high):
                continue

            if not (occder_score_low <= score_j <= occder_score_high):
                continue

            pairs.append({
                "occluded_query":      obj_i["query_id"],
                "occluder_query":      obj_j["query_id"],

                "occluded_gt_ann_id":  obj_i["gt_ann_id"],
                "occluder_gt_ann_id":  obj_j["gt_ann_id"],

                "occluded_class":      cls_i,
                "occluder_class":      cls_j,

                "occluded_score":      score_i,
                "occluder_score":      score_j,

                "occlusion_ratio":     occ_ratio,

                "same_class":          cls_i == cls_j,
            })

    return pairs


# ============================================================
# ADVANCED ANALYSIS
# ============================================================

def confidence_trajectory(cls_scores, q_idx):
    scores = cls_scores[:, q_idx]
    max_scores, _ = scores.max(dim=1)
    return max_scores.numpy()


def analyze_decoder_drop(data, q_idx):

    traj = confidence_trajectory(
        data["cls_scores"],
        q_idx
    )

    traj = traj.tolist()

    return {
        "trajectory": traj,

        "start_conf": float(traj[0]),
        "end_conf":   float(traj[-1]),

        "max_conf":   float(max(traj)),
        "min_conf":   float(min(traj)),

        "delta":      float(traj[-1] - traj[0]),

        # useful diagnostics
        "argmax_layer": int(np.argmax(traj)),
        "argmin_layer": int(np.argmin(traj)),
    }


# ============================================================
# MAIN FILE PROCESSOR
# ============================================================

def process_pt_file(pt_file, coco_images, coco_anns, args):

    data = torch.load(pt_file)

    img_name = Path(data["img_path"]).name

    # find image_id
    image_id = next(
        (k for k, v in coco_images.items() if v["file_name"] == img_name),
        None
    )
    if image_id is None:
        return []

    gt_annotations = coco_anns[image_id]
    img_w = coco_images[image_id]["width"]
    img_h = coco_images[image_id]["height"]

    # match GT → queries
    matched_queries = match_gt_to_queries(
        data,
        gt_annotations,
        img_w,
        img_h,
        iou_threshold=args.match_iou,
        layer=args.layer
    )

    if len(matched_queries) < 2:
        return []

    # class-pair mode
    same_class_only = {"same": True, "different": False}.get(args.mode, None)

    # occlusion pairs with FIXED range conditions
    pairs = analyze_occlusion_pairs(
        matched_queries,
        occ_ratio_threshold=args.occlusion_ratio,
        occ_score_low=args.occ_score_low,
        occ_score_high=args.occ_score_high,
        occder_score_low=args.occder_score_low,
        occder_score_high=args.occder_score_high,
        same_class_only=same_class_only
    )

    # decoder trajectory per pair
    for p in pairs:

        occ_stats = analyze_decoder_drop(data, p["occluded_query"])

        occder_stats = analyze_decoder_drop(data, p["occluder_query"])

        # ------------------------------------------------
        # occluded query trajectory
        # ------------------------------------------------
        p["occluded_scores_per_layer"] = occ_stats["trajectory"]

        p["occluded_start_conf"] = occ_stats["start_conf"]

        p["occluded_end_conf"] = occ_stats["end_conf"]

        p["occluded_max_conf"] = occ_stats["max_conf"]

        p["occluded_min_conf"] = occ_stats["min_conf"]

        p["occluded_delta"] = occ_stats["delta"]

        p["occluded_peak_layer"] = occ_stats["argmax_layer"]

        # ------------------------------------------------
        # occluder query trajectory
        # ------------------------------------------------
        p["occluder_scores_per_layer"] = occder_stats["trajectory"]

        p["occluder_start_conf"] = occder_stats["start_conf"]

        p["occluder_end_conf"] = occder_stats["end_conf"]

        p["occluder_max_conf"] = occder_stats["max_conf"]

        p["occluder_min_conf"] = occder_stats["min_conf"]

        p["occluder_delta"] = occder_stats["delta"]

        p["occluder_peak_layer"] = occder_stats["argmax_layer"]

        p["image_name"] = img_name

    return pairs


# ============================================================
# ENTRY
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Occlusion pair analysis for DETR-family models on RSUD."
    )

    parser.add_argument("--pt_dir",  type=str, required=True)
    parser.add_argument("--gt_json", type=str, required=True)
    parser.add_argument("--output",  type=str, default="occlusion_results.json")

    # ── geometry ──────────────────────────────────────────
    parser.add_argument(
        "--occlusion_ratio", type=float, default=0.5,
        help="Min fraction of occluded object area covered by occluder box."
    )
    parser.add_argument(
        "--match_iou", type=float, default=0.3,
        help="Min IoU to match a GT box to a query prediction."
    )

    # ── occluded object confidence range ──────────────────
    parser.add_argument(
        "--occ_score_low",  type=float, default=0.6,
        help="Lower bound of occluded object confidence (inclusive)."
    )
    parser.add_argument(
        "--occ_score_high", type=float, default=0.8,
        help="Upper bound of occluded object confidence (inclusive)."
    )

    # ── occluder object confidence range ──────────────────
    parser.add_argument(
        "--occder_score_low",  type=float, default=0.75,
        help="Lower bound of occluder object confidence (inclusive)."
    )
    parser.add_argument(
        "--occder_score_high", type=float, default=1.01,
        help="Upper bound of occluder object confidence (use 1.01 to include 1.0)."
    )

    # ── class-pair mode ───────────────────────────────────
    parser.add_argument(
        "--mode", type=str, default="same",
        choices=["same", "different", "all"],
        help=(
            "same      → only same-class pairs  (your primary use case)\n"
            "different → only cross-class pairs\n"
            "all       → no class filter"
        )
    )

    parser.add_argument(
        "--layer", type=int, default=-1,
        help="Decoder layer index to read predictions from (-1 = last)."
    )

    args = parser.parse_args()

    # load GT
    coco_images, coco_anns = load_coco_gt(args.gt_json)

    pt_files  = sorted(Path(args.pt_dir).glob("*.pt"))
    all_pairs = []

    for pt_file in pt_files:
        print(f"[INFO] processing {pt_file.name}")
        pairs = process_pt_file(pt_file, coco_images, coco_anns, args)
        all_pairs.extend(pairs)

    # summary
    same_count = sum(p["same_class"] for p in all_pairs)
    diff_count = len(all_pairs) - same_count

    print("\n================================================")
    print("SUMMARY")
    print("================================================")
    print(f"Total pairs        : {len(all_pairs)}")
    print(f"Same-class pairs   : {same_count}")
    print(f"Diff-class pairs   : {diff_count}")

    # breakdown by class
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