"""
fp_clustering_diagnosis.py
============================

For the Heavy_Occlusion subset, this script:

  1. Recomputes TP/FP with bbox + image_id kept (not just aggregate counts),
     for both baseline and updated, using the same official matching rules
     as CityPersons_mr_eval.py / cityperson_tp_fp_diagnosis.py.

  2. Splits UPDATED's FPs into:
       - "persistent" : a baseline FP exists at IoU>=0.5 in the same image
                         (both models produce it -- not new behavior)
       - "new"        : no such baseline FP nearby (this is what changed)

  3. For every "new" FP, checks its local GT neighborhood using a
     union-find clustering over ALL target-category GT boxes in that image
     (any occlusion level -- since the occluder is often another, more
     visible pedestrian):
       - "crowd"        : nearest overlapping GT belongs to a cluster of
                           size >= 2 (i.e. sits among overlapping people)
       - "isolated"      : nearest overlapping GT is in a cluster of size 1
       - "no_nearby_gt"  : no GT overlaps this FP at all (pure background
                           hallucination -- a different failure mode)

If "crowd" dominates the new FPs, that supports "queries locking onto
overlapping local regions in a crowd" over generic hallucination.

Usage:
    python3 fp_clustering_diagnosis.py \
        --gt val_instances_with_occlusion_visibility_ratio.json \
        --baseline results_citypersons_final_val.json \
        --updated results_citypersons_sampling_loss_val.json \
        --score-thresh 0.10
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

HEIGHT_RANGE = (50, 1e10)
VIS_RANGE = (0.20, 0.65)       # Heavy_Occlusion window
IOU_THRESHOLD = 0.5
EXP_FILTER = 1.25
NEW_FP_MATCH_IOU = 0.5         # FP-vs-FP: same image, IoU >= this -> "persistent"
FP_GT_OVERLAP_IOU = 0.02       # any overlap above this counts as "near a GT"
CLUSTER_IOU_THRESH = 0.05      # GT-GT IoU above this -> same crowd cluster


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def box_area(b):
    return max(0.0, b[2]) * max(0.0, b[3])


def iou(a, b):
    ax1, ay1, aw, ah = a
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = b
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def ioa(det_box, gt_box):
    dx1, dy1, dw, dh = det_box
    dx2, dy2 = dx1 + dw, dy1 + dh
    gx1, gy1, gw, gh = gt_box
    gx2, gy2 = gx1 + gw, gy1 + gh
    ix1, iy1 = max(dx1, gx1), max(dy1, gy1)
    ix2, iy2 = min(dx2, gx2), min(dy2, gy2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    det_area = box_area(det_box)
    return inter / det_area if det_area > 0 else 0.0


def get_target_category_ids(gt_full, names):
    names = {n.strip().lower() for n in names}
    return {c["id"] for c in gt_full.get("categories", [])
            if c.get("name", "").strip().lower() in names}


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build_gt_structures(gt_full, target_category_ids):
    """
    Returns:
      gt_by_image_subset : img_id -> [{'bbox':, 'ignore':}]  (Heavy_Occlusion
                            ignore flags, for the official TP/FP matching)
      gt_all_by_image     : img_id -> [bbox, ...]  ALL target-category GT
                            (any occlusion level) used purely for crowd
                            clustering context.
    """
    h_lo, h_hi = HEIGHT_RANGE
    v_lo, v_hi = VIS_RANGE

    gt_by_image_subset = {img["id"]: [] for img in gt_full["images"]}
    gt_all_by_image = {img["id"]: [] for img in gt_full["images"]}

    for ann in gt_full["annotations"]:
        img_id = ann["image_id"]
        if img_id not in gt_by_image_subset:
            continue
        is_target = ann["category_id"] in target_category_ids
        vis = ann.get("attributes", {}).get("visibility_ratio")
        h = ann["bbox"][3]

        if is_target:
            gt_all_by_image[img_id].append(ann["bbox"])

        if (not is_target) or (vis is None):
            ignore = 1
        else:
            in_range = (h_lo <= h <= h_hi) and (v_lo <= vis <= v_hi)
            ignore = 0 if in_range else 1
        gt_by_image_subset[img_id].append({"bbox": ann["bbox"], "ignore": ignore})

    for anns in gt_by_image_subset.values():
        anns.sort(key=lambda a: a["ignore"])

    return gt_by_image_subset, gt_all_by_image


def match_and_collect_fps(gt_by_image_subset, pred_list, target_category_ids,
                           score_thresh):
    """Official per-image greedy matcher, restricted to score>=thresh preds.
    Returns fp_records: list of dicts {img_id, bbox, score}."""
    h_lo, h_hi = HEIGHT_RANGE
    valid_ids = set(gt_by_image_subset.keys())
    dt_by_image = defaultdict(list)
    for p in pred_list:
        if p["image_id"] not in valid_ids or p["category_id"] not in target_category_ids:
            continue
        if p["score"] < score_thresh:
            continue
        dh = p["bbox"][3]
        if not (h_lo / EXP_FILTER <= dh < h_hi * EXP_FILTER):
            continue
        dt_by_image[p["image_id"]].append(p)

    fp_records = []
    for img_id, gts in gt_by_image_subset.items():
        dts = sorted(dt_by_image.get(img_id, []), key=lambda p: -p["score"])
        matched_gt = [False] * len(gts)
        for d in dts:
            best_ov, best_gi, best_kind = IOU_THRESHOLD, -2, -2
            for gi, g in enumerate(gts):
                if matched_gt[gi]:
                    continue
                if best_kind != -2 and g["ignore"] == 1:
                    break
                ov = iou(d["bbox"], g["bbox"]) if g["ignore"] == 0 else ioa(d["bbox"], g["bbox"])
                if ov < best_ov:
                    continue
                best_ov, best_gi = ov, gi
                best_kind = 1 if g["ignore"] == 0 else -1
            if best_gi == -2:
                fp_records.append({"img_id": img_id, "bbox": d["bbox"], "score": d["score"]})
            elif best_kind == 1:
                matched_gt[best_gi] = True
    return fp_records


def cluster_gt(gt_boxes, iou_thresh):
    """Union-find clustering of GT boxes by pairwise IoU. Returns list of
    cluster sizes, one per GT box (same order as gt_boxes)."""
    n = len(gt_boxes)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if iou(gt_boxes[i], gt_boxes[j]) > iou_thresh:
                uf.union(i, j)
    root_counts = defaultdict(int)
    roots = [uf.find(i) for i in range(n)]
    for r in roots:
        root_counts[r] += 1
    return [root_counts[r] for r in roots]


def classify_fp_neighborhood(fp_bbox, gt_boxes, cluster_sizes):
    if not gt_boxes:
        return "no_nearby_gt", 0.0, 0
    ious = [iou(fp_bbox, g) for g in gt_boxes]
    best_idx = int(np.argmax(ious))
    best_iou = ious[best_idx]
    if best_iou < FP_GT_OVERLAP_IOU:
        return "no_nearby_gt", best_iou, 0
    csize = cluster_sizes[best_idx]
    return ("crowd" if csize >= 2 else "isolated"), best_iou, csize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--updated", required=True)
    ap.add_argument("--target-categories", nargs="+",
                     default=["pedestrian", "rider", "sitting person"])
    ap.add_argument("--score-thresh", type=float, default=0.10)
    ap.add_argument("--top-k-examples", type=int, default=20,
                     help="Print the K highest-confidence 'new' FPs for follow-up visualization.")
    args = ap.parse_args()

    for p in (args.gt, args.baseline, args.updated):
        if not Path(p).exists():
            raise FileNotFoundError(p)

    print(f"Loading GT       : {args.gt}")
    gt_full = load_json(args.gt)
    target_category_ids = get_target_category_ids(gt_full, args.target_categories)
    if not target_category_ids:
        raise ValueError("No matching target categories found in GT.")

    print(f"Loading Baseline : {args.baseline}")
    b_preds = load_json(args.baseline)
    print(f"Loading Updated  : {args.updated}")
    u_preds = load_json(args.updated)

    gt_by_image_subset, gt_all_by_image = build_gt_structures(gt_full, target_category_ids)

    print("Matching baseline FPs (Heavy_Occlusion subset)...")
    b_fps = match_and_collect_fps(gt_by_image_subset, b_preds, target_category_ids, args.score_thresh)
    print("Matching updated FPs (Heavy_Occlusion subset)...")
    u_fps = match_and_collect_fps(gt_by_image_subset, u_preds, target_category_ids, args.score_thresh)

    b_fps_by_img = defaultdict(list)
    for r in b_fps:
        b_fps_by_img[r["img_id"]].append(r["bbox"])

    # classify updated FPs as persistent vs new
    new_fps, persistent_fps = [], []
    for r in u_fps:
        baseline_boxes = b_fps_by_img.get(r["img_id"], [])
        is_persistent = any(iou(r["bbox"], bb) >= NEW_FP_MATCH_IOU for bb in baseline_boxes)
        (persistent_fps if is_persistent else new_fps).append(r)

    print(f"\nUpdated Heavy_Occlusion FPs: {len(u_fps):,} total "
          f"({len(persistent_fps):,} persistent, {len(new_fps):,} new)")

    # precompute GT clusters per image once
    cluster_cache = {}

    def get_clusters(img_id):
        if img_id not in cluster_cache:
            boxes = gt_all_by_image.get(img_id, [])
            sizes = cluster_gt(boxes, CLUSTER_IOU_THRESH)
            cluster_cache[img_id] = (boxes, sizes)
        return cluster_cache[img_id]

    counts = defaultdict(int)
    detailed = []
    for r in new_fps:
        boxes, sizes = get_clusters(r["img_id"])
        label, best_iou, csize = classify_fp_neighborhood(r["bbox"], boxes, sizes)
        counts[label] += 1
        detailed.append({**r, "label": label, "nearest_gt_iou": round(best_iou, 3),
                          "cluster_size": csize})

    print("\n" + "=" * 78)
    print("Neighborhood breakdown of NEW Heavy_Occlusion FPs")
    print("=" * 78)
    total_new = len(new_fps)
    rows = []
    for label in ("crowd", "isolated", "no_nearby_gt"):
        n = counts.get(label, 0)
        pct = 100.0 * n / total_new if total_new else 0.0
        rows.append([label, n, f"{pct:.1f}%"])
    header = ["Category", "Count", "% of new FPs"]
    if HAS_TABULATE:
        print(tabulate(rows, headers=header, tablefmt="rounded_outline"))
    else:
        for row in rows:
            print(f"  {row[0]:<15} {row[1]:>6}  {row[2]:>8}")

    print("\nInterpretation:")
    print("  - 'crowd' dominant     -> new FPs sit among overlapping/clustered GT people:")
    print("                            supports 'queries locking onto overlapping local")
    print("                            regions in a crowd' (duplicate-in-crowd hypothesis).")
    print("  - 'isolated' dominant  -> new FPs appear near single occluded people with no")
    print("                            crowding: points more toward general precision loss")
    print("                            on hard occluded instances, not a crowd-specific issue.")
    print("  - 'no_nearby_gt' dominant -> new FPs are in empty background/occluder regions:")
    print("                            genuine hallucination, unrelated to crowding.")

    # top-K highest confidence new FPs for follow-up visualization
    detailed.sort(key=lambda r: -r["score"])
    print(f"\n{'='*78}\nTop {args.top_k_examples} highest-confidence NEW FPs (for visualization)\n{'='*78}")
    ex_rows = [[d["img_id"], [round(v, 1) for v in d["bbox"]], round(d["score"], 3),
                d["label"], d["nearest_gt_iou"], d["cluster_size"]]
               for d in detailed[:args.top_k_examples]]
    ex_header = ["img_id", "bbox", "score", "label", "nearest_gt_iou", "cluster_size"]
    if HAS_TABULATE:
        print(tabulate(ex_rows, headers=ex_header, tablefmt="rounded_outline"))
    else:
        for row in ex_rows:
            print(row)

    out_path = "new_fp_diagnosis.json"
    with open(out_path, "w") as f:
        json.dump(detailed, f, indent=2)
    print(f"\nFull per-FP detail written to: {out_path}")


if __name__ == "__main__":
    main()