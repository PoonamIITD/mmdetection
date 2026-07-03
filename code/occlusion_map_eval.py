"""
Occlusion-wise mAP Evaluation

Computes per-category and overall mAP broken down by occlusion level
(light, moderate, severe) using GT COCO JSON and a COCO-format predictions file.

Requirements:
    pip install pycocotools tabulate

Usage:
    python occlusion_map_eval.py \
        --gt    merged_gt.json \
        --pred  predictions.json \
        --iou   0.5              # optional, default 0.5  (use 0.5:0.95 for COCO standard)

Arguments:
    --gt        Path to GT COCO JSON file (annotations must have attributes.occlusion_level)
    --pred      Path to predictions JSON file (COCO results format)
    --iou       IoU threshold(s): single float e.g. 0.5, or range "0.5:0.95:0.05"
    --output    (optional) Save summary CSV to this path
"""

import json
import copy
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    raise ImportError("Install pycocotools:  pip install pycocotools")

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


OCCLUSION_LEVELS = ["light", "moderate", "severe"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_iou_thresholds(iou_str: str):
    """Parse '0.5' → [0.5]  or  '0.5:0.95:0.05' → [0.50, 0.55, …, 0.95]"""
    parts = iou_str.split(":")
    if len(parts) == 1:
        return [float(parts[0])]
    elif len(parts) == 3:
        start, stop, step = float(parts[0]), float(parts[1]), float(parts[2])
        thresholds = list(np.arange(start, stop + step / 2, step))
        return [round(t, 4) for t in thresholds]
    else:
        raise ValueError("--iou must be a float '0.5' or range 'start:stop:step' e.g. '0.5:0.95:0.05'")


def build_subset_gt(full_gt: dict, allowed_ann_ids: set) -> dict:
    """
    Return a new GT dict containing only annotations whose id is in allowed_ann_ids.
    Images and categories are kept intact (COCOeval needs the full image list).
    """
    subset = copy.deepcopy(full_gt)
    subset["annotations"] = [
        a for a in full_gt["annotations"] if a["id"] in allowed_ann_ids
    ]
    return subset


def run_cocoeval(gt_dict: dict, pred_list: list, iou_thresholds: list,
                 valid_image_ids: set = None) -> dict:
    """
    Run COCOeval on a (possibly filtered) GT dict and predictions list.
    Returns a dict with per-category AP and overall mAP.
    """
    # Load GT
    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()

    # Filter predictions to images that exist in this GT subset
    if valid_image_ids is None:
        valid_image_ids = {img["id"] for img in gt_dict["images"]}

    # Only keep predictions whose image_id exists in GT images
    # AND whose category exists in GT annotations for this subset
    gt_image_ids = {img["id"] for img in gt_dict["images"]}
    filtered_preds = [p for p in pred_list if p["image_id"] in gt_image_ids]

    if not filtered_preds:
        return {}

    coco_dt = coco_gt.loadRes(filtered_preds)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.iouThrs = np.array(iou_thresholds)
    coco_eval.evaluate()
    coco_eval.accumulate()

    results = {}

    # Overall mAP (all categories)
    coco_eval.summarize()
    results["__all__"] = float(coco_eval.stats[0])  # AP @ IoU thresholds

    # Per-category AP
    cat_ids = coco_gt.getCatIds()
    cat_id_to_name = {c["id"]: c["name"] for c in gt_dict["categories"]}

    for cat_id in cat_ids:
        coco_eval.params.catIds = [cat_id]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        cat_name = cat_id_to_name.get(cat_id, str(cat_id))
        results[cat_name] = float(coco_eval.stats[0])

    # Reset catIds for safety
    coco_eval.params.catIds = cat_ids

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def evaluate(gt_path: str, pred_path: str, iou_str: str = "0.5", output_csv: str = None):

    print(f"\nLoading GT   : {gt_path}")
    gt_full = load_json(gt_path)

    print(f"Loading Pred : {pred_path}")
    pred_list = load_json(pred_path)
    if not isinstance(pred_list, list):
        raise ValueError("Predictions file must be a JSON array of prediction objects.")

    iou_thresholds = parse_iou_thresholds(iou_str)
    print(f"IoU thresholds: {iou_thresholds}")

    # ── 1. Partition annotation IDs by occlusion level ──────────────────────
    level_to_ann_ids = defaultdict(set)
    missing_occlusion = 0

    for ann in gt_full["annotations"]:
        level = (
            ann.get("attributes", {}).get("occlusion_level", "")
            .strip().lower()
        )
        if level in OCCLUSION_LEVELS:
            level_to_ann_ids[level].add(ann["id"])
        else:
            missing_occlusion += 1

    if missing_occlusion:
        print(f"\n  WARNING: {missing_occlusion} annotation(s) have no/unknown "
              f"occlusion_level and will be excluded from occlusion-wise evaluation.")

    for lvl in OCCLUSION_LEVELS:
        print(f"  Annotations with occlusion='{lvl}': {len(level_to_ann_ids[lvl])}")

    # ── 2. Evaluate: ALL (baseline) ─────────────────────────────────────────
    print("\n" + "="*60)
    print("Evaluating: ALL annotations (baseline)")
    print("="*60)
    results_all = run_cocoeval(gt_full, pred_list, iou_thresholds)

    # ── 3. Evaluate per occlusion level ─────────────────────────────────────
    results_by_level = {}
    for level in OCCLUSION_LEVELS:
        ann_ids = level_to_ann_ids[level]
        if not ann_ids:
            print(f"\nSkipping '{level}' — no annotations found.")
            results_by_level[level] = {}
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: occlusion = '{level}'  ({len(ann_ids)} annotations)")
        print("="*60)

        subset_gt = build_subset_gt(gt_full, ann_ids)
        results_by_level[level] = run_cocoeval(subset_gt, pred_list, iou_thresholds)

    # ── 4. Print Summary Table ───────────────────────────────────────────────
    # Collect all category names
    all_cats = sorted(
        set(k for r in [results_all, *results_by_level.values()]
            for k in r if k != "__all__")
    )

    headers = ["Category", "ALL"] + [lvl.capitalize() for lvl in OCCLUSION_LEVELS]
    rows = []

    # Overall row first
    row = ["** mAP (all cats) **"]
    row.append(f"{results_all.get('__all__', float('nan')):.4f}")
    for lvl in OCCLUSION_LEVELS:
        val = results_by_level[lvl].get("__all__", float("nan"))
        row.append(f"{val:.4f}" if not np.isnan(val) else "  —  ")
    rows.append(row)

    # Per-category rows
    for cat in all_cats:
        row = [cat]
        row.append(f"{results_all.get(cat, float('nan')):.4f}")
        for lvl in OCCLUSION_LEVELS:
            val = results_by_level[lvl].get(cat, float("nan"))
            row.append(f"{val:.4f}" if not np.isnan(val) else "  —  ")
        rows.append(row)

    print("\n\n" + "="*60)
    print(f"  OCCLUSION-WISE mAP SUMMARY  (IoU: {iou_str})")
    print("="*60)
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    else:
        # Fallback plain-text table
        col_w = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
        fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
        print(fmt.format(*headers))
        print("  ".join("-" * w for w in col_w))
        for row in rows:
            print(fmt.format(*row))

    # ── 5. Optional CSV export ───────────────────────────────────────────────
    if output_csv:
        import csv
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        print(f"\nSummary saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute occlusion-wise mAP from GT COCO JSON + predictions JSON."
    )
    parser.add_argument("--gt",     required=True, help="Path to GT COCO JSON file")
    parser.add_argument("--pred",   required=True, help="Path to predictions JSON file")
    parser.add_argument("--iou",    default="0.5",
                        help="IoU threshold(s): '0.5' or '0.5:0.95:0.05' (default: 0.5)")
    parser.add_argument("--output", default=None,  help="Optional: save summary to CSV")
    args = parser.parse_args()

    for path in (args.gt, args.pred):
        if not Path(path).exists():
            raise FileNotFoundError(f"File not found: {path}")

    evaluate(args.gt, args.pred, args.iou, args.output)


if __name__ == "__main__":
    main()