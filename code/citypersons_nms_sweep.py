"""
nms_iou_sweep.py
==================
Sweeps post-hoc class-agnostic NMS over a range of IoU thresholds applied
to `updated`'s predictions, and for each threshold reports:

  - MR^-2 per subset (via the same official CityPersons eval code used in
    citypersons_mr_eval_real_official.py / citypersons_official_wrapper.py)
  - Heavy_Occlusion Matched-GT count (recall) at score-thresh 0.1, so you
    can see the recall/precision trade-off directly alongside MR^-2,
    rather than picking a threshold blind.

Produces a table + a dual-axis plot (Heavy_Occlusion MR^-2 vs IoU
threshold, with Matched-GT count on a secondary axis) to find the elbow.

Requires citypersons_official_wrapper.py in the same directory (reused
from the real-official MR^-2 script).

Usage:
    python3 nms_iou_sweep.py \
        --gt val_instances_with_occlusion_visibility_ratio.json \
        --baseline results_citypersons_baseline_val.json \
        --updated results_citypersons_sampling_loss_val.json \
        --iou-thresholds 0.5 0.55 0.6 0.65 0.7 0.75 0.8 \
        --score-thresh-recall 0.1 \
        --output nms_sweep_report
"""

import json
import argparse
import io
import contextlib
import tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from citypersons_official_wrapper import (
    convert_gt_to_official_format, convert_preds_to_official_format, run_subset,
)

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

STANDARD_SUBSETS = {
    "Reasonable":       {"height_range": (50, 1e5 ** 2), "vis_range": (0.65, 1.01)},
    "Reasonable_small": {"height_range": (50, 75),       "vis_range": (0.65, 1.01)},
    "Heavy_Occlusion":  {"height_range": (50, 1e5 ** 2), "vis_range": (0.20, 0.65)},
    "All":              {"height_range": (20, 1e5 ** 2), "vis_range": (0.20, 1.01)},
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_target_category_ids(gt_full, target_names):
    target_names = {n.strip().lower() for n in target_names}
    return {c["id"] for c in gt_full.get("categories", [])
            if c.get("name", "").strip().lower() in target_names}


def iou(a, b):
    ax1, ay1, aw, ah = a
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = b
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, aw) * max(0.0, ah)
    b_area = max(0.0, bw) * max(0.0, bh)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def apply_nms(preds, iou_thresh, class_agnostic=True):
    grouped = defaultdict(list)
    for p in preds:
        key = p["image_id"] if class_agnostic else (p["image_id"], p["category_id"])
        grouped[key].append(p)

    kept = []
    for key, group in grouped.items():
        group_sorted = sorted(group, key=lambda p: -p["score"])
        suppressed = [False] * len(group_sorted)
        for i in range(len(group_sorted)):
            if suppressed[i]:
                continue
            kept.append(group_sorted[i])
            for j in range(i + 1, len(group_sorted)):
                if suppressed[j]:
                    continue
                if iou(group_sorted[i]["bbox"], group_sorted[j]["bbox"]) >= iou_thresh:
                    suppressed[j] = True
    return kept


def official_greedy_match_recall(gt_by_image_subset, preds, target_category_ids,
                                  score_thresh, height_range, vis_range):
    """Lightweight recall/matched-GT counter at a fixed score threshold, using
    the same greedy IoU>=0.5-real / IoA>=0.5-ignore matching rule as the
    diagnostic script. Used purely to report recall alongside MR^-2 for
    each NMS threshold -- MR^-2 itself always uses the FULL prediction
    list (no score-thresh) via the official code, unaffected by this."""
    h_lo, h_hi = height_range
    v_lo, v_hi = vis_range
    IOU_THRESHOLD = 0.5

    valid_ids = set(gt_by_image_subset.keys())
    dt_by_image = defaultdict(list)
    for p in preds:
        if p["image_id"] not in valid_ids or p["category_id"] not in target_category_ids:
            continue
        if p["score"] < score_thresh:
            continue
        dt_by_image[p["image_id"]].append(p)

    total_matched, total_positives = 0, 0
    for img_id, gts in gt_by_image_subset.items():
        # gts: list of {"bbox":, "ignore":}, already built for this height/vis window
        total_positives += sum(1 for g in gts if g["ignore"] == 0)
        dts = sorted(dt_by_image.get(img_id, []), key=lambda p: -p["score"])
        matched_gt = [False] * len(gts)
        gts_sorted_idx = sorted(range(len(gts)), key=lambda i: gts[i]["ignore"])
        for d in dts:
            best_ov, best_gi, best_kind = IOU_THRESHOLD, -2, -2
            for gi in gts_sorted_idx:
                g = gts[gi]
                if matched_gt[gi]:
                    continue
                if best_kind != -2 and g["ignore"] == 1:
                    break
                ov = iou(d["bbox"], g["bbox"])
                if ov < best_ov:
                    continue
                best_ov, best_gi = ov, gi
                best_kind = 1 if g["ignore"] == 0 else -1
            if best_gi != -2 and best_kind == 1:
                matched_gt[best_gi] = True
        total_matched += sum(matched_gt)
    return total_matched, total_positives


def build_gt_window(gt_full, target_category_ids, height_range, vis_range):
    h_lo, h_hi = height_range
    v_lo, v_hi = vis_range
    out = {img["id"]: [] for img in gt_full["images"]}
    for ann in gt_full["annotations"]:
        img_id = ann["image_id"]
        if img_id not in out:
            continue
        is_target = ann["category_id"] in target_category_ids
        vis = ann.get("attributes", {}).get("visibility_ratio")
        h = ann["bbox"][3]
        if (not is_target) or (vis is None):
            ignore = 1
        else:
            in_range = (h_lo <= h <= h_hi) and (v_lo <= vis <= v_hi)
            ignore = 0 if in_range else 1
        out[img_id].append({"bbox": ann["bbox"], "ignore": ignore})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--updated", required=True)
    ap.add_argument("--target-categories", nargs="+",
                     default=["pedestrian", "rider", "sitting person"])
    ap.add_argument("--iou-thresholds", type=float, nargs="+",
                     default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
    ap.add_argument("--score-thresh-recall", type=float, default=0.1,
                     help="Score threshold used ONLY for the recall/matched-GT "
                          "side metric, not for MR^-2 itself.")
    ap.add_argument("--class-agnostic", action="store_true", default=True)
    ap.add_argument("--output", default="nms_sweep_report")
    args = ap.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    gt_full = load_json(args.gt)
    target_category_ids = get_target_category_ids(gt_full, args.target_categories)
    b_preds = load_json(args.baseline)
    u_preds = load_json(args.updated)

    tmp_dir = Path(tempfile.mkdtemp(prefix="nms_sweep_"))
    gt_official_path = tmp_dir / "gt_official.json"
    convert_gt_to_official_format(gt_full, target_category_ids, gt_official_path)

    b_preds_official = convert_preds_to_official_format(b_preds, target_category_ids)

    # baseline MR^-2 + recall per subset (fixed reference line)
    print("Evaluating baseline (reference)...")
    baseline_mr2 = {}
    baseline_matched = {}
    gt_windows = {}
    for subset_name, sd in STANDARD_SUBSETS.items():
        gt_windows[subset_name] = build_gt_window(gt_full, target_category_ids,
                                                    sd["height_range"], sd["vis_range"])
        with contextlib.redirect_stdout(io.StringIO()):
            cocoGt = COCO(str(gt_official_path))
            res = run_subset(cocoGt, b_preds_official, sd["height_range"], sd["vis_range"], subset_name)
        baseline_mr2[subset_name] = res["mr2"] if res else float("nan")
        matched, positives = official_greedy_match_recall(
            gt_windows[subset_name], b_preds, target_category_ids,
            args.score_thresh_recall, sd["height_range"], sd["vis_range"])
        baseline_matched[subset_name] = (matched, positives)
        print(f"  {subset_name}: MR2={baseline_mr2[subset_name]*100:.2f}%  "
              f"matched={matched}/{positives}")

    sweep_results = []  # list of dicts per iou_thresh
    for iou_thresh in args.iou_thresholds:
        print(f"\n{'='*70}\nNMS IoU threshold = {iou_thresh}\n{'='*70}")
        u_preds_nms = apply_nms(u_preds, iou_thresh, class_agnostic=args.class_agnostic)
        print(f"  predictions: {len(u_preds):,} -> {len(u_preds_nms):,} "
              f"(-{len(u_preds) - len(u_preds_nms):,})")
        u_preds_nms_official = convert_preds_to_official_format(u_preds_nms, target_category_ids)

        row = {"iou_thresh": iou_thresh, "n_preds": len(u_preds_nms)}
        for subset_name, sd in STANDARD_SUBSETS.items():
            with contextlib.redirect_stdout(io.StringIO()):
                cocoGt = COCO(str(gt_official_path))
                res = run_subset(cocoGt, u_preds_nms_official, sd["height_range"], sd["vis_range"], subset_name)
            mr2 = res["mr2"] if res else float("nan")
            matched, positives = official_greedy_match_recall(
                gt_windows[subset_name], u_preds_nms, target_category_ids,
                args.score_thresh_recall, sd["height_range"], sd["vis_range"])
            row[f"{subset_name}_mr2"] = mr2
            row[f"{subset_name}_matched"] = matched
            row[f"{subset_name}_positives"] = positives
            print(f"  {subset_name}: MR2={mr2*100:.2f}%  "
                  f"(base={baseline_mr2[subset_name]*100:.2f}%, "
                  f"Δ={100*(mr2-baseline_mr2[subset_name]):+.2f}pp)  "
                  f"matched={matched}/{positives} "
                  f"(base_matched={baseline_matched[subset_name][0]})")
        sweep_results.append(row)

    # ── summary table ──
    print(f"\n{'='*100}")
    print("SWEEP SUMMARY")
    print(f"{'='*100}")
    header = ["IoU thr", "#preds", "HO MR2%", "HO ΔMR2(pp)", "HO matched",
              "Reasonable MR2%", "Reas ΔMR2(pp)", "All MR2%", "All ΔMR2(pp)"]
    rows = []
    for row in sweep_results:
        rows.append([
            row["iou_thresh"], f"{row['n_preds']:,}",
            f"{row['Heavy_Occlusion_mr2']*100:.2f}",
            f"{100*(row['Heavy_Occlusion_mr2'] - baseline_mr2['Heavy_Occlusion']):+.2f}",
            f"{row['Heavy_Occlusion_matched']}/{row['Heavy_Occlusion_positives']}",
            f"{row['Reasonable_mr2']*100:.2f}",
            f"{100*(row['Reasonable_mr2'] - baseline_mr2['Reasonable']):+.2f}",
            f"{row['All_mr2']*100:.2f}",
            f"{100*(row['All_mr2'] - baseline_mr2['All']):+.2f}",
        ])
    if HAS_TABULATE:
        print(tabulate(rows, headers=header, tablefmt="rounded_outline"))
    else:
        for r in rows:
            print(r)
    print(f"\n(Reference) baseline Heavy_Occlusion matched: "
          f"{baseline_matched['Heavy_Occlusion'][0]}/{baseline_matched['Heavy_Occlusion'][1]}")

    # ── plot: Heavy_Occlusion MR2 vs IoU thresh, with matched-GT on secondary axis ──
    fig, ax1 = plt.subplots(figsize=(9, 6))
    x = [r["iou_thresh"] for r in sweep_results]
    y_mr2 = [r["Heavy_Occlusion_mr2"] * 100 for r in sweep_results]
    y_matched = [r["Heavy_Occlusion_matched"] for r in sweep_results]

    color1 = "#8E44AD"
    ax1.plot(x, y_mr2, "o-", color=color1, lw=2.5, label="Updated+NMS Heavy_Occlusion MR⁻²")
    ax1.axhline(baseline_mr2["Heavy_Occlusion"] * 100, color="#2ECC71", ls="--", lw=2,
                label=f"Baseline MR⁻² ({baseline_mr2['Heavy_Occlusion']*100:.2f}%)")
    ax1.set_xlabel("NMS IoU threshold", fontsize=11)
    ax1.set_ylabel("Heavy_Occlusion MR⁻² (%)", color=color1, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "#E67E22"
    ax2.plot(x, y_matched, "s--", color=color2, lw=2, alpha=0.8, label="Matched GT (recall)")
    ax2.axhline(baseline_matched["Heavy_Occlusion"][0], color="#34495E", ls=":", lw=1.5,
                label=f"Baseline matched ({baseline_matched['Heavy_Occlusion'][0]})")
    ax2.set_ylabel("Heavy_Occlusion Matched GT count", color=color2, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", fontsize=9)
    plt.title("Heavy_Occlusion: NMS IoU threshold trade-off", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = Path(args.output) / "nms_sweep_heavy_occlusion.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved plot: {out_path}")

    # save raw results
    out_json = Path(args.output) / "nms_sweep_results.json"
    with open(out_json, "w") as f:
        json.dump({
            "baseline_mr2": baseline_mr2,
            "baseline_matched": baseline_matched,
            "sweep": sweep_results,
        }, f, indent=2, default=str)
    print(f"Saved raw results: {out_json}")
    print("\nDone.")


if __name__ == "__main__":
    main()