"""
CityPersons subset-aware TP / FP / FN diagnostics.

Same idea as your RSUD pycocotools-based TP/FP script, but:
  - uses the CityPersons STANDARD subset definitions (Reasonable,
    Reasonable_small, Heavy_Occlusion, All) via height + visibility_ratio
    windows, instead of a single occlusion-level string match
  - uses a safe, verified-correct per-image greedy matcher (no COCOeval
    row-index bug -- see conversation history) instead of tapping into
    COCOeval.evalImgs internals
  - reports TP/FP *and* FN (missed GT), plus recall, per subset per model,
    so you can see e.g. "updated model has more FPs but not more TPs in
    Heavy_Occlusion" instead of only the aggregate MR^-2 number

Matching rules (identical to eval_MR_multisetup.py / the official algorithm):
  - IoU >= 0.5 required for a real ("non-ignore") GT match
  - IoA >= 0.5 (intersection / detection-area) for an ignore-region GT match
  - a detection matched to an ignore GT is dropped entirely (neither TP nor FP)
  - DT height cull: h_lo/expFilter <= dt_height < h_hi*expFilter (expFilter=1.25)
  - score_thresh only filters what goes into the printed averages, exactly
    like SCORE_THRESH in your RSUD script -- it does NOT affect matching
    itself (matching runs on all conf>=0 detections, same as the official
    benchmark), so recall/FN numbers stay faithful to the real MR^-2 run.

Usage:
    python citypersons_tp_fp_diagnostics.py \
        --gt        val_instances_with_occlusion_visibility_ratio.json \
        --baseline  results_citypersons_final_val.json \
        --updated   results_citypersons_sampling_loss_val.json \
        --target-categories pedestrian rider "sitting person" \
        --score-thresh 0.10 \
        --subsets standard
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# ── Official CityPersons standard subset windows ─────────────────────────────
STANDARD_SUBSETS = {
    "Reasonable":       {"height_range": (50,  1e10), "vis_range": (0.65, 1e10)},
    "Reasonable_small": {"height_range": (50,  75),   "vis_range": (0.65, 1e10)},
    "Heavy_Occlusion":  {"height_range": (50,  1e10), "vis_range": (0.20, 0.65)},
    "All":              {"height_range": (20,  1e10), "vis_range": (0.20, 1e10)},
}
IOU_THRESHOLD = 0.5
EXP_FILTER = 1.25


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_target_category_ids(gt_full, target_names):
    target_names = {n.strip().lower() for n in target_names}
    return {c["id"] for c in gt_full.get("categories", [])
            if c.get("name", "").strip().lower() in target_names}


def box_area(box):
    return max(0.0, box[2]) * max(0.0, box[3])


def iou(box_a, box_b):
    ax1, ay1, aw, ah = box_a
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = box_b
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = box_area(box_a) + box_area(box_b) - inter
    return inter / union if union > 0 else 0.0


def ioa(det_box, gt_box):
    """Intersection over the DETECTION's area (official ignore-region rule)."""
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


# ── Core: per-image matching + TP/FP/FN record collection ───────────────────

def evaluate_subset_with_diagnostics(gt_full, pred_list, target_category_ids,
                                      subset_def, iou_thr=IOU_THRESHOLD,
                                      exp_filter=EXP_FILTER, score_thresh=0.10):
    h_lo, h_hi = subset_def["height_range"]
    v_lo, v_hi = subset_def["vis_range"]

    # ---- build per-image GT with ignore flags (identical rule to MR eval) ----
    gt_by_image = {img["id"]: [] for img in gt_full["images"]}
    for ann in gt_full["annotations"]:
        img_id = ann["image_id"]
        if img_id not in gt_by_image:
            continue
        is_target_cat = ann["category_id"] in target_category_ids
        h = ann["bbox"][3]
        vis = ann.get("attributes", {}).get("visibility_ratio")
        if (not is_target_cat) or (vis is None):
            ignore = 1
        else:
            in_range = (h_lo <= h <= h_hi) and (v_lo <= vis <= v_hi)
            ignore = 0 if in_range else 1
        gt_by_image[img_id].append({"bbox": ann["bbox"], "ignore": ignore})

    total_positives = sum(1 for anns in gt_by_image.values() for a in anns if a["ignore"] == 0)
    num_images = len(gt_by_image)
    if total_positives == 0:
        return None

    for anns in gt_by_image.values():
        anns.sort(key=lambda a: a["ignore"])  # real GT first, ignore GT last

    # ---- bucket predictions by image, apply DT height cull ----
    valid_image_ids = set(gt_by_image.keys())
    dt_by_image = defaultdict(list)
    for p in pred_list:
        if p["image_id"] not in valid_image_ids or p["category_id"] not in target_category_ids:
            continue
        dh = p["bbox"][3]
        if not (h_lo / exp_filter <= dh < h_hi * exp_filter):
            continue
        dt_by_image[p["image_id"]].append(p)

    # ---- per-image greedy matching (official rule, score-sorted) ----
    tp_records, fp_records = [], []  # each: {'score':, 'iou':}
    matched_gt_count = 0

    for img_id, gts in gt_by_image.items():
        valid_gt_boxes = [g["bbox"] for g in gts if g["ignore"] == 0]
        dts = sorted(dt_by_image.get(img_id, []), key=lambda p: -p["score"])
        matched_gt = [False] * len(gts)

        for d in dts:
            best_ov, best_gi, best_kind = iou_thr, -2, -2
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
                # unmatched -> FP; report max IoU against any *valid* (real) GT
                # in this image, for diagnostic purposes (mirrors your RSUD script)
                max_iou = 0.0
                for gt_box in valid_gt_boxes:
                    v = iou(d["bbox"], gt_box)
                    if v > max_iou:
                        max_iou = v
                fp_records.append({"score": d["score"], "iou": max_iou})
            elif best_kind == 1:
                matched_gt[best_gi] = True
                matched_gt_count += 1
                tp_iou = iou(d["bbox"], gts[best_gi]["bbox"])
                tp_records.append({"score": d["score"], "iou": tp_iou})
            # else: matched an ignore-region GT -> dropped entirely (neither TP nor FP)

    fn_count = total_positives - matched_gt_count
    recall = matched_gt_count / total_positives if total_positives else 0.0

    # ---- apply score_thresh only for the printed averages ----
    tp_f = [r for r in tp_records if r["score"] >= score_thresh]
    fp_f = [r for r in fp_records if r["score"] >= score_thresh]

    return {
        "total_positives": total_positives,
        "num_images": num_images,
        "matched_gt_count": matched_gt_count,
        "fn_count": fn_count,
        "recall": recall,
        "TP_Count": len(tp_f),
        "TP_Conf": float(np.mean([r["score"] for r in tp_f])) if tp_f else 0.0,
        "TP_IoU": float(np.mean([r["iou"] for r in tp_f])) if tp_f else 0.0,
        "FP_Count": len(fp_f),
        "FP_Conf": float(np.mean([r["score"] for r in fp_f])) if fp_f else 0.0,
        "FP_IoU": float(np.mean([r["iou"] for r in fp_f])) if fp_f else 0.0,
    }


# ── Reporting ─────────────────────────────────────────────────────────────

def print_subset_report(subset_name, b_stats, c_stats):
    print(f"\n[{subset_name}]  (#positives={b_stats['total_positives']:,}, "
          f"#images={b_stats['num_images']:,})")
    print("-" * 90)
    rows = []
    keys = [
        ("Recall", "recall", "{:.4f}"),
        ("Matched GT (TP-GT)", "matched_gt_count", "{:,}"),
        ("Missed GT (FN)", "fn_count", "{:,}"),
        ("TP Count", "TP_Count", "{:,}"),
        ("Mean TP Confidence", "TP_Conf", "{:.4f}"),
        ("Mean TP IoU", "TP_IoU", "{:.4f}"),
        ("FP Count", "FP_Count", "{:,}"),
        ("Mean FP Confidence", "FP_Conf", "{:.4f}"),
        ("Mean FP max-IoU", "FP_IoU", "{:.4f}"),
    ]
    for label, key, fmt in keys:
        b_val, c_val = b_stats[key], c_stats[key]
        diff = c_val - b_val
        diff_str = (f"+{fmt.format(diff)}" if diff > 0 else fmt.format(diff))
        rows.append([label, fmt.format(b_val), fmt.format(c_val), diff_str])

    header = ["METRIC", "BASELINE", "UPDATED", "DELTA"]
    if HAS_TABULATE:
        print(tabulate(rows, headers=header, tablefmt="rounded_outline"))
    else:
        col_w = [max(len(h), max(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
        fmt_row = "  ".join(f"{{:<{w}}}" for w in col_w)
        print(fmt_row.format(*header))
        print("  ".join("-" * w for w in col_w))
        for row in rows:
            print(fmt_row.format(*[str(c) for c in row]))


def main():
    parser = argparse.ArgumentParser(description="CityPersons subset-aware TP/FP/FN diagnostics.")
    parser.add_argument("--gt", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--updated", required=True)
    parser.add_argument("--target-categories", nargs="+",
                         default=["pedestrian", "rider", "sitting person"])
    parser.add_argument("--score-thresh", type=float, default=0.10,
                         help="Min confidence for a detection to count toward the "
                              "printed TP/FP averages (matching itself is unaffected).")
    parser.add_argument("--subsets", choices=["standard"], default="standard")
    args = parser.parse_args()

    for p in (args.gt, args.baseline, args.updated):
        if not Path(p).exists():
            raise FileNotFoundError(f"Not found: {p}")

    print(f"Loading GT       : {args.gt}")
    gt_full = load_json(args.gt)
    cat_names = {c["id"]: c["name"] for c in gt_full.get("categories", [])}
    print("  Categories: " + ", ".join(f"{k}={v}" for k, v in sorted(cat_names.items())))

    target_category_ids = get_target_category_ids(gt_full, args.target_categories)
    if not target_category_ids:
        raise ValueError(f"None of {args.target_categories} matched a category name in GT: "
                          f"{list(cat_names.values())}")
    print("  Target category ids: " +
          ", ".join(f"{cid}={cat_names[cid]}" for cid in sorted(target_category_ids)))

    print(f"Loading Baseline : {args.baseline}")
    b_preds = load_json(args.baseline)
    print(f"Loading Updated  : {args.updated}")
    u_preds = load_json(args.updated)

    print(f"\n{'='*90}")
    print(f"  TRUE POSITIVE vs FALSE POSITIVE vs MISSED-GT DYNAMICS "
          f"(IoU >= {IOU_THRESHOLD} | Conf >= {args.score_thresh})")
    print(f"{'='*90}")

    for subset_name, subset_def in STANDARD_SUBSETS.items():
        b_stats = evaluate_subset_with_diagnostics(
            gt_full, b_preds, target_category_ids, subset_def, score_thresh=args.score_thresh)
        c_stats = evaluate_subset_with_diagnostics(
            gt_full, u_preds, target_category_ids, subset_def, score_thresh=args.score_thresh)
        if b_stats is None or c_stats is None:
            print(f"\n[{subset_name}]  -- skipped, no positive GT in this subset")
            continue
        print_subset_report(subset_name, b_stats, c_stats)

    print("\n" + "=" * 90)
    print("Done.")


if __name__ == "__main__":
    main()