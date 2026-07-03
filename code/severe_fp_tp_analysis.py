"""
TP / FP / FN analysis for a single occlusion level (e.g. "severe"),
built on top of COCOeval.evalImgs (the raw per-image match records that
accumulate() later collapses into the precision/recall arrays).

This lets you "unfold" any cell of the summary table:
    AP@.50, AP@.75, AP_small/medium/large, AR@maxDets=... etc.
into actual counts: TP, FP, FN, # GT, # Det, Precision, Recall.

Usage:
    python severe_tp_fp_analysis.py \
        --gt        instances_validation_merged_mannual.json \
        --baseline  results_baseline_GDINO_final_val.json \
        --updated   results_sampling_loss_final_val.json \
        --level     severe \
        --output    severe_tp_fp_report

Requires the same GT-masking trick as the original script: for the target
level, annotations are iscrowd=0 (active); everything else is iscrowd=1
(silently ignored by COCOeval, never counted as TP/FP/FN).
"""

import json
import copy
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

MAX_DETS = [100, 300, 1000]
# Custom area thresholds scaled for 1920x1080 RSUD images
# (COCO's 32²/96² were derived for ~640x480 images; scale factor ≈ sqrt(2,073,600/307,200) ≈ 2.6, 32*2.6=83, 96*2.6=250)
AREA_RNG = [
    [0 ** 2,   1e5 ** 2],   # all
    [0 ** 2,   83 ** 2],    # small   : area < 6,889 px²
    [83 ** 2,  250 ** 2],   # medium  : 6,889–62,500 px²
    [250 ** 2, 1e5 ** 2],   # large   : area > 62,500 px²
]
AREA_RNG_LBL = ["all", "small", "medium", "large"]
IOU_THRS = np.linspace(0.5, 0.95, 10)


# ── I/O ────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_gt_for_level(gt_full, target_level):
    gt_subset = copy.deepcopy(gt_full)
    for ann in gt_subset["annotations"]:
        level = (ann.get("attributes", {})
                    .get("occlusion_level", "unknown")
                    .strip().lower())
        ann["iscrowd"] = 0 if level == target_level else 1
    return gt_subset


def run_evaluate(gt_dict, pred_list):
    """Build COCOeval and stop right after evaluate() -- we need evalImgs,
    NOT the accumulated precision/recall arrays."""
    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()

    gt_image_ids   = {img["id"] for img in gt_dict["images"]}
    filtered_preds = [p for p in pred_list if p["image_id"] in gt_image_ids]
    if not filtered_preds:
        return None

    coco_dt   = coco_gt.loadRes(filtered_preds)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    
    coco_eval.params.maxDets = MAX_DETS
    coco_eval.params.areaRng    = AREA_RNG   
    coco_eval.params.areaRngLbl = AREA_RNG_LBL

    coco_eval.evaluate()       # populates coco_eval.evalImgs
    coco_eval.accumulate()     # not strictly needed here, but keeps params consistent
    return coco_eval


# ── Core: turn evalImgs into TP/FP/FN counts ────────────────────────────────

def _t_index(iou_thresh):
    idx = np.where(np.isclose(IOU_THRS, iou_thresh))[0]
    return int(idx[0]) if len(idx) else None


def collect_counts(coco_eval, iou_thresh=0.5, max_det=1000, area="all",
                    by_category=False, score_thresh=0.0):
    """
    Walk coco_eval.evalImgs and tally TP / FP / FN.

    evalImgs is a flat list indexed as catIds x areaRngs x imgIds (in that
    nested order, per COCOeval.evaluate()). Each non-None entry contains:
        dtMatches : [T, D]  -> matched gt_id per IoU thresh / detection (0 = no match)
        dtIgnore  : [T, D]  -> True if this detection is ignored at thresh t
        dtScores  : [D]
        gtIgnore  : [G]     -> True if this GT is ignored (outside area range,
                               or iscrowd=1 from another occlusion level)
        gtMatches : [T, G]  -> matched dt_id per IoU thresh / gt (0 = unmatched)
        aRng      : area range actually used for this record
        category_id, image_id
    """
    p = coco_eval.params
    t = _t_index(iou_thresh)
    if t is None:
        raise ValueError(f"IoU {iou_thresh} not in standard thresholds")

    ai = [i for i, lbl in enumerate(p.areaRngLbl) if lbl == area]
    if not ai:
        raise ValueError(f"Unknown area label: {area}")
    area_rng = p.areaRng[ai[0]]

    cat_names = {c["id"]: c["name"] for c in coco_eval.cocoGt.dataset["categories"]}

    agg = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "nGT": 0, "nDet": 0,
                                "fp_records": [], "fn_records": []})

    for ev in coco_eval.evalImgs:
        if ev is None:
            continue
        if list(ev["aRng"]) != list(area_rng):
            continue  # not the area bucket we want

        key = ev["category_id"] if by_category else "all"

        dt_ignore = np.asarray(ev["dtIgnore"])[t, :max_det]
        dt_matches = np.asarray(ev["dtMatches"])[t, :max_det]
        dt_scores  = np.asarray(ev["dtScores"])[:max_det]
        dt_ids     = np.asarray(ev["dtIds"])[:max_det]

        gt_ignore  = np.asarray(ev["gtIgnore"])
        gt_matches = np.asarray(ev["gtMatches"])[t, :]
        gt_ids     = np.asarray(ev["gtIds"])

        # detections: ignore the ones flagged ignore (e.g. matched an
        # iscrowd=1 GT from another occlusion level, or outside area range),
        # AND apply a score threshold for a human-readable FP count.
        valid_dt = ~dt_ignore.astype(bool) & (dt_scores >= score_thresh)
        tp_mask = valid_dt & (dt_matches > 0)
        fp_mask = valid_dt & (dt_matches == 0)

        # ground truth: only count non-ignored GT (i.e. target-level, in-area)
        valid_gt = ~gt_ignore.astype(bool)
        matched_gt = valid_gt & (gt_matches > 0)
        fn_mask = valid_gt & (gt_matches == 0)

        agg[key]["TP"]   += int(tp_mask.sum())
        agg[key]["FP"]   += int(fp_mask.sum())
        agg[key]["FN"]   += int(fn_mask.sum())
        agg[key]["nGT"]  += int(valid_gt.sum())
        agg[key]["nDet"] += int(valid_dt.sum())

        for d_id, score in zip(dt_ids[fp_mask], dt_scores[fp_mask]):
            agg[key]["fp_records"].append({
                "image_id": ev["image_id"], "det_id": int(d_id),
                "score": float(score), "category": cat_names.get(ev["category_id"])
            })
        for g_id in gt_ids[fn_mask]:
            agg[key]["fn_records"].append({
                "image_id": ev["image_id"], "gt_id": int(g_id),
                "category": cat_names.get(ev["category_id"])
            })

    out = {}
    for key, v in agg.items():
        precision = v["TP"] / v["nDet"] if v["nDet"] else float("nan")
        recall    = v["TP"] / v["nGT"]  if v["nGT"]  else float("nan")
        name = cat_names.get(key, key) if by_category else "ALL"
        out[name] = {
            "TP": v["TP"], "FP": v["FP"], "FN": v["FN"],
            "nGT": v["nGT"], "nDet": v["nDet"],
            "Precision": precision, "Recall": recall,
            "fp_records": v["fp_records"], "fn_records": v["fn_records"],
        }
    return out


# ── Tables ───────────────────────────────────────────────────────────────────

def print_table(rows, headers, title):
    print(f"\n── {title} ──")
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    else:
        col_w = [max(len(h), max(len(str(r[i])) for r in rows))
                 for i, h in enumerate(headers)]
        fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
        print(fmt.format(*headers))
        print("  ".join("-" * w for w in col_w))
        for row in rows:
            print(fmt.format(*[str(c) for c in row]))


def overall_tp_fp_fn_table(eval_b, eval_u, iou_thresh, max_det, area):
    cb = collect_counts(eval_b, iou_thresh, max_det, area)["ALL"]
    cu = collect_counts(eval_u, iou_thresh, max_det, area)["ALL"]
    rows = [
        ["GT count",    cb["nGT"],  cu["nGT"],  cu["nGT"]  - cb["nGT"]],
        ["Det count",   cb["nDet"], cu["nDet"], cu["nDet"] - cb["nDet"]],
        ["TP",          cb["TP"],   cu["TP"],   cu["TP"]   - cb["TP"]],
        ["FP",          cb["FP"],   cu["FP"],   cu["FP"]   - cb["FP"]],
        ["FN",          cb["FN"],   cu["FN"],   cu["FN"]   - cb["FN"]],
        ["Precision",   f"{cb['Precision']:.4f}", f"{cu['Precision']:.4f}",
         f"{cu['Precision']-cb['Precision']:+.4f}"],
        ["Recall",      f"{cb['Recall']:.4f}", f"{cu['Recall']:.4f}",
         f"{cu['Recall']-cb['Recall']:+.4f}"],
    ]
    print_table(rows, ["Metric", "Baseline", "Updated", "Δ"],
                f"Severe — IoU={iou_thresh}, maxDets={max_det}, area={area}")
    return cb, cu


def per_category_table(eval_b, eval_u, iou_thresh, max_det, area):
    cb = collect_counts(eval_b, iou_thresh, max_det, area, by_category=True)
    cu = collect_counts(eval_u, iou_thresh, max_det, area, by_category=True)
    cats = sorted(set(cb) | set(cu))
    rows = []
    for cat in cats:
        b = cb.get(cat, {"TP": 0, "FP": 0, "FN": 0, "nGT": 0})
        u = cu.get(cat, {"TP": 0, "FP": 0, "FN": 0, "nGT": 0})
        rows.append([
            cat, b["nGT"],
            b["TP"], b["FP"], b["FN"],
            u["TP"], u["FP"], u["FN"],
            (u["TP"] - b["TP"]),
        ])
    print_table(rows,
                ["Category", "nGT", "TP_b", "FP_b", "FN_b",
                 "TP_u", "FP_u", "FN_u", "ΔTP"],
                f"Per-category — Severe, IoU={iou_thresh}, maxDets={max_det}, area={area}")
    return cb, cu


def averaged_recall(coco_eval, max_det, area):
    """
    Mirrors how the original get_stats()/_ar() computes AR_small/medium/large:
    average the matched-recall across all 10 IoU thresholds (0.50..0.95),
    NOT just IoU=0.50. This is what makes AR look much lower than the
    naive "found it at IoU>=0.5" recall.
    """
    recalls = []
    for iou in IOU_THRS:
        c = collect_counts(coco_eval, iou_thresh=float(iou), max_det=max_det,
                            area=area)["ALL"]
        if c["nGT"]:
            recalls.append(c["TP"] / c["nGT"])
    return float(np.mean(recalls)) if recalls else float("nan")


def size_bucket_table_full(eval_b, eval_u, max_det=1000):
    """Like size_bucket_table but uses AR averaged over all IoU thresholds,
    so the numbers line up with the original summary table's AR_small/
    AR_medium/AR_large rows."""
    rows = []
    for area in ["small", "medium", "large"]:
        rb = averaged_recall(eval_b, max_det, area)
        ru = averaged_recall(eval_u, max_det, area)
        rows.append([area, f"{rb:.4f}", f"{ru:.4f}", f"{ru-rb:+.4f}"])
    print_table(rows, ["Size", "AR_b [.5:.95]", "AR_u [.5:.95]", "Δ"],
                "Size-bucket AR averaged over IoU 0.50-0.95 (matches summary table)")



def size_bucket_table(eval_b, eval_u, iou_thresh, max_det):
    """Explains AP_small / AP_medium / AP_large rows directly (naive recall
    at a single IoU threshold -- see size_bucket_table_full for the version
    that matches AR_small/medium/large in the summary table)."""
    rows = []
    for area in ["small", "medium", "large"]:
        cb = collect_counts(eval_b, iou_thresh, max_det, area)["ALL"]
        cu = collect_counts(eval_u, iou_thresh, max_det, area)["ALL"]
        rows.append([
            area, cb["nGT"],
            cb["TP"], cb["FP"], cb["FN"], f"{cb['Recall']:.4f}",
            cu["TP"], cu["FP"], cu["FN"], f"{cu['Recall']:.4f}",
        ])
    print_table(rows,
                ["Size", "nGT", "TP_b", "FP_b", "FN_b", "Rec_b",
                 "TP_u", "FP_u", "FN_u", "Rec_u"],
                f"Size-bucket breakdown — Severe, IoU={iou_thresh}, maxDets={max_det}")


def dump_fp_fn(counts, model_name, top_n=20):
    print(f"\n── Top {top_n} highest-confidence FALSE POSITIVES ({model_name}) ──")
    fps = sorted(counts["ALL"]["fp_records"], key=lambda r: -r["score"])[:top_n]
    rows = [[r["image_id"], r["det_id"], r["category"], f"{r['score']:.3f}"]
            for r in fps]
    print_table(rows, ["image_id", "det_id", "category", "score"],
                f"FP dump — {model_name}")

    print(f"\n── Sample MISSED ground truth (FN) ({model_name}) ──")
    fns = counts["ALL"]["fn_records"][:top_n]
    rows = [[r["image_id"], r["gt_id"], r["category"]] for r in fns]
    print_table(rows, ["image_id", "gt_id", "category"], f"FN dump — {model_name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TP/FP/FN breakdown for one occlusion level.")
    parser.add_argument("--gt",       required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--updated",  required=True)
    parser.add_argument("--level",    default="severe", choices=["light", "moderate", "severe"])
    parser.add_argument("--output",   default="severe_tp_fp_report")
    parser.add_argument("--dump-fp-fn", action="store_true",
                        help="Print raw FP/FN records (image_id, det/gt id, score)")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    gt_full = load_json(args.gt)
    b_preds = load_json(args.baseline)
    u_preds = load_json(args.updated)

    gt_subset = build_gt_for_level(gt_full, args.level)
    print(f"Level: {args.level}")
    print(f"Active (iscrowd=0): {sum(1 for a in gt_subset['annotations'] if a['iscrowd']==0):,}")

    eval_b = run_evaluate(gt_subset, b_preds)
    eval_u = run_evaluate(gt_subset, u_preds)

    if eval_b is None or eval_u is None:
        print("No matching predictions for this level/GT subset.")
        return

    # 1) Overall TP/FP/FN at IoU=.50 and IoU=.75, maxDets=1000 (area=all)
    #    -> explains AP@.50 / AP@.75 / AR@[.5:.95]_maxDets1000 rows
    for iou in [0.50, 0.75]:
        overall_tp_fp_fn_table(eval_b, eval_u, iou, max_det=1000, area="all")

    # 2) AR at each maxDets level (explains the 3 AR@[.5:.95]_maxDetsXXX rows)
    for md in MAX_DETS:
        overall_tp_fp_fn_table(eval_b, eval_u, iou_thresh=0.50, max_det=md, area="all")

    # 3) Size-bucket breakdown (raw, IoU=0.50 only -- naive recall)
    size_bucket_table(eval_b, eval_u, iou_thresh=0.50, max_det=1000)

    # 3b) Size-bucket AR averaged over all 10 IoU thresholds -- THIS is what
    #     matches your AR_small/AR_medium/AR_large summary table rows.
    size_bucket_table_full(eval_b, eval_u, max_det=1000)

    # 3c) Score-thresholded overall TP/FP table -- a human-readable FP count,
    #     since raw FP includes every near-zero-confidence box in the JSON.
    for st in [0.3, 0.5]:
        cb = collect_counts(eval_b, 0.50, 1000, "all", score_thresh=st)["ALL"]
        cu = collect_counts(eval_u, 0.50, 1000, "all", score_thresh=st)["ALL"]
        rows = [
            ["TP", cb["TP"], cu["TP"], cu["TP"] - cb["TP"]],
            ["FP", cb["FP"], cu["FP"], cu["FP"] - cb["FP"]],
            ["FN", cb["FN"], cu["FN"], cu["FN"] - cb["FN"]],
            ["Precision", f"{cb['Precision']:.4f}", f"{cu['Precision']:.4f}",
             f"{cu['Precision']-cb['Precision']:+.4f}"],
        ]
        print_table(rows, ["Metric", "Baseline", "Updated", "Δ"],
                    f"Score-thresholded (score >= {st}) -- Severe, IoU=0.50")

    # 4) Per-category breakdown at IoU=.50
    per_category_table(eval_b, eval_u, iou_thresh=0.50, max_det=1000, area="all")

    # 5) Optional: raw FP/FN dumps for manual image-level inspection
    if args.dump_fp_fn:
        cb = collect_counts(eval_b, 0.50, 1000, "all")
        cu = collect_counts(eval_u, 0.50, 1000, "all")
        dump_fp_fn(cb, "baseline")
        dump_fp_fn(cu, "updated")

    print(f"\nDone. (Use --dump-fp-fn to inspect individual missed/false detections.)")


if __name__ == "__main__":
    main()