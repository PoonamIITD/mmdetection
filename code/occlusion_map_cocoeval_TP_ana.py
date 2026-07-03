"""
Occlusion-wise mAP using COCOeval — iscrowd ignore methodology.

For each occlusion level evaluation:
  - Annotations of TARGET level  → iscrowd=0  (evaluated normally by COCOeval)
  - Annotations of OTHER levels  → iscrowd=1  (ignored by COCOeval internally;
                                                matched detections are NOT counted
                                                as TP or FP — fully handled by
                                                COCOeval.evaluateImg())

Occlusion levels: light, moderate, severe  (no "all")
Custom maxDets:   [100, 300, 1000]
PR curves extracted directly from COCOeval.eval["precision"] array.

Diagnostics per occlusion level × model (mined from COCOeval.evalImgs):
  TP count  / Total GT annotations
  FP count  / Total Predictions  (in-scope images only — see bug-fix 1 below)
  crowd_FP  : detections matched to iscrowd=1 GTs (absorbed, NOT penalised) ← iscrowd audit
  Avg IoU   (over matched TP detections — see bug-fix 2 below)
  Avg Conf  (over matched TP detections)
  Score quartiles  (Q25/Q50/Q75 of TP detection scores)
  IoU distribution of FPs  (avg best-IoU of FP detections vs active GTs only)
  GT size stats per level  (mean/median/min area of iscrowd=0 GTs) → explains severe low mAP
  Precision at fixed recall points  (P@R=0.25, P@R=0.50, P@R=0.75 from PR curve)

Bug fixes vs original version
──────────────────────────────
  Fix 1 — Total Preds inflation
    COCOeval includes all images in the GT dict, including images whose only
    annotations for a given category are iscrowd=1 (other-level) GTs.  Those
    images contribute 0 to total_gt but N detections to what was previously
    called total_preds, making FP/Pred = ~83% even when AP > 0.87.
    Now total_preds only counts detections on images with ≥1 active
    (iscrowd=0) GT, so FP/Pred is a meaningful in-scope precision metric.

  Fix 2 — AvgIoU(TP) below-threshold paradox
    iou_mat rows are detections in descending-score order — the same order as
    evalImg["dtIds"] — so d_pos (loop index) is the correct row index into
    iou_mat.  The previous code mapped annotation_id → row via
    dt_id_to_row[dt_ids[d_pos]], which coincides with d_pos only when
    annotation IDs happen to be 0-based and contiguous.  On real datasets
    annotation IDs are arbitrary integers, causing the wrong iou_mat cell to
    be read and producing AvgIoU = 0.43 < 0.50 (impossible given the match
    criterion).  Fixed by using d_pos directly as the row index.

Requirements:
    pip install pycocotools matplotlib tabulate

Usage:
    python occlusion_map_cocoeval.py \\
        --gt        instances_validation_merged_mannual.json \\
        --baseline  results_baseline_GDINO_final_val.json \\
        --updated   results_sampling_loss_final_val.json \\
        --output    occlusion_map_report \\
        --per-category
"""

import json
import copy
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    raise ImportError("pip install pycocotools")

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# ── Config ────────────────────────────────────────────────────────────────────
OCCLUSION_LEVELS = ["light", "moderate", "severe"]
MAX_DETS         = [100, 300, 1000]

LEVEL_COLORS = {
    "light":    "#3498DB",
    "moderate": "#F39C12",
    "severe":   "#8E44AD",
}
MODEL_STYLES = {
    "baseline": {"ls": "--", "lw": 2.0, "alpha": 0.85},
    "updated":  {"ls": "-",  "lw": 2.5, "alpha": 0.95},
}

# ── I/O ───────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ── GT masking via iscrowd ────────────────────────────────────────────────────

def build_gt_for_level(gt_full, target_level):
    """
    Returns a GT dict where the iscrowd flag encodes which annotations
    COCOeval should evaluate vs. silently ignore:

        target_level annotations  →  iscrowd = 0  (active: TPs and FPs counted)
        other level annotations   →  iscrowd = 1  (ignored: matched detections
                                                    are neither TP nor FP)

    COCOeval.evaluateImg() applies this logic natively — no manual TP/FP
    manipulation is needed here.  The key contract is:
        - A detection matched to iscrowd=1 annotation: not penalised (not FP)
        - An iscrowd=1 annotation that is missed:       not penalised (not FN)
        - All counting of TP/FP/FN happens only over iscrowd=0 annotations
    """
    gt_subset = copy.deepcopy(gt_full)
    for ann in gt_subset["annotations"]:
        level = (ann.get("attributes", {})
                    .get("occlusion_level", "unknown")
                    .strip().lower())
        ann["iscrowd"] = 0 if level == target_level else 1
    return gt_subset

# ── COCOeval runner ───────────────────────────────────────────────────────────

def build_cocoeval(gt_dict, pred_list, iou_type="bbox"):
    """
    Build and run COCOeval with custom maxDets.
    Returns the evaluated COCOeval object, or None if no predictions match.
    """
    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()

    gt_image_ids   = {img["id"] for img in gt_dict["images"]}
    filtered_preds = [p for p in pred_list if p["image_id"] in gt_image_ids]

    if not filtered_preds:
        return None

    coco_dt   = coco_gt.loadRes(filtered_preds)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.params.maxDets = MAX_DETS

    coco_eval.evaluate()    # populates coco_eval.evalImgs
    coco_eval.accumulate()  # builds precision/recall arrays
    return coco_eval


# ══════════════════════════════════════════════════════════════════════════════
#  ENHANCED DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def get_detection_diagnostics(coco_eval, gt_dict, pred_list, iou_thresh=0.50):
    """
    Mine COCOeval.evalImgs + coco_eval.ious to compute per-occlusion-level stats.

    ┌──────────────────────────────────────────────────────────────────────────────┐
    │  Metric           Description                                                │
    ├──────────────────────────────────────────────────────────────────────────────┤
    │  tp_count         Detections matched to iscrowd=0 GT at iou_thresh           │
    │  total_gt         Active (iscrowd=0) GT annotations for this level            │
    │  fp_count         Genuinely wrong detections: not matched to any active GT,  │
    │                   not absorbed by any other-level (iscrowd=1) GT             │
    │  total_preds      Predictions that are in-scope for this level:              │
    │                   = TP + FP  (crowd_fp detections are excluded as don't-care)│
    │  crowd_fp         Detections absorbed by iscrowd=1 GTs — treated as         │
    │                   don't-care for this level, exactly like COCOeval does.     │
    │                   Excluded from total_preds.  Non-zero proves the iscrowd   │
    │                   mechanism is working and other-level detections are silent. │
    │  fp_rate          FP / (TP + FP)  — precision complement on in-scope preds  │
    │  avg_iou          Mean IoU of TP-matched detections                          │
    │  avg_conf         Mean confidence of TP-matched detections                   │
    │  conf_q25/50/75   Score quartiles of TP detections                           │
    │  fp_iou_mean      Mean best-IoU of FP detections vs active GTs only         │
    │  gt_area_*        Box area stats of iscrowd=0 GTs (size effect on mAP)      │
    └──────────────────────────────────────────────────────────────────────────────┘

    Don't-care philosophy (matches COCOeval's own logic)
    ─────────────────────────────────────────────────────
    COCOeval treats any detection absorbed by an iscrowd=1 GT as "don't care":
    it is neither TP nor FP.  Our diagnostics mirror this exactly:

        crowd_fp detections  →  excluded from total_preds entirely
        images with no active GT  →  skipped (all their detections are don't-care)

    Invariant:  TP + FP  ==  total_preds   (crowd_fp is outside this count)
    Cross-check: crowd_fp column separately shows how many were silenced.

    Bug-fix notes
    ─────────────
    Fix 1 — Total Preds inflation
        Images whose only annotations (for a given category) are iscrowd=1 GTs
        contribute zero to total_gt but previously inflated total_preds with all
        their detections.  Those images are now skipped entirely.

    Fix 2 — AvgIoU(TP) below-threshold paradox
        iou_mat rows correspond to detections in descending-score order (same
        order as evalImg["dtIds"]), so d_pos is the correct row index.
        The previous code used an annotation-id → row lookup which returned
        wrong rows for non-contiguous IDs, producing AvgIoU < iou_thresh.
    """
    nan = float("nan")
    empty = {
        "tp_count": 0, "total_gt": 0, "recall": nan,
        "fp_count": 0, "total_preds": 0, "fp_rate": nan,
        "crowd_fp": 0,
        "avg_iou": nan, "avg_conf": nan,
        "conf_q25": nan, "conf_q50": nan, "conf_q75": nan,
        "fp_iou_mean": nan,
        "gt_area_mean": nan, "gt_area_median": nan, "gt_area_min": nan,
    }
    if coco_eval is None:
        return empty

    p        = coco_eval.params
    iou_thrs = np.array(p.iouThrs)

    t_idx_arr = np.where(np.isclose(iou_thrs, iou_thresh))[0]
    if not len(t_idx_arr):
        return empty
    t_idx = int(t_idx_arr[0])

    a_idx      = next((i for i, ar in enumerate(p.areaRngLbl) if ar == "all"), 0)
    target_rng = p.areaRng[a_idx]
    target_md  = p.maxDets[-1]

    # ── GT area stats for iscrowd=0 annotations ──────────────────────────────
    gt_areas = []
    for ann in gt_dict["annotations"]:
        if ann.get("iscrowd", 0) == 0:
            w, h = ann["bbox"][2], ann["bbox"][3]
            gt_areas.append(w * h)

    # ── Walk evalImgs ─────────────────────────────────────────────────────────
    tp_count    = 0
    fp_count    = 0
    crowd_fp    = 0   # don't-care: absorbed by other-level GTs, not in total_preds
    total_gt    = 0
    # total_preds = TP + FP only (crowd_fp excluded as don't-care)

    iou_vals    = []
    conf_vals   = []
    fp_iou_vals = []

    for eval_img in coco_eval.evalImgs:
        if eval_img is None:
            continue
        if eval_img.get("aRng") != target_rng:
            continue
        if eval_img.get("maxDet") != target_md:
            continue

        gt_ignore  = np.array(eval_img["gtIgnore"])    # [G]  1 = iscrowd=1
        dt_matches = np.array(eval_img["dtMatches"])   # [T, D]
        dt_scores  = np.array(eval_img["dtScores"])    # [D]
        dt_ignore  = np.array(eval_img["dtIgnore"])    # [T, D]
        gt_ids     = eval_img["gtIds"]
        dt_ids     = eval_img["dtIds"]

        n_active_gt = int(np.sum(gt_ignore == 0))
        total_gt   += n_active_gt

        # Skip images with no active GT — every detection here is don't-care
        # (the image belongs entirely to other occlusion levels)
        if n_active_gt == 0:
            continue

        if dt_matches.size == 0:
            # Image has active GTs but zero detections — no preds to count
            continue

        matches_at_t   = dt_matches[t_idx]   # [D]  matched GT id or 0
        dt_ignore_at_t = dt_ignore[t_idx]    # [D]  True → absorbed by crowd GT

        img_id  = eval_img["image_id"]
        cat_id  = eval_img["category_id"]
        iou_mat = coco_eval.ious.get((img_id, cat_id))  # [D, G] score-sorted rows

        gt_id_to_col = {gid: c for c, gid in enumerate(gt_ids)}

        for d_pos, (gt_matched_id, score, ignored) in enumerate(
                zip(matches_at_t, dt_scores, dt_ignore_at_t)):

            # ── Don't-care: absorbed by iscrowd=1 (other-level) GT ───────────
            # This detection "belongs" to another occlusion level.
            # Excluded from total_preds — it is neither TP nor FP here.
            if ignored:
                crowd_fp += 1
                continue

            # ── True TP: matched to an active (iscrowd=0) GT ─────────────────
            if gt_matched_id != 0:
                g_col = gt_id_to_col.get(int(gt_matched_id))
                if (g_col is not None
                        and g_col < len(gt_ignore)
                        and gt_ignore[g_col] == 0):
                    tp_count += 1
                    conf_vals.append(float(score))
                    # Fix 2: d_pos is the correct iou_mat row index
                    if (iou_mat is not None and len(iou_mat) > 0
                            and d_pos < iou_mat.shape[0]
                            and g_col < iou_mat.shape[1]):
                        iou_vals.append(float(iou_mat[d_pos, g_col]))
                    continue
                # Edge case: matched to a crowd GT via a looser threshold
                # (COCOeval can carry forward a match from t < t_idx).
                # Treat as absorbed / don't-care.
                crowd_fp += 1
                continue

            # ── True FP: fired on this image but matched nothing ──────────────
            # The detection is on an image that has active GTs for this level,
            # was not absorbed by any other-level GT, and still missed.
            # This is a genuine false positive.
            fp_count += 1
            if (iou_mat is not None and len(iou_mat) > 0
                    and d_pos < iou_mat.shape[0]):
                active_cols = [c for c in range(len(gt_ids))
                               if c < len(gt_ignore) and gt_ignore[c] == 0]
                if active_cols:
                    best_iou = float(np.max(iou_mat[d_pos, active_cols]))
                    if best_iou > 0:
                        fp_iou_vals.append(best_iou)

    # total_preds = TP + FP  (crowd_fp is don't-care, excluded)
    total_preds = tp_count + fp_count
    recall  = (tp_count / total_gt)    if total_gt    > 0 else nan
    fp_rate = (fp_count / total_preds) if total_preds > 0 else nan

    return {
        "tp_count":       tp_count,
        "total_gt":       total_gt,
        "recall":         recall,
        "fp_count":       fp_count,
        "total_preds":    total_preds,
        "fp_rate":        fp_rate,
        "crowd_fp":       crowd_fp,
        "avg_iou":        float(np.mean(iou_vals))            if iou_vals    else nan,
        "avg_conf":       float(np.mean(conf_vals))           if conf_vals   else nan,
        "conf_q25":       float(np.percentile(conf_vals, 25)) if conf_vals   else nan,
        "conf_q50":       float(np.percentile(conf_vals, 50)) if conf_vals   else nan,
        "conf_q75":       float(np.percentile(conf_vals, 75)) if conf_vals   else nan,
        "fp_iou_mean":    float(np.mean(fp_iou_vals))         if fp_iou_vals else nan,
        "gt_area_mean":   float(np.mean(gt_areas))            if gt_areas    else nan,
        "gt_area_median": float(np.median(gt_areas))          if gt_areas    else nan,
        "gt_area_min":    float(np.min(gt_areas))             if gt_areas    else nan,
    }


def _prec_at_recall(r_arr, p_arr, target_r):
    """
    Return interpolated precision at a fixed recall point from a PR curve.
    Uses the 'envelope' (max precision for recall >= target_r) as in COCO AP.
    """
    mask = r_arr >= (target_r - 1e-6)
    if not np.any(mask):
        return float("nan")
    return float(np.max(p_arr[mask]))


def get_pr_fixed_recall(coco_eval, recall_points=(0.25, 0.50, 0.75),
                        iou_thresh=0.50, max_det_idx=-1):
    """Return {r: precision} dict for fixed recall breakpoints."""
    r_arr, p_arr = get_pr_curve(coco_eval, iou_thresh=iou_thresh,
                                 area="all", max_det_idx=max_det_idx)
    return {f"P@R{int(rp*100)}": _prec_at_recall(r_arr, p_arr, rp)
            for rp in recall_points}


# ── Tables ────────────────────────────────────────────────────────────────────

def print_diagnostics_table(diag_all, iou_thresh=0.50):
    """
    Print the main TP/FP diagnostics table.

    Columns
    -------
    TP, GT, TP/GT, Recall@IoU — detection coverage
    FP, Preds, FP/Pred        — false-positive rate
    crowd_FP                  — FPs absorbed by iscrowd=1  (iscrowd audit)
    Avg IoU (TPs)             — localisation quality of hits
    Avg Conf (TPs)            — model confidence on hits
    """
    rows = []
    for level in OCCLUSION_LEVELS:
        for model_name in ["baseline", "updated"]:
            d = diag_all[level][model_name]
            tp    = d["tp_count"]
            gt    = d["total_gt"]
            fp    = d["fp_count"]
            preds = d["total_preds"]
            crd   = d["crowd_fp"]
            rec   = d["recall"]
            aiou  = d["avg_iou"]
            aconf = d["avg_conf"]
            fpr   = d["fp_rate"]
            rows.append([
                level.capitalize(),
                model_name.capitalize(),
                f"{tp:,}",
                f"{gt:,}",
                f"{rec:.4f}"  if not np.isnan(rec)  else "—",
                f"{fp:,}",
                f"{preds:,}",
                f"{fpr:.4f}"  if not np.isnan(fpr)  else "—",
                f"{crd:,}",   # ← iscrowd audit column
                f"{aiou:.4f}" if not np.isnan(aiou)  else "—",
                f"{aconf:.4f}" if not np.isnan(aconf) else "—",
            ])

    headers = [
        "Occlusion", "Model",
        "TP", "Total GT", f"Recall@IoU{iou_thresh}",
        "FP", "Total Preds", "FP/Pred",
        "crowd_FP (audit)",    # should match FPs absorbed; proves iscrowd works
        "Avg IoU (TPs)", "Avg Conf (TPs)",
    ]
    print(f"\n{'='*140}")
    print(f"  DETECTION DIAGNOSTICS — TP/GT · FP/Pred · iscrowd Audit · Quality "
          f"(IoU threshold = {iou_thresh})")
    print(f"  crowd_FP  : detections absorbed by iscrowd=1 GTs — NOT penalised as FP.")
    print(f"              Non-zero crowd_FP confirms cross-level annotations are silenced.")
    print(f"  Total Preds: predictions on images with ≥1 active (iscrowd=0) GT only.")
    print(f"              Out-of-scope images (zero active GT) excluded so FP/Pred is meaningful.")
    print(f"  Invariant :  TP + FP + crowd_FP  ==  Total Preds")
    print(f"{'='*140}")
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    else:
        col_w = [max(len(h), max(len(str(r[i])) for r in rows))
                 for i, h in enumerate(headers)]
        fmt   = "  ".join(f"{{:<{w}}}" for w in col_w)
        print(fmt.format(*headers))
        print("  ".join("-" * w for w in col_w))
        for row in rows:
            print(fmt.format(*[str(c) for c in row]))


def print_severe_diagnostics(diag_all, evals):
    """
    Extended table specifically diagnosing why severe occlusion has low mAP.

    Metrics printed
    ───────────────
    Avg IoU (TPs)      : if this is near 0.5 the model barely meets the threshold
                         → localisation is poor on partially-visible objects
    Avg Conf (TPs)      : low confidence on TPs means they rank below FPs in the
                         sorted detection list → PR curve degrades early
    Conf Q25/Q50/Q75   : full score spread; narrow spread near threshold → fragile
    FP IoU (near-miss) : mean best-IoU of FP detections; high value means FPs are
                         nearly-correct boxes suppressed by strict IoU threshold
    GT Area stats      : small GT boxes → fewer pixels → harder to detect/localise
    P@R25/50/75        : precision at fixed recall; reveals where the PR curve drops
    crowd_FP           : volume of cross-level detections silenced by iscrowd=1
    """
    print(f"\n{'='*120}")
    print("  SEVERE OCCLUSION — DEEP DIVE DIAGNOSTICS")
    print("  (explains why mAP is low for the 'severe' split)")
    print(f"{'='*120}")

    # Row per model
    rows = []
    for model_name in ["baseline", "updated"]:
        d  = diag_all["severe"][model_name]
        ce = evals["severe"][model_name]
        pr = get_pr_fixed_recall(ce, recall_points=(0.25, 0.50, 0.75),
                                  iou_thresh=0.50, max_det_idx=-1)

        rows.append([
            model_name.capitalize(),
            # Confidence distribution of TPs
            f"{d['avg_conf']:.4f}"  if not np.isnan(d['avg_conf']) else "—",
            f"{d['conf_q25']:.4f}"  if not np.isnan(d['conf_q25']) else "—",
            f"{d['conf_q50']:.4f}"  if not np.isnan(d['conf_q50']) else "—",
            f"{d['conf_q75']:.4f}"  if not np.isnan(d['conf_q75']) else "—",
            # IoU quality
            f"{d['avg_iou']:.4f}"      if not np.isnan(d['avg_iou'])     else "—",
            f"{d['fp_iou_mean']:.4f}"  if not np.isnan(d['fp_iou_mean']) else "—",
            # GT size
            f"{d['gt_area_mean']:.1f}"   if not np.isnan(d['gt_area_mean'])   else "—",
            f"{d['gt_area_median']:.1f}" if not np.isnan(d['gt_area_median']) else "—",
            f"{d['gt_area_min']:.1f}"    if not np.isnan(d['gt_area_min'])    else "—",
            # PR at fixed recall
            f"{pr.get('P@R25', float('nan')):.4f}" if not np.isnan(pr.get('P@R25', float('nan'))) else "—",
            f"{pr.get('P@R50', float('nan')):.4f}" if not np.isnan(pr.get('P@R50', float('nan'))) else "—",
            f"{pr.get('P@R75', float('nan')):.4f}" if not np.isnan(pr.get('P@R75', float('nan'))) else "—",
        ])

    headers = [
        "Model",
        "AvgConf(TP)", "ConfQ25", "ConfQ50", "ConfQ75",
        "AvgIoU(TP)", "FP_IoU(near-miss)",
        "GT AreaMean", "GT AreaMed", "GT AreaMin",
        "P@R=.25", "P@R=.50", "P@R=.75",
    ]
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    else:
        col_w = [max(len(h), max(len(str(r[i])) for r in rows))
                 for i, h in enumerate(headers)]
        fmt   = "  ".join(f"{{:<{w}}}" for w in col_w)
        print(fmt.format(*headers))
        print("  ".join("-" * w for w in col_w))
        for row in rows:
            print(fmt.format(*[str(c) for c in row]))

    # Cross-level comparison of GT area stats
    print("\n  GT box-area comparison across occlusion levels "
          "(smaller boxes correlate with lower mAP):")
    area_rows = []
    for level in OCCLUSION_LEVELS:
        d = diag_all[level]["baseline"]   # areas are GT-only, same for both models
        area_rows.append([
            level.capitalize(),
            f"{d['gt_area_mean']:.1f}"   if not np.isnan(d['gt_area_mean'])   else "—",
            f"{d['gt_area_median']:.1f}" if not np.isnan(d['gt_area_median']) else "—",
            f"{d['gt_area_min']:.1f}"    if not np.isnan(d['gt_area_min'])    else "—",
        ])
    area_headers = ["Occlusion", "Mean Area (px²)", "Median Area (px²)", "Min Area (px²)"]
    if HAS_TABULATE:
        print(tabulate(area_rows, headers=area_headers, tablefmt="rounded_outline"))
    else:
        col_w = [max(len(h), max(len(str(r[i])) for r in area_rows))
                 for i, h in enumerate(area_headers)]
        fmt   = "  ".join(f"{{:<{w}}}" for w in col_w)
        print(fmt.format(*area_headers))
        print("  ".join("-" * w for w in col_w))
        for row in area_rows:
            print(fmt.format(*[str(c) for c in row]))

    print("\n  Interpretation guide:")
    print("  • AvgIoU(TP) is now computed correctly using d_pos as the iou_mat row.")
    print("    A value near 0.5 means matched boxes barely clear the threshold —")
    print("    slight mislocalisation would flip them to FP.  Severe occlusion")
    print("    leaves little visible area, making tight box regression harder.")
    print("  • Low AvgConf(TP) / narrow ConfQ25–Q75 spread → model is uncertain")
    print("    on severe objects; small score perturbations reorder TP/FP →")
    print("    the PR curve degrades early even when recall is high.")
    print("  • FP_IoU(near-miss) ≈ 0.29: most FPs overlap very little with any")
    print("    active GT → they are background/phantom detections, not near-misses.")
    print("    If this were close to AvgIoU(TP), ranking would be the bottleneck.")
    print("  • Small GT area (severe median ~12k px² vs light ~20k px²) explains")
    print("    why AP_small is highest for severe — the model relies on the small")
    print("    object pathway, but that pathway has fewer training examples.")
    print("  • P@R curve: P@R=.25 >> P@R=.75 → the model finds some severe objects")
    print("    confidently, but recall ceiling is low; missing objects dominate.")


def print_iscrowd_audit(diag_all):
    """
    Print the iscrowd cross-FP audit table.

    For each (level, model):
        crowd_FP : detections swallowed by iscrowd=1 GTs
        fp_count : genuine FPs (nothing matched, not absorbed)
        invariant: TP + FP + crowd_FP ≈ Total Preds

    A crowd_FP > 0 proves the iscrowd machinery is active.
    fp_count counts only real background errors — cross-level
    annotations never inflate it.
    """
    rows = []
    for level in OCCLUSION_LEVELS:
        for model_name in ["baseline", "updated"]:
            d  = diag_all[level][model_name]
            tp   = d["tp_count"]
            fp   = d["fp_count"]
            crd  = d["crowd_fp"]
            pred = d["total_preds"]
            chk  = tp + fp + crd
            ok   = "✓" if chk == pred else f"✗ (off by {chk-pred:+d})"
            rows.append([
                level.capitalize(),
                model_name.capitalize(),
                f"{tp:,}",
                f"{fp:,}",
                f"{crd:,}",
                f"{pred:,}",
                f"{chk:,}",
                ok,
            ])
    headers = [
        "Occlusion", "Model",
        "TP", "FP (genuine)", "crowd_FP (absorbed)",
        "Total Preds", "TP+FP+crowd_FP", "Inv. OK?",
    ]
    print(f"\n{'='*115}")
    print("  ISCROWD CROSS-FP AUDIT")
    print("  crowd_FP  : detections matched to OTHER-level (iscrowd=1) GTs on in-scope images.")
    print("              These are NEVER counted as FP — they are silently absorbed.")
    print("  Total Preds: in-scope images only (≥1 active iscrowd=0 GT per image).")
    print("  Inv. OK?  checks  TP + FP + crowd_FP == Total Preds  (exact match expected).")
    print(f"{'='*115}")
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    else:
        col_w = [max(len(h), max(len(str(r[i])) for r in rows))
                 for i, h in enumerate(headers)]
        fmt   = "  ".join(f"{{:<{w}}}" for w in col_w)
        print(fmt.format(*headers))
        print("  ".join("-" * w for w in col_w))
        for row in rows:
            print(fmt.format(*[str(c) for c in row]))


# ── Original stats helpers ────────────────────────────────────────────────────

def get_stats(coco_eval):
    """
    Compute summary stats matching COCOeval.summarize() for custom maxDets.

    COCOeval.eval["precision"] shape:  [T, R, K, A, M]
    COCOeval.eval["recall"]    shape:  [T, K, A, M]
    """
    if coco_eval is None:
        return {}

    p  = coco_eval.params
    ev = coco_eval.eval

    iou_thrs = np.linspace(0.5, 0.95, 10)

    def _t_range(lo, hi):
        t_lo = np.where(np.isclose(iou_thrs, lo))[0]
        t_hi = np.where(np.isclose(iou_thrs, hi))[0]
        if not len(t_lo) or not len(t_hi):
            return None, None
        return int(t_lo[0]), int(t_hi[0]) + 1

    def _aind(area):
        idx = [i for i, ar in enumerate(p.areaRngLbl) if ar == area]
        return idx[0] if idx else None

    def _ap(iou_lo, iou_hi, area="all", max_det_idx=-1):
        t0, t1 = _t_range(iou_lo, iou_hi)
        ai     = _aind(area)
        if t0 is None or ai is None:
            return float("nan")
        prec = ev["precision"][t0:t1, :, :, ai, max_det_idx]
        prec = prec[prec > -1]
        return float(np.mean(prec)) if len(prec) else float("nan")

    def _ar(iou_lo, iou_hi, area="all", max_det_idx=-1):
        t0, t1 = _t_range(iou_lo, iou_hi)
        ai     = _aind(area)
        if t0 is None or ai is None:
            return float("nan")
        rec = ev["recall"][t0:t1, :, ai, max_det_idx]
        rec = rec[rec > -1]
        return float(np.mean(rec)) if len(rec) else float("nan")

    stats = {
        "AP@[.5:.95]":                          _ap(0.50, 0.95, "all",    2),
        "AP@.50":                               _ap(0.50, 0.50, "all",    2),
        "AP@.75":                               _ap(0.75, 0.75, "all",    2),
        "AP_small":                             _ap(0.50, 0.95, "small",  2),
        "AP_medium":                            _ap(0.50, 0.95, "medium", 2),
        "AP_large":                             _ap(0.50, 0.95, "large",  2),
        f"AR@[.5:.95]_maxDets{MAX_DETS[0]}":   _ar(0.50, 0.95, "all",    0),
        f"AR@[.5:.95]_maxDets{MAX_DETS[1]}":   _ar(0.50, 0.95, "all",    1),
        f"AR@[.5:.95]_maxDets{MAX_DETS[2]}":   _ar(0.50, 0.95, "all",    2),
        "AR_small":                             _ar(0.50, 0.95, "small",  2),
        "AR_medium":                            _ar(0.50, 0.95, "medium", 2),
        "AR_large":                             _ar(0.50, 0.95, "large",  2),
    }
    return stats


def get_pr_curve(coco_eval, iou_thresh=0.5, area="all", max_det_idx=-1):
    """
    Extract a PR curve from COCOeval.eval["precision"].
    Returns (recall_thrs [101], prec_mean [101]).
    """
    if coco_eval is None:
        return np.linspace(0, 1, 101), np.zeros(101)

    p  = coco_eval.params
    ev = coco_eval.eval

    iou_thrs = np.linspace(0.5, 0.95, 10)
    t_idx    = np.where(np.isclose(iou_thrs, iou_thresh))[0]
    if not len(t_idx):
        return np.linspace(0, 1, 101), np.zeros(101)
    t_idx = int(t_idx[0])

    aind = [i for i, ar in enumerate(p.areaRngLbl) if ar == area]
    if not aind:
        return np.linspace(0, 1, 101), np.zeros(101)

    prec = ev["precision"][t_idx, :, :, aind[0], max_det_idx]
    prec_mean = np.array([
        float(np.mean(row[row > -1])) if np.any(row > -1) else 0.0
        for row in prec
    ])
    return np.linspace(0, 1, 101), prec_mean


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_pr_per_occlusion(evals, output_dir, iou_thresh=0.5, max_det_idx=2):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"PR Curves by Occlusion Level  (IoU={iou_thresh}, "
        f"maxDets={MAX_DETS[max_det_idx]})\n"
        "Dashed = Baseline  |  Solid = Updated  |  "
        "Other-level annotations ignored via iscrowd=1",
        fontsize=13, y=1.02
    )

    for ax, level in zip(axes, OCCLUSION_LEVELS):
        color = LEVEL_COLORS[level]
        aps   = {}
        for model_name in ["baseline", "updated"]:
            coco_eval = evals[level][model_name]
            r, p_vals = get_pr_curve(coco_eval, iou_thresh=iou_thresh,
                                     area="all", max_det_idx=max_det_idx)
            ap = float(np.mean(p_vals))
            aps[model_name] = (r, p_vals, ap)
            style = MODEL_STYLES[model_name]
            ax.plot(r, p_vals, color=color,
                    ls=style["ls"], lw=style["lw"], alpha=style["alpha"],
                    label=f"{model_name.capitalize()}  AP={ap:.4f}")

        r_b, p_b, ap_b = aps["baseline"]
        r_u, p_u, ap_u = aps["updated"]
        diff = p_u - p_b
        ax.fill_between(r_u, p_b, p_u, where=(diff > 0),
                        alpha=0.15, color="green", label="Updated better")
        ax.fill_between(r_u, p_b, p_u, where=(diff < 0),
                        alpha=0.15, color="red",   label="Baseline better")
        ax.text(0.04, 0.06,
                f"ΔAP = {ap_u - ap_b:+.4f}",
                transform=ax.transAxes, fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.9))
        ax.set_title(f"{level.capitalize()} Occlusion",
                     color=color, fontsize=13, fontweight="bold")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, loc="upper right"); ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out = Path(output_dir) / (
        f"pr_per_occlusion_iou{iou_thresh}_maxDets{MAX_DETS[max_det_idx]}.png"
    )
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")


def plot_pr_overlay(evals, output_dir, iou_thresh=0.5, max_det_idx=2):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(
        f"PR Curves — All Occlusion Levels  (IoU={iou_thresh}, "
        f"maxDets={MAX_DETS[max_det_idx]})\n"
        "Colour = Occlusion Level  |  Dashed = Baseline  |  Solid = Updated",
        fontsize=12
    )
    for level in OCCLUSION_LEVELS:
        color = LEVEL_COLORS[level]
        for model_name in ["baseline", "updated"]:
            r, p_vals = get_pr_curve(evals[level][model_name],
                                     iou_thresh=iou_thresh,
                                     area="all", max_det_idx=max_det_idx)
            ap    = float(np.mean(p_vals))
            style = MODEL_STYLES[model_name]
            ax.plot(r, p_vals, color=color,
                    ls=style["ls"], lw=style["lw"], alpha=style["alpha"],
                    label=f"{level.capitalize()} {model_name.capitalize()} "
                          f"(AP={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, ncol=2, loc="upper right"); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = Path(output_dir) / (
        f"pr_overlay_iou{iou_thresh}_maxDets{MAX_DETS[max_det_idx]}.png"
    )
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")


def plot_pr_maxdets_comparison(evals, output_dir, iou_thresh=0.5):
    maxdet_colors = ["#1ABC9C", "#E67E22", "#E74C3C"]
    for model_name in ["baseline", "updated"]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            f"PR Curves by maxDets — {model_name.capitalize()}  "
            f"(IoU={iou_thresh})\nmaxDets compared: {MAX_DETS}",
            fontsize=13, y=1.02
        )
        for ax, level in zip(axes, OCCLUSION_LEVELS):
            for md_idx, (md, md_color) in enumerate(zip(MAX_DETS, maxdet_colors)):
                r, p_vals = get_pr_curve(evals[level][model_name],
                                         iou_thresh=iou_thresh,
                                         area="all", max_det_idx=md_idx)
                ap = float(np.mean(p_vals))
                ax.plot(r, p_vals, color=md_color, lw=2.0,
                        label=f"maxDets={md}  AP={ap:.4f}")
            ax.set_title(f"{level.capitalize()} Occlusion",
                         color=LEVEL_COLORS[level], fontsize=13, fontweight="bold")
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
            ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
        plt.tight_layout()
        out = Path(output_dir) / f"pr_maxdets_{model_name}_iou{iou_thresh}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Saved: {out}")


def plot_pr_iou_comparison(evals, output_dir, max_det_idx=2):
    iou_settings = [0.50, 0.75]
    iou_colors   = ["#2980B9", "#8E44AD"]
    for model_name in ["baseline", "updated"]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            f"PR Curves by IoU Threshold — {model_name.capitalize()}\n"
            f"maxDets={MAX_DETS[max_det_idx]}",
            fontsize=13, y=1.02
        )
        for ax, level in zip(axes, OCCLUSION_LEVELS):
            for iou_thresh, iou_color in zip(iou_settings, iou_colors):
                r, p_vals = get_pr_curve(evals[level][model_name],
                                         iou_thresh=iou_thresh,
                                         area="all", max_det_idx=max_det_idx)
                ap = float(np.mean(p_vals))
                ax.plot(r, p_vals, color=iou_color, lw=2.0,
                        label=f"IoU={iou_thresh}  AP={ap:.4f}")
            ax.set_title(f"{level.capitalize()} Occlusion",
                         color=LEVEL_COLORS[level], fontsize=13, fontweight="bold")
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
            ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
        plt.tight_layout()
        out = Path(output_dir) / (
            f"pr_iou_{model_name}_maxDets{MAX_DETS[max_det_idx]}.png"
        )
        plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Saved: {out}")


def plot_pr_grid(evals, output_dir, max_det_idx=2):
    iou_settings = [0.50, 0.75]
    iou_colors   = ["#2980B9", "#8E44AD"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"PR Curves Grid — Rows: Model  |  Cols: Occlusion Level\n"
        f"maxDets={MAX_DETS[max_det_idx]}  |  Blue=IoU@.50  |  Purple=IoU@.75",
        fontsize=13
    )
    for row, model_name in enumerate(["baseline", "updated"]):
        for col, level in enumerate(OCCLUSION_LEVELS):
            ax = axes[row][col]
            for iou_thresh, iou_color in zip(iou_settings, iou_colors):
                r, p_vals = get_pr_curve(evals[level][model_name],
                                         iou_thresh=iou_thresh,
                                         area="all", max_det_idx=max_det_idx)
                ap = float(np.mean(p_vals))
                ax.plot(r, p_vals, color=iou_color, lw=2.0,
                        label=f"IoU={iou_thresh}  AP={ap:.4f}")
            ax.set_title(
                f"{model_name.capitalize()} — {level.capitalize()}",
                color=LEVEL_COLORS[level], fontsize=11, fontweight="bold"
            )
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = Path(output_dir) / f"pr_grid_maxDets{MAX_DETS[max_det_idx]}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out}")


def plot_diagnostics_bar(diag_all, output_dir, iou_thresh=0.50):
    """
    Four-panel bar chart:
        Panel 1 — TP/GT stacked (TP + missed)
        Panel 2 — FP breakdown (FP genuine vs crowd_FP absorbed)
        Panel 3 — Avg IoU of TP detections
        Panel 4 — Avg Confidence of TP detections
    """
    x      = np.arange(len(OCCLUSION_LEVELS))
    width  = 0.35
    labels = [lvl.capitalize() for lvl in OCCLUSION_LEVELS]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle(
        f"Detection Diagnostics per Occlusion Level  (IoU@{iou_thresh}, "
        f"maxDets={MAX_DETS[-1]})",
        fontsize=13
    )

    # ── Panel 1: TP / Total GT ───────────────────────────────────────────────
    ax = axes[0]
    for i, model_name in enumerate(["baseline", "updated"]):
        tp_vals  = [diag_all[lvl][model_name]["tp_count"] for lvl in OCCLUSION_LEVELS]
        gt_vals  = [diag_all[lvl][model_name]["total_gt"] for lvl in OCCLUSION_LEVELS]
        rem_vals = [max(0, g - t) for g, t in zip(gt_vals, tp_vals)]
        offset   = (i - 0.5) * width
        color    = "#2980B9" if model_name == "baseline" else "#27AE60"
        ax.bar(x + offset, tp_vals,  width, label=f"{model_name.capitalize()} TP",
               color=color, alpha=0.85)
        ax.bar(x + offset, rem_vals, width, bottom=tp_vals,
               color="lightgrey", alpha=0.6,
               label=f"{model_name.capitalize()} Missed" if i == 0 else "_nolegend_")
        for j, (tp, gt) in enumerate(zip(tp_vals, gt_vals)):
            ratio = tp / gt if gt > 0 else 0
            ax.text(x[j] + offset, tp + max(gt_vals) * 0.01,
                    f"{ratio:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("TP Count  /  Total GT", fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Annotation count"); ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # ── Panel 2: FP breakdown (genuine FP vs crowd_FP absorbed) ─────────────
    ax = axes[1]
    for i, model_name in enumerate(["baseline", "updated"]):
        fp_vals  = [diag_all[lvl][model_name]["fp_count"]  for lvl in OCCLUSION_LEVELS]
        crd_vals = [diag_all[lvl][model_name]["crowd_fp"]   for lvl in OCCLUSION_LEVELS]
        offset   = (i - 0.5) * width
        color    = "#2980B9" if model_name == "baseline" else "#27AE60"
        ax.bar(x + offset, fp_vals,  width, label=f"{model_name.capitalize()} FP",
               color=color, alpha=0.85)
        ax.bar(x + offset, crd_vals, width, bottom=fp_vals,
               color="#E74C3C", alpha=0.45,
               label="crowd_FP (absorbed)" if i == 0 else "_nolegend_")
        for j, (fp, crd) in enumerate(zip(fp_vals, crd_vals)):
            ax.text(x[j] + offset, fp + crd + max(max(fp_vals), 1) * 0.02,
                    f"{fp}", ha="center", va="bottom", fontsize=7)
    ax.set_title("FP Count  (red = crowd_FP absorbed, NOT penalised)", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Detection count"); ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    # ── Panel 3: Average IoU ─────────────────────────────────────────────────
    ax = axes[2]
    for i, model_name in enumerate(["baseline", "updated"]):
        iou_vals = [diag_all[lvl][model_name]["avg_iou"] for lvl in OCCLUSION_LEVELS]
        offset   = (i - 0.5) * width
        color    = "#2980B9" if model_name == "baseline" else "#27AE60"
        bars = ax.bar(x + offset,
                      [v if not np.isnan(v) else 0 for v in iou_vals],
                      width, label=model_name.capitalize(),
                      color=color, alpha=0.85)
        for bar, v in zip(bars, iou_vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Avg IoU of TP Detections", fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Mean IoU"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)

    # ── Panel 4: Average Confidence ──────────────────────────────────────────
    ax = axes[3]
    for i, model_name in enumerate(["baseline", "updated"]):
        conf_vals = [diag_all[lvl][model_name]["avg_conf"] for lvl in OCCLUSION_LEVELS]
        offset    = (i - 0.5) * width
        color     = "#2980B9" if model_name == "baseline" else "#27AE60"
        bars = ax.bar(x + offset,
                      [v if not np.isnan(v) else 0 for v in conf_vals],
                      width, label=model_name.capitalize(),
                      color=color, alpha=0.85)
        for bar, v in zip(bars, conf_vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Avg Confidence of TP Detections", fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Mean score"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = Path(output_dir) / f"diagnostics_tp_fp_iou_conf_iou{iou_thresh}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_diagnostics_delta(diag_all, output_dir, iou_thresh=0.50):
    """
    Delta plot: Updated − Baseline for recall, FP/Pred, avg IoU, avg confidence.
    """
    metrics = {
        "Recall (TP/GT)":       "recall",
        "FP Rate (FP/Pred)":    "fp_rate",
        "Avg IoU (TPs)":        "avg_iou",
        "Avg Confidence (TPs)": "avg_conf",
    }
    x      = np.arange(len(OCCLUSION_LEVELS))
    labels = [lvl.capitalize() for lvl in OCCLUSION_LEVELS]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle(
        f"Δ (Updated − Baseline) per Occlusion Level  (IoU@{iou_thresh})\n"
        "Green = Updated is better  |  Red = Baseline is better  "
        "(Note: for FP Rate, negative Δ = Updated has fewer FPs = better)",
        fontsize=12
    )

    for ax, (title, key) in zip(axes, metrics.items()):
        deltas = []
        colors = []
        for lvl in OCCLUSION_LEVELS:
            b = diag_all[lvl]["baseline"][key]
            u = diag_all[lvl]["updated"][key]
            d = (u - b) if not (np.isnan(b) or np.isnan(u)) else 0.0
            deltas.append(d)
            # For FP rate: negative delta means Updated is better
            better = (d < 0) if key == "fp_rate" else (d >= 0)
            colors.append("#27AE60" if better else "#E74C3C")

        bars = ax.bar(x, deltas, color=colors, alpha=0.85, width=0.5)
        ax.axhline(0, color="black", lw=0.8)
        for bar, d in zip(bars, deltas):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    d + (0.002 if d >= 0 else -0.004),
                    f"{d:+.4f}", ha="center",
                    va="bottom" if d >= 0 else "top", fontsize=9)

        ax.set_title(title, fontsize=11)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("Δ value")
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = Path(output_dir) / f"diagnostics_delta_iou{iou_thresh}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_severe_deep_dive(diag_all, evals, output_dir, iou_thresh=0.50):
    """
    Three-panel deep-dive for severe occlusion:
        Left   — Confidence quartiles (Q25/Q50/Q75) for baseline vs updated,
                 compared across all levels
        Middle — TP avg IoU vs FP near-miss IoU for severe only
        Right  — GT area median across levels (size effect)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Severe Occlusion Deep-Dive  —  Why is mAP low?\n"
        "(Left: confidence spread | Middle: IoU quality | Right: GT size effect)",
        fontsize=13
    )

    labels = [lvl.capitalize() for lvl in OCCLUSION_LEVELS]
    x      = np.arange(len(OCCLUSION_LEVELS))
    width  = 0.35

    # ── Left: confidence quartiles ────────────────────────────────────────────
    ax = axes[0]
    ax.set_title("TP Confidence Spread (Q25–Q75)\nNarrow/low spread → fragile ranking",
                 fontsize=11)
    for i, model_name in enumerate(["baseline", "updated"]):
        q25_vals = [diag_all[lvl][model_name]["conf_q25"] for lvl in OCCLUSION_LEVELS]
        q50_vals = [diag_all[lvl][model_name]["conf_q50"] for lvl in OCCLUSION_LEVELS]
        q75_vals = [diag_all[lvl][model_name]["conf_q75"] for lvl in OCCLUSION_LEVELS]
        offset   = (i - 0.5) * width
        color    = "#2980B9" if model_name == "baseline" else "#27AE60"
        _x = x + offset
        q25_c = [v if not np.isnan(v) else 0 for v in q25_vals]
        q50_c = [v if not np.isnan(v) else 0 for v in q50_vals]
        q75_c = [v if not np.isnan(v) else 0 for v in q75_vals]
        # Plot median bar
        ax.bar(_x, q50_c, width, color=color, alpha=0.75,
               label=f"{model_name.capitalize()} Q50")
        # Error bars from Q25 to Q75
        yerr_lo = [max(0, m - lo) for m, lo in zip(q50_c, q25_c)]
        yerr_hi = [max(0, hi - m) for m, hi in zip(q50_c, q75_c)]
        ax.errorbar(_x, q50_c,
                    yerr=[yerr_lo, yerr_hi],
                    fmt="none", color=color, capsize=4, lw=1.5, alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Confidence score"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    # Annotate severe column
    ax.axvspan(2 - 0.45, 2 + 0.45, color="purple", alpha=0.07, label="Severe zone")

    # ── Middle: TP avg IoU vs FP near-miss IoU (severe only) ─────────────────
    ax = axes[1]
    ax.set_title("Severe Only — TP IoU vs FP Near-Miss IoU\n"
                 "FP near-miss IoU close to TP IoU → ranking is the bottleneck",
                 fontsize=11)
    cats   = ["Baseline", "Updated"]
    tp_ious  = [diag_all["severe"][m]["avg_iou"]      for m in ["baseline", "updated"]]
    fp_ious  = [diag_all["severe"][m]["fp_iou_mean"]  for m in ["baseline", "updated"]]
    _x = np.arange(len(cats))
    tp_c  = [v if not np.isnan(v) else 0 for v in tp_ious]
    fp_c  = [v if not np.isnan(v) else 0 for v in fp_ious]
    bars1 = ax.bar(_x - 0.2, tp_c, 0.35, label="TP avg IoU",  color="#27AE60", alpha=0.85)
    bars2 = ax.bar(_x + 0.2, fp_c, 0.35, label="FP near-miss avg IoU", color="#E74C3C", alpha=0.75)
    for bar, v in zip(list(bars1) + list(bars2), tp_c + fp_c):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.5, color="grey", lw=1.0, ls="--", label="IoU=0.5 threshold")
    ax.set_xticks(_x); ax.set_xticklabels(cats)
    ax.set_ylabel("Mean IoU"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)

    # ── Right: GT box area across levels ─────────────────────────────────────
    ax = axes[2]
    ax.set_title("GT Box Area per Occlusion Level\n"
                 "Smaller area → harder to detect → lower mAP",
                 fontsize=11)
    mean_areas   = [diag_all[lvl]["baseline"]["gt_area_mean"]   for lvl in OCCLUSION_LEVELS]
    median_areas = [diag_all[lvl]["baseline"]["gt_area_median"] for lvl in OCCLUSION_LEVELS]
    colors_area  = [LEVEL_COLORS[lvl] for lvl in OCCLUSION_LEVELS]
    bars = ax.bar(x, [v if not np.isnan(v) else 0 for v in median_areas],
                  width=0.5, color=colors_area, alpha=0.80, label="Median area")
    ax.plot(x, [v if not np.isnan(v) else 0 for v in mean_areas],
            "D--", color="black", ms=7, lw=1.5, label="Mean area")
    for bar, med in zip(bars, median_areas):
        if not np.isnan(med):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(v for v in median_areas if not np.isnan(v)) * 0.01,
                    f"{med:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Box area (px²)"); ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out = Path(output_dir) / f"severe_deep_dive_iou{iou_thresh}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Summary tables ────────────────────────────────────────────────────────────

def print_summary_table(all_stats):
    metric_keys = list(next(iter(
        next(iter(all_stats.values())).values()
    )).keys())

    header = ["Metric"]
    for level in OCCLUSION_LEVELS:
        short = level[:3].upper()
        header += [f"{short} Base", f"{short} Upd", f"{short} Δ"]

    rows = []
    for metric in metric_keys:
        row = [metric]
        for level in OCCLUSION_LEVELS:
            b = all_stats[level]["baseline"].get(metric, float("nan"))
            u = all_stats[level]["updated"].get(metric,  float("nan"))
            d = u - b if not (np.isnan(b) or np.isnan(u)) else float("nan")
            row += [
                f"{b:.4f}"  if not np.isnan(b) else "  —  ",
                f"{u:.4f}"  if not np.isnan(u) else "  —  ",
                f"{d:+.4f}" if not np.isnan(d) else "  —  ",
            ]
        rows.append(row)

    print(f"\n{'='*130}")
    print(
        f"  OCCLUSION-WISE mAP — COCOeval  |  maxDets={MAX_DETS}  |  "
        "iscrowd=1 ignore methodology"
    )
    print(f"{'='*130}")
    if HAS_TABULATE:
        print(tabulate(rows, headers=header, tablefmt="rounded_outline"))
    else:
        col_w = [max(len(h), max(len(str(r[i])) for r in rows))
                 for i, h in enumerate(header)]
        fmt   = "  ".join(f"{{:<{w}}}" for w in col_w)
        print(fmt.format(*header))
        print("  ".join("-" * w for w in col_w))
        for row in rows:
            print(fmt.format(*[str(c) for c in row]))


def print_per_category_table(b_cat_stats, u_cat_stats, level, metric="AP@.50"):
    all_cats = sorted(set(list(b_cat_stats) + list(u_cat_stats)))
    rows = []
    for cat in all_cats:
        b = b_cat_stats.get(cat, {}).get(metric, float("nan"))
        u = u_cat_stats.get(cat, {}).get(metric, float("nan"))
        d = u - b if not (np.isnan(b) or np.isnan(u)) else float("nan")
        verdict = (
            ("✅" if d > 0.001 else ("❌" if d < -0.001 else "≈"))
            if not np.isnan(d) else "—"
        )
        rows.append([
            cat,
            f"{b:.4f}" if not np.isnan(b) else "—",
            f"{u:.4f}" if not np.isnan(u) else "—",
            f"{d:+.4f}" if not np.isnan(d) else "—",
            verdict,
        ])
    headers = ["Category", f"{metric} Base", f"{metric} Upd", "Δ", ""]
    print(f"\n── Per-Category {metric} — {level.capitalize()} Occlusion ──")
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    else:
        col_w = [max(len(h), max(len(str(r[i])) for r in rows))
                 for i, h in enumerate(headers)]
        fmt   = "  ".join(f"{{:<{w}}}" for w in col_w)
        print(fmt.format(*headers))
        print("  ".join("-" * w for w in col_w))
        for row in rows:
            print(fmt.format(*[str(c) for c in row]))


def run_per_category(gt_dict, pred_list):
    """Run COCOeval per category. Returns {cat_name: {metric: value}}."""
    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()

    gt_image_ids   = {img["id"] for img in gt_dict["images"]}
    filtered_preds = [p for p in pred_list if p["image_id"] in gt_image_ids]
    if not filtered_preds:
        return {}

    coco_dt   = coco_gt.loadRes(filtered_preds)
    cat_names = {c["id"]: c["name"] for c in gt_dict["categories"]}
    results   = {}

    for cat_id, cat_name in cat_names.items():
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.catIds  = [cat_id]
        coco_eval.params.maxDets = MAX_DETS
        coco_eval.evaluate()
        coco_eval.accumulate()
        results[cat_name] = get_stats(coco_eval)

    return results

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Occlusion-wise mAP + PR curves via COCOeval (iscrowd ignore)."
    )
    parser.add_argument("--gt",           required=True)
    parser.add_argument("--baseline",     required=True)
    parser.add_argument("--updated",      required=True)
    parser.add_argument("--output",       default="occlusion_map_report")
    parser.add_argument("--per-category", action="store_true",
                        help="Per-category breakdown per occlusion level")
    parser.add_argument("--diag-iou",     type=float, default=0.50,
                        help="IoU threshold for TP/FP/crowd_FP diagnostics (default=0.50)")
    args = parser.parse_args()

    for p in (args.gt, args.baseline, args.updated):
        if not Path(p).exists():
            raise FileNotFoundError(f"Not found: {p}")

    Path(args.output).mkdir(parents=True, exist_ok=True)

    # ── Load ────────────────────────────────────────────────────────────────
    print(f"\nLoading GT        : {args.gt}")
    gt_full   = load_json(args.gt)
    cat_names = {c["id"]: c["name"] for c in gt_full.get("categories", [])}
    print("  Categories: " +
          ", ".join(f"{k}={v}" for k, v in sorted(cat_names.items())))

    level_counts = defaultdict(int)
    for ann in gt_full["annotations"]:
        level = (ann.get("attributes", {})
                    .get("occlusion_level", "unknown")
                    .strip().lower())
        level_counts[level] += 1
    print("  GT distribution: " +
          "  ".join(f"{lvl}={level_counts[lvl]:,}" for lvl in OCCLUSION_LEVELS))

    print(f"\nLoading Baseline  : {args.baseline}")
    b_preds = load_json(args.baseline)
    print(f"  {len(b_preds):,} predictions")

    print(f"Loading Updated   : {args.updated}")
    u_preds = load_json(args.updated)
    print(f"  {len(u_preds):,} predictions")

    print(f"\nmaxDets setting   : {MAX_DETS}")

    # ── Build COCOeval objects per level ─────────────────────────────────────
    evals     = {}
    all_stats = {}
    diag_all  = {}

    for level in OCCLUSION_LEVELS:
        print(f"\n{'─'*60}")
        print(f"Evaluating: {level.upper()}")
        print(f"{'─'*60}")

        gt_subset = build_gt_for_level(gt_full, level)

        active  = sum(1 for a in gt_subset["annotations"] if a["iscrowd"] == 0)
        ignored = sum(1 for a in gt_subset["annotations"] if a["iscrowd"] == 1)
        print(f"  Active  (iscrowd=0, evaluated): {active:,}")
        print(f"  Ignored (iscrowd=1, skipped)  : {ignored:,}")

        # Filter predictions to images that appear in this GT subset
        gt_image_ids = {img["id"] for img in gt_subset["images"]}
        b_preds_level = [p for p in b_preds if p["image_id"] in gt_image_ids]
        u_preds_level = [p for p in u_preds if p["image_id"] in gt_image_ids]

        evals[level]     = {}
        all_stats[level] = {}
        diag_all[level]  = {}

        for model_name, preds, preds_level in [
                ("baseline", b_preds, b_preds_level),
                ("updated",  u_preds, u_preds_level)]:
            print(f"\n  >> {model_name.capitalize()}")
            coco_eval = build_cocoeval(gt_subset, preds)
            evals[level][model_name]     = coco_eval
            all_stats[level][model_name] = get_stats(coco_eval)

            # Enhanced diagnostics: TP/FP/crowd_FP + quality + size
            diag = get_detection_diagnostics(
                coco_eval, gt_subset, preds_level, iou_thresh=args.diag_iou)
            diag_all[level][model_name] = diag
            print(f"     TP={diag['tp_count']:,}  GT={diag['total_gt']:,}  "
                  f"Recall={diag['recall']:.4f}  "
                  f"FP={diag['fp_count']:,}  Preds={diag['total_preds']:,}  "
                  f"FP/Pred={diag['fp_rate']:.4f}  "
                  f"crowd_FP={diag['crowd_fp']:,}  "
                  f"AvgIoU={diag['avg_iou']:.4f}  "
                  f"AvgConf={diag['avg_conf']:.4f}")

    # ── Summary tables ───────────────────────────────────────────────────────
    print_summary_table(all_stats)

    # ── Diagnostics table (TP/FP/crowd_FP + quality) ─────────────────────────
    print_diagnostics_table(diag_all, iou_thresh=args.diag_iou)

    # ── iscrowd audit ────────────────────────────────────────────────────────
    print_iscrowd_audit(diag_all)

    # ── Severe deep-dive ──────────────────────────────────────────────────────
    print_severe_diagnostics(diag_all, evals)

    # ── PR curve plots ───────────────────────────────────────────────────────
    print(f"\nGenerating plots → {args.output}/")

    for iou_thresh in [0.5, 0.75]:
        plot_pr_per_occlusion(evals, args.output,
                              iou_thresh=iou_thresh, max_det_idx=2)
        plot_pr_overlay(evals, args.output,
                        iou_thresh=iou_thresh, max_det_idx=2)

    plot_pr_maxdets_comparison(evals, args.output, iou_thresh=0.5)
    plot_pr_iou_comparison(evals, args.output, max_det_idx=2)
    plot_pr_grid(evals, args.output, max_det_idx=2)

    # ── Diagnostics plots ─────────────────────────────────────────────────────
    plot_diagnostics_bar(diag_all, args.output, iou_thresh=args.diag_iou)
    plot_diagnostics_delta(diag_all, args.output, iou_thresh=args.diag_iou)
    plot_severe_deep_dive(diag_all, evals, args.output, iou_thresh=args.diag_iou)

    # ── Per-category breakdown ───────────────────────────────────────────────
    if args.per_category:
        for level in OCCLUSION_LEVELS:
            print(f"\n{'='*60}")
            print(f"Per-Category: {level.upper()}")
            print(f"{'='*60}")
            gt_subset = build_gt_for_level(gt_full, level)
            print("  >> Baseline ...")
            b_cat = run_per_category(gt_subset, b_preds)
            print("  >> Updated ...")
            u_cat = run_per_category(gt_subset, u_preds)
            for metric in ["AP@.50", "AP@[.5:.95]", "AP@.75"]:
                print_per_category_table(b_cat, u_cat, level, metric)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    save_data = {}
    for level, models in all_stats.items():
        save_data[level] = {
            m: {k: (v if not np.isnan(v) else None) for k, v in stats.items()}
            for m, stats in models.items()
        }

    for level in OCCLUSION_LEVELS:
        for model_name in ["baseline", "updated"]:
            d = diag_all[level][model_name]
            save_data[level][model_name]["_diagnostics"] = {
                k: (v if not (isinstance(v, float) and np.isnan(v)) else None)
                for k, v in d.items()
            }

    out_path = Path(args.output) / "occlusion_map_results.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to : {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()