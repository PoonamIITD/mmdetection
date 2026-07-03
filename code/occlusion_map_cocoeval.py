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

Requirements:
    pip install pycocotools matplotlib tabulate

Usage:
    python occlusion_map_cocoeval.py \
        --gt        instances_validation_merged_mannual.json \
        --baseline  results_baseline_GDINO_final_val.json \
        --updated   results_sampling_loss_final_val.json \
        --output    occlusion_map_report \
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
# Only the 3 real occlusion levels — no synthetic "all"
OCCLUSION_LEVELS = ["light", "moderate", "severe"]
MAX_DETS         = [100, 300, 1000]

# Custom area thresholds scaled for 1920x1080 RSUD images
# (COCO's 32²/96² were derived for ~640x480 images; scale factor ≈ sqrt(2,073,600/307,200) ≈ 2.6, 32*2.6=83, 96*2.6=250)
AREA_RNG = [
    [0 ** 2,   1e5 ** 2],   # all
    [0 ** 2,   83 ** 2],    # small   : area < 6,889 px²
    [83 ** 2,  250 ** 2],   # medium  : 6,889–62,500 px²
    [250 ** 2, 1e5 ** 2],   # large   : area > 62,500 px²
]
AREA_RNG_LBL = ["all", "small", "medium", "large"]

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

    Parameters
    ----------
    gt_full      : dict  – full COCO-format ground-truth loaded from JSON
    target_level : str   – one of {"light", "moderate", "severe"}
    """
    gt_subset = copy.deepcopy(gt_full)
    for ann in gt_subset["annotations"]:
        level = (ann.get("attributes", {})
                    .get("occlusion_level", "unknown")
                    .strip().lower())
        # iscrowd=0 → COCOeval counts this annotation normally
        # iscrowd=1 → COCOeval silently ignores this annotation
        ann["iscrowd"] = 0 if level == target_level else 1
    return gt_subset

# ── COCOeval runner ───────────────────────────────────────────────────────────

def build_cocoeval(gt_dict, pred_list, iou_type="bbox"):
    """
    Build and run COCOeval with custom maxDets.

    The iscrowd flags set by build_gt_for_level() are respected automatically:
    COCOeval.evaluateImg() skips iscrowd=1 annotations from TP/FP accounting.

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

    # Custom maxDets — replace COCOeval default maxDets and areaRng
    coco_eval.params.maxDets = MAX_DETS
    coco_eval.params.areaRng    = AREA_RNG   
    coco_eval.params.areaRngLbl = AREA_RNG_LBL


    coco_eval.evaluate()    # calls evaluateImg() per image/category/area/maxDet
    coco_eval.accumulate()  # builds precision/recall arrays from evaluateImg results
    return coco_eval


def get_stats(coco_eval):
    """
    Compute summary stats matching COCOeval.summarize() for custom maxDets.

    COCOeval.eval["precision"] shape:
        [T, R, K, A, M]
        T = IoU thresholds  (10: 0.50 … 0.95)
        R = recall points   (101: 0 … 1)
        K = categories
        A = area ranges     (all, small, medium, large)
        M = maxDets levels  (indices into MAX_DETS)

    COCOeval.eval["recall"] shape:
        [T, K, A, M]
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

    # maxDets indices: 0 → MAX_DETS[0], 1 → MAX_DETS[1], 2 → MAX_DETS[2]
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
    Extract a PR curve directly from COCOeval.eval["precision"].

    precision array dimensions:
        [T, R, K, A, M]
        T = IoU thresholds, R = recall points (101), K = categories,
        A = area ranges,    M = maxDets

    Returns
    -------
    recall_thrs  : np.ndarray  shape (101,)   0 … 1
    prec_mean    : np.ndarray  shape (101,)   mean precision across categories
                               (-1 entries, meaning "not evaluated", are excluded)
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

    # precision[t, :, :, area, maxdet]  →  shape [101, K]
    prec = ev["precision"][t_idx, :, :, aind[0], max_det_idx]

    # Average across categories; -1 means "not evaluated for this category"
    prec_mean = np.array([
        float(np.mean(row[row > -1])) if np.any(row > -1) else 0.0
        for row in prec          # prec shape is [101, K]; iterate over recall points
    ])

    return np.linspace(0, 1, 101), prec_mean

# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_pr_per_occlusion(evals, output_dir, iou_thresh=0.5, max_det_idx=2):
    """
    3-panel figure: one panel per occlusion level (light / moderate / severe).
    Each panel shows baseline (dashed) vs updated (solid).
    Green/red fill shows where updated is better/worse.
    """
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
        ax.set_xlabel("Recall",    fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out = Path(output_dir) / (
        f"pr_per_occlusion_iou{iou_thresh}_maxDets{MAX_DETS[max_det_idx]}.png"
    )
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_pr_overlay(evals, output_dir, iou_thresh=0.5, max_det_idx=2):
    """
    All 3 occlusion levels on one plot.
    Colour = occlusion level  |  style = model (dashed baseline, solid updated).
    """
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

    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = Path(output_dir) / (
        f"pr_overlay_iou{iou_thresh}_maxDets{MAX_DETS[max_det_idx]}.png"
    )
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_pr_maxdets_comparison(evals, output_dir, iou_thresh=0.5):
    """
    3-panel figure per model: each panel is one occlusion level,
    showing PR curves at each of the 3 maxDets values.
    """
    maxdet_colors = ["#1ABC9C", "#E67E22", "#E74C3C"]

    for model_name in ["baseline", "updated"]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            f"PR Curves by maxDets — {model_name.capitalize()}  "
            f"(IoU={iou_thresh})\n"
            f"maxDets compared: {MAX_DETS}",
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
            ax.set_xlabel("Recall",    fontsize=11)
            ax.set_ylabel("Precision", fontsize=11)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.25)

        plt.tight_layout()
        out = Path(output_dir) / f"pr_maxdets_{model_name}_iou{iou_thresh}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")


def plot_pr_iou_comparison(evals, output_dir, max_det_idx=2):
    """
    3-panel figure per model: each panel is one occlusion level,
    showing PR curves at IoU=0.50 and IoU=0.75.
    """
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
            ax.set_xlabel("Recall",    fontsize=11)
            ax.set_ylabel("Precision", fontsize=11)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.25)

        plt.tight_layout()
        out = Path(output_dir) / (
            f"pr_iou_{model_name}_maxDets{MAX_DETS[max_det_idx]}.png"
        )
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")


def plot_pr_grid(evals, output_dir, max_det_idx=2):
    """
    2 × 3 grid: rows = baseline / updated, cols = light / moderate / severe.
    Both IoU=0.50 and IoU=0.75 overlaid per panel.
    """
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
            ax.set_xlabel("Recall",    fontsize=10)
            ax.set_ylabel("Precision", fontsize=10)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out = Path(output_dir) / f"pr_grid_maxDets{MAX_DETS[max_det_idx]}.png"
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

    for level in OCCLUSION_LEVELS:
        print(f"\n{'─'*60}")
        print(f"Evaluating: {level.upper()}")
        print(f"{'─'*60}")

        # Set iscrowd=0 for target level, iscrowd=1 for all others.
        # COCOeval.evaluateImg() uses this to silently ignore non-target
        # annotations — no manual TP/FP adjustments needed.
        gt_subset = build_gt_for_level(gt_full, level)

        active  = sum(1 for a in gt_subset["annotations"] if a["iscrowd"] == 0)
        ignored = sum(1 for a in gt_subset["annotations"] if a["iscrowd"] == 1)
        print(f"  Active  (iscrowd=0, evaluated): {active:,}")
        print(f"  Ignored (iscrowd=1, skipped)  : {ignored:,}")

        evals[level]     = {}
        all_stats[level] = {}

        for model_name, preds in [("baseline", b_preds), ("updated", u_preds)]:
            print(f"\n  >> {model_name.capitalize()}")
            coco_eval = build_cocoeval(gt_subset, preds)
            evals[level][model_name]     = coco_eval
            all_stats[level][model_name] = get_stats(coco_eval)

    # ── Summary table ────────────────────────────────────────────────────────
    print_summary_table(all_stats)

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
    out_path = Path(args.output) / "occlusion_map_results.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to : {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()