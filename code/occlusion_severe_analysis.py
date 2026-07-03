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

NEW (severe deep-dive):
  --severe-analysis   Runs three extra analyses on the "severe" split only:
    1. Per-category AP breakdown sorted ascending — shows which categories
       drag overall severe AP down.
    2. TP score distribution — histograms of detection confidence scores for
       true-positive detections (matched to iscrowd=0 severe annotations at
       IoU≥0.50).  Clustering near 0.25 implies the model barely fires on them.
    3. GT count vs AP scatter — plots per-category annotation count against
       AP@.50 to expose underrepresented classes.

Requirements:
    pip install pycocotools matplotlib tabulate

Usage:
    python occlusion_map_cocoeval.py \
        --gt        instances_validation_merged_mannual.json \
        --baseline  results_baseline_GDINO_final_val.json \
        --updated   results_sampling_loss_final_val.json \
        --output    occlusion_map_report \
        --per-category \
        --severe-analysis
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
    coco_eval.evaluate()
    coco_eval.accumulate()
    return coco_eval


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
    Extract a PR curve directly from COCOeval.eval["precision"].
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

# ── Plots (original) ──────────────────────────────────────────────────────────

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


# ══════════════════════════════════════════════════════════════════════════════
# ── NEW: Severe deep-dive analysis ───────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _get_severe_gt_ann_ids(gt_dict):
    """
    Return the set of annotation IDs whose occlusion_level == "severe"
    AND iscrowd == 0 (i.e. the active target annotations, not the ignored ones).
    Call this on the already-masked gt_dict from build_gt_for_level("severe").
    """
    return {
        ann["id"]
        for ann in gt_dict["annotations"]
        if ann.get("iscrowd", 1) == 0
    }


# ── Analysis 1: Per-category AP sorted ascending ──────────────────────────────

def plot_severe_per_category_ap(gt_dict, b_preds, u_preds, output_dir):
    """
    Bar chart: per-category AP@.50 for severe occlusion, sorted ascending.
    Baseline (red) vs Updated (blue) side-by-side.
    Immediately shows which categories drag overall severe AP down.
    """
    print("\n  [Severe deep-dive] Running per-category AP (severe) ...")
    b_cat = run_per_category(gt_dict, b_preds)
    u_cat = run_per_category(gt_dict, u_preds)

    categories = sorted(b_cat.keys())
    metric     = "AP@.50"

    b_vals = [b_cat[c].get(metric, float("nan")) for c in categories]
    u_vals = [u_cat[c].get(metric, float("nan")) for c in categories]

    # Sort by baseline AP ascending — worst categories first on the left
    order      = np.argsort([v if not np.isnan(v) else -1 for v in b_vals])
    categories = [categories[i] for i in order]
    b_vals     = [b_vals[i]     for i in order]
    u_vals     = [u_vals[i]     for i in order]

    x   = np.arange(len(categories))
    w   = 0.38
    fig, ax = plt.subplots(figsize=(max(10, len(categories) * 1.1), 6))

    bars_b = ax.bar(x - w / 2, b_vals, w, label="Baseline",
                    color="#E74C3C", alpha=0.85, zorder=3)
    bars_u = ax.bar(x + w / 2, u_vals, w, label="Updated",
                    color="#2980B9", alpha=0.85, zorder=3)

    # Annotate delta above the taller bar
    for i, (bv, uv) in enumerate(zip(b_vals, u_vals)):
        if not (np.isnan(bv) or np.isnan(uv)):
            delta = uv - bv
            top   = max(bv, uv) + 0.01
            color = "green" if delta >= 0 else "red"
            ax.text(i, min(top + 0.02, 1.02), f"{delta:+.2f}",
                    ha="center", va="bottom", fontsize=7.5,
                    color=color, fontweight="bold")

    ax.axhline(0.25, color="orange", ls=":", lw=1.5,
               label="Score threshold ≈ 0.25 reference")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("AP @ IoU=0.50", fontsize=11)
    ax.set_title(
        "Per-Category AP@.50 — Severe Occlusion  (sorted by Baseline AP)\n"
        "Red = Baseline  |  Blue = Updated  |  Δ annotated above bars",
        fontsize=12
    )
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    plt.tight_layout()

    out = Path(output_dir) / "severe_per_category_ap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out}")

    # ── Console table sorted the same way ───────────────────────────────────
    print(f"\n── Per-Category AP@.50 — SEVERE (sorted ascending by Baseline) ──")
    rows = []
    for cat, bv, uv in zip(categories, b_vals, u_vals):
        d = uv - bv if not (np.isnan(bv) or np.isnan(uv)) else float("nan")
        verdict = ("✅" if d > 0.001 else ("❌" if d < -0.001 else "≈")) \
                  if not np.isnan(d) else "—"
        rows.append([
            cat,
            f"{bv:.4f}" if not np.isnan(bv) else "—",
            f"{uv:.4f}" if not np.isnan(uv) else "—",
            f"{d:+.4f}" if not np.isnan(d) else "—",
            verdict,
        ])
    hdrs = ["Category", "Base AP@.50", "Upd AP@.50", "Δ", ""]
    if HAS_TABULATE:
        print(tabulate(rows, headers=hdrs, tablefmt="rounded_outline"))
    else:
        for row in rows:
            print("  ".join(str(c) for c in row))

    return b_cat, u_cat   # returned for reuse by scatter plot


# ── Analysis 2: TP score distribution ────────────────────────────────────────

def _collect_tp_scores(coco_eval, iou_thresh=0.50, area="all", max_det_idx=2):
    """
    Collect TP/FP scores from a single evalImgs slice to avoid double-counting.
    Restricts to one area range and one maxDets level.
    """
    if coco_eval is None:
        return [], []

    p        = coco_eval.params
    iou_thrs = np.array(p.iouThrs)
    t_idx    = np.where(np.isclose(iou_thrs, iou_thresh))[0]
    if not len(t_idx):
        return [], []
    t = int(t_idx[0])

    # ── Resolve the area-range and maxDets indices we want ──────────────────
    try:
        a_idx = p.areaRngLbl.index(area)        # e.g. "all" → 0
    except ValueError:
        a_idx = 0

    # evalImgs is indexed as:
    #   [img_idx * (n_cats * n_areas * n_maxdets)
    #    + cat_idx * (n_areas * n_maxdets)
    #    + area_idx * n_maxdets
    #    + maxdet_idx]
    # We only want area_idx == a_idx and maxdet_idx == max_det_idx.
    n_cats    = len(p.catIds)   if p.catIds   else len(coco_eval.cocoGt.getCatIds())
    n_areas   = len(p.areaRng)
    n_maxdets = len(p.maxDets)

    score_lookup = {
        ann_id: ann["score"]
        for ann_id, ann in coco_eval.cocoDt.anns.items()
    }

    tp_scores, fp_scores = [], []
    seen_det_ids = set()   # guard against any residual duplicates

    for ei_idx, ei in enumerate(coco_eval.evalImgs):
        if ei is None:
            continue

        # ── Filter to the single (area, maxDets) slice we care about ────────
        # Position of this evalImg within the (area × maxDets) block
        slot = ei_idx % (n_areas * n_maxdets)
        this_area_idx   = slot // n_maxdets
        this_maxdet_idx = slot  % n_maxdets

        if this_area_idx != a_idx or this_maxdet_idx != max_det_idx:
            continue   # skip all other slices — avoids double-counting

        dt_ids     = ei["dtIds"]
        dt_matches = ei["dtMatches"]   # shape [T, D]
        dt_ignore  = ei["dtIgnore"]    # shape [T, D]

        for d, det_id in enumerate(dt_ids):
            if det_id in seen_det_ids:
                continue
            seen_det_ids.add(det_id)

            score = score_lookup.get(det_id, float("nan"))
            if np.isnan(score):
                continue

            ignored = dt_ignore[t, d]
            matched = dt_matches[t, d]

            if ignored:
                continue
            elif matched > 0:
                tp_scores.append(float(score))
            else:
                fp_scores.append(float(score))

    return tp_scores, fp_scores

def plot_severe_tp_score_distribution(gt_dict, b_preds, u_preds, output_dir,
                                      iou_thresh=0.50, score_threshold=0.25):
    """
    Histogram of TP detection confidence scores for severe occlusion.

    Key diagnostic: if TP scores cluster near `score_threshold` (default 0.25),
    the model is barely detecting severe objects — it fires just above the
    minimum confidence required to be counted.

    Layout:  2 side-by-side histograms (Baseline | Updated).
    Each panel shows:
        - Blue bars  = TP score distribution
        - Red bars   = FP score distribution (stacked / overlaid)
        - Orange dashed vertical line at `score_threshold`
        - Stats text: mean TP score, % TPs below threshold+0.1 buffer zone
    """
    print("\n  [Severe deep-dive] Collecting TP/FP scores (severe) ...")

    results = {}
    for model_name, preds in [("baseline", b_preds), ("updated", u_preds)]:
        coco_gt = COCO()
        coco_gt.dataset = gt_dict
        coco_gt.createIndex()

        gt_img_ids = {img["id"] for img in gt_dict["images"]}
        filtered   = [p for p in preds if p["image_id"] in gt_img_ids]
        if not filtered:
            results[model_name] = ([], [])
            continue

        coco_dt   = coco_gt.loadRes(filtered)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.maxDets = MAX_DETS
        coco_eval.evaluate()
        coco_eval.accumulate()

        tp, fp = _collect_tp_scores(coco_eval, iou_thresh=iou_thresh)
        results[model_name] = (tp, fp)
        print(f"    {model_name}: {len(tp)} TPs, {len(fp)} FPs at IoU≥{iou_thresh}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"TP / FP Score Distributions — Severe Occlusion  (IoU≥{iou_thresh})\n"
        f"Orange dashed = score threshold ({score_threshold}).  "
        "Clustering of TPs near threshold → model barely detects severe objects.",
        fontsize=12
    )

    bins = np.linspace(0.0, 1.0, 41)   # 0.025-wide bins

    for ax, model_name in zip(axes, ["baseline", "updated"]):
        tp_s, fp_s = results[model_name]

        ax.hist(fp_s, bins=bins, color="#E74C3C", alpha=0.55,
                label=f"FP  n={len(fp_s)}", zorder=2)
        ax.hist(tp_s, bins=bins, color="#2980B9", alpha=0.75,
                label=f"TP  n={len(tp_s)}", zorder=3)

        ax.axvline(score_threshold, color="orange", ls="--", lw=2.0,
                   label=f"Score threshold = {score_threshold}")

        # "Near-threshold" zone: fraction of TPs within [threshold, threshold+0.15]
        buffer = 0.15
        near   = sum(score_threshold <= s <= score_threshold + buffer
                     for s in tp_s)
        pct    = 100 * near / len(tp_s) if tp_s else 0
        mean_s = float(np.mean(tp_s)) if tp_s else float("nan")

        ax.axvspan(score_threshold, score_threshold + buffer,
                   alpha=0.08, color="orange",
                   label=f"Near-threshold zone ({pct:.1f}% of TPs)")

        ax.text(0.97, 0.97,
                f"Mean TP score: {mean_s:.3f}\n"
                f"TPs in [{score_threshold:.2f}–{score_threshold+buffer:.2f}]: "
                f"{near}/{len(tp_s)}  ({pct:.1f}%)",
                transform=ax.transAxes, fontsize=9, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.9))

        ax.set_title(f"{model_name.capitalize()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Detection confidence score", fontsize=11)
        ax.set_ylabel("Count",                      fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    out = Path(output_dir) / f"severe_tp_score_distribution_iou{iou_thresh}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out}")

    # ── Per-category breakdown of TP score stats ─────────────────────────────
    _print_tp_score_stats_by_category(gt_dict, b_preds, u_preds,
                                      iou_thresh, score_threshold)


def _print_tp_score_stats_by_category(gt_dict, b_preds, u_preds,
                                      iou_thresh=0.50, score_threshold=0.25):
    """
    For each category, print: n_TP, mean TP score, % near threshold.
    Helps identify whether low-AP categories also have low-confidence TPs.
    """
    cat_names = {c["id"]: c["name"] for c in gt_dict["categories"]}

    print(f"\n── TP Score Stats per Category — Severe (IoU≥{iou_thresh}) ──")

    for model_name, preds in [("baseline", b_preds), ("updated", u_preds)]:
        coco_gt = COCO()
        coco_gt.dataset = gt_dict
        coco_gt.createIndex()

        gt_img_ids = {img["id"] for img in gt_dict["images"]}
        filtered   = [p for p in preds if p["image_id"] in gt_img_ids]
        if not filtered:
            continue

        coco_dt   = coco_gt.loadRes(filtered)

        rows = []
        for cat_id, cat_name in sorted(cat_names.items(), key=lambda x: x[1]):
            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
            coco_eval.params.catIds  = [cat_id]
            coco_eval.params.maxDets = MAX_DETS
            coco_eval.evaluate()
            coco_eval.accumulate()

            tp_s, _ = _collect_tp_scores(coco_eval, iou_thresh=iou_thresh)
            if not tp_s:
                rows.append([cat_name, 0, "—", "—", "—"])
                continue

            mean_s = float(np.mean(tp_s))
            buffer = 0.15
            near   = sum(score_threshold <= s <= score_threshold + buffer
                         for s in tp_s)
            pct    = 100 * near / len(tp_s)

            # Flag: if mean TP score is within 0.10 of threshold → concerning
            flag = " ⚠️" if mean_s < score_threshold + 0.10 else ""
            rows.append([
                cat_name + flag,
                len(tp_s),
                f"{mean_s:.3f}",
                f"{pct:.1f}%",
                f"{np.percentile(tp_s, 25):.3f} / "
                f"{np.percentile(tp_s, 50):.3f} / "
                f"{np.percentile(tp_s, 75):.3f}",
            ])

        hdrs = ["Category", "n_TP", "Mean Score",
                f"% in [{score_threshold:.2f}–{score_threshold+0.15:.2f}]",
                "Q25/Q50/Q75"]
        print(f"\n  ── {model_name.capitalize()} ──")
        if HAS_TABULATE:
            print(tabulate(rows, headers=hdrs, tablefmt="rounded_outline"))
        else:
            for row in rows:
                print("  ".join(str(c) for c in row))


# ── Analysis 3: GT count vs AP scatter (data sufficiency) ────────────────────

def plot_severe_gt_count_vs_ap(gt_dict, b_cat_stats, u_cat_stats, output_dir):
    """
    Scatter plot: x = GT annotation count for severe level per category,
                  y = AP@.50 for that category.

    Interpretation:
      - Categories in the bottom-left quadrant (few GTs, low AP) are
        *both* underrepresented in training signal *and* performing poorly.
      - A negative correlation across the scatter confirms that count is a
        limiting factor; flat/no correlation suggests other issues.
      - A vertical dashed line at a "minimum viable count" (default 50)
        flags categories at risk of being underrepresented.

    Both models plotted together (baseline=circles, updated=triangles).
    """
    print("\n  [Severe deep-dive] Building GT count vs AP scatter (severe) ...")

    # Count severe GT annotations per category from the masked gt_dict
    # (iscrowd=0 annotations = severe target level)
    cat_names  = {c["id"]: c["name"] for c in gt_dict["categories"]}
    cat_counts = defaultdict(int)
    for ann in gt_dict["annotations"]:
        if ann.get("iscrowd", 1) == 0:
            cat_counts[ann["category_id"]] += 1

    # Build (count, ap) pairs
    data = {}
    for cat_id, cat_name in cat_names.items():
        count = cat_counts.get(cat_id, 0)
        b_ap  = b_cat_stats.get(cat_name, {}).get("AP@.50", float("nan"))
        u_ap  = u_cat_stats.get(cat_name, {}).get("AP@.50", float("nan"))
        data[cat_name] = {"count": count, "baseline_ap": b_ap, "updated_ap": u_ap}

    cats   = sorted(data.keys())
    counts = [data[c]["count"]      for c in cats]
    b_aps  = [data[c]["baseline_ap"] for c in cats]
    u_aps  = [data[c]["updated_ap"]  for c in cats]

    MIN_VIABLE_COUNT = 50   # heuristic: fewer than this risks underfitting

    fig, ax = plt.subplots(figsize=(11, 7))

    # Scatter: baseline (circles) and updated (triangles)
    sc_b = ax.scatter(counts, b_aps, marker="o", s=90,
                      color="#E74C3C", alpha=0.85, zorder=4, label="Baseline")
    sc_u = ax.scatter(counts, u_aps, marker="^", s=90,
                      color="#2980B9", alpha=0.85, zorder=4, label="Updated")

    # Connect each category's two points with a thin line to show Δ
    for cnt, b, u in zip(counts, b_aps, u_aps):
        if not (np.isnan(b) or np.isnan(u)):
            color = "green" if u >= b else "red"
            ax.plot([cnt, cnt], [b, u], color=color, lw=1.2, alpha=0.5, zorder=3)

    # Label every point with category name
    for cat, cnt, b, u in zip(cats, counts, b_aps, u_aps):
        best_y = max(v for v in [b, u] if not np.isnan(v)) if \
                 any(not np.isnan(v) for v in [b, u]) else 0
        ax.annotate(cat, (cnt, best_y),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=7.5, color="#2C3E50", zorder=5)

    # "Minimum viable count" reference line
    ax.axvline(MIN_VIABLE_COUNT, color="orange", ls="--", lw=1.8,
               label=f"Min viable GT count = {MIN_VIABLE_COUNT}")

    # Trend line (baseline) via polyfit — only finite points
    valid = [(c, b) for c, b in zip(counts, b_aps) if not np.isnan(b) and c > 0]
    if len(valid) >= 3:
        xv, yv   = zip(*valid)
        z        = np.polyfit(np.log1p(xv), yv, 1)   # log scale for count
        x_smooth = np.linspace(0, max(xv), 200)
        ax.plot(x_smooth,
                np.polyval(z, np.log1p(x_smooth)),
                color="#E74C3C", ls=":", lw=1.5, alpha=0.7,
                label="Baseline trend (log-linear fit)")

    ax.set_xlabel("Severe GT annotation count (per category)", fontsize=11)
    ax.set_ylabel("AP @ IoU=0.50",                            fontsize=11)
    ax.set_title(
        "GT Count vs AP@.50 — Severe Occlusion\n"
        "Bottom-left = underrepresented AND poor performance  |  "
        "Vertical line = min viable count\n"
        "Red circle = Baseline  |  Blue triangle = Updated  |  "
        "Green/red connector = improvement/regression",
        fontsize=11
    )
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()

    out = Path(output_dir) / "severe_gt_count_vs_ap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out}")

    # ── Console table: count, AP, and underrepresentation flag ──────────────
    print(f"\n── GT Count vs AP — SEVERE ──")
    rows = []
    for cat in cats:
        cnt  = data[cat]["count"]
        b_ap = data[cat]["baseline_ap"]
        u_ap = data[cat]["updated_ap"]
        d    = u_ap - b_ap if not (np.isnan(b_ap) or np.isnan(u_ap)) else float("nan")
        under = "⚠️ UNDERREP" if cnt < MIN_VIABLE_COUNT else ""
        rows.append([
            cat,
            cnt,
            f"{b_ap:.4f}" if not np.isnan(b_ap) else "—",
            f"{u_ap:.4f}" if not np.isnan(u_ap) else "—",
            f"{d:+.4f}"   if not np.isnan(d)    else "—",
            under,
        ])
    # Sort by count ascending so underrepresented categories are at the top
    rows.sort(key=lambda r: r[1])
    hdrs = ["Category", "GT Count", "Base AP@.50", "Upd AP@.50", "Δ", "Flag"]
    if HAS_TABULATE:
        print(tabulate(rows, headers=hdrs, tablefmt="rounded_outline"))
    else:
        for row in rows:
            print("  ".join(str(c) for c in row))


# ── Entry-point for the three severe analyses ─────────────────────────────────

def run_severe_deep_dive(gt_full, b_preds, u_preds, output_dir,
                         iou_thresh=0.50, score_threshold=0.25):
    """
    Orchestrates all three severe deep-dive analyses.

    Builds the iscrowd-masked GT for the severe level once, then runs:
      1. plot_severe_per_category_ap      — which categories have low AP?
      2. plot_severe_tp_score_distribution — are TPs clustering near threshold?
      3. plot_severe_gt_count_vs_ap       — are some classes underrepresented?

    Parameters
    ----------
    gt_full          : full COCO GT dict (before iscrowd masking)
    b_preds          : baseline prediction list
    u_preds          : updated prediction list
    output_dir       : str / Path — all plots saved here
    iou_thresh       : IoU threshold used for TP matching (default 0.50)
    score_threshold  : confidence threshold to flag near-threshold TPs (default 0.25)
    """
    print(f"\n{'═'*70}")
    print("  SEVERE DEEP-DIVE ANALYSIS")
    print(f"{'═'*70}")

    # Build iscrowd-masked GT: severe annotations → iscrowd=0, others → iscrowd=1
    gt_severe = build_gt_for_level(gt_full, "severe")

    n_active  = sum(1 for a in gt_severe["annotations"] if a["iscrowd"] == 0)
    n_ignored = sum(1 for a in gt_severe["annotations"] if a["iscrowd"] == 1)
    print(f"  Severe GT (active  iscrowd=0): {n_active:,}")
    print(f"  Other  GT (ignored iscrowd=1): {n_ignored:,}")

    # ── 1. Per-category AP sorted ascending ──────────────────────────────────
    b_cat_stats, u_cat_stats = plot_severe_per_category_ap(
        gt_severe, b_preds, u_preds, output_dir
    )

    # ── 2. TP score distribution ──────────────────────────────────────────────
    plot_severe_tp_score_distribution(
        gt_severe, b_preds, u_preds, output_dir,
        iou_thresh=iou_thresh,
        score_threshold=score_threshold,
    )

    # ── 3. GT count vs AP scatter ─────────────────────────────────────────────
    plot_severe_gt_count_vs_ap(gt_severe, b_cat_stats, u_cat_stats, output_dir)

    print(f"\n  Severe deep-dive complete.  Plots saved to: {output_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# ── Main ─────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Occlusion-wise mAP + PR curves via COCOeval (iscrowd ignore)."
    )
    parser.add_argument("--gt",               required=True)
    parser.add_argument("--baseline",         required=True)
    parser.add_argument("--updated",          required=True)
    parser.add_argument("--output",           default="occlusion_map_report")
    parser.add_argument("--per-category",     action="store_true",
                        help="Per-category breakdown per occlusion level")
    # ── NEW ──────────────────────────────────────────────────────────────────
    parser.add_argument("--severe-analysis",  action="store_true",
                        help=(
                            "Run three deep-dive analyses on severe occlusion: "
                            "(1) per-category AP sorted ascending, "
                            "(2) TP score distribution, "
                            "(3) GT count vs AP scatter."
                        ))
    parser.add_argument("--score-threshold",  type=float, default=0.05,
                        help="Confidence threshold for near-threshold TP flagging "
                             "(default: 0.25)")
    parser.add_argument("--iou-thresh",       type=float, default=0.50,
                        help="IoU threshold used for TP matching in deep-dive "
                             "(default: 0.50)")
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

    # ── NEW: Severe deep-dive ────────────────────────────────────────────────
    if args.severe_analysis:
        run_severe_deep_dive(
            gt_full,
            b_preds,
            u_preds,
            output_dir      = args.output,
            iou_thresh      = args.iou_thresh,
            score_threshold = args.score_threshold,
        )

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