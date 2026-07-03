"""
TP Ranking Analysis — Baseline vs Updated Model

Answers: "Does the updated model push True Positives higher in the confidence ranking?"

For each model's predictions (sorted by score descending), we:
1. Match each prediction to a GT box (IoU >= threshold) → label as TP or FP
2. Record the rank at which each TP was found
3. Compare rank distributions between baseline and updated model

Outputs:
  - Rank distribution histograms
  - Cumulative TP curve (how many TPs found in top-K predictions)
  - Per-occlusion-level TP rank comparison
  - Summary statistics (median rank, mean rank, TPs in top-N)

Requirements:
    pip install pycocotools matplotlib numpy tabulate

Usage:
    python tp_rank_analysis.py \
        --gt        merged_gt.json \
        --baseline  predictions_baseline.json \
        --updated   predictions_updated.json \
        --iou       0.5 \
        --output    tp_rank_report
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

try:
    from pycocotools.coco import COCO
    from pycocotools import mask as maskUtils
except ImportError:
    raise ImportError("pip install pycocotools")

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

OCCLUSION_LEVELS = ["light", "moderate", "severe"]
COLORS = {"baseline": "#E74C3C", "updated": "#2ECC71"}


# ── I/O ──────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── IoU ──────────────────────────────────────────────────────────────────────

def bbox_iou(boxA, boxB):
    """COCO bbox format: [x, y, w, h]"""
    ax1, ay1 = boxA[0], boxA[1]
    ax2, ay2 = boxA[0] + boxA[2], boxA[1] + boxA[3]
    bx1, by1 = boxB[0], boxB[1]
    bx2, by2 = boxB[0] + boxB[2], boxB[1] + boxB[3]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]
    return inter / (areaA + areaB - inter)


# ── Core: label predictions as TP/FP ─────────────────────────────────────────

def label_predictions(preds, gt_anns, iou_threshold=0.5):
    """
    Sort predictions by score descending.
    Match greedily to GT boxes (per image, per category).
    Returns list of dicts with rank, score, is_tp, matched_gt_id, occlusion_level.
    """
    # Index GT by (image_id, category_id)
    gt_index = defaultdict(list)
    gt_meta  = {}   # ann_id → {occlusion_level, bbox}
    for ann in gt_anns:
        gt_index[(ann["image_id"], ann["category_id"])].append(ann)
        level = ann.get("attributes", {}).get("occlusion_level", "unknown").strip().lower()
        gt_meta[ann["id"]] = {"occlusion_level": level, "bbox": ann["bbox"]}

    # Track which GT boxes have been matched
    matched_gt = set()

    # Sort predictions by descending score
    sorted_preds = sorted(preds, key=lambda p: p["score"], reverse=True)

    labeled = []
    for rank, pred in enumerate(sorted_preds, start=1):
        key = (pred["image_id"], pred["category_id"])
        candidates = gt_index.get(key, [])

        best_iou   = 0.0
        best_gt_id = None
        for gt in candidates:
            if gt["id"] in matched_gt:
                continue
            iou = bbox_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou   = iou
                best_gt_id = gt["id"]

        is_tp = best_iou >= iou_threshold and best_gt_id is not None
        if is_tp:
            matched_gt.add(best_gt_id)

        labeled.append({
            "rank":             rank,
            "score":            pred["score"],
            "is_tp":            is_tp,
            "matched_gt_id":    best_gt_id if is_tp else None,
            "occlusion_level":  gt_meta[best_gt_id]["occlusion_level"] if is_tp else "fp",
            "image_id":         pred["image_id"],
            "category_id":      pred["category_id"],
        })

    return labeled


# ── Stats helpers ─────────────────────────────────────────────────────────────

def tp_ranks(labeled):
    return [x["rank"] for x in labeled if x["is_tp"]]


def cumulative_tp_curve(labeled, total_preds):
    ranks = sorted(tp_ranks(labeled))
    if not ranks:
        return np.arange(total_preds + 1), np.zeros(total_preds + 1)
    xs = np.arange(total_preds + 1)
    ys = np.searchsorted(ranks, xs, side="right")
    return xs, ys


def summary_stats(labeled, label="model"):
    ranks = tp_ranks(labeled)
    total_preds = len(labeled)
    total_tp    = len(ranks)

    if not ranks:
        return {"label": label, "total_preds": total_preds, "total_tp": 0}

    ranks_arr = np.array(ranks)
    thresholds = [100, 200, 500, 1000, 2000, 5000]

    stats = {
        "label":        label,
        "total_preds":  total_preds,
        "total_tp":     total_tp,
        "median_rank":  float(np.median(ranks_arr)),
        "mean_rank":    float(np.mean(ranks_arr)),
        "p25_rank":     float(np.percentile(ranks_arr, 25)),
        "p75_rank":     float(np.percentile(ranks_arr, 75)),
        "p90_rank":     float(np.percentile(ranks_arr, 90)),
    }
    for t in thresholds:
        stats[f"tp_in_top{t}"] = int((ranks_arr <= t).sum())

    return stats


def occlusion_rank_stats(labeled):
    by_level = defaultdict(list)
    for x in labeled:
        if x["is_tp"]:
            by_level[x["occlusion_level"]].append(x["rank"])
    return by_level


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_cumulative_tp(b_labeled, u_labeled, output_dir, total_gt):
    total_preds = max(len(b_labeled), len(u_labeled))

    bx, by = cumulative_tp_curve(b_labeled, total_preds)
    ux, uy = cumulative_tp_curve(u_labeled, total_preds)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(bx, by, color=COLORS["baseline"], lw=2, label="Baseline")
    ax.plot(ux, uy, color=COLORS["updated"],  lw=2, label="Updated")
    ax.axhline(total_gt, color="gray", ls="--", lw=1, label=f"Total GT = {total_gt}")
    ax.set_xlabel("Number of Predictions Considered (Rank)")
    ax.set_ylabel("Cumulative True Positives Found")
    ax.set_title("Cumulative TP Found vs Prediction Rank\n(Steeper = Better — TPs ranked higher)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate area difference
    diff_area = np.trapz(uy[:total_preds+1], ux[:total_preds+1]) - \
                np.trapz(by[:total_preds+1], bx[:total_preds+1])
    ax.text(0.98, 0.05, f"Area gain (updated − baseline): {diff_area:+,.0f}",
            transform=ax.transAxes, ha="right", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    plt.tight_layout()
    path = Path(output_dir) / "cumulative_tp_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_rank_histogram(b_labeled, u_labeled, output_dir):
    b_ranks = tp_ranks(b_labeled)
    u_ranks = tp_ranks(u_labeled)
    max_rank = max(max(b_ranks, default=1), max(u_ranks, default=1))
    bins = np.linspace(0, min(max_rank, 5000), 50)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, ranks, label, color in [
        (axes[0], b_ranks, "Baseline", COLORS["baseline"]),
        (axes[1], u_ranks, "Updated",  COLORS["updated"]),
    ]:
        ax.hist(ranks, bins=bins, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(np.median(ranks), color="black", ls="--", lw=1.5,
                   label=f"Median={np.median(ranks):.0f}")
        ax.set_title(f"{label} — TP Rank Distribution")
        ax.set_xlabel("Rank of TP (lower = better)")
        ax.set_ylabel("Count of TPs")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Where in the Ranked List do TPs Appear?", fontsize=13, y=1.02)
    plt.tight_layout()
    path = Path(output_dir) / "tp_rank_histogram.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_occlusion_rank_boxplot(b_labeled, u_labeled, output_dir):
    b_by_level = occlusion_rank_stats(b_labeled)
    u_by_level = occlusion_rank_stats(u_labeled)

    levels = [l for l in OCCLUSION_LEVELS if b_by_level[l] or u_by_level[l]]
    if not levels:
        return

    x = np.arange(len(levels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bp1 = ax.boxplot([b_by_level[l] for l in levels],
                     positions=x - width/2, widths=width*0.9,
                     patch_artist=True,
                     boxprops=dict(facecolor=COLORS["baseline"], alpha=0.7),
                     medianprops=dict(color="black", lw=2),
                     showfliers=False)
    bp2 = ax.boxplot([u_by_level[l] for l in levels],
                     positions=x + width/2, widths=width*0.9,
                     patch_artist=True,
                     boxprops=dict(facecolor=COLORS["updated"], alpha=0.7),
                     medianprops=dict(color="black", lw=2),
                     showfliers=False)

    ax.set_xticks(x)
    ax.set_xticklabels([l.capitalize() for l in levels], fontsize=12)
    ax.set_ylabel("Rank of TP (lower = better)")
    ax.set_title("TP Rank Distribution by Occlusion Level\n(Lower box = TPs found earlier = model is more confident)")
    ax.legend(
        handles=[
            mpatches.Patch(color=COLORS["baseline"], label="Baseline"),
            mpatches.Patch(color=COLORS["updated"],  label="Updated"),
        ]
    )
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = Path(output_dir) / "occlusion_rank_boxplot.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_tp_in_topk(b_stats, u_stats, output_dir, total_gt):
    thresholds = [k for k in [100, 200, 500, 1000, 2000, 5000]
                  if f"tp_in_top{k}" in b_stats]
    b_vals = [b_stats[f"tp_in_top{k}"] for k in thresholds]
    u_vals = [u_stats[f"tp_in_top{k}"] for k in thresholds]

    x = np.arange(len(thresholds))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, b_vals, width, label="Baseline",
                   color=COLORS["baseline"], alpha=0.85)
    bars2 = ax.bar(x + width/2, u_vals, width, label="Updated",
                   color=COLORS["updated"],  alpha=0.85)

    ax.axhline(total_gt, color="gray", ls="--", lw=1.5,
               label=f"Total GT = {total_gt}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Top-{k}" for k in thresholds])
    ax.set_ylabel("TPs Found")
    ax.set_title("How Many TPs are in the Top-K Predictions?\n(Higher = model ranks TPs earlier)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate difference
    for i, (bv, uv) in enumerate(zip(b_vals, u_vals)):
        diff = uv - bv
        if diff != 0:
            ax.text(i + width/2, uv + 5, f"{diff:+d}",
                    ha="center", va="bottom", fontsize=9,
                    color="darkgreen" if diff > 0 else "red")

    plt.tight_layout()
    path = Path(output_dir) / "tp_in_topk.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Print tables ──────────────────────────────────────────────────────────────

def print_summary_table(b_stats, u_stats, total_gt):
    thresholds = [k for k in [100, 200, 500, 1000, 2000, 5000]
                  if f"tp_in_top{k}" in b_stats]

    rows = [
        ["Total Predictions",  b_stats["total_preds"],   u_stats["total_preds"],   "—"],
        ["Total TPs Matched",  b_stats["total_tp"],      u_stats["total_tp"],
         f"{u_stats['total_tp'] - b_stats['total_tp']:+d}"],
        ["Median TP Rank ↓",   f"{b_stats['median_rank']:.1f}",
         f"{u_stats['median_rank']:.1f}",
         f"{u_stats['median_rank'] - b_stats['median_rank']:+.1f}"],
        ["Mean TP Rank ↓",     f"{b_stats['mean_rank']:.1f}",
         f"{u_stats['mean_rank']:.1f}",
         f"{u_stats['mean_rank'] - b_stats['mean_rank']:+.1f}"],
        ["P25 Rank ↓",         f"{b_stats['p25_rank']:.1f}",
         f"{u_stats['p25_rank']:.1f}",
         f"{u_stats['p25_rank'] - b_stats['p25_rank']:+.1f}"],
        ["P75 Rank ↓",         f"{b_stats['p75_rank']:.1f}",
         f"{u_stats['p75_rank']:.1f}",
         f"{u_stats['p75_rank'] - b_stats['p75_rank']:+.1f}"],
        ["P90 Rank ↓",         f"{b_stats['p90_rank']:.1f}",
         f"{u_stats['p90_rank']:.1f}",
         f"{u_stats['p90_rank'] - b_stats['p90_rank']:+.1f}"],
    ]
    for k in thresholds:
        bv = b_stats[f"tp_in_top{k}"]
        uv = u_stats[f"tp_in_top{k}"]
        pct_b = bv / total_gt * 100
        pct_u = uv / total_gt * 100
        rows.append([
            f"TPs in Top-{k} (% of GT)",
            f"{bv} ({pct_b:.1f}%)",
            f"{uv} ({pct_u:.1f}%)",
            f"{uv-bv:+d} ({pct_u-pct_b:+.1f}%)"
        ])

    headers = ["Metric", "Baseline", "Updated", "Δ (Updated − Baseline)"]
    print("\n" + "="*70)
    print("  TP RANKING COMPARISON: BASELINE vs UPDATED MODEL")
    print("="*70)
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    else:
        col_w = [max(len(h), max(len(str(r[i])) for r in rows))
                 for i, h in enumerate(headers)]
        fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
        print(fmt.format(*headers))
        print("  ".join("-"*w for w in col_w))
        for row in rows:
            print(fmt.format(*[str(c) for c in row]))


def print_occlusion_table(b_labeled, u_labeled):
    b_by_level = occlusion_rank_stats(b_labeled)
    u_by_level = occlusion_rank_stats(u_labeled)

    rows = []
    for level in OCCLUSION_LEVELS:
        br = np.array(b_by_level[level]) if b_by_level[level] else np.array([np.nan])
        ur = np.array(u_by_level[level]) if u_by_level[level] else np.array([np.nan])
        rows.append([
            level.capitalize(),
            len(b_by_level[level]),
            len(u_by_level[level]),
            f"{np.median(br):.1f}" if not np.isnan(br).all() else "—",
            f"{np.median(ur):.1f}" if not np.isnan(ur).all() else "—",
            f"{np.median(ur) - np.median(br):+.1f}"
              if not (np.isnan(br).all() or np.isnan(ur).all()) else "—",
        ])

    headers = ["Occlusion", "B TPs", "U TPs",
               "Median Rank (B) ↓", "Median Rank (U) ↓", "Δ Median Rank"]
    print("\n" + "="*70)
    print("  TP RANKING BY OCCLUSION LEVEL")
    print("="*70)
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    else:
        col_w = [max(len(h), max(len(str(r[i])) for r in rows))
                 for i, h in enumerate(headers)]
        fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
        print(fmt.format(*headers))
        print("  ".join("-"*w for w in col_w))
        for row in rows:
            print(fmt.format(*[str(c) for c in row]))

# Add this after print_occlusion_table() in tp_rank_analysis.py

def print_occlusion_mean_table(b_labeled, u_labeled):
    b_by_level = occlusion_rank_stats(b_labeled)  # returns ranks per level
    u_by_level = occlusion_rank_stats(u_labeled)

    rows = []
    for level in OCCLUSION_LEVELS:
        br = np.array(b_by_level[level]) if b_by_level[level] else np.array([np.nan])
        ur = np.array(u_by_level[level]) if u_by_level[level] else np.array([np.nan])

        b_median = np.median(br)
        u_median = np.median(ur)
        b_mean   = np.mean(br)
        u_mean   = np.mean(ur)
        b_std    = np.std(br)
        u_std    = np.std(ur)

        rows.append([
            level.capitalize(),
            f"{b_median:,.1f}",   f"{u_median:,.1f}",  f"{u_median - b_median:+,.1f}",
            f"{b_mean:,.1f}",     f"{u_mean:,.1f}",    f"{u_mean - b_mean:+,.1f}",
            f"{b_std:,.1f}",      f"{u_std:,.1f}",     f"{u_std - b_std:+,.1f}",
        ])

    headers = [
        "Occlusion",
        "Median(B)", "Median(U)", "Δ Median",
        "Mean(B)",   "Mean(U)",   "Δ Mean",
        "Std(B)",    "Std(U)",    "Δ Std",
    ]
    print("\n" + "="*95)
    print("  MEDIAN vs MEAN vs STD RANK — PER OCCLUSION LEVEL")
    print("="*95)
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    else:
        col_w = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
        fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
        print(fmt.format(*headers))
        print("  ".join("-"*w for w in col_w))
        for row in rows:
            print(fmt.format(*row))

    print("\n  Interpretation guide:")
    print("  Δ Median → improvement for the TYPICAL TP")
    print("  Δ Mean   → improvement averaged across ALL TPs (sensitive to hard outliers)")
    print("  Δ Std    → negative = ranks more tightly clustered = more consistent model")

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare TP ranking between baseline and updated model predictions."
    )
    parser.add_argument("--gt",       required=True, help="GT COCO JSON file")
    parser.add_argument("--baseline", required=True, help="Baseline predictions JSON")
    parser.add_argument("--updated",  required=True, help="Updated model predictions JSON")
    parser.add_argument("--iou",      type=float, default=0.5, help="IoU threshold (default 0.5)")
    parser.add_argument("--output",   default="tp_rank_report", help="Output directory for plots")
    args = parser.parse_args()

    for path in (args.gt, args.baseline, args.updated):
        if not Path(path).exists():
            raise FileNotFoundError(f"Not found: {path}")

    Path(args.output).mkdir(parents=True, exist_ok=True)

    print(f"\nLoading GT        : {args.gt}")
    gt = load_json(args.gt)
    gt_anns   = gt["annotations"]
    total_gt  = len(gt_anns)
    print(f"  Total GT annotations: {total_gt}")

    print(f"Loading Baseline  : {args.baseline}")
    b_preds = load_json(args.baseline)
    print(f"  Total predictions: {len(b_preds)}")

    print(f"Loading Updated   : {args.updated}")
    u_preds = load_json(args.updated)
    print(f"  Total predictions: {len(u_preds)}")

    # Add this before calling label_predictions in the script
    b_preds = [p for p in b_preds if p["score"] >= 0.05]
    u_preds = [p for p in u_preds if p["score"] >= 0.05]
    
    print(f"\nLabeling predictions (IoU >= {args.iou}) ...")
    print("  Labeling baseline ...")
    b_labeled = label_predictions(b_preds, gt_anns, args.iou)
    print("  Labeling updated  ...")
    u_labeled = label_predictions(u_preds, gt_anns, args.iou)

    # Stats
    b_stats = summary_stats(b_labeled, "Baseline")
    u_stats = summary_stats(u_labeled, "Updated")

    # Print tables
    print_summary_table(b_stats, u_stats, total_gt)
    # print_occlusion_table(b_labeled, u_labeled)
    print_occlusion_mean_table(b_labeled,u_labeled)

    # Plots
    print(f"\nGenerating plots → {args.output}/")
    plot_cumulative_tp(b_labeled, u_labeled, args.output, total_gt)
    plot_rank_histogram(b_labeled, u_labeled, args.output)
    plot_occlusion_rank_boxplot(b_labeled, u_labeled, args.output)
    plot_tp_in_topk(b_stats, u_stats, args.output, total_gt)

    print("\nDone.")


if __name__ == "__main__":
    main()