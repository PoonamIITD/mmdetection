"""
Plot TP rank distributions per occlusion level: Baseline vs Updated.
Generates a combined figure with KDE + histogram + median markers.

Usage:
    python plot_occlusion_rank_dist.py \
        --gt        instances_validation_merged_mannual.json \
        --baseline  results_baseline_GDINO_final_val.json \
        --updated   results_sampling_loss_final_val.json \
        --conf      0.05 \
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
from matplotlib.gridspec import GridSpec
from collections import defaultdict
from pathlib import Path
from scipy.stats import gaussian_kde

# ── Colors ────────────────────────────────────────────────────────────────────
COLORS = {
    "baseline": "#E74C3C",   # red
    "updated":  "#2ECC71",   # green
}
OCCLUSION_LEVELS = ["light", "moderate", "severe"]
LEVEL_COLORS     = {"light": "#3498DB", "moderate": "#F39C12", "severe": "#8E44AD"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0]+a[2], a[1]+a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0]+b[2], b[1]+b[3]
    ix = max(0, min(ax2,bx2) - max(ax1,bx1))
    iy = max(0, min(ay2,by2) - max(ay1,by1))
    inter = ix * iy
    if inter == 0: return 0.0
    return inter / (a[2]*a[3] + b[2]*b[3] - inter)

def label_predictions(preds, gt_anns, iou_threshold=0.5):
    gt_index = defaultdict(list)
    gt_meta  = {}
    for ann in gt_anns:
        gt_index[(ann["image_id"], ann["category_id"])].append(ann)
        level = ann.get("attributes", {}).get("occlusion_level", "unknown").strip().lower()
        gt_meta[ann["id"]] = level

    matched_gt = set()
    sorted_preds = sorted(preds, key=lambda p: p["score"], reverse=True)
    labeled = []
    for rank, pred in enumerate(sorted_preds, start=1):
        key = (pred["image_id"], pred["category_id"])
        best_iou, best_gt_id = 0.0, None
        for gt in gt_index.get(key, []):
            if gt["id"] in matched_gt: continue
            iou = bbox_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou, best_gt_id = iou, gt["id"]
        is_tp = best_iou >= iou_threshold and best_gt_id is not None
        if is_tp:
            matched_gt.add(best_gt_id)
        labeled.append({
            "rank":            rank,
            "score":           pred["score"],
            "is_tp":           is_tp,
            "occlusion_level": gt_meta[best_gt_id] if is_tp else "fp",
        })
    return labeled

def get_tp_ranks_by_level(labeled):
    by_level = defaultdict(list)
    for x in labeled:
        if x["is_tp"] and x["occlusion_level"] in OCCLUSION_LEVELS:
            by_level[x["occlusion_level"]].append(x["rank"])
    return by_level

# ── Plot 1: KDE + Rug per occlusion level (3×1 grid) ─────────────────────────

def plot_kde_grid(b_by_level, u_by_level, output_dir, total_preds):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    fig.suptitle(
        "TP Rank Distribution by Occlusion Level — Baseline vs Updated\n"
        "(Left-shifted = TPs ranked higher = model more confident on those objects)",
        fontsize=13, y=1.02
    )

    for ax, level in zip(axes, OCCLUSION_LEVELS):
        b_ranks = np.array(b_by_level[level])
        u_ranks = np.array(u_by_level[level])

        if len(b_ranks) < 2 or len(u_ranks) < 2:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", transform=ax.transAxes)
            ax.set_title(level.capitalize())
            continue

        x = np.linspace(0, total_preds, 1000)

        # KDE
        kde_b = gaussian_kde(b_ranks, bw_method=0.15)
        kde_u = gaussian_kde(u_ranks, bw_method=0.15)

        ax.fill_between(x, kde_b(x), alpha=0.35, color=COLORS["baseline"])
        ax.fill_between(x, kde_u(x), alpha=0.35, color=COLORS["updated"])
        ax.plot(x, kde_b(x), color=COLORS["baseline"], lw=2, label="Baseline")
        ax.plot(x, kde_u(x), color=COLORS["updated"],  lw=2, label="Updated")

        # Median lines
        bm, um = np.median(b_ranks), np.median(u_ranks)
        ax.axvline(bm, color=COLORS["baseline"], ls="--", lw=1.5,
                   label=f"Median B: {bm:,.0f}")
        ax.axvline(um, color=COLORS["updated"],  ls="--", lw=1.5,
                   label=f"Median U: {um:,.0f}")

        # Shade improvement region between medians
        ax.axvspan(um, bm, alpha=0.12, color="gold",
                   label=f"Δ = {bm-um:+.0f}")

        # Stats box
        delta = bm - um
        pct   = delta / bm * 100
        ax.text(0.97, 0.97,
                f"n(B)={len(b_ranks):,}\nn(U)={len(u_ranks):,}\n"
                f"Δ median={delta:+.0f}\n({pct:+.1f}%)",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))

        ax.set_title(f"{level.capitalize()} Occlusion", fontsize=12,
                     color=LEVEL_COLORS[level], fontweight="bold")
        ax.set_xlabel("Rank (lower = higher confidence)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0, total_preds)

    plt.tight_layout()
    path = Path(output_dir) / "occlusion_rank_kde.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# ── Plot 2: Cumulative TP curve per occlusion level ───────────────────────────

def plot_cumulative_per_occlusion(b_by_level, u_by_level, output_dir, total_preds):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    fig.suptitle(
        "Cumulative TPs Found vs Rank — Per Occlusion Level\n"
        "(Steeper/higher curve = more TPs found at lower ranks)",
        fontsize=13, y=1.02
    )

    for ax, level in zip(axes, OCCLUSION_LEVELS):
        b_ranks = sorted(b_by_level[level])
        u_ranks = sorted(u_by_level[level])
        total_b = len(b_ranks)
        total_u = len(u_ranks)

        xs = np.arange(0, total_preds + 1, max(1, total_preds // 2000))

        by = np.searchsorted(b_ranks, xs, side="right")
        uy = np.searchsorted(u_ranks, xs, side="right")

        ax.plot(xs, by, color=COLORS["baseline"], lw=2, label=f"Baseline (n={total_b:,})")
        ax.plot(xs, uy, color=COLORS["updated"],  lw=2, label=f"Updated  (n={total_u:,})")

        # Fill area between curves
        ax.fill_between(xs, by, uy,
                        where=(uy >= by),
                        alpha=0.2, color=COLORS["updated"],
                        label="Updated ahead")
        ax.fill_between(xs, by, uy,
                        where=(uy < by),
                        alpha=0.2, color=COLORS["baseline"],
                        label="Baseline ahead")

        # Mark rank where 50% of TPs found
        for ranks, color, tag in [(b_ranks, COLORS["baseline"], "B"),
                                   (u_ranks, COLORS["updated"],  "U")]:
            if ranks:
                half_idx = len(ranks) // 2
                half_rank = ranks[half_idx]
                ax.axvline(half_rank, color=color, ls=":", lw=1.2,
                           label=f"50% TPs @ rank {half_rank:,} ({tag})")

        ax.set_title(f"{level.capitalize()} Occlusion", fontsize=12,
                     color=LEVEL_COLORS[level], fontweight="bold")
        ax.set_xlabel("Prediction Rank", fontsize=10)
        ax.set_ylabel("Cumulative TPs Found", fontsize=10)
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0, total_preds)

    plt.tight_layout()
    path = Path(output_dir) / "occlusion_cumulative_tp.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# ── Plot 3: Overlaid histogram comparison (all 3 levels, side by side) ────────

def plot_combined_histogram(b_by_level, u_by_level, output_dir, total_preds):
    fig = plt.figure(figsize=(18, 10))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Row 1: per-level histograms
    for col, level in enumerate(OCCLUSION_LEVELS):
        ax = fig.add_subplot(gs[0, col])
        b_ranks = b_by_level[level]
        u_ranks = u_by_level[level]
        bins = np.linspace(0, total_preds, 60)

        ax.hist(b_ranks, bins=bins, color=COLORS["baseline"],
                alpha=0.6, label="Baseline", density=True)
        ax.hist(u_ranks, bins=bins, color=COLORS["updated"],
                alpha=0.6, label="Updated",  density=True)

        bm = np.median(b_ranks) if b_ranks else 0
        um = np.median(u_ranks) if u_ranks else 0
        ax.axvline(bm, color=COLORS["baseline"], lw=2, ls="--")
        ax.axvline(um, color=COLORS["updated"],  lw=2, ls="--")

        ax.set_title(f"{level.capitalize()}  (Δ median = {bm-um:+.0f})",
                     color=LEVEL_COLORS[level], fontweight="bold")
        ax.set_xlabel("Rank")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    # Row 2: single overlay of all 3 levels, updated only (to show relative difficulty)
    ax_all = fig.add_subplot(gs[1, :])
    for level in OCCLUSION_LEVELS:
        b_ranks = b_by_level[level]
        u_ranks = u_by_level[level]
        if not b_ranks: continue
        bins = np.linspace(0, total_preds, 80)

        ax_all.hist(b_ranks, bins=bins, alpha=0.25,
                    color=LEVEL_COLORS[level], density=True,
                    label=f"Baseline {level}")
        ax_all.hist(u_ranks, bins=bins, alpha=0.50,
                    color=LEVEL_COLORS[level], density=True,
                    histtype="step", lw=2,
                    label=f"Updated {level}")

    ax_all.set_title(
        "All Occlusion Levels Overlaid — Filled=Baseline, Outline=Updated\n"
        "Updated outline shifting left vs filled = improvement",
        fontsize=11
    )
    ax_all.set_xlabel("Rank (lower = better)")
    ax_all.set_ylabel("Density")
    ax_all.legend(fontsize=8, ncol=3)
    ax_all.grid(True, alpha=0.25)

    fig.suptitle("TP Rank Histograms by Occlusion Level", fontsize=14, y=1.01)
    path = Path(output_dir) / "occlusion_rank_histograms.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# ── Plot 4: Score distributions of TPs per occlusion level ───────────────────

def plot_score_distributions(b_labeled, u_labeled, output_dir):
    b_by_level = defaultdict(list)
    u_by_level = defaultdict(list)
    for x in b_labeled:
        if x["is_tp"] and x["occlusion_level"] in OCCLUSION_LEVELS:
            b_by_level[x["occlusion_level"]].append(x["score"])
    for x in u_labeled:
        if x["is_tp"] and x["occlusion_level"] in OCCLUSION_LEVELS:
            u_by_level[x["occlusion_level"]].append(x["score"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Confidence Score Distribution of TPs by Occlusion Level\n"
        "(Right-shifted = model more confident on real objects = better)",
        fontsize=13, y=1.02
    )

    for ax, level in zip(axes, OCCLUSION_LEVELS):
        b_scores = np.array(b_by_level[level])
        u_scores = np.array(u_by_level[level])
        if len(b_scores) < 2 or len(u_scores) < 2:
            continue

        bins = np.linspace(0, 1, 50)
        ax.hist(b_scores, bins=bins, color=COLORS["baseline"],
                alpha=0.6, density=True, label="Baseline")
        ax.hist(u_scores, bins=bins, color=COLORS["updated"],
                alpha=0.6, density=True, label="Updated")

        bm, um = np.median(b_scores), np.median(u_scores)
        ax.axvline(bm, color=COLORS["baseline"], lw=2, ls="--",
                   label=f"Median B: {bm:.3f}")
        ax.axvline(um, color=COLORS["updated"],  lw=2, ls="--",
                   label=f"Median U: {um:.3f}")

        ax.text(0.03, 0.97,
                f"Δ median = {um-bm:+.4f}\n({(um-bm)/bm*100:+.1f}%)",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

        ax.set_title(f"{level.capitalize()} Occlusion",
                     color=LEVEL_COLORS[level], fontweight="bold")
        ax.set_xlabel("Confidence Score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = Path(output_dir) / "tp_score_distribution_by_occlusion.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt",       required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--updated",  required=True)
    parser.add_argument("--conf",     type=float, default=0.05)
    parser.add_argument("--iou",      type=float, default=0.5)
    parser.add_argument("--output",   default="tp_rank_report")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    print(f"Loading GT        : {args.gt}")
    gt      = load_json(args.gt)
    gt_anns = gt["annotations"]

    print(f"Loading Baseline  : {args.baseline}")
    b_preds = [p for p in load_json(args.baseline) if p["score"] >= args.conf]
    print(f"  Kept {len(b_preds):,} predictions (conf >= {args.conf})")

    print(f"Loading Updated   : {args.updated}")
    u_preds = [p for p in load_json(args.updated) if p["score"] >= args.conf]
    print(f"  Kept {len(u_preds):,} predictions (conf >= {args.conf})")

    total_preds = max(len(b_preds), len(u_preds))

    print("\nLabeling predictions ...")
    b_labeled = label_predictions(b_preds, gt_anns, args.iou)
    u_labeled = label_predictions(u_preds, gt_anns, args.iou)

    b_by_level = get_tp_ranks_by_level(b_labeled)
    u_by_level = get_tp_ranks_by_level(u_labeled)

    print(f"\nGenerating plots → {args.output}/")
    plot_kde_grid(b_by_level, u_by_level, args.output, total_preds)
    plot_cumulative_per_occlusion(b_by_level, u_by_level, args.output, total_preds)
    plot_combined_histogram(b_by_level, u_by_level, args.output, total_preds)
    plot_score_distributions(b_labeled, u_labeled, args.output)

    # Print median score per occlusion level
    print("\n── Median TP Confidence Score by Occlusion Level ────────────────")
    print(f"{'Occlusion':<12} {'Median B':>10} {'Median U':>10} {'Δ':>10} {'% Δ':>8}")
    print("-" * 54)
    for level in OCCLUSION_LEVELS:
        bs = [x["score"] for x in b_labeled if x["is_tp"] and x["occlusion_level"]==level]
        us = [x["score"] for x in u_labeled if x["is_tp"] and x["occlusion_level"]==level]
        if bs and us:
            bm, um = np.median(bs), np.median(us)
            print(f"{level:<12} {bm:>10.4f} {um:>10.4f} {um-bm:>+10.4f} {(um-bm)/bm*100:>+7.1f}%")

    print("\nDone.")

if __name__ == "__main__":
    main()