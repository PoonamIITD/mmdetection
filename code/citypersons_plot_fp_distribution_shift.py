"""
plot_fp_distribution_shift.py
================================
Visualizes the FULL distribution shift (not just the mean) behind the
Mann-Whitney results in solidify_heavy_occlusion_fp_comparison.py:

  - FP confidence: baseline vs updated (overlaid histograms + KDE)
  - FP max-IoU-to-GT: baseline vs updated (overlaid histograms + KDE)

for the Heavy_Occlusion subset.

Usage:
    python3 plot_fp_distribution_shift.py \
        --gt val_instances_with_occlusion_visibility_ratio.json \
        --baseline results_citypersons_baseline_val.json \
        --updated results_citypersons_sampling_loss_val.json \
        --score-thresh 0.1 \
        --out-dir fp_distribution_plots
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from citypersons_fp_clustering_diagnosis import (
    load_json, get_target_category_ids, build_gt_structures,
    match_and_collect_fps, cluster_gt, classify_fp_neighborhood,
    CLUSTER_IOU_THRESH,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--updated", required=True)
    ap.add_argument("--target-categories", nargs="+",
                     default=["pedestrian", "rider", "sitting person"])
    ap.add_argument("--score-thresh", type=float, default=0.1)
    ap.add_argument("--out-dir", default="fp_distribution_plots")
    args = ap.parse_args()

    gt_full = load_json(args.gt)
    target_category_ids = get_target_category_ids(gt_full, args.target_categories)
    b_preds = load_json(args.baseline)
    u_preds = load_json(args.updated)

    gt_by_image_subset, gt_all_by_image = build_gt_structures(gt_full, target_category_ids)

    print("Matching baseline FPs...")
    b_fps = match_and_collect_fps(gt_by_image_subset, b_preds, target_category_ids, args.score_thresh)
    print("Matching updated FPs...")
    u_fps = match_and_collect_fps(gt_by_image_subset, u_preds, target_category_ids, args.score_thresh)

    cluster_cache = {}

    def get_clusters(img_id):
        if img_id not in cluster_cache:
            boxes = gt_all_by_image.get(img_id, [])
            sizes = cluster_gt(boxes, CLUSTER_IOU_THRESH)
            cluster_cache[img_id] = (boxes, sizes)
        return cluster_cache[img_id]

    def attach_max_iou(fp_records):
        out = []
        for r in fp_records:
            boxes, sizes = get_clusters(r["img_id"])
            _, best_iou, _ = classify_fp_neighborhood(r["bbox"], boxes, sizes)
            out.append(best_iou)
        return np.array(out)

    b_scores = np.array([r["score"] for r in b_fps])
    u_scores = np.array([r["score"] for r in u_fps])
    b_ious = attach_max_iou(b_fps)
    u_ious = attach_max_iou(u_fps)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- FP confidence ---
    ax = axes[0]
    bins = np.linspace(args.score_thresh, max(b_scores.max(), u_scores.max()), 40)
    ax.hist(b_scores, bins=bins, density=True, alpha=0.5, color="#3498DB",
            label=f"Baseline (n={len(b_scores):,}, mean={b_scores.mean():.3f})")
    ax.hist(u_scores, bins=bins, density=True, alpha=0.5, color="#E74C3C",
            label=f"Updated (n={len(u_scores):,}, mean={u_scores.mean():.3f})")
    ax.axvline(b_scores.mean(), color="#2980B9", ls="--", lw=1.5)
    ax.axvline(u_scores.mean(), color="#C0392B", ls="--", lw=1.5)
    ax.set_xlabel("FP confidence score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Heavy_Occlusion FP confidence distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # --- FP max-IoU to GT ---
    ax = axes[1]
    bins = np.linspace(0, 1, 40)
    ax.hist(b_ious, bins=bins, density=True, alpha=0.5, color="#3498DB",
            label=f"Baseline (n={len(b_ious):,}, mean={b_ious.mean():.3f})")
    ax.hist(u_ious, bins=bins, density=True, alpha=0.5, color="#E74C3C",
            label=f"Updated (n={len(u_ious):,}, mean={u_ious.mean():.3f})")
    ax.axvline(b_ious.mean(), color="#2980B9", ls="--", lw=1.5)
    ax.axvline(u_ious.mean(), color="#C0392B", ls="--", lw=1.5)
    ax.set_xlabel("Max IoU to nearest GT box", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Heavy_Occlusion FP max-IoU-to-GT distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out_path = out_dir / "fp_confidence_and_iou_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")

    # --- bonus: per-image FP count distribution (paired) ---
    from collections import defaultdict
    b_by_img = defaultdict(int)
    for r in b_fps:
        b_by_img[r["img_id"]] += 1
    u_by_img = defaultdict(int)
    for r in u_fps:
        u_by_img[r["img_id"]] += 1
    all_img_ids = sorted(gt_by_image_subset.keys())
    b_counts = np.array([b_by_img.get(i, 0) for i in all_img_ids])
    u_counts = np.array([u_by_img.get(i, 0) for i in all_img_ids])

    fig, ax = plt.subplots(figsize=(7, 6))
    max_c = max(b_counts.max(), u_counts.max())
    ax.scatter(b_counts, u_counts, alpha=0.4, s=25, color="#8E44AD")
    ax.plot([0, max_c], [0, max_c], "k--", lw=1, alpha=0.6, label="y = x (no change)")
    ax.set_xlabel("Baseline FP count per image", fontsize=11)
    ax.set_ylabel("Updated FP count per image", fontsize=11)
    ax.set_title("Per-image FP count: baseline vs updated (Heavy_Occlusion)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out_path2 = out_dir / "per_image_fp_count_scatter.png"
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path2}")

    print("\nDone.")


if __name__ == "__main__":
    main()