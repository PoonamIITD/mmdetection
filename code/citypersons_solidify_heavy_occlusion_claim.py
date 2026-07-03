# """
# solidify_heavy_occlusion_fp_comparison.py
# ===========================================
# Rigorously confirms (not just point-estimates) that:

#   (1) Updated produces MORE false positives than baseline over the
#       Heavy_Occlusion GT subset -- tested per-image with a paired
#       Wilcoxon signed-rank test (robust to the non-normal, skewed
#       per-image FP-count distribution) AND a paired t-test for reference.

#   (2) Mean FP confidence and mean FP max-IoU-to-GT differ between
#       baseline and updated -- reported with bootstrap 95% confidence
#       intervals (since the underlying FP-level distributions are not
#       normal / not independent within an image) plus a Mann-Whitney U
#       test on the FP-level (not per-image) score/IoU distributions.

# Reuses match_and_collect_fps / classify_fp_neighborhood from
# citypersons_fp_clustering_diagnosis.py so results are guaranteed
# consistent with every earlier diagnostic script.

# Usage:
#     python3 solidify_heavy_occlusion_fp_comparison.py \
#         --gt val_instances_with_occlusion_visibility_ratio.json \
#         --baseline results_citypersons_baseline_val.json \
#         --updated results_citypersons_sampling_loss_val.json \
#         --score-thresh 0.1 \
#         --n-bootstrap 5000
# """

# import argparse
# from pathlib import Path
# from collections import defaultdict

# import numpy as np
# from scipy import stats

# try:
#     from tabulate import tabulate
#     HAS_TABULATE = True
# except ImportError:
#     HAS_TABULATE = False

# from citypersons_fp_clustering_diagnosis import (
#     load_json, get_target_category_ids, build_gt_structures,
#     match_and_collect_fps, cluster_gt, classify_fp_neighborhood,
#     CLUSTER_IOU_THRESH,
# )


# def print_table(rows, header):
#     if HAS_TABULATE:
#         print(tabulate(rows, headers=header, tablefmt="rounded_outline"))
#     else:
#         widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
#         fmt = "  ".join(f"{{:<{w}}}" for w in widths)
#         print(fmt.format(*header))
#         for row in rows:
#             print(fmt.format(*[str(c) for c in row]))


# def bootstrap_mean_ci(values, n_bootstrap=5000, ci=95, seed=0):
#     """Bootstrap CI for the mean of a 1D array."""
#     rng = np.random.default_rng(seed)
#     values = np.asarray(values)
#     n = len(values)
#     if n == 0:
#         return float("nan"), float("nan"), float("nan")
#     boot_means = np.empty(n_bootstrap)
#     for i in range(n_bootstrap):
#         sample = rng.choice(values, size=n, replace=True)
#         boot_means[i] = sample.mean()
#     lo = np.percentile(boot_means, (100 - ci) / 2)
#     hi = np.percentile(boot_means, 100 - (100 - ci) / 2)
#     return values.mean(), lo, hi


# def per_image_fp_counts(fp_records, all_image_ids):
#     counts = defaultdict(int)
#     for r in fp_records:
#         counts[r["img_id"]] += 1
#     return np.array([counts.get(img_id, 0) for img_id in all_image_ids])


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--gt", required=True)
#     ap.add_argument("--baseline", required=True)
#     ap.add_argument("--updated", required=True)
#     ap.add_argument("--target-categories", nargs="+",
#                      default=["pedestrian", "rider", "sitting person"])
#     ap.add_argument("--score-thresh", type=float, default=0.1)
#     ap.add_argument("--n-bootstrap", type=int, default=5000)
#     args = ap.parse_args()

#     for p in (args.gt, args.baseline, args.updated):
#         if not Path(p).exists():
#             raise FileNotFoundError(p)

#     gt_full = load_json(args.gt)
#     target_category_ids = get_target_category_ids(gt_full, args.target_categories)
#     b_preds = load_json(args.baseline)
#     u_preds = load_json(args.updated)

#     gt_by_image_subset, gt_all_by_image = build_gt_structures(gt_full, target_category_ids)
#     all_image_ids = sorted(gt_by_image_subset.keys())

#     print("Matching baseline FPs (Heavy_Occlusion subset)...")
#     b_fps = match_and_collect_fps(gt_by_image_subset, b_preds, target_category_ids, args.score_thresh)
#     print("Matching updated FPs (Heavy_Occlusion subset)...")
#     u_fps = match_and_collect_fps(gt_by_image_subset, u_preds, target_category_ids, args.score_thresh)

#     # ══════════════════════════════════════════════════════════════
#     # (1) PAIRED PER-IMAGE FP COUNT TEST
#     # ══════════════════════════════════════════════════════════════
#     print("\n" + "=" * 78)
#     print("(1) Paired per-image FP count: baseline vs updated (Heavy_Occlusion)")
#     print("=" * 78)

#     b_counts = per_image_fp_counts(b_fps, all_image_ids)
#     u_counts = per_image_fp_counts(u_fps, all_image_ids)
#     diffs = u_counts - b_counts

#     n_images = len(all_image_ids)
#     n_more = int((diffs > 0).sum())
#     n_fewer = int((diffs < 0).sum())
#     n_equal = int((diffs == 0).sum())

#     print(f"Images evaluated: {n_images}")
#     print(f"Total FPs -- baseline: {b_counts.sum():,}   updated: {u_counts.sum():,}   "
#           f"(mean/image: {b_counts.mean():.3f} vs {u_counts.mean():.3f})")
#     print(f"\nPer-image comparison:")
#     print(f"  images where updated has MORE FPs than baseline : {n_more} ({100*n_more/n_images:.1f}%)")
#     print(f"  images where updated has FEWER FPs than baseline: {n_fewer} ({100*n_fewer/n_images:.1f}%)")
#     print(f"  images with equal FP count                       : {n_equal} ({100*n_equal/n_images:.1f}%)")

#     # Wilcoxon signed-rank test (paired, robust to skew) -- drops zero-diff pairs automatically
#     if n_more > 0 or n_fewer > 0:
#         try:
#             wilcoxon_stat, wilcoxon_p = stats.wilcoxon(u_counts, b_counts, alternative="greater")
#         except ValueError as e:
#             wilcoxon_stat, wilcoxon_p = float("nan"), float("nan")
#             print(f"  [Wilcoxon test could not run: {e}]")
#     else:
#         wilcoxon_stat, wilcoxon_p = float("nan"), float("nan")

#     # paired t-test for reference
#     t_stat, t_p = stats.ttest_rel(u_counts, b_counts, alternative="greater")

#     print(f"\nWilcoxon signed-rank test (H1: updated > baseline, per image) : "
#           f"stat={wilcoxon_stat:.1f}  p={wilcoxon_p:.3g}")
#     print(f"Paired t-test          (H1: updated > baseline, per image) : "
#           f"t={t_stat:.3f}  p={t_p:.3g}")

#     mean_diff, lo, hi = bootstrap_mean_ci(diffs, args.n_bootstrap)
#     print(f"\nMean per-image FP increase (updated - baseline): {mean_diff:+.3f}  "
#           f"[95% CI: {lo:+.3f}, {hi:+.3f}]")

#     if wilcoxon_p < 0.05 and lo > 0:
#         print("\n-> CONFIRMED: updated produces significantly more FPs per image than")
#         print("   baseline in Heavy_Occlusion. This is a systematic, image-wide effect,")
#         print("   not driven by a handful of outlier images.")
#     else:
#         print("\n-> NOT confirmed at p<0.05 with a CI excluding zero -- effect may be")
#         print("   driven by a subset of images or could be noise. Inspect further.")

#     # ══════════════════════════════════════════════════════════════
#     # (2) FP-LEVEL CONFIDENCE AND IoU COMPARISON
#     # ══════════════════════════════════════════════════════════════
#     print("\n" + "=" * 78)
#     print("(2) FP confidence and max-IoU-to-GT: baseline vs updated (Heavy_Occlusion)")
#     print("=" * 78)

#     cluster_cache = {}

#     def get_clusters(img_id):
#         if img_id not in cluster_cache:
#             boxes = gt_all_by_image.get(img_id, [])
#             sizes = cluster_gt(boxes, CLUSTER_IOU_THRESH)
#             cluster_cache[img_id] = (boxes, sizes)
#         return cluster_cache[img_id]

#     def attach_max_iou(fp_records):
#         out = []
#         for r in fp_records:
#             boxes, sizes = get_clusters(r["img_id"])
#             label, best_iou, csize = classify_fp_neighborhood(r["bbox"], boxes, sizes)
#             out.append({**r, "max_iou": best_iou, "label": label})
#         return out

#     b_fps_full = attach_max_iou(b_fps)
#     u_fps_full = attach_max_iou(u_fps)

#     b_scores = np.array([r["score"] for r in b_fps_full])
#     u_scores = np.array([r["score"] for r in u_fps_full])
#     b_ious = np.array([r["max_iou"] for r in b_fps_full])
#     u_ious = np.array([r["max_iou"] for r in u_fps_full])

#     rows = []
#     for name, b_arr, u_arr in [("Mean FP confidence", b_scores, u_scores),
#                                 ("Mean FP max-IoU to GT", b_ious, u_ious)]:
#         b_mean, b_lo, b_hi = bootstrap_mean_ci(b_arr, args.n_bootstrap, seed=1)
#         u_mean, u_lo, u_hi = bootstrap_mean_ci(u_arr, args.n_bootstrap, seed=2)
#         rows.append([name,
#                      f"{b_mean:.4f} [{b_lo:.4f}, {b_hi:.4f}]",
#                      f"{u_mean:.4f} [{u_lo:.4f}, {u_hi:.4f}]",
#                      f"{u_mean - b_mean:+.4f}"])
#     print_table(rows, ["Metric", "Baseline (mean [95% CI])", "Updated (mean [95% CI])", "Delta"])

#     # Mann-Whitney U test on the raw FP-level distributions (independent samples --
#     # FPs are pooled across images, not paired 1-to-1 like the per-image count test)
#     u_stat_score, p_score = stats.mannwhitneyu(u_scores, b_scores, alternative="two-sided")
#     u_stat_iou, p_iou = stats.mannwhitneyu(u_ious, b_ious, alternative="two-sided")

#     print(f"\nMann-Whitney U test, FP confidence  (baseline vs updated distributions): p={p_score:.3g}")
#     print(f"Mann-Whitney U test, FP max-IoU      (baseline vs updated distributions): p={p_iou:.3g}")

#     print(f"\nFP count for reference -- baseline: {len(b_fps_full):,}   updated: {len(u_fps_full):,}")

#     if p_score < 0.05:
#         direction = "LOWER" if u_scores.mean() < b_scores.mean() else "HIGHER"
#         print(f"-> Updated's FP confidence is significantly {direction} than baseline's "
#               f"(p={p_score:.3g}).")
#     if p_iou < 0.05:
#         direction = "HIGHER" if u_ious.mean() > b_ious.mean() else "LOWER"
#         print(f"-> Updated's FP max-IoU-to-GT is significantly {direction} than baseline's "
#               f"(p={p_iou:.3g}) -- updated's false positives sit closer to real GT boxes.")

#     print("\nDone.")


# if __name__ == "__main__":
#     main()



"""
solidify_heavy_occlusion_fp_comparison.py
===========================================
Rigorously confirms (not just point-estimates) that:

  (1) Updated produces MORE false positives than baseline over the
      Heavy_Occlusion GT subset -- tested per-image with a paired
      Wilcoxon signed-rank test (robust to the non-normal, skewed
      per-image FP-count distribution) AND a paired t-test for reference.

  (2) Mean FP confidence and mean FP max-IoU-to-GT differ between
      baseline and updated -- reported with bootstrap 95% confidence
      intervals (since the underlying FP-level distributions are not
      normal / not independent within an image) plus a Mann-Whitney U
      test on the FP-level (not per-image) score/IoU distributions.

Both analyses are now run TWICE:
  - ALL FPs (unchanged from the original script)
  - NEAR-GT FPs ONLY: FPs whose max-IoU to the nearest GT box is >=
    --near-gt-iou. This isolates the "duplication at real people" /
    crowd-adjacent failure mode from generic background hallucination,
    so the significance tests speak directly to that specific mechanism
    instead of being diluted by unrelated background noise.

Reuses match_and_collect_fps / classify_fp_neighborhood from
citypersons_fp_clustering_diagnosis.py so results are guaranteed
consistent with every earlier diagnostic script.

Usage:
    python3 solidify_heavy_occlusion_fp_comparison.py \
        --gt val_instances_with_occlusion_visibility_ratio.json \
        --baseline results_citypersons_baseline_val.json \
        --updated results_citypersons_sampling_loss_val.json \
        --score-thresh 0.1 \
        --near-gt-iou 0.1 \
        --n-bootstrap 5000
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

from citypersons_fp_clustering_diagnosis import (
    load_json, get_target_category_ids, build_gt_structures,
    match_and_collect_fps, cluster_gt, classify_fp_neighborhood,
    CLUSTER_IOU_THRESH,
)


def print_table(rows, header):
    if HAS_TABULATE:
        print(tabulate(rows, headers=header, tablefmt="rounded_outline"))
    else:
        widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
        fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        print(fmt.format(*header))
        for row in rows:
            print(fmt.format(*[str(c) for c in row]))


def bootstrap_mean_ci(values, n_bootstrap=5000, ci=95, seed=0):
    """Bootstrap CI for the mean of a 1D array."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()
    lo = np.percentile(boot_means, (100 - ci) / 2)
    hi = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return values.mean(), lo, hi


def per_image_fp_counts(fp_records, all_image_ids):
    counts = defaultdict(int)
    for r in fp_records:
        counts[r["img_id"]] += 1
    return np.array([counts.get(img_id, 0) for img_id in all_image_ids])


def run_paired_fp_count_test(b_fps, u_fps, all_image_ids, n_bootstrap, section_label):
    """(1) Paired per-image FP count test -- factored out so it can run on
    either the full FP set or a near-GT-only filtered subset."""
    print("\n" + "=" * 78)
    print(f"(1) Paired per-image FP count: baseline vs updated  [{section_label}]")
    print("=" * 78)

    b_counts = per_image_fp_counts(b_fps, all_image_ids)
    u_counts = per_image_fp_counts(u_fps, all_image_ids)
    diffs = u_counts - b_counts

    n_images = len(all_image_ids)
    n_more = int((diffs > 0).sum())
    n_fewer = int((diffs < 0).sum())
    n_equal = int((diffs == 0).sum())

    print(f"Images evaluated: {n_images}")
    print(f"Total FPs -- baseline: {b_counts.sum():,}   updated: {u_counts.sum():,}   "
          f"(mean/image: {b_counts.mean():.3f} vs {u_counts.mean():.3f})")
    print(f"\nPer-image comparison:")
    print(f"  images where updated has MORE FPs than baseline : {n_more} ({100*n_more/n_images:.1f}%)")
    print(f"  images where updated has FEWER FPs than baseline: {n_fewer} ({100*n_fewer/n_images:.1f}%)")
    print(f"  images with equal FP count                       : {n_equal} ({100*n_equal/n_images:.1f}%)")

    if n_more > 0 or n_fewer > 0:
        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(u_counts, b_counts, alternative="greater")
        except ValueError as e:
            wilcoxon_stat, wilcoxon_p = float("nan"), float("nan")
            print(f"  [Wilcoxon test could not run: {e}]")
    else:
        wilcoxon_stat, wilcoxon_p = float("nan"), float("nan")

    t_stat, t_p = stats.ttest_rel(u_counts, b_counts, alternative="greater")

    print(f"\nWilcoxon signed-rank test (H1: updated > baseline, per image) : "
          f"stat={wilcoxon_stat:.1f}  p={wilcoxon_p:.3g}")
    print(f"Paired t-test          (H1: updated > baseline, per image) : "
          f"t={t_stat:.3f}  p={t_p:.3g}")

    mean_diff, lo, hi = bootstrap_mean_ci(diffs, n_bootstrap)
    print(f"\nMean per-image FP increase (updated - baseline): {mean_diff:+.3f}  "
          f"[95% CI: {lo:+.3f}, {hi:+.3f}]")

    if wilcoxon_p < 0.05 and lo > 0:
        print(f"\n-> CONFIRMED [{section_label}]: updated produces significantly more FPs")
        print("   per image than baseline. Systematic, image-wide effect.")
    else:
        print(f"\n-> NOT confirmed at p<0.05 with a CI excluding zero [{section_label}].")

    return {"b_counts": b_counts, "u_counts": u_counts, "wilcoxon_p": wilcoxon_p,
            "mean_diff": mean_diff, "ci": (lo, hi)}


def run_confidence_iou_test(b_fps_full, u_fps_full, n_bootstrap, section_label):
    """(2) FP confidence / max-IoU-to-GT comparison -- factored out so it can
    run on either the full FP set or a near-GT-only filtered subset."""
    print("\n" + "=" * 78)
    print(f"(2) FP confidence and max-IoU-to-GT: baseline vs updated  [{section_label}]")
    print("=" * 78)

    if len(b_fps_full) == 0 or len(u_fps_full) == 0:
        print("  [Skipped -- one or both FP lists are empty for this subset.]")
        return

    b_scores = np.array([r["score"] for r in b_fps_full])
    u_scores = np.array([r["score"] for r in u_fps_full])
    b_ious = np.array([r["max_iou"] for r in b_fps_full])
    u_ious = np.array([r["max_iou"] for r in u_fps_full])

    rows = []
    for name, b_arr, u_arr in [("Mean FP confidence", b_scores, u_scores),
                                ("Mean FP max-IoU to GT", b_ious, u_ious)]:
        b_mean, b_lo, b_hi = bootstrap_mean_ci(b_arr, n_bootstrap, seed=1)
        u_mean, u_lo, u_hi = bootstrap_mean_ci(u_arr, n_bootstrap, seed=2)
        rows.append([name,
                     f"{b_mean:.4f} [{b_lo:.4f}, {b_hi:.4f}]",
                     f"{u_mean:.4f} [{u_lo:.4f}, {u_hi:.4f}]",
                     f"{u_mean - b_mean:+.4f}"])
    print_table(rows, ["Metric", "Baseline (mean [95% CI])", "Updated (mean [95% CI])", "Delta"])

    u_stat_score, p_score = stats.mannwhitneyu(u_scores, b_scores, alternative="two-sided")
    u_stat_iou, p_iou = stats.mannwhitneyu(u_ious, b_ious, alternative="two-sided")

    print(f"\nMann-Whitney U test, FP confidence  (baseline vs updated distributions): p={p_score:.3g}")
    print(f"Mann-Whitney U test, FP max-IoU      (baseline vs updated distributions): p={p_iou:.3g}")

    print(f"\nFP count for reference [{section_label}] -- baseline: {len(b_fps_full):,}   "
          f"updated: {len(u_fps_full):,}")

    if p_score < 0.05:
        direction = "LOWER" if u_scores.mean() < b_scores.mean() else "HIGHER"
        print(f"-> Updated's FP confidence is significantly {direction} than baseline's "
              f"(p={p_score:.3g}).")
    if p_iou < 0.05:
        direction = "HIGHER" if u_ious.mean() > b_ious.mean() else "LOWER"
        print(f"-> Updated's FP max-IoU-to-GT is significantly {direction} than baseline's "
              f"(p={p_iou:.3g}).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--updated", required=True)
    ap.add_argument("--target-categories", nargs="+",
                     default=["pedestrian", "rider", "sitting person"])
    ap.add_argument("--score-thresh", type=float, default=0.1)
    ap.add_argument("--near-gt-iou", type=float, default=0.1,
                     help="Max-IoU-to-nearest-GT threshold used to define the "
                          "'near-GT-only' FP subset (the second, filtered set "
                          "of tests). FPs with max_iou below this are treated "
                          "as background/no-nearby-GT and excluded from that "
                          "subset's stats.")
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    args = ap.parse_args()

    for p in (args.gt, args.baseline, args.updated):
        if not Path(p).exists():
            raise FileNotFoundError(p)

    gt_full = load_json(args.gt)
    target_category_ids = get_target_category_ids(gt_full, args.target_categories)
    b_preds = load_json(args.baseline)
    u_preds = load_json(args.updated)

    gt_by_image_subset, gt_all_by_image = build_gt_structures(gt_full, target_category_ids)
    all_image_ids = sorted(gt_by_image_subset.keys())

    print("Matching baseline FPs (Heavy_Occlusion subset)...")
    b_fps = match_and_collect_fps(gt_by_image_subset, b_preds, target_category_ids, args.score_thresh)
    print("Matching updated FPs (Heavy_Occlusion subset)...")
    u_fps = match_and_collect_fps(gt_by_image_subset, u_preds, target_category_ids, args.score_thresh)

    # attach max-IoU-to-nearest-GT to every FP up front, once, so both the
    # "all FPs" and "near-GT-only" analyses can reuse it consistently
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
            label, best_iou, csize = classify_fp_neighborhood(r["bbox"], boxes, sizes)
            out.append({**r, "max_iou": best_iou, "label": label})
        return out

    b_fps_full = attach_max_iou(b_fps)
    u_fps_full = attach_max_iou(u_fps)

    # near-GT-only subsets: FPs whose max-IoU to the nearest GT box clears
    # --near-gt-iou (i.e. they sit ON or immediately adjacent to a real,
    # possibly crowd-clustered person -- NOT background/occluder noise)
    b_fps_near = [r for r in b_fps_full if r["max_iou"] >= args.near_gt_iou]
    u_fps_near = [r for r in u_fps_full if r["max_iou"] >= args.near_gt_iou]

    print(f"\nNear-GT filter: max_iou >= {args.near_gt_iou}")
    print(f"  Baseline FPs: {len(b_fps_full):,} total -> {len(b_fps_near):,} near-GT "
          f"({100*len(b_fps_near)/max(len(b_fps_full),1):.1f}%)")
    print(f"  Updated  FPs: {len(u_fps_full):,} total -> {len(u_fps_near):,} near-GT "
          f"({100*len(u_fps_near)/max(len(u_fps_full),1):.1f}%)")

    # ══════════════════════════════════════════════════════════════
    # Run both tests on ALL FPs first (reference / matches earlier runs)
    # ══════════════════════════════════════════════════════════════
    run_paired_fp_count_test(b_fps_full, u_fps_full, all_image_ids, args.n_bootstrap,
                              section_label="ALL FPs")
    run_confidence_iou_test(b_fps_full, u_fps_full, args.n_bootstrap,
                             section_label="ALL FPs")

    # ══════════════════════════════════════════════════════════════
    # Run both tests again, restricted to NEAR-GT FPs only
    # ══════════════════════════════════════════════════════════════
    run_paired_fp_count_test(b_fps_near, u_fps_near, all_image_ids, args.n_bootstrap,
                              section_label=f"NEAR-GT ONLY, IoU>={args.near_gt_iou}")
    run_confidence_iou_test(b_fps_near, u_fps_near, args.n_bootstrap,
                             section_label=f"NEAR-GT ONLY, IoU>={args.near_gt_iou}")

    print("\nDone.")


if __name__ == "__main__":
    main()