"""
analyze_tp_confidence_distribution.py
========================================
Mirrors solidify_heavy_occlusion_fp_comparison.py, but for TRUE POSITIVES
instead of false positives -- directly tests whether updated's TP
confidence is really lower than baseline's (not just a mean-difference
point estimate), and visualizes the distribution shift.

Reports, for the Heavy_Occlusion subset:
  (1) Mean TP confidence & mean TP IoU-to-matched-GT, baseline vs updated,
      with bootstrap 95% CIs + Mann-Whitney U test (FP-level distributions,
      pooled across images).
  (2) Paired per-image mean-TP-confidence test (Wilcoxon signed-rank),
      restricted to images where BOTH models have >=1 TP, so the pairing
      is meaningful.
  (3) A histogram plot overlaying baseline vs updated TP confidence
      (and TP IoU) distributions.

Uses the SAME official per-image greedy matching rule as
citypersons_fp_clustering_diagnosis.py's match_and_collect_fps (IoU>=0.5
real match / IoA>=0.5 ignore, exp-filtered height window), just collecting
the matched TRUE positives instead of the false positives, so results are
guaranteed consistent with every earlier diagnostic script.

Usage:
    python3 analyze_tp_confidence_distribution.py \
        --gt val_instances_with_occlusion_visibility_ratio.json \
        --baseline results_citypersons_baseline_val.json \
        --updated results_citypersons_sampling_loss_val.json \
        --score-thresh 0.1 \
        --n-bootstrap 5000 \
        --out-dir tp_confidence_report
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

from citypersons_fp_clustering_diagnosis import (
    load_json, get_target_category_ids, build_gt_structures,
    iou, ioa, IOU_THRESHOLD, EXP_FILTER, HEIGHT_RANGE,
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


def match_and_collect_tps(gt_by_image_subset, pred_list, target_category_ids, score_thresh):
    """Same official per-image greedy matcher as match_and_collect_fps in
    citypersons_fp_clustering_diagnosis.py, but returns the matched TRUE
    POSITIVES (detection matched to a non-ignore GT box) instead of FPs.
    Each record: {img_id, bbox, score, gt_bbox, gt_iou}."""
    h_lo, h_hi = HEIGHT_RANGE
    valid_ids = set(gt_by_image_subset.keys())
    dt_by_image = defaultdict(list)
    for p in pred_list:
        if p["image_id"] not in valid_ids or p["category_id"] not in target_category_ids:
            continue
        if p["score"] < score_thresh:
            continue
        dh = p["bbox"][3]
        if not (h_lo / EXP_FILTER <= dh < h_hi * EXP_FILTER):
            continue
        dt_by_image[p["image_id"]].append(p)

    tp_records = []
    for img_id, gts in gt_by_image_subset.items():
        dts = sorted(dt_by_image.get(img_id, []), key=lambda p: -p["score"])
        matched_gt = [False] * len(gts)
        for d in dts:
            best_ov, best_gi, best_kind = IOU_THRESHOLD, -2, -2
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
                continue  # FP -- not collected here
            elif best_kind == 1:
                matched_gt[best_gi] = True
                tp_records.append({"img_id": img_id, "bbox": d["bbox"], "score": d["score"],
                                    "gt_bbox": gts[best_gi]["bbox"], "gt_iou": best_ov})
            # best_kind == -1 (matched an ignore region): neither TP nor FP, skip
    return tp_records


def per_image_mean_conf(tp_records, image_ids):
    by_img = defaultdict(list)
    for r in tp_records:
        by_img[r["img_id"]].append(r["score"])
    means = {}
    for img_id in image_ids:
        scores = by_img.get(img_id, [])
        if scores:
            means[img_id] = float(np.mean(scores))
    return means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--updated", required=True)
    ap.add_argument("--target-categories", nargs="+",
                     default=["pedestrian", "rider", "sitting person"])
    ap.add_argument("--score-thresh", type=float, default=0.1)
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    ap.add_argument("--out-dir", default="tp_confidence_report")
    args = ap.parse_args()

    for p in (args.gt, args.baseline, args.updated):
        if not Path(p).exists():
            raise FileNotFoundError(p)

    gt_full = load_json(args.gt)
    target_category_ids = get_target_category_ids(gt_full, args.target_categories)
    b_preds = load_json(args.baseline)
    u_preds = load_json(args.updated)

    gt_by_image_subset, _ = build_gt_structures(gt_full, target_category_ids)
    all_image_ids = sorted(gt_by_image_subset.keys())

    print("Matching baseline TPs (Heavy_Occlusion subset)...")
    b_tps = match_and_collect_tps(gt_by_image_subset, b_preds, target_category_ids, args.score_thresh)
    print("Matching updated TPs (Heavy_Occlusion subset)...")
    u_tps = match_and_collect_tps(gt_by_image_subset, u_preds, target_category_ids, args.score_thresh)

    b_scores = np.array([r["score"] for r in b_tps])
    u_scores = np.array([r["score"] for r in u_tps])
    b_ious = np.array([r["gt_iou"] for r in b_tps])
    u_ious = np.array([r["gt_iou"] for r in u_tps])

    # ══════════════════════════════════════════════════════════════
    # (1) TP-level confidence / IoU comparison, pooled across images
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("(1) TP confidence and TP IoU-to-matched-GT: baseline vs updated")
    print("=" * 78)
    print(f"TP count -- baseline: {len(b_tps):,}   updated: {len(u_tps):,}")

    rows = []
    for name, b_arr, u_arr in [("Mean TP confidence", b_scores, u_scores),
                                ("Mean TP IoU to matched GT", b_ious, u_ious)]:
        b_mean, b_lo, b_hi = bootstrap_mean_ci(b_arr, args.n_bootstrap, seed=1)
        u_mean, u_lo, u_hi = bootstrap_mean_ci(u_arr, args.n_bootstrap, seed=2)
        rows.append([name,
                     f"{b_mean:.4f} [{b_lo:.4f}, {b_hi:.4f}]",
                     f"{u_mean:.4f} [{u_lo:.4f}, {u_hi:.4f}]",
                     f"{u_mean - b_mean:+.4f}"])
    print_table(rows, ["Metric", "Baseline (mean [95% CI])", "Updated (mean [95% CI])", "Delta"])

    _, p_score = stats.mannwhitneyu(u_scores, b_scores, alternative="two-sided")
    _, p_iou = stats.mannwhitneyu(u_ious, b_ious, alternative="two-sided")
    print(f"\nMann-Whitney U test, TP confidence (baseline vs updated): p={p_score:.3g}")
    print(f"Mann-Whitney U test, TP IoU          (baseline vs updated): p={p_iou:.3g}")

    if p_score < 0.05:
        direction = "LOWER" if u_scores.mean() < b_scores.mean() else "HIGHER"
        print(f"-> Updated's TP confidence is significantly {direction} than baseline's "
              f"(p={p_score:.3g}) -- this is a genuine confidence shift, not sampling noise.")
    if p_iou < 0.05:
        direction = "LOWER" if u_ious.mean() < b_ious.mean() else "HIGHER"
        print(f"-> Updated's TP box-GT IoU is significantly {direction} than baseline's "
              f"(p={p_iou:.3g}) -- localization quality on correct detections has shifted too.")

    # score percentile breakdown -- useful to see if the drop is uniform or
    # concentrated near the score threshold (would matter for deployment)
    print("\nTP confidence percentiles:")
    pct_rows = []
    for pct in [10, 25, 50, 75, 90]:
        pct_rows.append([f"p{pct}", f"{np.percentile(b_scores, pct):.4f}",
                          f"{np.percentile(u_scores, pct):.4f}",
                          f"{np.percentile(u_scores, pct) - np.percentile(b_scores, pct):+.4f}"])
    print_table(pct_rows, ["Percentile", "Baseline", "Updated", "Delta"])

    # ══════════════════════════════════════════════════════════════
    # (2) Paired per-image mean TP confidence test
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("(2) Paired per-image MEAN TP confidence: baseline vs updated")
    print("    (images with >=1 TP in BOTH models only, for valid pairing)")
    print("=" * 78)

    b_means = per_image_mean_conf(b_tps, all_image_ids)
    u_means = per_image_mean_conf(u_tps, all_image_ids)
    common_images = sorted(set(b_means) & set(u_means))
    print(f"Images with >=1 TP in both models: {len(common_images)} / {len(all_image_ids)}")

    if len(common_images) >= 2:
        b_arr = np.array([b_means[i] for i in common_images])
        u_arr = np.array([u_means[i] for i in common_images])
        diffs = u_arr - b_arr

        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(u_arr, b_arr)
        mean_diff, lo, hi = bootstrap_mean_ci(diffs, args.n_bootstrap, seed=3)
        n_lower = int((diffs < 0).sum())
        n_higher = int((diffs > 0).sum())

        print(f"\nImages where updated's mean TP confidence is LOWER : {n_lower} "
              f"({100*n_lower/len(common_images):.1f}%)")
        print(f"Images where updated's mean TP confidence is HIGHER: {n_higher} "
              f"({100*n_higher/len(common_images):.1f}%)")
        print(f"\nWilcoxon signed-rank test (two-sided): stat={wilcoxon_stat:.1f}  p={wilcoxon_p:.3g}")
        print(f"Mean per-image TP-confidence change (updated - baseline): {mean_diff:+.4f}  "
              f"[95% CI: {lo:+.4f}, {hi:+.4f}]")

        if wilcoxon_p < 0.05 and hi < 0:
            print("\n-> CONFIRMED: updated's TP confidence is systematically lower per image,")
            print("   not just in the pooled aggregate -- this is a broad, image-wide")
            print("   confidence compression, not a few outlier detections dragging the mean.")
    else:
        print("Not enough paired images to run the test.")

    # ══════════════════════════════════════════════════════════════
    # (3) Distribution plot
    # ══════════════════════════════════════════════════════════════
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    bins = np.linspace(args.score_thresh, 1.0, 40)
    ax.hist(b_scores, bins=bins, density=True, alpha=0.5, color="#2ECC71",
            label=f"Baseline (n={len(b_scores):,}, mean={b_scores.mean():.3f})")
    ax.hist(u_scores, bins=bins, density=True, alpha=0.5, color="#9B59B6",
            label=f"Updated (n={len(u_scores):,}, mean={u_scores.mean():.3f})")
    ax.axvline(b_scores.mean(), color="#1E8449", ls="--", lw=1.5)
    ax.axvline(u_scores.mean(), color="#6C3483", ls="--", lw=1.5)
    ax.set_xlabel("TP confidence score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Heavy_Occlusion TP confidence distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    bins = np.linspace(0.5, 1.0, 40)
    ax.hist(b_ious, bins=bins, density=True, alpha=0.5, color="#2ECC71",
            label=f"Baseline (n={len(b_ious):,}, mean={b_ious.mean():.3f})")
    ax.hist(u_ious, bins=bins, density=True, alpha=0.5, color="#9B59B6",
            label=f"Updated (n={len(u_ious):,}, mean={u_ious.mean():.3f})")
    ax.axvline(b_ious.mean(), color="#1E8449", ls="--", lw=1.5)
    ax.axvline(u_ious.mean(), color="#6C3483", ls="--", lw=1.5)
    ax.set_xlabel("TP box IoU to matched GT", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Heavy_Occlusion TP localization (IoU) distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out_path = out_dir / "tp_confidence_and_iou_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")

    # bonus: TP confidence vs FP confidence separation plot -- the key
    # visual for "is the classifier still separating TP from FP well"
    print("\nDone.")


if __name__ == "__main__":
    main()