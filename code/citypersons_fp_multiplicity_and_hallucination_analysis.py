"""
fp_multiplicity_and_hallucination_analysis.py
================================================

Two follow-ups to fp_clustering_diagnosis.py's finding that 54.6% of NEW
Heavy_Occlusion FPs have no nearby GT at all, while duplication is likely
hiding in the "persistent" bucket instead:

  (A) PERSISTENT-LOCATION MULTIPLICITY
      For every baseline FP location, count how many updated FPs match it
      (IoU >= 0.5, same image). If updated frequently produces 2-3 FPs
      where baseline produced 1, that's the duplicate-box-in-hard-region
      signature -- just located where baseline already had a spurious box,
      not in a fresh crowd region.

  (B) no_nearby_gt HALLUCINATION CHARACTERIZATION
      For the "new" FPs with no nearby GT (pure background/occluder
      hallucination): score distribution, box-height distribution, and
      whether they concentrate in images with higher occlusion density
      (i.e. "sampling loss loses grounding specifically in cluttered
      scenes") vs scattering randomly across all images.

Requires fp_clustering_diagnosis.py in the same directory (imports its
matching/clustering functions directly, so both scripts stay consistent).

Usage:
    python3 fp_multiplicity_and_hallucination_analysis.py \
        --gt val_instances_with_occlusion_visibility_ratio.json \
        --baseline results_citypersons_final_val.json \
        --updated results_citypersons_sampling_loss_val.json \
        --score-thresh 0.10
"""

import argparse
from pathlib import Path
from collections import defaultdict
import json

import numpy as np
from scipy import stats

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

from citypersons_fp_clustering_diagnosis import (
    load_json, get_target_category_ids, build_gt_structures,
    match_and_collect_fps, iou, cluster_gt, classify_fp_neighborhood,
    NEW_FP_MATCH_IOU, CLUSTER_IOU_THRESH, VIS_RANGE,
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


# ────────────────────────────────────────────────────────────────────────
# (A) Persistent-location FP multiplicity
# ────────────────────────────────────────────────────────────────────────

def analyze_multiplicity(b_fps, u_fps, out_json="high_multiplicity_locations.json"):
    b_by_img = defaultdict(list)
    for i, r in enumerate(b_fps):
        b_by_img[r["img_id"]].append((i, r["bbox"]))

    # for each updated FP, find its best-matching baseline FP (if any, IoU>=thresh)
    mult_counter = defaultdict(int)          # baseline_fp_global_id -> count
    mult_matches = defaultdict(list)         # baseline_fp_global_id -> list of updated FP records
    unmatched_updated = 0

    for r in u_fps:
        candidates = b_by_img.get(r["img_id"], [])
        best_i, best_iou = None, NEW_FP_MATCH_IOU
        for (i, bbox) in candidates:
            ov = iou(r["bbox"], bbox)
            if ov >= best_iou:
                best_iou, best_i = ov, i
        if best_i is not None:
            mult_counter[best_i] += 1
            mult_matches[best_i].append({"bbox": r["bbox"], "score": r["score"]})
        else:
            unmatched_updated += 1  # these are the "new" FPs from the prior script

    total_baseline_fps = len(b_fps)
    matched_baseline_fps = len(mult_counter)
    unmatched_baseline_fps = total_baseline_fps - matched_baseline_fps

    mults = list(mult_counter.values())
    hist = defaultdict(int)
    for m in mults:
        bucket = m if m <= 3 else "4+"
        hist[bucket] += 1

    print("\n" + "=" * 78)
    print("(A) Persistent-location FP multiplicity")
    print("=" * 78)
    print(f"Baseline FPs total          : {total_baseline_fps:,}")
    print(f"  -> matched by >=1 updated FP : {matched_baseline_fps:,} "
          f"({100*matched_baseline_fps/total_baseline_fps:.1f}%)")
    print(f"  -> matched by 0 updated FPs  : {unmatched_baseline_fps:,} "
          f"(baseline FP that updated no longer produces)")
    print(f"Updated FPs with no baseline match (= 'new' FPs) : {unmatched_updated:,}")

    print("\nDistribution: how many updated FPs land on each single baseline FP location")
    rows = []
    for k in [1, 2, 3, "4+"]:
        n = hist.get(k, 0)
        pct = 100.0 * n / matched_baseline_fps if matched_baseline_fps else 0.0
        rows.append([f"{k} updated FP(s)", n, f"{pct:.1f}%"])
    print_table(rows, ["Multiplicity", "# baseline FP locations", "% of matched locations"])

    if mults:
        excess = sum(m - 1 for m in mults)
        mean_mult = np.mean(mults)
        print(f"\nMean multiplicity at matched locations : {mean_mult:.2f}")
        print(f"Excess updated FPs from duplication      : {excess:,} "
              f"(updated FPs beyond a 1-to-1 mapping with baseline FP locations)")
        if mean_mult > 1.15:
            print("-> Meaningful duplication: updated often places multiple FPs where")
            print("   baseline placed one. This IS the duplicate-box-in-hard-region")
            print("   signature -- just at pre-existing hard locations, not new ones.")
        else:
            print("-> Multiplicity is close to 1-to-1; duplication at existing FP")
            print("   locations does not look like the dominant driver either.")

    # ── dump per-location detail, sorted by multiplicity descending ──
    detail = []
    for baseline_i, updated_list in mult_matches.items():
        img_id, baseline_bbox = b_fps[baseline_i]["img_id"], b_fps[baseline_i]["bbox"]
        detail.append({
            "img_id": img_id,
            "baseline_bbox": baseline_bbox,
            "baseline_score": b_fps[baseline_i]["score"],
            "multiplicity": len(updated_list),
            "updated_fps": sorted(updated_list, key=lambda r: -r["score"]),
        })
    detail.sort(key=lambda d: -d["multiplicity"])

    with open(out_json, "w") as f:
        json.dump(detail, f, indent=2)
    print(f"\nFull per-location multiplicity detail written to: {out_json}")

    return detail

# ────────────────────────────────────────────────────────────────────────
# (B) no_nearby_gt hallucination characterization
# ────────────────────────────────────────────────────────────────────────

def analyze_hallucinations(new_fps, gt_by_image_subset, gt_all_by_image, target_category_ids):
    cluster_cache = {}

    def get_clusters(img_id):
        if img_id not in cluster_cache:
            boxes = gt_all_by_image.get(img_id, [])
            sizes = cluster_gt(boxes, CLUSTER_IOU_THRESH)
            cluster_cache[img_id] = (boxes, sizes)
        return cluster_cache[img_id]

    hallucinations = []
    for r in new_fps:
        boxes, sizes = get_clusters(r["img_id"])
        label, best_iou, csize = classify_fp_neighborhood(r["bbox"], boxes, sizes)
        if label == "no_nearby_gt":
            hallucinations.append(r)

    print("\n" + "=" * 78)
    print(f"(B) Characterizing {len(hallucinations):,} no_nearby_gt hallucinations")
    print("=" * 78)

    scores = np.array([h["score"] for h in hallucinations])
    heights = np.array([h["bbox"][3] for h in hallucinations])

    print("\nScore distribution:")
    print(f"  mean={scores.mean():.3f}  median={np.median(scores):.3f}  "
          f"p75={np.percentile(scores,75):.3f}  p90={np.percentile(scores,90):.3f}  "
          f"max={scores.max():.3f}")
    bins = [0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.01]
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        n = int(((scores >= lo) & (scores < hi)).sum())
        rows.append([f"[{lo:.2f}, {hi:.2f})", n, f"{100*n/len(scores):.1f}%"])
    print_table(rows, ["Score range", "Count", "% of hallucinations"])

    print("\nBox-height distribution (px):")
    print(f"  mean={heights.mean():.1f}  median={np.median(heights):.1f}  "
          f"p10={np.percentile(heights,10):.1f}  p90={np.percentile(heights,90):.1f}")
    h_bins = [40, 50, 75, 100, 150, 250, 10000]
    rows = []
    for lo, hi in zip(h_bins[:-1], h_bins[1:]):
        n = int(((heights >= lo) & (heights < hi)).sum())
        rows.append([f"[{lo}, {hi})", n, f"{100*n/len(heights):.1f}%"])
    print_table(rows, ["Height range (px)", "Count", "% of hallucinations"])

    if scores.mean() < 0.20 and np.percentile(heights, 50) < 75:
        print("\n-> Mostly low-confidence, near-minimum-size boxes: consistent with")
        print("   generic low-grade noise near the detection threshold, not a strong")
        print("   confident failure mode. A higher deployment score threshold would")
        print("   likely suppress most of these cheaply.")
    else:
        print("\n-> Meaningful mass at higher confidence/size: these are not just")
        print("   threshold noise -- worth visualizing directly.")

    # occlusion-density correlation, at the image level
    print("\n" + "-" * 78)
    print("Per-image occlusion density vs hallucination count")
    print("-" * 78)

    halluc_by_img = defaultdict(int)
    for h in hallucinations:
        halluc_by_img[h["img_id"]] += 1

    img_ids = list(gt_by_image_subset.keys())
    density, halluc_count = [], []
    for img_id in img_ids:
        all_gt = gt_all_by_image.get(img_id, [])
        if not all_gt:
            continue
        occ_gt = [a for a in gt_by_image_subset[img_id]
                  if a["ignore"] == 0]  # already restricted to Heavy_Occlusion window
        dens = len(occ_gt) / len(all_gt)
        density.append(dens)
        halluc_count.append(halluc_by_img.get(img_id, 0))

    density = np.array(density)
    halluc_count = np.array(halluc_count)

    rho, p = stats.spearmanr(density, halluc_count)
    print(f"Spearman correlation (occlusion density vs hallucination count per image): "
          f"rho={rho:.3f}  p={p:.3g}")

    with_h = density[halluc_count > 0]
    without_h = density[halluc_count == 0]
    print(f"\nMean occlusion density | images WITH >=1 hallucination : {with_h.mean():.3f} (n={len(with_h)})")
    print(f"Mean occlusion density | images WITH 0 hallucinations    : {without_h.mean():.3f} (n={len(without_h)})")

    if p < 0.05 and rho > 0.1:
        print("\n-> Significant positive correlation: hallucinations concentrate in more")
        print("   occlusion-dense images. Supports 'sampling loss loses grounding")
        print("   specifically in cluttered/occluded scenes' (image-level effect).")
    else:
        print("\n-> No strong correlation: hallucinations look roughly independent of")
        print("   how occlusion-dense the image is -- more consistent with general,")
        print("   scattered noise than a scene-clutter-specific failure mode.")

    return hallucinations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--updated", required=True)
    ap.add_argument("--target-categories", nargs="+",
                     default=["pedestrian", "rider", "sitting person"])
    ap.add_argument("--score-thresh", type=float, default=0.10)
    args = ap.parse_args()

    for p in (args.gt, args.baseline, args.updated):
        if not Path(p).exists():
            raise FileNotFoundError(p)

    gt_full = load_json(args.gt)
    target_category_ids = get_target_category_ids(gt_full, args.target_categories)
    b_preds = load_json(args.baseline)
    u_preds = load_json(args.updated)

    gt_by_image_subset, gt_all_by_image = build_gt_structures(gt_full, target_category_ids)

    print("Matching baseline FPs...")
    b_fps = match_and_collect_fps(gt_by_image_subset, b_preds, target_category_ids, args.score_thresh)
    print("Matching updated FPs...")
    u_fps = match_and_collect_fps(gt_by_image_subset, u_preds, target_category_ids, args.score_thresh)

    analyze_multiplicity(b_fps, u_fps)

    b_fps_by_img = defaultdict(list)
    for r in b_fps:
        b_fps_by_img[r["img_id"]].append(r["bbox"])
    new_fps = [r for r in u_fps
               if not any(iou(r["bbox"], bb) >= NEW_FP_MATCH_IOU for bb in b_fps_by_img.get(r["img_id"], []))]

    analyze_hallucinations(new_fps, gt_by_image_subset, gt_all_by_image, target_category_ids)

    print("\nDone.")


if __name__ == "__main__":
    main()