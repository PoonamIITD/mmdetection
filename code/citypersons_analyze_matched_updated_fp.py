"""
analyze_matched_location_confidence.py
========================================
Uses high_multiplicity_locations.json (from
fp_multiplicity_and_hallucination_analysis.py) to compare:

  - baseline FP confidence at matched locations (always 1 FP/location,
    by construction of the greedy per-image matcher)
  - updated FP confidence at those SAME locations (all boxes matched
    to that baseline location, IoU >= NEW_FP_MATCH_IOU)

Breaks the comparison down by multiplicity bucket, since a location with
4+ updated FPs stacked on it likely tells a different confidence story
than one with exactly 1.

Usage:
    python3 analyze_matched_location_confidence.py \
        --detail-json high_multiplicity_locations.json
"""

import argparse
import json
from collections import defaultdict

import numpy as np

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


def print_table(rows, header):
    if HAS_TABULATE:
        print(tabulate(rows, headers=header, tablefmt="rounded_outline"))
    else:
        widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
        fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        print(fmt.format(*header))
        for row in rows:
            print(fmt.format(*[str(c) for c in row]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail-json", default="high_multiplicity_locations.json")
    args = ap.parse_args()

    with open(args.detail_json) as f:
        detail = json.load(f)

    print("\n" + "=" * 78)
    print("Baseline vs Updated FP confidence at matched (duplicated) locations")
    print("=" * 78)
    print(f"Total matched locations: {len(detail):,}")

    # ── overall: baseline conf at these locations vs ALL updated FPs matched there ──
    baseline_scores = np.array([d["baseline_score"] for d in detail])
    all_updated_scores = np.array([u["score"] for d in detail for u in d["updated_fps"]])
    # updated FP with the HIGHEST confidence at each location (the "best duplicate")
    top_updated_scores = np.array([max(u["score"] for u in d["updated_fps"]) for d in detail])
    # updated FP with the LOWEST confidence at each location (the "weakest duplicate")
    min_updated_scores = np.array([min(u["score"] for u in d["updated_fps"]) for d in detail])
    # mean confidence of updated FPs per location, then averaged across locations
    mean_updated_per_loc = np.array([np.mean([u["score"] for u in d["updated_fps"]]) for d in detail])

    print("\nOverall (n = %d matched locations):" % len(detail))
    rows = [
        ["Baseline FP (1 per location)", f"{baseline_scores.mean():.4f}", f"{np.median(baseline_scores):.4f}"],
        ["Updated FPs - ALL individual boxes", f"{all_updated_scores.mean():.4f}", f"{np.median(all_updated_scores):.4f}"],
        ["Updated FPs - highest conf per location", f"{top_updated_scores.mean():.4f}", f"{np.median(top_updated_scores):.4f}"],
        ["Updated FPs - lowest conf per location", f"{min_updated_scores.mean():.4f}", f"{np.median(min_updated_scores):.4f}"],
        ["Updated FPs - mean-per-location", f"{mean_updated_per_loc.mean():.4f}", f"{np.median(mean_updated_per_loc):.4f}"],
    ]
    print_table(rows, ["Group", "Mean score", "Median score"])

    delta_top = top_updated_scores - baseline_scores
    delta_mean = mean_updated_per_loc - baseline_scores
    print(f"\nMean(highest updated FP conf - baseline conf) per location : {delta_top.mean():+.4f}")
    print(f"Mean(avg updated FP conf   - baseline conf) per location    : {delta_mean.mean():+.4f}")
    pct_top_higher = 100.0 * (delta_top > 0).sum() / len(delta_top)
    print(f"% locations where the BEST updated duplicate outscores baseline : {pct_top_higher:.1f}%")

    # ── breakdown by multiplicity bucket ──
    print("\n" + "-" * 78)
    print("Breakdown by multiplicity bucket")
    print("-" * 78)

    buckets = defaultdict(list)
    for d in detail:
        m = d["multiplicity"]
        key = m if m <= 3 else "4+"
        buckets[key].append(d)

    rows = []
    for key in [1, 2, 3, "4+"]:
        items = buckets.get(key, [])
        if not items:
            continue
        b_scores = np.array([d["baseline_score"] for d in items])
        u_scores_flat = np.array([u["score"] for d in items for u in d["updated_fps"]])
        rows.append([
            f"{key} updated FP(s)",
            len(items),
            f"{b_scores.mean():.4f}",
            f"{u_scores_flat.mean():.4f}",
            f"{u_scores_flat.mean() - b_scores.mean():+.4f}",
        ])
    print_table(rows, ["Multiplicity", "# locations", "Mean baseline conf",
                        "Mean updated conf (all boxes)", "Delta"])

    # ── how many updated duplicates actually beat baseline's own confidence ──
    print("\n" + "-" * 78)
    print("Do the EXTRA (duplicate) boxes tend to be higher or lower confidence")
    print("than baseline's single box at that location?")
    print("-" * 78)

    higher, lower, equal_ish = 0, 0, 0
    all_diffs = []
    for d in detail:
        for u in d["updated_fps"]:
            diff = u["score"] - d["baseline_score"]
            all_diffs.append(diff)
            if diff > 0.01:
                higher += 1
            elif diff < -0.01:
                lower += 1
            else:
                equal_ish += 1
    all_diffs = np.array(all_diffs)
    total = len(all_diffs)
    print(f"Total updated FP boxes at matched locations: {total:,}")
    print(f"  higher confidence than baseline's box : {higher:,} ({100*higher/total:.1f}%)")
    print(f"  lower confidence than baseline's box  : {lower:,} ({100*lower/total:.1f}%)")
    print(f"  ~equal (within 0.01)                  : {equal_ish:,} ({100*equal_ish/total:.1f}%)")
    print(f"Mean(updated box conf - baseline conf), all boxes : {all_diffs.mean():+.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()