"""
fn_suppression_analysis.py  (v2)
=================================
Partitions False Negative GT objects into FOUR cleanly separated groups
using the pairs JSON as the single source of truth for occlusion.

  Group A  — In pairs + suppression condition MET + in FN list
             → occluded query was suppressed AND object was missed
             → primary claim group

  Group B  — In pairs + suppression condition NOT MET + in FN list
             → occluded, same occluder confidence range, same class mode
               BUT decoder did not suppress confidence
             → strongest possible control (same pipeline, different outcome)

  Group B2 — Geometrically occluded in GT (ratio >= threshold) but never
             captured in pairs list (occluder below conf threshold, or
             cross-class when running same-class mode) + in FN list
             → occluded but outside your analysis scope

  Group C  — Not occluded at all + in FN list
             → pure detection failure baseline

The key comparison is A vs B:
  If A FN rate > B FN rate  -> suppression specifically causes missed detections
  If A FN rate ~ B FN rate  -> occlusion is the driver, not suppression per se

Usage
-----
python fn_suppression_analysis.py \
    --fn_json         fn_detections.json \
    --gt_json         val_annotations.json \
    --pairs_json      occlusion_results.json \
    [--occlusion_ratio   0.3] \
    [--output_json    fn_group_analysis.json]
"""

import json
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path


# ============================================================
# BOX UTILS
# ============================================================

def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def compute_occlusion_ratio(box_occ, box_occder):
    xA = max(box_occ[0], box_occder[0])
    yA = max(box_occ[1], box_occder[1])
    xB = min(box_occ[2], box_occder[2])
    yB = min(box_occ[3], box_occder[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area  = max(0, box_occ[2]-box_occ[0]) * max(0, box_occ[3]-box_occ[1]) + 1e-6
    return inter / area


def compute_iou(box1, box2):
    xA = max(box1[0], box2[0]); yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2]); yB = min(box1[3], box2[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    a1 = max(0, box1[2]-box1[0]) * max(0, box1[3]-box1[1])
    a2 = max(0, box2[2]-box2[0]) * max(0, box2[3]-box2[1])
    return inter / (a1 + a2 - inter + 1e-6)


# ============================================================
# LOADERS
# ============================================================

def load_coco(json_path):
    with open(json_path) as f:
        data = json.load(f)
    images         = {img["id"]: img for img in data["images"]}
    categories     = {cat["id"]: cat["name"] for cat in data["categories"]}
    anns_by_id     = {}
    anns_per_image = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_id[ann["id"]] = ann
        anns_per_image[ann["image_id"]].append(ann)
    return images, anns_by_id, anns_per_image, categories


def load_pairs(json_path):
    with open(json_path) as f:
        return json.load(f)


# ============================================================
# SUPPRESSION CONDITION
# (mirrors your main analysis script exactly)
# ============================================================

def occluded_not_maintained(scores, eps=1e-6):
    """
    Suppressed if:
      peak confidence occurs before the final layer
      OR final confidence is lower than starting confidence.
    Captures: rise-then-fall, continuous fall, oscillating decay.
    """
    end  = scores[-1]
    peak = max(scores)
    return (peak > end + eps) or (end < scores[0] - eps)


def occluder_increasing(scores, eps=1e-6):
    return scores[-1] > scores[0] + eps

def diagnose_b2_objects(b2_fn_ann_ids, gt_anns_by_id,
                         gt_anns_per_image, pairs, occder_score_low=0.75,
                         occ_ratio_threshold=0.3):
    """
    For each FN object in B2, find exactly why it never formed a pair.
    """
    reasons = defaultdict(int)

    # build set of ann_ids that DO appear as occluders in pairs
    occluder_ann_ids_in_pairs = {p["occluder_gt_ann_id"] for p in pairs}

    for ann_id in b2_fn_ann_ids:
        ann      = gt_anns_by_id[ann_id]
        image_id = ann["image_id"]
        box_i    = xywh_to_xyxy(ann["bbox"])

        # get all other GT objects in same image
        siblings = [a for a in gt_anns_per_image[image_id]
                    if a["id"] != ann_id]

        has_geometric_occluder      = False
        has_matched_occluder        = False
        has_conf_occluder           = False

        for sib in siblings:
            box_j = xywh_to_xyxy(sib["bbox"])
            ratio = compute_occlusion_ratio(box_i, box_j)

            if ratio >= occ_ratio_threshold:
                has_geometric_occluder = True

                # is this sibling's ann_id in any pair as occluder?
                if sib["id"] in occluder_ann_ids_in_pairs:
                    has_matched_occluder = True
                    has_conf_occluder    = True   # if in pairs, conf was ok

        if not has_geometric_occluder:
            reasons["no_geometric_occluder"] += 1
        elif not has_matched_occluder:
            reasons["occluder_query_not_matched_or_low_conf"] += 1
        else:
            reasons["unknown"] += 1

    return dict(reasons)
# ============================================================
# BUILD FOUR-WAY INDEX FROM PAIRS JSON
# Single source of truth — no mixing with separate geometry check.
# ============================================================

def build_pair_indices(pairs):
    """
    Returns
    -------
    suppressed_ann_ids     : set
        ann_ids where suppression condition is MET in at least one pair
        AND occluder is increasing in that pair.
    not_suppressed_ann_ids : set
        ann_ids that appear in pairs but suppression condition never met.
    all_in_pairs_ann_ids   : set
        every ann_id that appears as occluded_gt_ann_id.
    pair_by_ann_id         : dict  ann_id -> representative pair record
        Uses the pair with the LOWEST final occluded confidence
        (worst suppression case) as the representative.
    all_pairs_for_ann_id   : dict  ann_id -> list of all pairs
    """
    suppressed_ann_ids     = set()
    not_suppressed_ann_ids = set()
    all_in_pairs_ann_ids   = set()
    pair_by_ann_id         = {}
    all_pairs_for_ann_id   = defaultdict(list)

    for p in pairs:
        aid         = p["occluded_gt_ann_id"]
        occ_scores  = p.get("occluded_scores_per_layer", [])
        occr_scores = p.get("occluder_scores_per_layer", [])

        all_in_pairs_ann_ids.add(aid)
        all_pairs_for_ann_id[aid].append(p)

        # representative = pair with lowest final occluded confidence
        if aid not in pair_by_ann_id:
            pair_by_ann_id[aid] = p
        else:
            existing_end = pair_by_ann_id[aid].get(
                "occluded_scores_per_layer", [1.0])[-1]
            new_end = occ_scores[-1] if occ_scores else 1.0
            if new_end < existing_end:
                pair_by_ann_id[aid] = p

        # suppression: occluded degraded AND occluder improved
        if occ_scores and occr_scores:
            if occluded_not_maintained(occ_scores) and occluder_increasing(occr_scores):
                suppressed_ann_ids.add(aid)
            else:
                if aid not in suppressed_ann_ids:
                    not_suppressed_ann_ids.add(aid)

    # clean not_suppressed: any ann_id suppressed in ANY pair is Group A
    not_suppressed_ann_ids -= suppressed_ann_ids

    return (
        suppressed_ann_ids,
        not_suppressed_ann_ids,
        all_in_pairs_ann_ids,
        pair_by_ann_id,
        dict(all_pairs_for_ann_id),
    )


# ============================================================
# GEOMETRIC OCCLUSION CHECK  (Group B2 only)
# ============================================================

def find_gt_occluded_ann_ids(gt_anns_per_image, occ_ratio_threshold):
    gt_occluded_ids = set()
    for image_id, anns in gt_anns_per_image.items():
        boxes = {a["id"]: xywh_to_xyxy(a["bbox"]) for a in anns}
        ids   = list(boxes.keys())
        for i, id_i in enumerate(ids):
            for j, id_j in enumerate(ids):
                if i == j:
                    continue
                if compute_occlusion_ratio(boxes[id_i], boxes[id_j]) >= occ_ratio_threshold:
                    gt_occluded_ids.add(id_i)
    return gt_occluded_ids


# ============================================================
# TRAJECTORY STATISTICS
# ============================================================

def traj_stats(trajectories):
    if not trajectories:
        return {}
    arr          = np.array(trajectories)
    deltas       = arr[:, -1] - arr[:, 0]
    peak_layers  = arr.argmax(axis=1)
    layer_deltas = np.diff(arr, axis=1).mean(axis=0).tolist()
    return {
        "n":                       len(trajectories),
        "mean_trajectory":         [round(v, 4) for v in arr.mean(axis=0).tolist()],
        "std_trajectory":          [round(v, 4) for v in arr.std(axis=0).tolist()],
        "mean_layer_deltas":       [round(v, 5) for v in layer_deltas],
        "mean_net_delta":          round(float(deltas.mean()), 4),
        "pct_net_decrease":        round(float((deltas < 0).mean() * 100), 2),
        "peak_layer_distribution": {
            str(l): int((peak_layers == l).sum()) for l in range(6)
        },
        "pct_peak_at_L0":          round(float((peak_layers == 0).mean() * 100), 2),
        "pct_peak_at_L5":          round(float((peak_layers == 5).mean() * 100), 2),
    }


# ============================================================
# SAMPLING OVERLAP HELPERS
# ============================================================

def mean_if_present(pairs_list, key):
    vals = [p[key] for p in pairs_list if p.get(key) is not None]
    return round(float(np.mean(vals)), 6) if vals else None


def layer_mean(pairs_list, key):
    arrays = [p[key] for p in pairs_list if p.get(key) is not None]
    if not arrays:
        return None
    return [round(v, 6) for v in np.mean(arrays, axis=0).tolist()]


def overlap_stats(pairs_list):
    return {
        "mean_unweighted":         mean_if_present(pairs_list, "mean_sampling_overlap"),
        "mean_attn_weighted":      mean_if_present(pairs_list, "mean_attn_weighted_overlap"),
        "per_layer_unweighted":    layer_mean(pairs_list, "sampling_overlap_per_layer"),
        "per_layer_attn_weighted": layer_mean(pairs_list, "attn_weighted_overlap_per_layer"),
    }


# ============================================================
# FN RATE HELPER
# ============================================================

def fn_rate(group_ann_ids, fn_ann_ids):
    total = len(group_ann_ids)
    in_fn = len(set(group_ann_ids) & fn_ann_ids)
    rate  = in_fn / max(total, 1)
    return total, in_fn, rate


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fn_json",         required=True)
    parser.add_argument("--gt_json",         required=True)
    parser.add_argument("--pairs_json",      required=True)
    parser.add_argument("--occlusion_ratio", type=float, default=0.3,
                        help="Threshold for geometric occlusion (Group B2 only)")
    parser.add_argument("--output_json",     default="fn_group_analysis.json")
    args = parser.parse_args()

    # ── load ─────────────────────────────────────────────
    print("[INFO] Loading files...")
    fn_images,  fn_anns_by_id,  fn_anns_per_image,  _ = load_coco(args.fn_json)
    gt_images,  gt_anns_by_id,  gt_anns_per_image,  _ = load_coco(args.gt_json)
    pairs = load_pairs(args.pairs_json)

    print(f"  FN annotations           : {len(fn_anns_by_id)}")
    print(f"  GT annotations (full)    : {len(gt_anns_by_id)}")
    print(f"  Occlusion pairs loaded   : {len(pairs)}")

    # ── build indices (pairs JSON = single source of truth) ──
    (
        suppressed_ann_ids,
        not_suppressed_ann_ids,
        all_in_pairs_ann_ids,
        pair_by_ann_id,
        all_pairs_for_ann_id,
    ) = build_pair_indices(pairs)

    print(f"\n  From pairs JSON (single source of truth):")
    print(f"    Unique occluded ann_ids      : {len(all_in_pairs_ann_ids)}")
    print(f"    Suppression condition MET    : {len(suppressed_ann_ids)}")
    print(f"    Suppression condition NOT met: {len(not_suppressed_ann_ids)}")
    overlap_check = len(suppressed_ann_ids & not_suppressed_ann_ids)
    print(f"    Overlap A∩B (must be 0)      : {overlap_check}")

    # ── Group B2: geometrically occluded, not in pairs at all ──
    print(f"\n[INFO] Finding geometrically occluded GT objects (for B2)...")
    gt_occluded_ids = find_gt_occluded_ann_ids(
        gt_anns_per_image, args.occlusion_ratio
    )
    b2_gt_ids = gt_occluded_ids - all_in_pairs_ann_ids
    print(f"  GT-occluded (ratio>={args.occlusion_ratio}) : {len(gt_occluded_ids)}")
    print(f"  Of which in pairs list         : {len(gt_occluded_ids & all_in_pairs_ann_ids)}")
    print(f"  Group B2 pool (not in pairs)   : {len(b2_gt_ids)}")

    fn_ann_ids  = set(fn_anns_by_id.keys())
    group_C_gt  = set(gt_anns_by_id.keys()) - gt_occluded_ids - all_in_pairs_ann_ids

    # ── FN rates ─────────────────────────────────────────
    A_total,  A_fn,  A_rate  = fn_rate(suppressed_ann_ids,     fn_ann_ids)
    B_total,  B_fn,  B_rate  = fn_rate(not_suppressed_ann_ids, fn_ann_ids)
    B2_total, B2_fn, B2_rate = fn_rate(b2_gt_ids,              fn_ann_ids)
    C_total,  C_fn,  C_rate  = fn_rate(group_C_gt,             fn_ann_ids)

    total_gt  = len(gt_anns_by_id)
    total_fn  = len(fn_ann_ids)
    base_rate = total_fn / max(total_gt, 1)

    rr_A_vs_C    = A_rate / max(C_rate,    1e-9)
    rr_A_vs_B    = A_rate / max(B_rate,    1e-9)
    rr_B_vs_C    = B_rate / max(C_rate,    1e-9)
    rr_A_vs_base = A_rate / max(base_rate, 1e-9)

    # ── partition FN objects ──────────────────────────────
    fn_A, fn_B, fn_B2, fn_C, fn_unk = [], [], [], [], []
    for ann_id in fn_ann_ids:
        if   ann_id in suppressed_ann_ids:     fn_A.append(ann_id)
        elif ann_id in not_suppressed_ann_ids: fn_B.append(ann_id)
        elif ann_id in b2_gt_ids:              fn_B2.append(ann_id)
        elif ann_id in group_C_gt:             fn_C.append(ann_id)
        else:                                  fn_unk.append(ann_id)

    print("Diagnosis of B2 :", diagnose_b2_objects(fn_B2, gt_anns_by_id, gt_anns_per_image, pairs, occder_score_low=0.75,
                        occ_ratio_threshold=0.3))
    # ── trajectory stats ─────────────────────────────────
    def get_trajs(ann_id_list, key):
        result = []
        for aid in ann_id_list:
            p = pair_by_ann_id.get(aid)
            if p and p.get(key) is not None:
                result.append(p[key])
        return result

    stats_A    = traj_stats(get_trajs(fn_A, "occluded_scores_per_layer"))
    stats_B    = traj_stats(get_trajs(fn_B, "occluded_scores_per_layer"))

    all_supp_trajs = [
        p["occluded_scores_per_layer"] for p in pairs
        if p.get("occluded_scores_per_layer")
        and p.get("occluder_scores_per_layer")
        and occluded_not_maintained(p["occluded_scores_per_layer"])
        and occluder_increasing(p["occluder_scores_per_layer"])
    ]
    stats_all_supp = traj_stats(all_supp_trajs)

    all_occr_trajs = [
        p["occluder_scores_per_layer"] for p in pairs
        if p.get("occluder_scores_per_layer")
    ]
    stats_occluder = traj_stats(all_occr_trajs)

    # ── sampling overlap ─────────────────────────────────
    pairs_A = [pair_by_ann_id[aid] for aid in fn_A  if aid in pair_by_ann_id]
    pairs_B = [pair_by_ann_id[aid] for aid in fn_B  if aid in pair_by_ann_id]
    ov_A    = overlap_stats(pairs_A)
    ov_B    = overlap_stats(pairs_B)
    ov_all  = overlap_stats(list(pair_by_ann_id.values()))

    # ── verdict ───────────────────────────────────────────
    if rr_A_vs_B > 1.2:
        verdict = "SUPPORTS: suppression adds FN risk beyond occlusion alone"
    elif rr_A_vs_B > 1.0:
        verdict = "WEAK: marginal difference, occlusion likely primary driver"
    else:
        verdict = "DOES NOT SUPPORT: suppression not adding FN risk beyond occlusion"

    # ── print report ─────────────────────────────────────
    sep  = "=" * 64
    dash = "-" * 64

    print(f"\n{sep}")
    print(f"  FALSE NEGATIVE SUPPRESSION ANALYSIS  (v2)")
    print(sep)

    print(f"\n  Total GT objects               : {total_gt:>6}")
    print(f"  Total FN objects               : {total_fn:>6}")
    print(f"  Base FN rate (all GT)          : {base_rate*100:>6.2f}%")

    print(f"\n{dash}")
    print(f"  GROUP SIZES (GT denominators)")
    print(f"{dash}")
    print(f"  A  — suppressed (pairs, cond MET)    : {A_total:>6}")
    print(f"  B  — in pairs, cond NOT met          : {B_total:>6}")
    print(f"  B2 — geom. occluded, not in pairs    : {B2_total:>6}")
    print(f"  C  — not occluded (clean baseline)   : {C_total:>6}")
    accounted = A_total + B_total + B2_total + C_total
    print(f"  Total accounted                      : {accounted:>6}  "
          f"(of {total_gt}, diff={total_gt-accounted})")

    print(f"\n{dash}")
    print(f"  FN RATES BY GROUP")
    print(f"{dash}")
    print(f"  Group A  (suppressed)        : {A_fn:>4} / {A_total:<6} = {A_rate*100:>6.2f}%")
    print(f"  Group B  (not suppressed)    : {B_fn:>4} / {B_total:<6} = {B_rate*100:>6.2f}%")
    print(f"  Group B2 (occ, not in pairs) : {B2_fn:>4} / {B2_total:<6} = {B2_rate*100:>6.2f}%")
    print(f"  Group C  (non-occluded)      : {C_fn:>4} / {C_total:<6} = {C_rate*100:>6.2f}%")
    print(f"  Base rate (all GT)           :        {total_fn:<6} = {base_rate*100:>6.2f}%")

    print(f"\n{dash}")
    print(f"  RELATIVE RISK")
    print(f"{dash}")
    print(f"  A vs B  (key: supp. vs not-supp.)    : {rr_A_vs_B:>6.2f}x")
    print(f"  A vs C  (supp. vs clean baseline)    : {rr_A_vs_C:>6.2f}x")
    print(f"  B vs C  (in-pairs vs clean)          : {rr_B_vs_C:>6.2f}x")
    print(f"  A vs base                            : {rr_A_vs_base:>6.2f}x")
    print(f"\n  Verdict: {verdict}")

    print(f"\n{dash}")
    print(f"  FN PARTITION  (of {total_fn} total FN objects)")
    print(f"{dash}")
    for label, grp in [
        ("A  suppressed occluded     ", fn_A),
        ("B  in-pairs not suppressed ", fn_B),
        ("B2 occ not in pairs        ", fn_B2),
        ("C  non-occluded baseline   ", fn_C),
        ("?  uncategorised           ", fn_unk),
    ]:
        pct = len(grp) / max(total_fn, 1) * 100
        print(f"  Group {label}: {len(grp):>5}  ({pct:.1f}%)")

    print(f"\n{dash}")
    print(f"  CONFIDENCE TRAJECTORY STATS")
    print(f"{dash}")
    for label, stats in [
        ("Group A  suppressed + FN        ", stats_A),
        ("Group B  not-suppressed + FN    ", stats_B),
        ("All suppressed (incl. TP)       ", stats_all_supp),
        ("Occluder contrast               ", stats_occluder),
    ]:
        if not stats:
            print(f"\n  [{label.strip()}]  -- no data")
            continue
        print(f"\n  [{label.strip()}]  n={stats['n']}")
        print(f"    Mean trajectory  : {stats['mean_trajectory']}")
        print(f"    Mean layer Dconf : {stats['mean_layer_deltas']}")
        print(f"    Mean net delta   : {stats['mean_net_delta']:+.4f}")
        print(f"    % net decrease   : {stats['pct_net_decrease']:.1f}%")
        print(f"    % peak at L0     : {stats['pct_peak_at_L0']:.1f}%")
        print(f"    % peak at L5     : {stats['pct_peak_at_L5']:.1f}%")
        print(f"    Peak layer dist  : {stats['peak_layer_distribution']}")

    print(f"\n{dash}")
    print(f"  SAMPLING OVERLAP")
    print(f"{dash}")
    for label, ov in [
        ("Group A  suppressed + FN    ", ov_A),
        ("Group B  not-suppressed + FN", ov_B),
        ("All pairs                   ", ov_all),
    ]:
        if ov.get("mean_unweighted") is None:
            continue
        print(f"\n  [{label.strip()}]")
        print(f"    Mean unweighted       : {ov['mean_unweighted']:.4f}")
        print(f"    Mean attn-weighted    : {ov['mean_attn_weighted']:.6f}")
        if ov["per_layer_unweighted"]:
            row = "  ".join(
                f"L{i}:{v:.4f}" for i, v in enumerate(ov["per_layer_unweighted"])
            )
            print(f"    Per-layer unweighted  : {row}")
        if ov["per_layer_attn_weighted"]:
            row = "  ".join(
                f"L{i}:{v:.6f}" for i, v in enumerate(ov["per_layer_attn_weighted"])
            )
            print(f"    Per-layer attn-wtd    : {row}")

    print(f"\n{sep}\n")

    # ── save JSON ─────────────────────────────────────────
    output = {
        "config": {
            "occlusion_ratio_threshold": args.occlusion_ratio,
            "suppression_condition": (
                "occluded: peak_before_end OR end < start  "
                "AND occluder: end > start"
            ),
            "group_B_definition": (
                "In pairs JSON but suppression condition never met. "
                "Same occluder confidence range + class mode as Group A. "
                "Strongest control group."
            ),
        },
        "summary": {
            "total_gt":    total_gt,
            "total_fn":    total_fn,
            "base_fn_rate": round(base_rate, 4),
            "group_A":  {"total": A_total,  "fn": A_fn,  "fn_rate": round(A_rate,  4)},
            "group_B":  {"total": B_total,  "fn": B_fn,  "fn_rate": round(B_rate,  4)},
            "group_B2": {"total": B2_total, "fn": B2_fn, "fn_rate": round(B2_rate, 4)},
            "group_C":  {"total": C_total,  "fn": C_fn,  "fn_rate": round(C_rate,  4)},
            "relative_risk": {
                "A_vs_B":    round(rr_A_vs_B,    3),
                "A_vs_C":    round(rr_A_vs_C,    3),
                "B_vs_C":    round(rr_B_vs_C,    3),
                "A_vs_base": round(rr_A_vs_base, 3),
            },
            "verdict": verdict,
        },
        "fn_partition": {
            "group_A_ann_ids":  sorted(fn_A),
            "group_B_ann_ids":  sorted(fn_B),
            "group_B2_ann_ids": sorted(fn_B2),
            "group_C_ann_ids":  sorted(fn_C),
        },
        "trajectory_stats": {
            "group_A":           stats_A,
            "group_B":           stats_B,
            "all_suppressed":    stats_all_supp,
            "occluder_contrast": stats_occluder,
        },
        "sampling_overlap": {
            "group_A":   ov_A,
            "group_B":   ov_B,
            "all_pairs": ov_all,
        },
    }

    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[SAVED] {args.output_json}")


if __name__ == "__main__":
    main()