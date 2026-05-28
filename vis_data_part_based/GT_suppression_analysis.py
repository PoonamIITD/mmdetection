"""
gt_suppression_analysis.py
===========================
Analyses occlusion suppression patterns across ALL GT objects
(not just false negatives). Partitions every GT object into four
groups using the pairs JSON as the single source of truth.

  Group A  — In pairs + suppression condition MET
             → occluded query was suppressed
             → split further into A_tp (model kept it) and A_fn (model missed it)
               IF an optional --fn_json is provided

  Group B  — In pairs + suppression condition NOT MET
             → occluded, same occluder confidence range, but decoder
               did NOT suppress confidence
             → control group: same pipeline, different outcome

  Group B2 — Geometrically occluded (ratio >= threshold) but never
             captured in pairs list
             → occluded but outside analysis scope

  Group C  — Not occluded at all
             → clean baseline

Key comparisons
---------------
  A vs B       : does suppression condition change trajectory shape?
  A_tp vs A_fn : what trajectory feature separates recovery from failure?
                 (only available when --fn_json is supplied)
  B vs C       : does being in a pair at all affect trajectory?

Usage
-----
python gt_suppression_analysis.py \
    --gt_json        val_annotations.json \
    --pairs_json     occlusion_results.json \
    [--fn_json       fn_detections.json]   \
    [--occlusion_ratio  0.3] \
    [--output_json   gt_group_analysis.json]
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
# SUPPRESSION CONDITIONS  (identical to original script)
# ============================================================

def occluded_not_maintained(scores, eps=1e-6):
    """
    Suppressed if peak confidence occurs before the final layer
    OR final confidence is lower than starting confidence.
    Captures: rise-then-fall, continuous fall, oscillating decay.
    """
    end  = scores[-1]
    peak = max(scores)
    return (peak > end + eps) or (end < scores[0] - eps)


def occluder_increasing(scores, eps=1e-6):
    return scores[-1] > scores[0] + eps


# ============================================================
# BUILD FOUR-WAY INDEX FROM PAIRS JSON
# ============================================================

def build_pair_indices(pairs):
    """
    Returns
    -------
    suppressed_ann_ids     : set  — suppression condition MET in ≥1 pair
    not_suppressed_ann_ids : set  — in pairs but condition never met
    all_in_pairs_ann_ids   : set  — every ann_id appearing as occluded
    pair_by_ann_id         : dict — representative pair per ann_id
                                    (lowest final occluded confidence)
    all_pairs_for_ann_id   : dict — all pairs per ann_id
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

        if occ_scores and occr_scores:
            if occluded_not_maintained(occ_scores) and occluder_increasing(occr_scores):
                suppressed_ann_ids.add(aid)
            else:
                if aid not in suppressed_ann_ids:
                    not_suppressed_ann_ids.add(aid)

    # any ann_id suppressed in ANY pair goes to Group A
    not_suppressed_ann_ids -= suppressed_ann_ids

    return (
        suppressed_ann_ids,
        not_suppressed_ann_ids,
        all_in_pairs_ann_ids,
        pair_by_ann_id,
        dict(all_pairs_for_ann_id),
    )


# ============================================================
# GEOMETRIC OCCLUSION CHECK  (Group B2)
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

def traj_stats(trajectories, label=""):
    if not trajectories:
        return {}
    arr          = np.array(trajectories)
    deltas       = arr[:, -1] - arr[:, 0]
    peak_layers  = arr.argmax(axis=1)
    layer_deltas = np.diff(arr, axis=1).mean(axis=0).tolist()

    # per-layer confidence mean and std
    layer_means = arr.mean(axis=0).tolist()
    layer_stds  = arr.std(axis=0).tolist()

    return {
        "n":                       len(trajectories),
        "mean_trajectory":         [round(v, 4) for v in layer_means],
        "std_trajectory":          [round(v, 4) for v in layer_stds],
        "mean_layer_deltas":       [round(v, 5) for v in layer_deltas],
        "mean_net_delta":          round(float(deltas.mean()), 4),
        "std_net_delta":           round(float(deltas.std()),  4),
        "pct_net_decrease":        round(float((deltas < 0).mean() * 100), 2),
        "pct_net_increase":        round(float((deltas > 0).mean() * 100), 2),
        "peak_layer_distribution": {
            str(l): int((peak_layers == l).sum()) for l in range(arr.shape[1])
        },
        "pct_peak_at_L0":          round(float((peak_layers == 0).mean() * 100), 2),
        "pct_peak_at_last":        round(float((peak_layers == arr.shape[1]-1).mean() * 100), 2),
        # recovery metric: how often does conf end above the midpoint of its range?
        "pct_recovered":           round(
            float(((arr[:, -1] - arr.min(axis=1)) /
                   (arr.max(axis=1) - arr.min(axis=1) + 1e-6) > 0.5).mean() * 100), 2
        ),
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
# PRINT HELPERS
# ============================================================

def print_traj(label, stats):
    if not stats:
        print(f"\n  [{label}]  -- no data")
        return
    print(f"\n  [{label}]  n={stats['n']}")
    print(f"    Mean trajectory    : {stats['mean_trajectory']}")
    print(f"    Std  trajectory    : {stats['std_trajectory']}")
    print(f"    Mean layer Δconf   : {stats['mean_layer_deltas']}")
    print(f"    Mean net delta     : {stats['mean_net_delta']:+.4f}  "
          f"(std={stats['std_net_delta']:.4f})")
    print(f"    % net decrease     : {stats['pct_net_decrease']:.1f}%")
    print(f"    % net increase     : {stats['pct_net_increase']:.1f}%")
    print(f"    % recovered        : {stats['pct_recovered']:.1f}%")
    print(f"    % peak at L0       : {stats['pct_peak_at_L0']:.1f}%")
    print(f"    % peak at last     : {stats['pct_peak_at_last']:.1f}%")
    print(f"    Peak layer dist    : {stats['peak_layer_distribution']}")


def print_overlap(label, ov):
    if ov.get("mean_unweighted") is None:
        return
    print(f"\n  [{label}]")
    print(f"    Mean unweighted       : {ov['mean_unweighted']:.4f}")
    if ov.get("mean_attn_weighted") is not None:
        print(f"    Mean attn-weighted    : {ov['mean_attn_weighted']:.6f}")
    if ov["per_layer_unweighted"]:
        row = "  ".join(
            f"L{i}:{v:.4f}" for i, v in enumerate(ov["per_layer_unweighted"])
        )
        print(f"    Per-layer unweighted  : {row}")
    if ov.get("per_layer_attn_weighted"):
        row = "  ".join(
            f"L{i}:{v:.6f}" for i, v in enumerate(ov["per_layer_attn_weighted"])
        )
        print(f"    Per-layer attn-wtd    : {row}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_json",         required=True,
                        help="Full COCO ground truth JSON")
    parser.add_argument("--pairs_json",      required=True,
                        help="Occlusion pairs JSON from script 2")
    parser.add_argument("--fn_json",         default=None,
                        help="Optional: COCO FN JSON. When supplied, "
                             "Group A is split into A_tp and A_fn so you "
                             "can compare recovered vs failed trajectories.")
    parser.add_argument("--occlusion_ratio", type=float, default=0.3,
                        help="Geometric occlusion threshold for Group B2")
    parser.add_argument("--output_json",     default="gt_group_analysis.json")
    args = parser.parse_args()

    # ── load ─────────────────────────────────────────────
    print("[INFO] Loading files...")
    gt_images, gt_anns_by_id, gt_anns_per_image, _ = load_coco(args.gt_json)
    pairs = load_pairs(args.pairs_json)

    print(f"  GT annotations           : {len(gt_anns_by_id)}")
    print(f"  Occlusion pairs loaded   : {len(pairs)}")

    # optional FN ids
    fn_ann_ids = set()
    if args.fn_json:
        _, fn_anns_by_id, _, _ = load_coco(args.fn_json)
        fn_ann_ids = set(fn_anns_by_id.keys())
        print(f"  FN annotations (optional): {len(fn_ann_ids)}")

    # ── build group indices ───────────────────────────────
    (
        suppressed_ann_ids,
        not_suppressed_ann_ids,
        all_in_pairs_ann_ids,
        pair_by_ann_id,
        all_pairs_for_ann_id,
    ) = build_pair_indices(pairs)

    print(f"\n  From pairs JSON:")
    print(f"    Unique occluded ann_ids      : {len(all_in_pairs_ann_ids)}")
    print(f"    Suppression condition MET    : {len(suppressed_ann_ids)}")
    print(f"    Suppression condition NOT met: {len(not_suppressed_ann_ids)}")
    print(f"    Overlap A∩B (must be 0)      : "
          f"{len(suppressed_ann_ids & not_suppressed_ann_ids)}")

    # ── Group B2 ─────────────────────────────────────────
    print(f"\n[INFO] Finding geometrically occluded GT objects (for B2)...")
    gt_occluded_ids = find_gt_occluded_ann_ids(
        gt_anns_per_image, args.occlusion_ratio
    )
    b2_gt_ids  = gt_occluded_ids - all_in_pairs_ann_ids
    group_C_gt = set(gt_anns_by_id.keys()) - gt_occluded_ids - all_in_pairs_ann_ids

    print(f"  GT-occluded (ratio>={args.occlusion_ratio}) : {len(gt_occluded_ids)}")
    print(f"  Of which in pairs list         : {len(gt_occluded_ids & all_in_pairs_ann_ids)}")
    print(f"  Group B2 pool                  : {len(b2_gt_ids)}")
    print(f"  Group C  pool                  : {len(group_C_gt)}")

    # ── partition ALL GT objects into groups ──────────────
    grp = {}   # ann_id -> group label
    for aid in gt_anns_by_id:
        if   aid in suppressed_ann_ids:     grp[aid] = "A"
        elif aid in not_suppressed_ann_ids: grp[aid] = "B"
        elif aid in b2_gt_ids:              grp[aid] = "B2"
        elif aid in group_C_gt:             grp[aid] = "C"
        else:                               grp[aid] = "?"

    # Group A sub-split (only meaningful if fn_json provided)
    A_tp_ids = suppressed_ann_ids - fn_ann_ids   # suppressed but model kept it
    A_fn_ids = suppressed_ann_ids & fn_ann_ids   # suppressed and model missed it

    # ── trajectory extraction helper ─────────────────────
    def get_trajs(ann_id_iter, key="occluded_scores_per_layer"):
        result = []
        for aid in ann_id_iter:
            p = pair_by_ann_id.get(aid)
            if p and p.get(key) is not None:
                result.append(p[key])
        return result

    # ── compute trajectory stats per group ───────────────
    stats = {
        "A":    traj_stats(get_trajs(suppressed_ann_ids)),
        "A_tp": traj_stats(get_trajs(A_tp_ids)) if fn_ann_ids else {},
        "A_fn": traj_stats(get_trajs(A_fn_ids)) if fn_ann_ids else {},
        "B":    traj_stats(get_trajs(not_suppressed_ann_ids)),
        "occluder_A":  traj_stats(get_trajs(suppressed_ann_ids,
                                             "occluder_scores_per_layer")),
        "occluder_B":  traj_stats(get_trajs(not_suppressed_ann_ids,
                                             "occluder_scores_per_layer")),
        "occluder_all": traj_stats([
            p["occluder_scores_per_layer"] for p in pairs
            if p.get("occluder_scores_per_layer")
        ]),
    }

    # ── sampling overlap per group ────────────────────────
    pairs_A    = [pair_by_ann_id[a] for a in suppressed_ann_ids     if a in pair_by_ann_id]
    pairs_A_tp = [pair_by_ann_id[a] for a in A_tp_ids               if a in pair_by_ann_id]
    pairs_A_fn = [pair_by_ann_id[a] for a in A_fn_ids               if a in pair_by_ann_id]
    pairs_B    = [pair_by_ann_id[a] for a in not_suppressed_ann_ids if a in pair_by_ann_id]
    ov = {
        "A":    overlap_stats(pairs_A),
        "A_tp": overlap_stats(pairs_A_tp) if fn_ann_ids else {},
        "A_fn": overlap_stats(pairs_A_fn) if fn_ann_ids else {},
        "B":    overlap_stats(pairs_B),
        "all":  overlap_stats(list(pair_by_ann_id.values())),
    }

    
    # ── group size summary ────────────────────────────────
    total_gt = len(gt_anns_by_id)
    sep  = "=" * 64
    dash = "-" * 64

    print(f"\n{sep}")
    print(f"  GT-WIDE SUPPRESSION ANALYSIS  (all GT objects)")
    print(sep)

    print(f"\n  Total GT objects               : {total_gt:>6}")

    print(f"\n{dash}")
    print(f"  GROUP SIZES  (GT denominators)")
    print(f"{dash}")
    print(f"  A   suppressed (pairs, cond MET)     : {len(suppressed_ann_ids):>6}")
    if fn_ann_ids:
        print(f"    A_tp  suppressed + TP (recovered)  : {len(A_tp_ids):>6}")
        print(f"    A_fn  suppressed + FN (missed)     : {len(A_fn_ids):>6}")
    print(f"  B   in pairs, cond NOT met           : {len(not_suppressed_ann_ids):>6}")
    print(f"  B2  geom. occluded, not in pairs     : {len(b2_gt_ids):>6}")
    print(f"  C   not occluded (clean baseline)    : {len(group_C_gt):>6}")
    accounted = (len(suppressed_ann_ids) + len(not_suppressed_ann_ids)
                 + len(b2_gt_ids) + len(group_C_gt))
    print(f"  Unaccounted (?)                      : {total_gt - accounted:>6}")

    # ── trajectory stats ─────────────────────────────────
    print(f"\n{dash}")
    print(f"  CONFIDENCE TRAJECTORY STATS  (occluded query)")
    print(f"{dash}")
    print_traj("Group A  — all suppressed",           stats["A"])
    if fn_ann_ids:
        print_traj("Group A_tp — suppressed + TP (recovered)", stats["A_tp"])
        print_traj("Group A_fn — suppressed + FN (missed)",    stats["A_fn"])
    print_traj("Group B  — in pairs, not suppressed", stats["B"])

    print(f"\n{dash}")
    print(f"  CONFIDENCE TRAJECTORY STATS  (occluder query)")
    print(f"{dash}")
    print_traj("Occluder in Group A pairs", stats["occluder_A"])
    print_traj("Occluder in Group B pairs", stats["occluder_B"])
    print_traj("Occluder all pairs",        stats["occluder_all"])

    # ── key contrasts ─────────────────────────────────────
    print(f"\n{dash}")
    print(f"  KEY CONTRASTS")
    print(f"{dash}")

    def safe_delta(s1, s2, key):
        if s1 and s2 and key in s1 and key in s2:
            return round(s1[key] - s2[key], 4)
        return None

    d_AB = safe_delta(stats["A"], stats["B"], "mean_net_delta")
    if d_AB is not None:
        print(f"  A mean_net_delta − B mean_net_delta : {d_AB:+.4f}  "
              f"({'A drops more' if d_AB < 0 else 'B drops more or A recovers'})")

    if fn_ann_ids and stats["A_tp"] and stats["A_fn"]:
        d_tp_fn = safe_delta(stats["A_tp"], stats["A_fn"], "mean_net_delta")
        if d_tp_fn is not None:
            print(f"  A_tp net_delta − A_fn net_delta     : {d_tp_fn:+.4f}  "
                  f"({'TP recovers more' if d_tp_fn > 0 else 'FN has stronger drop'})")

        # compare recovery rates
        tp_rec = stats["A_tp"].get("pct_recovered", 0)
        fn_rec = stats["A_fn"].get("pct_recovered", 0)
        print(f"  % recovered — A_tp: {tp_rec:.1f}%   A_fn: {fn_rec:.1f}%  "
              f"(gap={tp_rec-fn_rec:+.1f}pp)")

        # peak layer comparison
        print(f"  Peak at L0  — A_tp: {stats['A_tp'].get('pct_peak_at_L0',0):.1f}%"
              f"   A_fn: {stats['A_fn'].get('pct_peak_at_L0',0):.1f}%")

    # ── sampling overlap ─────────────────────────────────
    print(f"\n{dash}")
    print(f"  SAMPLING OVERLAP")
    print(f"{dash}")
    print_overlap("Group A  — all suppressed",           ov["A"])
    if fn_ann_ids:
        print_overlap("Group A_tp — suppressed + TP",        ov["A_tp"])
        print_overlap("Group A_fn — suppressed + FN",        ov["A_fn"])
    print_overlap("Group B  — not suppressed",           ov["B"])
    print_overlap("All pairs",                           ov["all"])

    print(f"\n{sep}\n")

    # ── save JSON ─────────────────────────────────────────
    output = {
        "config": {
            "occlusion_ratio_threshold": args.occlusion_ratio,
            "suppression_condition": (
                "occluded: peak_before_end OR end < start  "
                "AND occluder: end > start"
            ),
            "fn_json_provided": args.fn_json is not None,
            "note": (
                "All counts and trajectory stats are over the FULL GT pool, "
                "not just false negatives. A_tp/A_fn split only present when "
                "--fn_json is supplied."
            ),
        },
        "summary": {
            "total_gt":   total_gt,
            "group_A":    len(suppressed_ann_ids),
            "group_A_tp": len(A_tp_ids) if fn_ann_ids else None,
            "group_A_fn": len(A_fn_ids) if fn_ann_ids else None,
            "group_B":    len(not_suppressed_ann_ids),
            "group_B2":   len(b2_gt_ids),
            "group_C":    len(group_C_gt),
        },
        "group_ann_ids": {
            "A":    sorted(suppressed_ann_ids),
            "A_tp": sorted(A_tp_ids) if fn_ann_ids else [],
            "A_fn": sorted(A_fn_ids) if fn_ann_ids else [],
            "B":    sorted(not_suppressed_ann_ids),
            "B2":   sorted(b2_gt_ids),
            "C":    sorted(group_C_gt),
        },
        "trajectory_stats": {
            "occluded": {
                "A":    stats["A"],
                "A_tp": stats["A_tp"],
                "A_fn": stats["A_fn"],
                "B":    stats["B"],
            },
            "occluder": {
                "A_pairs":   stats["occluder_A"],
                "B_pairs":   stats["occluder_B"],
                "all_pairs": stats["occluder_all"],
            },
        },
        "sampling_overlap": {
            "A":    ov["A"],
            "A_tp": ov["A_tp"],
            "A_fn": ov["A_fn"],
            "B":    ov["B"],
            "all":  ov["all"],
        },
    }

    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[SAVED] {args.output_json}")


if __name__ == "__main__":
    main()