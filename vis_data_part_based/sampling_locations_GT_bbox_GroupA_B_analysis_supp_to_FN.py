"""
Sampling Location BBox Analysis  (v2 — with occlusion pair group splitting)
============================================================================
For each GT annotation, find the best-matched query (highest cls confidence),
then compute:
  - fraction of sampling locations inside its own GT bbox
  - fraction of sampling locations inside any other GT bbox
  - for occluded queries: fraction specifically inside the PAIRED OCCLUDER bbox
  - how confidence trajectory varies with these fractions

Groups (from occlusion pairs JSON):
  A  — occluded AND suppressed   (suppressed_to_fn == True  OR  occluded_final_score < threshold)
       NOTE: the pairs JSON field "suppressed_to_fn" marks suppression directly.
       If absent we fall back to: occluded_final_score < occluder_score * 0.5
  B  — occluded BUT NOT suppressed  (in pairs, suppressed_to_fn == False)
  C  — not in any pair (clean baseline)

Usage:
    python sampling_location_bbox_analysis.py \
        --debug_dir   /path/to/debug_outputs_complete_val_set \
        --coco_json   /path/to/instances_val.json \
        --pairs_json  /path/to/occlusion_pairs.json \
        --out_dir     ./sampling_bbox_analysis_out \
        [--last_layer_only] \
        [--max_images N]
"""

import os
import json
import argparse
import glob
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def xywh_to_x1y1x2y2_norm(bbox_xywh, img_w, img_h):
    """Convert COCO [x,y,w,h] pixel bbox to normalised [x1,y1,x2,y2]."""
    x, y, w, h = bbox_xywh
    return np.array([x / img_w, y / img_h,
                     (x + w) / img_w, (y + h) / img_h], dtype=np.float32)


def points_in_box(pts_xy, box_x1y1x2y2):
    """
    pts_xy        : (N, 2)  normalised (x, y)
    box_x1y1x2y2  : (4,)    normalised [x1, y1, x2, y2]
    Returns boolean mask (N,).
    """
    x1, y1, x2, y2 = box_x1y1x2y2
    return ((pts_xy[:, 0] >= x1) & (pts_xy[:, 0] <= x2) &
            (pts_xy[:, 1] >= y1) & (pts_xy[:, 1] <= y2))


def flatten_sampling_locs(sampling_locs_query):
    """
    sampling_locs_query : (H, Lv, P, 2)  normalised [0,1]
    Returns pts : (H*Lv*P, 2) clipped to [0,1].
    """
    pts = sampling_locs_query.reshape(-1, 2).float().numpy()
    return np.clip(pts, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Confidence helper
# ─────────────────────────────────────────────────────────────────────────────

def get_conf_trajectory(cls_scores_all_layers, query_idx, class_id):
    """cls_scores_all_layers: (L, Q, C) → (L,) sigmoid confidence."""
    scores = cls_scores_all_layers[:, query_idx, class_id].float()
    return torch.sigmoid(scores).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Load & index occlusion pairs
# ─────────────────────────────────────────────────────────────────────────────

def load_pairs(pairs_json_path):
    """
    Returns:
      suppressed_ann_ids  : set of occluded ann_ids where suppressed_to_fn=True
      not_suppressed_ids  : set of occluded ann_ids where suppressed_to_fn=False
      pair_by_occluded    : dict  occluded_ann_id → pair dict
                            (used to look up the occluder ann_id per occluded query)
    """
    with open(pairs_json_path) as f:
        pairs = json.load(f)

    suppressed_ann_ids   = set()
    not_suppressed_ids   = set()
    pair_by_occluded     = {}   # occluded_gt_ann_id → pair entry

    for p in pairs:
        occ_id = p["occluded_gt_ann_id"]
        pair_by_occluded[occ_id] = p
        if p.get("suppressed_to_fn", False):
            suppressed_ann_ids.add(occ_id)
        else:
            not_suppressed_ids.add(occ_id)

    print(f"[INFO] Pairs loaded: {len(pairs)} total | "
          f"suppressed (A)={len(suppressed_ann_ids)} | "
          f"not-suppressed (B)={len(not_suppressed_ids)}")
    return suppressed_ann_ids, not_suppressed_ids, pair_by_occluded


# ─────────────────────────────────────────────────────────────────────────────
# Per-image analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_image(data, gt_anns, img_w, img_h,
                  suppressed_ids, not_suppressed_ids, pair_by_occluded,
                  ann_to_box_norm,   # full val-set ann_id → norm box (pre-built)
                  last_layer_only=False):
    """
    Returns list of per-GT result dicts, each tagged with group A/B/C
    and (for A/B) the leakage fraction into the paired occluder bbox.
    """
    cls_scores    = data["cls_scores"]          # (L, Q, C)
    sampling_locs = data["sampling_locations"]  # (L, Q, H, Lv, P, 2)
    attn_weights  = data["attention_weights"]   # (L, Q, H, Lv, P)
    spatial_shapes = data["spatial_shapes"]     # (Lv, 2)  — kept for reference

    L, Q, C = cls_scores.shape
    layers   = [L - 1] if last_layer_only else list(range(L))

    last_layer_scores = torch.sigmoid(cls_scores[-1].float())  # (Q, C)

    # Build per-image normalised boxes
    gt_boxes_norm = []
    for ann in gt_anns:
        cat_id  = ann["category_id"] - 1   # 0-indexed
        box_n   = xywh_to_x1y1x2y2_norm(ann["bbox"], img_w, img_h)
        gt_boxes_norm.append((ann, cat_id, box_n))

    results = []

    for ann, cat_id, own_box in gt_boxes_norm:
        ann_id = ann["id"]

        # ── group tag ────────────────────────────────────────────────────────
        if ann_id in suppressed_ids:
            group = "A"
        elif ann_id in not_suppressed_ids:
            group = "B"
        else:
            group = "C"

        # Paired occluder ann_id (None for group C)
        occluder_ann_id = None
        occluder_box    = None
        if group in ("A", "B") and ann_id in pair_by_occluded:
            occluder_ann_id = pair_by_occluded[ann_id]["occluder_gt_ann_id"]
            occluder_box    = ann_to_box_norm.get(occluder_ann_id)

        # ── best query for this GT ────────────────────────────────────────────
        # Use the query index stored in the pairs JSON if available (most accurate),
        # otherwise fall back to highest-confidence query at last layer.
        if group in ("A", "B") and ann_id in pair_by_occluded:
            best_q = int(pair_by_occluded[ann_id]["occluded_query"])
        else:
            cat_scores = last_layer_scores[:, cat_id]
            best_q     = int(cat_scores.argmax().item())

        best_conf = float(last_layer_scores[best_q, cat_id].item())
        conf_traj = get_conf_trajectory(cls_scores, best_q, cat_id)  # (L,)

        # ── per-layer sampling location fractions ─────────────────────────────
        layer_results = {}
        for l_idx in layers:
            sl_q    = sampling_locs[l_idx, best_q]          # (H, Lv, P, 2)
            aw_q    = attn_weights[l_idx, best_q]           # (H, Lv, P)
            pts     = flatten_sampling_locs(sl_q)            # (N, 2)
            aw_flat = aw_q.float().numpy().reshape(-1)       # (N,)
            n_pts   = len(pts)

            # own bbox
            mask_own        = points_in_box(pts, own_box)
            frac_own_unwtd  = float(mask_own.sum()) / n_pts
            frac_own_wtd    = float(aw_flat[mask_own].sum())

            # paired occluder bbox (A/B only)
            frac_occluder_unwtd = np.nan
            frac_occluder_wtd   = np.nan
            if occluder_box is not None:
                mask_occ            = points_in_box(pts, occluder_box)
                frac_occluder_unwtd = float(mask_occ.sum()) / n_pts
                frac_occluder_wtd   = float(aw_flat[mask_occ].sum())

            # any other GT bbox on this image (excluding own)
            mask_other_any = np.zeros(n_pts, dtype=bool)
            for other_ann, _, other_box in gt_boxes_norm:
                if other_ann["id"] == ann_id:
                    continue
                mask_other_any |= points_in_box(pts, other_box)

            frac_other_unwtd = float(mask_other_any.sum()) / n_pts
            frac_other_wtd   = float(aw_flat[mask_other_any].sum())
            frac_bg          = max(0.0, 1.0 - frac_own_unwtd - frac_other_unwtd)

            layer_results[l_idx] = {
                "frac_own_unwtd":        frac_own_unwtd,
                "frac_own_wtd":          frac_own_wtd,
                "frac_other_unwtd":      frac_other_unwtd,
                "frac_other_wtd":        frac_other_wtd,
                "frac_occluder_unwtd":   frac_occluder_unwtd,   # NaN for group C
                "frac_occluder_wtd":     frac_occluder_wtd,
                "frac_bg":               frac_bg,
                "conf":                  float(conf_traj[l_idx]),
            }

        results.append({
            "ann_id":          ann_id,
            "cat_id":          cat_id,
            "group":           group,
            "occluder_ann_id": occluder_ann_id,
            "best_query":      best_q,
            "best_conf_L":     best_conf,
            "conf_traj":       conf_traj.tolist(),
            "layer_results":   layer_results,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────

def collect_per_layer(results_list, layers):
    """Collect per-layer metric arrays from a list of per-GT result dicts."""
    d = {l: defaultdict(list) for l in layers}
    for r in results_list:
        for l in layers:
            lr = r["layer_results"][l]
            d[l]["frac_own_unwtd"].append(lr["frac_own_unwtd"])
            d[l]["frac_own_wtd"].append(lr["frac_own_wtd"])
            d[l]["frac_other_unwtd"].append(lr["frac_other_unwtd"])
            d[l]["frac_other_wtd"].append(lr["frac_other_wtd"])
            d[l]["frac_bg"].append(lr["frac_bg"])
            d[l]["conf"].append(lr["conf"])
            if not np.isnan(lr["frac_occluder_unwtd"]):
                d[l]["frac_occluder_unwtd"].append(lr["frac_occluder_unwtd"])
                d[l]["frac_occluder_wtd"].append(lr["frac_occluder_wtd"])
    return d


def safe_mean(lst):
    return float(np.mean(lst)) if lst else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_layer_distribution(per_layer_data, layers, title, out_path, color_own="steelblue",
                             color_other="darkorange", color_occ="red", color_bg="gray"):
    """Grouped bar chart: own / other / occluder / bg fractions + mean conf."""
    mean_own   = [safe_mean(per_layer_data[l]["frac_own_unwtd"])   for l in layers]
    mean_other = [safe_mean(per_layer_data[l]["frac_other_unwtd"]) for l in layers]
    mean_occ   = [safe_mean(per_layer_data[l].get("frac_occluder_unwtd", []))  for l in layers]
    mean_bg    = [safe_mean(per_layer_data[l]["frac_bg"])          for l in layers]
    mean_conf  = [safe_mean(per_layer_data[l]["conf"])             for l in layers]

    has_occ = any(not np.isnan(v) for v in mean_occ)
    n_bars  = 4 if has_occ else 3
    x = np.arange(len(layers))
    bw = 0.18

    fig, ax = plt.subplots(figsize=(12, 5))
    offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * bw

    ax.bar(x + offsets[0], mean_own,   bw, label="Own bbox",        color=color_own)
    ax.bar(x + offsets[1], mean_other, bw, label="Other bbox (any)", color=color_other)
    if has_occ:
        ax.bar(x + offsets[2], mean_occ, bw, label="Paired occluder", color=color_occ, alpha=0.75)
        ax.bar(x + offsets[3], mean_bg,  bw, label="Background",       color=color_bg)
    else:
        ax.bar(x + offsets[2], mean_bg,  bw, label="Background",       color=color_bg)

    ax2 = ax.twinx()
    ax2.plot(x, mean_conf, "k-o", linewidth=2, label="Mean conf")
    ax2.set_ylabel("Mean confidence")

    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_ylabel("Mean fraction of sampling points")
    ax.set_title(title)
    ax.set_ylim(0, 1.0)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_ab_comparison(per_layer_A, per_layer_B, layers, metric, ylabel, title, out_path):
    """Side-by-side line plot comparing Group A vs B across decoder layers."""
    vals_A = [safe_mean(per_layer_A[l][metric]) for l in layers]
    vals_B = [safe_mean(per_layer_B[l][metric]) for l in layers]

    std_A  = [float(np.std(per_layer_A[l][metric])) if per_layer_A[l][metric] else np.nan for l in layers]
    std_B  = [float(np.std(per_layer_B[l][metric])) if per_layer_B[l][metric] else np.nan for l in layers]

    x = np.arange(len(layers))
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.errorbar(x, vals_A, yerr=std_A, fmt="-o", color="crimson",
                label="Group A (suppressed)", linewidth=2, capsize=4)
    ax.errorbar(x, vals_B, yerr=std_B, fmt="-s", color="steelblue",
                label="Group B (not suppressed)", linewidth=2, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[SAVED] {out_path}")


def plot_occluder_leakage_ab(per_layer_A, per_layer_B, layers, out_path):
    """
    Stacked area / grouped bars showing what fraction of other-bbox leakage
    goes specifically to the PAIRED occluder vs the rest, for A vs B.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, per_layer, label, color in zip(
            axes,
            [per_layer_A, per_layer_B],
            ["Group A (suppressed)", "Group B (not suppressed)"],
            ["crimson", "steelblue"]):

        mean_occ   = [safe_mean(per_layer[l].get("frac_occluder_unwtd", [])) for l in layers]
        mean_other = [safe_mean(per_layer[l]["frac_other_unwtd"]) for l in layers]
        # other-excluding-occluder
        mean_rest  = [max(0.0, o - oc) if not np.isnan(oc) else o
                      for o, oc in zip(mean_other, mean_occ)]

        x  = np.arange(len(layers))
        bw = 0.35
        ax.bar(x, mean_occ,  bw, label="Into paired occluder",  color=color,     alpha=0.9)
        ax.bar(x, mean_rest, bw, label="Into other GT (rest)",   color="orange",  alpha=0.6,
               bottom=mean_occ)

        ax.set_xticks(x)
        ax.set_xticklabels([f"L{l}" for l in layers])
        ax.set_ylabel("Mean fraction of sampling points")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.0)

    fig.suptitle("Other-bbox leakage: paired occluder vs rest  (A vs B)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[SAVED] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Master aggregation + print + save
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_and_plot(all_results, out_dir, last_layer_only):
    os.makedirs(out_dir, exist_ok=True)

    if not all_results:
        print("[WARN] No results to aggregate.")
        return

    layers = sorted(all_results[0]["layer_results"].keys())
    last_l = layers[-1]

    # ── split by group ────────────────────────────────────────────────────────
    grp = {"A": [], "B": [], "C": []}
    for r in all_results:
        grp[r["group"]].append(r)

    print(f"\n{'='*72}")
    print("  SAMPLING LOCATION BBOX ANALYSIS  —  GROUP SPLIT")
    print(f"{'='*72}")
    print(f"  Group A (suppressed)      : {len(grp['A'])}")
    print(f"  Group B (not suppressed)  : {len(grp['B'])}")
    print(f"  Group C (no pair)         : {len(grp['C'])}")
    print(f"  Total                     : {len(all_results)}")

    pl_all = collect_per_layer(all_results,  layers)
    pl_A   = collect_per_layer(grp["A"],     layers)
    pl_B   = collect_per_layer(grp["B"],     layers)
    pl_C   = collect_per_layer(grp["C"],     layers)

    # ── print table ───────────────────────────────────────────────────────────
    for gname, pl, glist in [("ALL", pl_all, all_results),
                               ("A  (suppressed)", pl_A, grp["A"]),
                               ("B  (not suppr.) ", pl_B, grp["B"]),
                               ("C  (clean)      ", pl_C, grp["C"])]:
        if not glist:
            continue
        print(f"\n  --- Group {gname}  (n={len(glist)}) ---")
        print(f"  {'Layer':<6} {'Own(U)':<9} {'Other(U)':<10} {'Occ(U)':<9} {'BG':<9} {'Conf':<8}")
        print(f"  {'-'*55}")
        for l in layers:
            print(f"  L{l:<5} "
                  f"{safe_mean(pl[l]['frac_own_unwtd']):.4f}   "
                  f"{safe_mean(pl[l]['frac_other_unwtd']):.4f}    "
                  f"{safe_mean(pl[l].get('frac_occluder_unwtd',[])) if pl[l].get('frac_occluder_unwtd') else float('nan'):.4f}   "
                  f"{safe_mean(pl[l]['frac_bg']):.4f}   "
                  f"{safe_mean(pl[l]['conf']):.4f}")

    # ── A vs B gap summary at last layer ─────────────────────────────────────
    if grp["A"] and grp["B"]:
        print(f"\n  --- A vs B gap at L{last_l} ---")
        for metric, label in [("frac_own_unwtd",       "Own bbox (unwtd)  "),
                               ("frac_other_unwtd",     "Other bbox (unwtd)"),
                               ("frac_occluder_unwtd",  "Paired occ (unwtd)"),
                               ("frac_bg",              "Background        "),
                               ("conf",                 "Confidence        ")]:
            vA = safe_mean(pl_A[last_l].get(metric, []))
            vB = safe_mean(pl_B[last_l].get(metric, []))
            gap = vA - vB if not (np.isnan(vA) or np.isnan(vB)) else np.nan
            print(f"    {label}: A={vA:.4f}  B={vB:.4f}  gap(A-B)={gap:+.4f}")
    print(f"\n{'='*72}")

    # ── plots ─────────────────────────────────────────────────────────────────

    # 1. Overall distribution (all GTs)
    plot_layer_distribution(
        pl_all, layers,
        "Sampling location distribution — ALL GT objects",
        os.path.join(out_dir, "layer_dist_ALL.png"))

    # 2. Group A
    if grp["A"]:
        plot_layer_distribution(
            pl_A, layers,
            "Sampling location distribution — Group A (suppressed)",
            os.path.join(out_dir, "layer_dist_A.png"),
            color_own="crimson")

    # 3. Group B
    if grp["B"]:
        plot_layer_distribution(
            pl_B, layers,
            "Sampling location distribution — Group B (not suppressed)",
            os.path.join(out_dir, "layer_dist_B.png"),
            color_own="steelblue")

    # 4. A vs B — own bbox fraction across layers
    if grp["A"] and grp["B"]:
        plot_ab_comparison(
            pl_A, pl_B, layers,
            metric="frac_own_unwtd",
            ylabel="Mean fraction of pts inside OWN GT bbox",
            title="Own-bbox coverage: Group A vs B across decoder layers",
            out_path=os.path.join(out_dir, "ab_own_bbox_coverage.png"))

        # 5. A vs B — other bbox leakage
        plot_ab_comparison(
            pl_A, pl_B, layers,
            metric="frac_other_unwtd",
            ylabel="Mean fraction of pts inside OTHER GT bbox",
            title="Other-bbox leakage: Group A vs B across decoder layers",
            out_path=os.path.join(out_dir, "ab_other_bbox_leakage.png"))

        # 6. A vs B — specifically into paired occluder
        if pl_A[last_l].get("frac_occluder_unwtd") and pl_B[last_l].get("frac_occluder_unwtd"):
            plot_ab_comparison(
                pl_A, pl_B, layers,
                metric="frac_occluder_unwtd",
                ylabel="Mean fraction of pts inside PAIRED OCCLUDER bbox",
                title="Leakage into paired occluder: Group A vs B",
                out_path=os.path.join(out_dir, "ab_occluder_leakage.png"))

            # 7. Stacked: occluder vs rest of other-bbox
            plot_occluder_leakage_ab(
                pl_A, pl_B, layers,
                os.path.join(out_dir, "ab_occluder_vs_rest_stacked.png"))

        # 8. A vs B — confidence trajectory
        plot_ab_comparison(
            pl_A, pl_B, layers,
            metric="conf",
            ylabel="Mean confidence",
            title="Confidence trajectory: Group A vs B",
            out_path=os.path.join(out_dir, "ab_confidence_trajectory.png"))

        # 9. A vs B — background fraction
        plot_ab_comparison(
            pl_A, pl_B, layers,
            metric="frac_bg",
            ylabel="Mean fraction of pts in background",
            title="Background scatter: Group A vs B across decoder layers",
            out_path=os.path.join(out_dir, "ab_background_fraction.png"))

    # ── scatter: frac_own vs conf at last layer, coloured by group ───────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    group_styles = {"A": ("crimson",    "A suppressed"),
                    "B": ("steelblue",  "B not-suppr."),
                    "C": ("gray",       "C clean")}
    for ax_idx, (xmetric, xlabel) in enumerate([
            ("frac_own_unwtd",   "Fraction inside own GT bbox"),
            ("frac_other_unwtd", "Fraction inside other GT bbox")]):
        ax = axes[ax_idx]
        for g, (col, lbl) in group_styles.items():
            if not grp[g]:
                continue
            xs = [r["layer_results"][last_l][xmetric] for r in grp[g]]
            ys = [r["layer_results"][last_l]["conf"]   for r in grp[g]]
            ax.scatter(xs, ys, alpha=0.25, s=8, c=col, label=lbl)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"Confidence at L{last_l}")
        ax.set_title(f"{xlabel} vs confidence (L{last_l})")
        ax.legend(markerscale=2, fontsize=8)
    plt.tight_layout()
    scatter_path = os.path.join(out_dir, "scatter_frac_vs_conf_groups.png")
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"[SAVED] {scatter_path}")

    # ── JSON summary ──────────────────────────────────────────────────────────
    def layer_summary(pl):
        return {
            str(l): {
                "frac_own_unwtd":        safe_mean(pl[l]["frac_own_unwtd"]),
                "frac_own_wtd":          safe_mean(pl[l]["frac_own_wtd"]),
                "frac_other_unwtd":      safe_mean(pl[l]["frac_other_unwtd"]),
                "frac_other_wtd":        safe_mean(pl[l]["frac_other_wtd"]),
                "frac_occluder_unwtd":   safe_mean(pl[l].get("frac_occluder_unwtd", [])),
                "frac_occluder_wtd":     safe_mean(pl[l].get("frac_occluder_wtd",   [])),
                "frac_bg":               safe_mean(pl[l]["frac_bg"]),
                "mean_conf":             safe_mean(pl[l]["conf"]),
            }
            for l in layers
        }

    summary = {
        "group_counts":   {g: len(v) for g, v in grp.items()},
        "layers":         layers,
        "ALL":  layer_summary(pl_all),
        "A":    layer_summary(pl_A),
        "B":    layer_summary(pl_B),
        "C":    layer_summary(pl_C),
    }
    json_path = os.path.join(out_dir, "sampling_bbox_analysis_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVED] {json_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sampling location bbox analysis v2")
    parser.add_argument("--debug_dir",       required=True)
    parser.add_argument("--coco_json",       required=True)
    parser.add_argument("--pairs_json",      required=True,
                        help="Occlusion pairs JSON file")
    parser.add_argument("--out_dir",         default="./sampling_bbox_analysis_out")
    parser.add_argument("--last_layer_only", action="store_true")
    parser.add_argument("--max_images",      type=int, default=None)
    args = parser.parse_args()

    # ── load COCO GT ──────────────────────────────────────────────────────────
    print(f"[INFO] Loading COCO GT from {args.coco_json} ...")
    with open(args.coco_json) as f:
        coco = json.load(f)

    img_info       = {img["file_name"]: img for img in coco["images"]}
    img_info_by_id = {img["id"]: img         for img in coco["images"]}

    anns_by_img    = defaultdict(list)
    ann_by_id      = {}
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)
        ann_by_id[ann["id"]] = ann

    # Pre-build a val-set-wide ann_id → normalised box mapping
    # (needed to look up occluder boxes that may be on the same image)
    # We'll fill this lazily per image during processing instead, but we need
    # image dimensions. Build a full map upfront.
    ann_to_box_norm = {}
    for ann in coco["annotations"]:
        img_meta = img_info_by_id.get(ann["image_id"])
        if img_meta is None:
            continue
        ann_to_box_norm[ann["id"]] = xywh_to_x1y1x2y2_norm(
            ann["bbox"], img_meta["width"], img_meta["height"])

    # ── load pairs ────────────────────────────────────────────────────────────
    suppressed_ids, not_suppressed_ids, pair_by_occluded = load_pairs(args.pairs_json)

    # ── find .pt files ────────────────────────────────────────────────────────
    pt_files = sorted(glob.glob(os.path.join(args.debug_dir, "*.pt")))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files in {args.debug_dir}")
    if args.max_images:
        pt_files = pt_files[:args.max_images]
    print(f"[INFO] {len(pt_files)} .pt files to process")

    # ── process ───────────────────────────────────────────────────────────────
    all_results = []
    skipped     = 0

    for pt_path in tqdm(pt_files, desc="Images"):
        data = torch.load(pt_path, map_location="cpu",
                          weights_only=False)

        img_path_str = str(data.get("img_path", ""))
        img_basename = os.path.basename(img_path_str)

        if img_basename in img_info:
            iinfo = img_info[img_basename]
        else:
            matched = [v for k, v in img_info.items() if img_basename in k]
            if matched:
                iinfo = matched[0]
            else:
                skipped += 1
                continue

        img_id  = iinfo["id"]
        img_w   = iinfo["width"]
        img_h   = iinfo["height"]
        gt_anns = anns_by_img.get(img_id, [])
        if not gt_anns:
            continue

        img_results = analyse_image(
            data, gt_anns, img_w, img_h,
            suppressed_ids, not_suppressed_ids, pair_by_occluded,
            ann_to_box_norm,
            last_layer_only=args.last_layer_only,
        )
        all_results.extend(img_results)

    print(f"[INFO] {len(all_results)} GT objects processed | {skipped} images skipped")

    if not all_results:
        print("[ERROR] No results — check img_path vs COCO filenames.")
        return

    aggregate_and_plot(all_results, args.out_dir, args.last_layer_only)


if __name__ == "__main__":
    main()