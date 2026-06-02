"""
Sampling Location BBox Analysis
================================
For each GT annotation, find the best-matched query (highest cls confidence),
then compute:
  - fraction of sampling locations inside its own GT bbox
  - fraction of sampling locations inside any other GT bbox
  - how confidence trajectory varies with these fractions

Sampling locations are in normalised [0,1] coords relative to the image.
GT bboxes from COCO JSON are in [x, y, w, h] pixel coords → convert to [x1,y1,x2,y2]
then normalise by image W, H.

Usage:
    python sampling_location_bbox_analysis.py \
        --debug_dir  /path/to/debug_outputs_complete_val_set \
        --coco_json  /path/to/instances_val.json \
        --out_dir    ./sampling_bbox_analysis_out \
        [--last_layer_only]   # only analyse final decoder layer (default: all layers)
        [--max_images N]      # limit to N images for quick testing
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
    return np.array([x / img_w,
                     y / img_h,
                     (x + w) / img_w,
                     (y + h) / img_h], dtype=np.float32)


def points_in_box(pts_xy, box_x1y1x2y2):
    """
    pts_xy          : (N, 2)  normalised (x, y)
    box_x1y1x2y2    : (4,)    normalised [x1, y1, x2, y2]
    Returns boolean mask of shape (N,).
    """
    x1, y1, x2, y2 = box_x1y1x2y2
    inside = (pts_xy[:, 0] >= x1) & (pts_xy[:, 0] <= x2) & \
             (pts_xy[:, 1] >= y1) & (pts_xy[:, 1] <= y2)
    return inside


def flatten_sampling_locs(sampling_locs_query, spatial_shapes, valid_ratios=None):
    """
    Flatten sampling locations for ONE query across all heads/levels/points.

    sampling_locs_query : (H, Lv, P, 2)   normalised [0,1] coords
    spatial_shapes      : (Lv, 2)          [h, w] per level
    valid_ratios        : (Lv, 2)          optional, used to clip to valid region

    Returns  pts : (H*Lv*P, 2)  in true normalised image coords [0,1]
    """
    H, Lv, P, _ = sampling_locs_query.shape
    pts = sampling_locs_query.reshape(-1, 2).float().numpy()  # (H*Lv*P, 2)

    # Clip to [0, 1] — points outside image are meaningless
    pts = np.clip(pts, 0.0, 1.0)
    return pts


# ─────────────────────────────────────────────────────────────────────────────
# Confidence helper
# ─────────────────────────────────────────────────────────────────────────────

def get_cls_score_trajectory(cls_scores_all_layers, query_idx, class_id):
    """
    cls_scores_all_layers : (L, Q, C)  full precision
    Returns (L,) confidence trajectory for the given query and class.
    """
    # cls_scores may be logits → sigmoid
    scores = cls_scores_all_layers[:, query_idx, class_id].float()
    return torch.sigmoid(scores).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Per-image analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_image(data, gt_anns, img_w, img_h, last_layer_only=False):
    """
    data      : dict loaded from .pt file
    gt_anns   : list of COCO annotation dicts for this image
    Returns list of per-GT result dicts.
    """
    cls_scores      = data["cls_scores"]          # (L, Q, C)
    sampling_locs   = data["sampling_locations"]  # (L, Q, H, Lv, P, 2)
    attn_weights    = data["attention_weights"]   # (L, Q, H, Lv, P)
    spatial_shapes  = data["spatial_shapes"]      # (Lv, 2)
    valid_ratios    = data.get("valid_ratios")     # (Lv, 2) or None

    L, Q, C = cls_scores.shape
    layers   = [L - 1] if last_layer_only else list(range(L))

    # ── best query per GT ────────────────────────────────────────────────────
    # Use LAST layer confidence to pick the best query for each GT.
    last_layer_scores = torch.sigmoid(cls_scores[-1].float())  # (Q, C)

    # Build normalised GT bboxes
    gt_boxes_norm = []
    for ann in gt_anns:
        cat_id_zero = ann["category_id"] - 1          # 0-indexed
        box_norm    = xywh_to_x1y1x2y2_norm(ann["bbox"], img_w, img_h)
        gt_boxes_norm.append((ann, cat_id_zero, box_norm))

    results = []

    for ann, cat_id, own_box in gt_boxes_norm:
        # Best query: highest confidence for this category at last layer
        cat_scores = last_layer_scores[:, cat_id]     # (Q,)
        best_q     = int(cat_scores.argmax().item())
        best_conf  = float(cat_scores[best_q].item())

        # Confidence trajectory across layers
        conf_traj = get_cls_score_trajectory(cls_scores, best_q, cat_id)   # (L,)

        # ── per-layer sampling location analysis ─────────────────────────────
        layer_results = {}
        for l_idx in layers:
            sl_q = sampling_locs[l_idx, best_q]  # (H, Lv, P, 2)
            aw_q = attn_weights[l_idx, best_q]   # (H, Lv, P)

            pts      = flatten_sampling_locs(sl_q, spatial_shapes, valid_ratios)  # (N,2)
            aw_flat  = aw_q.float().numpy().reshape(-1)  # (N,)

            n_pts = len(pts)

            # ── inside own bbox ──────────────────────────────────────────────
            mask_own = points_in_box(pts, own_box)
            frac_own_unwtd  = float(mask_own.sum()) / n_pts
            frac_own_wtd    = float(aw_flat[mask_own].sum())  # attn already sums ~1 across pts

            # ── inside any OTHER GT bbox ─────────────────────────────────────
            mask_other_any = np.zeros(n_pts, dtype=bool)
            other_box_hits = {}  # ann_id → (frac_unwtd, frac_wtd)
            for other_ann, other_cat, other_box in gt_boxes_norm:
                if other_ann["id"] == ann["id"]:
                    continue
                m = points_in_box(pts, other_box)
                other_box_hits[other_ann["id"]] = {
                    "frac_unwtd": float(m.sum()) / n_pts,
                    "frac_wtd":   float(aw_flat[m].sum()),
                    "other_cat":  other_cat,
                }
                mask_other_any |= m

            frac_other_any_unwtd = float(mask_other_any.sum()) / n_pts
            frac_other_any_wtd   = float(aw_flat[mask_other_any].sum())
            frac_background      = 1.0 - frac_own_unwtd - frac_other_any_unwtd

            layer_results[l_idx] = {
                "frac_own_unwtd":        frac_own_unwtd,
                "frac_own_wtd":          frac_own_wtd,
                "frac_other_any_unwtd":  frac_other_any_unwtd,
                "frac_other_any_wtd":    frac_other_any_wtd,
                "frac_background":       frac_background,
                "other_box_hits":        other_box_hits,
                "conf":                  float(conf_traj[l_idx]),
            }

        results.append({
            "ann_id":       ann["id"],
            "cat_id":       cat_id,
            "best_query":   best_q,
            "best_conf_L":  best_conf,
            "conf_traj":    conf_traj.tolist(),
            "layer_results": layer_results,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation & plotting
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_and_plot(all_results, out_dir, last_layer_only):
    """
    all_results : list of per-GT dicts (flattened across all images)
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── determine available layers ───────────────────────────────────────────
    if all_results:
        layers = sorted(all_results[0]["layer_results"].keys())
    else:
        print("[WARN] No results to aggregate.")
        return

    # ── collect arrays per layer ─────────────────────────────────────────────
    per_layer = {l: defaultdict(list) for l in layers}

    for r in all_results:
        for l in layers:
            lr = r["layer_results"][l]
            per_layer[l]["frac_own_unwtd"].append(lr["frac_own_unwtd"])
            per_layer[l]["frac_own_wtd"].append(lr["frac_own_wtd"])
            per_layer[l]["frac_other_unwtd"].append(lr["frac_other_any_unwtd"])
            per_layer[l]["frac_other_wtd"].append(lr["frac_other_any_wtd"])
            per_layer[l]["frac_bg"].append(lr["frac_background"])
            per_layer[l]["conf"].append(lr["conf"])

    # ── print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SAMPLING LOCATION BBOX ANALYSIS")
    print("=" * 70)
    print(f"  Total GT objects analysed : {len(all_results)}")
    print(f"  Layers analysed           : {layers}")
    print("-" * 70)
    print(f"{'Layer':<8} {'Own(U)':<10} {'Own(W)':<10} {'Other(U)':<10} {'Other(W)':<10} {'BG(U)':<10} {'MeanConf':<10}")
    print("-" * 70)
    for l in layers:
        d = per_layer[l]
        print(f"  L{l:<5} "
              f"{np.mean(d['frac_own_unwtd']):.4f}    "
              f"{np.mean(d['frac_own_wtd']):.4f}    "
              f"{np.mean(d['frac_other_unwtd']):.4f}    "
              f"{np.mean(d['frac_other_wtd']):.4f}    "
              f"{np.mean(d['frac_bg']):.4f}    "
              f"{np.mean(d['conf']):.4f}")
    print("=" * 70)

    # ── scatter: frac_own vs confidence (last layer) ─────────────────────────
    last_l = layers[-1]
    frac_own_last   = np.array(per_layer[last_l]["frac_own_unwtd"])
    frac_other_last = np.array(per_layer[last_l]["frac_other_unwtd"])
    conf_last       = np.array(per_layer[last_l]["conf"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(frac_own_last, conf_last, alpha=0.3, s=10, c="steelblue")
    axes[0].set_xlabel("Fraction of sampling pts inside OWN GT bbox (unweighted)")
    axes[0].set_ylabel(f"Confidence at L{last_l}")
    axes[0].set_title("Own-bbox coverage vs Confidence")
    # trend line
    if len(frac_own_last) > 5:
        z = np.polyfit(frac_own_last, conf_last, 1)
        xp = np.linspace(frac_own_last.min(), frac_own_last.max(), 100)
        axes[0].plot(xp, np.poly1d(z)(xp), "r--", linewidth=1.5,
                     label=f"slope={z[0]:.3f}")
        axes[0].legend()

    axes[1].scatter(frac_other_last, conf_last, alpha=0.3, s=10, c="darkorange")
    axes[1].set_xlabel("Fraction of sampling pts inside OTHER GT bboxes (unweighted)")
    axes[1].set_ylabel(f"Confidence at L{last_l}")
    axes[1].set_title("Other-bbox leakage vs Confidence")
    if len(frac_other_last) > 5:
        z2 = np.polyfit(frac_other_last, conf_last, 1)
        xp2 = np.linspace(frac_other_last.min(), frac_other_last.max(), 100)
        axes[1].plot(xp2, np.poly1d(z2)(xp2), "r--", linewidth=1.5,
                     label=f"slope={z2[0]:.3f}")
        axes[1].legend()

    plt.tight_layout()
    scatter_path = os.path.join(out_dir, "scatter_frac_vs_conf.png")
    plt.savefig(scatter_path, dpi=150)
    plt.close() 
    print(f"[SAVED] {scatter_path}")

    # ── mean fractions across layers ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    mean_own   = [np.mean(per_layer[l]["frac_own_unwtd"])   for l in layers]
    mean_other = [np.mean(per_layer[l]["frac_other_unwtd"]) for l in layers]
    mean_bg    = [np.mean(per_layer[l]["frac_bg"])          for l in layers]
    mean_conf  = [np.mean(per_layer[l]["conf"])             for l in layers]

    x = np.arange(len(layers))
    w = 0.25
    ax.bar(x - w, mean_own,   w, label="Own bbox",   color="steelblue")
    ax.bar(x,     mean_other, w, label="Other bbox",  color="darkorange")
    ax.bar(x + w, mean_bg,    w, label="Background",  color="gray")
    ax2 = ax.twinx()
    ax2.plot(x, mean_conf, "r-o", label="Mean conf", linewidth=2)
    ax2.set_ylabel("Mean confidence")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_ylabel("Mean fraction of sampling points")
    ax.set_title("Sampling location distribution across decoder layers")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    bar_path = os.path.join(out_dir, "layer_sampling_distribution.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"[SAVED] {bar_path}")

    # ── confidence binned by own-bbox fraction (last layer) ──────────────────
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_conf_mean  = []
    bin_conf_std   = []
    bin_counts     = []

    for i in range(len(bins) - 1):
        mask = (frac_own_last >= bins[i]) & (frac_own_last < bins[i + 1])
        c    = conf_last[mask]
        bin_conf_mean.append(c.mean() if len(c) > 0 else np.nan)
        bin_conf_std.append(c.std()  if len(c) > 0 else np.nan)
        bin_counts.append(len(c))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(bin_centers, bin_conf_mean, width=0.08, color="steelblue",
           alpha=0.7, yerr=bin_conf_std, capsize=3, label="mean±std conf")
    ax2 = ax.twinx()
    ax2.plot(bin_centers, bin_counts, "r-o", label="count", linewidth=1.5)
    ax2.set_ylabel("# GT objects")
    ax.set_xlabel("Fraction of sampling pts inside own GT bbox")
    ax.set_ylabel("Mean confidence at last layer")
    ax.set_title("Confidence vs own-bbox sampling coverage")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    bin_path = os.path.join(out_dir, "conf_vs_own_frac_binned.png")
    plt.savefig(bin_path, dpi=150)
    plt.close()
    print(f"[SAVED] {bin_path}")

    # ── same binned plot but for other-bbox leakage ───────────────────────────
    bin_conf_mean2  = []
    bin_conf_std2   = []
    bin_counts2     = []
    for i in range(len(bins) - 1):
        mask = (frac_other_last >= bins[i]) & (frac_other_last < bins[i + 1])
        c    = conf_last[mask]
        bin_conf_mean2.append(c.mean() if len(c) > 0 else np.nan)
        bin_conf_std2.append(c.std()  if len(c) > 0 else np.nan)
        bin_counts2.append(len(c))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(bin_centers, bin_conf_mean2, width=0.08, color="darkorange",
           alpha=0.7, yerr=bin_conf_std2, capsize=3, label="mean±std conf")
    ax2 = ax.twinx()
    ax2.plot(bin_centers, bin_counts2, "r-o", label="count", linewidth=1.5)
    ax2.set_ylabel("# GT objects")
    ax2.set_xlabel("Fraction of sampling pts inside other GT bboxes")
    ax.set_xlabel("Fraction of sampling pts inside other GT bboxes")
    ax.set_ylabel("Mean confidence at last layer")
    ax.set_title("Confidence vs other-bbox leakage")
    ax.legend(loc="upper right")
    ax2.legend(loc="center right")
    plt.tight_layout()
    bin_path2 = os.path.join(out_dir, "conf_vs_other_frac_binned.png")
    plt.savefig(bin_path2, dpi=150)
    plt.close()
    print(f"[SAVED] {bin_path2}")

    # ── save JSON summary ─────────────────────────────────────────────────────
    summary = {
        "n_gt_objects": len(all_results),
        "layers": layers,
        "per_layer_means": {
            str(l): {
                "frac_own_unwtd":   float(np.mean(per_layer[l]["frac_own_unwtd"])),
                "frac_own_wtd":     float(np.mean(per_layer[l]["frac_own_wtd"])),
                "frac_other_unwtd": float(np.mean(per_layer[l]["frac_other_unwtd"])),
                "frac_other_wtd":   float(np.mean(per_layer[l]["frac_other_wtd"])),
                "frac_bg":          float(np.mean(per_layer[l]["frac_bg"])),
                "mean_conf":        float(np.mean(per_layer[l]["conf"])),
            }
            for l in layers
        },
        "correlation_own_vs_conf_lastlayer": float(np.corrcoef(frac_own_last, conf_last)[0, 1]),
        "correlation_other_vs_conf_lastlayer": float(np.corrcoef(frac_other_last, conf_last)[0, 1]),
    }
    json_path = os.path.join(out_dir, "sampling_bbox_analysis_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVED] {json_path}")

    print("\n  CORRELATION SUMMARY")
    print(f"  frac_own  vs conf (L{last_l}): {summary['correlation_own_vs_conf_lastlayer']:+.4f}")
    print(f"  frac_other vs conf (L{last_l}): {summary['correlation_other_vs_conf_lastlayer']:+.4f}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sampling location bbox analysis")
    parser.add_argument("--debug_dir",       required=True,
                        help="Directory containing per-image .pt debug files")
    parser.add_argument("--coco_json",       required=True,
                        help="COCO format GT annotation JSON")
    parser.add_argument("--out_dir",         default="./sampling_bbox_analysis_out")
    parser.add_argument("--last_layer_only", action="store_true",
                        help="Only analyse the final decoder layer")
    parser.add_argument("--max_images",      type=int, default=None,
                        help="Limit number of images (for quick testing)")
    args = parser.parse_args()

    # ── load COCO GT ──────────────────────────────────────────────────────────
    print(f"[INFO] Loading COCO GT from {args.coco_json} ...")
    with open(args.coco_json) as f:
        coco = json.load(f)

    # map image filename → (image_id, width, height)
    img_info = {
        img["file_name"]: img
        for img in coco["images"]
    }
    # also map by id
    img_info_by_id = {img["id"]: img for img in coco["images"]}

    # map image_id → list of annotations
    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    # ── find .pt files ────────────────────────────────────────────────────────
    pt_files = sorted(glob.glob(os.path.join(args.debug_dir, "*.pt")))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {args.debug_dir}")
    if args.max_images:
        pt_files = pt_files[:args.max_images]
    print(f"[INFO] Found {len(pt_files)} .pt files (processing {len(pt_files)})")

    # ── process each image ────────────────────────────────────────────────────
    all_results = []
    skipped     = 0

    for pt_path in tqdm(pt_files, desc="Images"):
        data = torch.load(pt_path, map_location="cpu")

        # resolve image info from img_path stored in the .pt
        img_path_str = str(data.get("img_path", ""))
        img_basename = os.path.basename(img_path_str)

        # try to match in COCO
        if img_basename in img_info:
            iinfo = img_info[img_basename]
        else:
            # fall back: try matching by numeric id in filename
            matched = [v for k, v in img_info.items() if img_basename in k]
            if matched:
                iinfo = matched[0]
            else:
                skipped += 1
                continue

        img_id = iinfo["id"]
        img_w  = iinfo["width"]
        img_h  = iinfo["height"]
        gt_anns = anns_by_img.get(img_id, [])
        if not gt_anns:
            continue

        img_results = analyse_image(
            data, gt_anns, img_w, img_h,
            last_layer_only=args.last_layer_only
        )
        all_results.extend(img_results)

    print(f"[INFO] Processed {len(all_results)} GT objects | skipped {skipped} images (no COCO match)")

    if not all_results:
        print("[ERROR] No results. Check that img_path in .pt files matches COCO filenames.")
        return

    aggregate_and_plot(all_results, args.out_dir, args.last_layer_only)


if __name__ == "__main__":
    main()