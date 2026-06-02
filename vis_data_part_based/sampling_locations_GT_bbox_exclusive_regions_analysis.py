"""
Sampling Location BBox Analysis (Set-Theoretic Decomposition)
=============================================================
For each GT annotation, find the best-matched query (highest cls confidence),
then compute a proper set-theoretic decomposition of sampling locations:

  Region A  : Exclusively inside own GT bbox (own ∩ ¬others)
  Region A∩B: Inside own GT bbox AND at least one other GT bbox (overlap/intersection)
  Region B  : Exclusively inside other GT bboxes (¬own ∩ others)
  Region BG : Background (outside all GT bboxes)

These four regions are mutually exclusive and exhaustive:
  A + A∩B + B + BG = 1.0  (for unweighted fractions)

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
    return np.array([
        x / img_w,
        y / img_h,
        (x + w) / img_w,
        (y + h) / img_h
    ], dtype=np.float32)


def cxcywh_to_xyxy(boxes):
    """
    boxes : (...,4) normalized [cx,cy,w,h]

    returns (...,4) normalized [x1,y1,x2,y2]
    """
    cx = boxes[..., 0]
    cy = boxes[..., 1]
    w  = boxes[..., 2]
    h  = boxes[..., 3]

    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    return np.stack([x1, y1, x2, y2], axis=-1)


def compute_iou(pred_boxes_xyxy, gt_box_xyxy):
    """
    pred_boxes_xyxy : (Q,4)
    gt_box_xyxy     : (4,)

    returns:
        ious : (Q,)
    """

    ix1 = np.maximum(pred_boxes_xyxy[:, 0], gt_box_xyxy[0])
    iy1 = np.maximum(pred_boxes_xyxy[:, 1], gt_box_xyxy[1])
    ix2 = np.minimum(pred_boxes_xyxy[:, 2], gt_box_xyxy[2])
    iy2 = np.minimum(pred_boxes_xyxy[:, 3], gt_box_xyxy[3])

    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)

    inter = iw * ih

    pred_area = (
        (pred_boxes_xyxy[:, 2] - pred_boxes_xyxy[:, 0]) *
        (pred_boxes_xyxy[:, 3] - pred_boxes_xyxy[:, 1])
    )

    gt_area = (
        (gt_box_xyxy[2] - gt_box_xyxy[0]) *
        (gt_box_xyxy[3] - gt_box_xyxy[1])
    )

    union = pred_area + gt_area - inter

    return inter / np.maximum(union, 1e-8)


def points_in_box(pts_xy, box_x1y1x2y2):
    """
    pts_xy          : (N, 2)  normalised (x, y)
    box_x1y1x2y2    : (4,)    normalised [x1, y1, x2, y2]
    Returns boolean mask of shape (N,).
    """
    x1, y1, x2, y2 = box_x1y1x2y2
    return (
        (pts_xy[:, 0] >= x1) & (pts_xy[:, 0] <= x2) &
        (pts_xy[:, 1] >= y1) & (pts_xy[:, 1] <= y2)
    )


def flatten_sampling_locs(sampling_locs_query):
    """
    Flatten sampling locations for ONE query across all heads/levels/points.

    sampling_locs_query : (H, Lv, P, 2)   normalised [0,1] coords

    Returns pts : (H*Lv*P, 2)  clipped to valid image coords [0,1]
    """
    pts = sampling_locs_query.reshape(-1, 2).float().numpy()
    pts = np.clip(pts, 0.0, 1.0)
    return pts


def compute_set_theoretic_regions(pts, aw_flat, own_box, other_boxes_norm):
    """
    Given sampling points and attention weights for one query at one layer,
    compute the four mutually exclusive, exhaustive regions.

    Parameters
    ----------
    pts             : (N, 2)  normalised sampling point coords
    aw_flat         : (N,)    attention weights (should sum to ~1)
    own_box         : (4,)    own GT bbox in normalised [x1,y1,x2,y2]
    other_boxes_norm: list of (4,) arrays for all OTHER GT bboxes in this image

    Returns dict with keys:
        exclusive_own       → inside own bbox only
        overlap             → inside own bbox AND ≥1 other bbox
        exclusive_other     → inside ≥1 other bbox but NOT own bbox
        background          → outside all GT bboxes
        [each key has _unwtd and _wtd sub-fields]

    Sanity check: exclusive_own + overlap + exclusive_other + background ≈ 1.0
    """
    n_pts = len(pts)

    # Primary masks
    mask_own   = points_in_box(pts, own_box)                    # in own bbox
    mask_other = np.zeros(n_pts, dtype=bool)
    for ob in other_boxes_norm:
        mask_other |= points_in_box(pts, ob)                    # in ANY other bbox

    # Set-theoretic decomposition (mutually exclusive)
    mask_excl_own   = mask_own & ~mask_other    # A \ B
    mask_overlap    = mask_own &  mask_other    # A ∩ B
    mask_excl_other = ~mask_own & mask_other   # B \ A
    mask_bg         = ~mask_own & ~mask_other  # background

    # Sanity check
    total = (mask_excl_own | mask_overlap | mask_excl_other | mask_bg).sum()
    assert total == n_pts, "Partition is not exhaustive — logic error"

    def fracs(mask):
        unwtd = float(mask.sum()) / n_pts
        wtd   = float(aw_flat[mask].sum())
        return unwtd, wtd

    excl_own_u,   excl_own_w   = fracs(mask_excl_own)
    overlap_u,    overlap_w    = fracs(mask_overlap)
    excl_other_u, excl_other_w = fracs(mask_excl_other)
    bg_u,         bg_w         = fracs(mask_bg)

    return {
        "exclusive_own_unwtd":   excl_own_u,
        "exclusive_own_wtd":     excl_own_w,
        "overlap_unwtd":         overlap_u,
        "overlap_wtd":           overlap_w,
        "exclusive_other_unwtd": excl_other_u,
        "exclusive_other_wtd":   excl_other_w,
        "background_unwtd":      bg_u,
        "background_wtd":        bg_w,
        # convenience: full own = excl_own + overlap  (for backward compat)
        "full_own_unwtd":        excl_own_u + overlap_u,
        "full_own_wtd":          excl_own_w + overlap_w,
        # convenience: full other = excl_other + overlap
        "full_other_unwtd":      excl_other_u + overlap_u,
        "full_other_wtd":        excl_other_w + overlap_w,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Confidence helper
# ─────────────────────────────────────────────────────────────────────────────

def get_cls_score_trajectory(cls_scores_all_layers, query_idx, class_id):
    """
    cls_scores_all_layers : (L, Q, C)  raw logits
    Returns (L,) sigmoid confidence trajectory for the given query and class.
    """
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
    cls_scores    = data["cls_scores"]
    pred_bboxes   = data["pred_bboxes"]      # [L,Q,4]
    sampling_locs = data["sampling_locations"]
    attn_weights  = data["attention_weights"]

    L, Q, C = cls_scores.shape
    layers   = [L - 1] if last_layer_only else list(range(L))

    # Use last-layer confidence to assign the best query to each GT object
    last_layer_scores = torch.sigmoid(cls_scores[-1].float())  # (Q, C)

    last_layer_boxes = pred_bboxes[-1].numpy()      # [Q,4] cxcywh

    last_layer_boxes_xyxy = cxcywh_to_xyxy(
        last_layer_boxes
    )

    last_layer_boxes_xyxy = np.clip(
        last_layer_boxes_xyxy,
        0.0,
        1.0
    )

    # Pre-compute normalised GT bboxes once per image
    gt_boxes_norm = []
    for ann in gt_anns:
        cat_id_zero = ann["category_id"] - 1
        box_norm    = xywh_to_x1y1x2y2_norm(ann["bbox"], img_w, img_h)
        gt_boxes_norm.append((ann, cat_id_zero, box_norm))

    results = []

    for ann, cat_id, own_box in gt_boxes_norm:

        # ------------------------------------------------------------
        # Combined IoU × confidence query selection
        # ------------------------------------------------------------

        cat_scores = last_layer_scores[:, cat_id].numpy()

        ious = compute_iou(
            last_layer_boxes_xyxy,
            own_box
        )

        valid = cat_scores > 0.05

        combined_scores = np.where(
            valid,
            ious * cat_scores,
            -1.0
        )

        best_q = int(np.argmax(combined_scores))

        best_conf     = float(cat_scores[best_q])
        best_iou      = float(ious[best_q])
        best_combined = float(combined_scores[best_q])

        # All OTHER normalised boxes
        other_boxes = [
            ob for (a, _, ob) in gt_boxes_norm
            if a["id"] != ann["id"]
        ]

        # Confidence trajectory across layers
        conf_traj = get_cls_score_trajectory(
            cls_scores,
            best_q,
            cat_id
        )

        # ------------------------------------------------------------
        # Per-layer analysis
        # ------------------------------------------------------------

        layer_results = {}

        for l_idx in layers:

            sl_q = sampling_locs[l_idx, best_q]
            aw_q = attn_weights[l_idx, best_q]

            pts = flatten_sampling_locs(sl_q)
            aw_flat = aw_q.float().numpy().reshape(-1)

            regions = compute_set_theoretic_regions(
                pts,
                aw_flat,
                own_box,
                other_boxes
            )

            regions["conf"] = float(conf_traj[l_idx])

            layer_results[l_idx] = regions

        results.append({
            "ann_id":         ann["id"],
            "cat_id":         cat_id,
            "best_query":     best_q,
            "best_conf_L":    best_conf,
            "best_iou_L":     best_iou,
            "best_combined":  best_combined,
            "conf_traj":      conf_traj.tolist(),
            "has_neighbors":  len(other_boxes) > 0,
            "layer_results":  layer_results,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation & plotting
# ─────────────────────────────────────────────────────────────────────────────

REGION_KEYS = [
    ("exclusive_own_unwtd",   "Excl. Own",   "steelblue"),
    ("overlap_unwtd",         "Overlap",     "mediumpurple"),
    ("exclusive_other_unwtd", "Excl. Other", "darkorange"),
    ("background_unwtd",      "Background",  "gray"),
]


def aggregate_and_plot(all_results, out_dir, last_layer_only):
    os.makedirs(out_dir, exist_ok=True)

    if not all_results:
        print("[WARN] No results to aggregate.")
        return

    layers = sorted(all_results[0]["layer_results"].keys())
    last_l = layers[-1]

    # ── collect scalars per layer ─────────────────────────────────────────────
    per_layer = {l: defaultdict(list) for l in layers}
    for r in all_results:
        for l in layers:
            lr = r["layer_results"][l]
            for key, _, _ in REGION_KEYS:
                per_layer[l][key].append(lr[key])
            per_layer[l]["conf"].append(lr["conf"])
            # weighted equivalents
            per_layer[l]["exclusive_own_wtd"].append(lr["exclusive_own_wtd"])
            per_layer[l]["overlap_wtd"].append(lr["overlap_wtd"])
            per_layer[l]["exclusive_other_wtd"].append(lr["exclusive_other_wtd"])
            per_layer[l]["background_wtd"].append(lr["background_wtd"])

    # ── sanity check: fractions sum to 1 ─────────────────────────────────────
    for l in layers:
        means = [np.mean(per_layer[l][k]) for k, _, _ in REGION_KEYS]
        total = sum(means)
        if abs(total - 1.0) > 1e-3:
            print(f"[WARN] Layer {l}: region fractions sum to {total:.4f} (expected 1.0)")

    # ── print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  SAMPLING LOCATION SET-THEORETIC ANALYSIS")
    print("=" * 90)
    print(f"  Total GT objects : {len(all_results)}")
    print(f"  Layers analysed  : {layers}")
    print(f"\n  NOTE: Excl.Own + Overlap + Excl.Other + BG = 1.0 (mutually exclusive)")
    print("-" * 90)
    hdr = f"{'Layer':<8} {'Excl.Own(U)':<14} {'Overlap(U)':<14} {'Excl.Other(U)':<16} {'BG(U)':<12} {'MeanConf':<10}"
    print(hdr)
    print("-" * 90)
    for l in layers:
        d = per_layer[l]
        print(f"  L{l:<5} "
              f"{np.mean(d['exclusive_own_unwtd']):.4f}         "
              f"{np.mean(d['overlap_unwtd']):.4f}         "
              f"{np.mean(d['exclusive_other_unwtd']):.4f}            "
              f"{np.mean(d['background_unwtd']):.4f}       "
              f"{np.mean(d['conf']):.4f}")
    print("=" * 90)

    # ── also split by images with/without neighbouring GT boxes ──────────────
    isolated = [r for r in all_results if not r["has_neighbors"]]
    crowded  = [r for r in all_results if r["has_neighbors"]]
    print(f"\n  GT objects with no neighbours (isolated) : {len(isolated)}")
    print(f"  GT objects with ≥1 neighbour  (crowded)  : {len(crowded)}")
    print(f"  [For isolated objects, Overlap and Excl.Other are always 0 by definition]")

    # ── PLOT 1: stacked bar — mean region fractions across layers ─────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(layers))
    bottoms = np.zeros(len(layers))

    for key, label, color in REGION_KEYS:
        vals = np.array([np.mean(per_layer[l][key]) for l in layers])
        ax.bar(x, vals, bottom=bottoms, label=label, color=color, alpha=0.85)
        bottoms += vals

    ax2 = ax.twinx()
    mean_conf = [np.mean(per_layer[l]["conf"]) for l in layers]
    ax2.plot(x, mean_conf, "r-o", label="Mean conf", linewidth=2)
    ax2.set_ylabel("Mean confidence")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_ylabel("Fraction of sampling points (unweighted)")
    ax.set_title("Set-theoretic sampling decomposition across decoder layers\n"
                 "(Excl.Own + Overlap + Excl.Other + BG = 1.0)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    path = os.path.join(out_dir, "layer_set_theoretic_stacked.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"\n[SAVED] {path}")

    # ── PLOT 2: scatter — each region fraction vs confidence (last layer) ─────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    plot_regions = [
        ("exclusive_own_unwtd",   "Excl. Own",   "steelblue",    axes[0]),
        ("overlap_unwtd",         "Overlap",     "mediumpurple",  axes[1]),
        ("exclusive_other_unwtd", "Excl. Other", "darkorange",   axes[2]),
    ]

    last_conf = np.array(per_layer[last_l]["conf"])

    for key, label, color, ax in plot_regions:
        frac = np.array(per_layer[last_l][key])
        ax.scatter(frac, last_conf, alpha=0.3, s=10, c=color)
        ax.set_xlabel(f"Fraction of pts in {label}")
        ax.set_ylabel(f"Confidence at L{last_l}")
        ax.set_title(f"{label} coverage vs Confidence")
        if len(frac) > 5 and frac.std() > 1e-6:
            z  = np.polyfit(frac, last_conf, 1)
            xp = np.linspace(frac.min(), frac.max(), 100)
            corr = float(np.corrcoef(frac, last_conf)[0, 1])
            ax.plot(xp, np.poly1d(z)(xp), "r--", linewidth=1.5,
                    label=f"slope={z[0]:.3f}\nr={corr:.3f}")
            ax.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "scatter_regions_vs_conf.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[SAVED] {path}")

    # ── PLOT 3: binned confidence per region (last layer) ─────────────────────
    bins        = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    plot_regions_binned = [
        ("exclusive_own_unwtd",   "Excl. Own",   "steelblue"),
        ("overlap_unwtd",         "Overlap",     "mediumpurple"),
        ("exclusive_other_unwtd", "Excl. Other", "darkorange"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for (key, label, color), ax in zip(plot_regions_binned, axes):
        frac = np.array(per_layer[last_l][key])
        means, stds, counts = [], [], []
        for i in range(len(bins) - 1):
            mask = (frac >= bins[i]) & (frac < bins[i + 1])
            c    = last_conf[mask]
            means.append(c.mean() if len(c) > 0 else np.nan)
            stds.append(c.std()   if len(c) > 0 else np.nan)
            counts.append(len(c))

        ax.bar(bin_centers, means, width=0.08, color=color, alpha=0.75,
               yerr=stds, capsize=3, label="mean±std conf")
        ax2 = ax.twinx()
        ax2.plot(bin_centers, counts, "r-o", linewidth=1.5, label="count")
        ax2.set_ylabel("# GT objects")
        ax.set_xlabel(f"Fraction of pts in {label}")
        ax.set_ylabel("Mean confidence")
        ax.set_title(f"Confidence vs {label} fraction\n(L{last_l})")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")

    plt.tight_layout()
    path = os.path.join(out_dir, "conf_vs_regions_binned.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"[SAVED] {path}")

    # ── PLOT 4: crowded-only — overlap fraction across layers ─────────────────
    if crowded:
        crowded_per_layer = {l: defaultdict(list) for l in layers}
        for r in crowded:
            for l in layers:
                lr = r["layer_results"][l]
                crowded_per_layer[l]["overlap_unwtd"].append(lr["overlap_unwtd"])
                crowded_per_layer[l]["exclusive_own_unwtd"].append(lr["exclusive_own_unwtd"])
                crowded_per_layer[l]["exclusive_other_unwtd"].append(lr["exclusive_other_unwtd"])
                crowded_per_layer[l]["conf"].append(lr["conf"])

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(layers))
        for key, label, color in [
            ("exclusive_own_unwtd",   "Excl. Own",   "steelblue"),
            ("overlap_unwtd",         "Overlap",     "mediumpurple"),
            ("exclusive_other_unwtd", "Excl. Other", "darkorange"),
        ]:
            vals = [np.mean(crowded_per_layer[l][key]) for l in layers]
            ax.plot(x, vals, "-o", label=label, color=color)
        ax2 = ax.twinx()
        ax2.plot(x, [np.mean(crowded_per_layer[l]["conf"]) for l in layers],
                 "r--s", label="Mean conf", linewidth=2)
        ax2.set_ylabel("Mean confidence")
        ax.set_xticks(x); ax.set_xticklabels([f"L{l}" for l in layers])
        ax.set_ylabel("Mean unweighted fraction")
        ax.set_title(f"Crowded GT objects only (n={len(crowded)})\nRegion fractions across layers")
        ax.legend(loc="upper left"); ax2.legend(loc="upper right")
        plt.tight_layout()
        path = os.path.join(out_dir, "crowded_region_fractions_across_layers.png")
        plt.savefig(path, dpi=150); plt.close()
        print(f"[SAVED] {path}")

    # ── save JSON summary ─────────────────────────────────────────────────────
    def corr(a, b):
        a, b = np.array(a), np.array(b)
        if a.std() < 1e-9 or b.std() < 1e-9:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    summary = {
        "n_gt_objects":   len(all_results),
        "n_isolated":     len(isolated),
        "n_crowded":      len(crowded),
        "layers":         layers,
        "per_layer_means": {
            str(l): {
                "exclusive_own_unwtd":   float(np.mean(per_layer[l]["exclusive_own_unwtd"])),
                "exclusive_own_wtd":     float(np.mean(per_layer[l]["exclusive_own_wtd"])),
                "overlap_unwtd":         float(np.mean(per_layer[l]["overlap_unwtd"])),
                "overlap_wtd":           float(np.mean(per_layer[l]["overlap_wtd"])),
                "exclusive_other_unwtd": float(np.mean(per_layer[l]["exclusive_other_unwtd"])),
                "exclusive_other_wtd":   float(np.mean(per_layer[l]["exclusive_other_wtd"])),
                "background_unwtd":      float(np.mean(per_layer[l]["background_unwtd"])),
                "background_wtd":        float(np.mean(per_layer[l]["background_wtd"])),
                "mean_conf":             float(np.mean(per_layer[l]["conf"])),
                "partition_sum_check":   float(
                    np.mean(per_layer[l]["exclusive_own_unwtd"]) +
                    np.mean(per_layer[l]["overlap_unwtd"]) +
                    np.mean(per_layer[l]["exclusive_other_unwtd"]) +
                    np.mean(per_layer[l]["background_unwtd"])
                ),
            }
            for l in layers
        },
        "correlations_last_layer": {
            "excl_own_vs_conf":   corr(per_layer[last_l]["exclusive_own_unwtd"],   per_layer[last_l]["conf"]),
            "overlap_vs_conf":    corr(per_layer[last_l]["overlap_unwtd"],          per_layer[last_l]["conf"]),
            "excl_other_vs_conf": corr(per_layer[last_l]["exclusive_other_unwtd"], per_layer[last_l]["conf"]),
        },
    }
    json_path = os.path.join(out_dir, "sampling_bbox_analysis_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVED] {json_path}")

    print("\n  CORRELATION SUMMARY (last layer, unweighted fracs)")
    for k, v in summary["correlations_last_layer"].items():
        print(f"    {k:<30}: {v:+.4f}")
    print("=" * 90)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sampling location bbox set-theoretic analysis")
    parser.add_argument("--debug_dir",       required=True)
    parser.add_argument("--coco_json",       required=True)
    parser.add_argument("--out_dir",         default="./sampling_bbox_analysis_out")
    parser.add_argument("--last_layer_only", action="store_true")
    parser.add_argument("--max_images",      type=int, default=None)
    args = parser.parse_args()

    print(f"[INFO] Loading COCO GT from {args.coco_json} ...")
    with open(args.coco_json) as f:
        coco = json.load(f)

    img_info       = {img["file_name"]: img for img in coco["images"]}
    anns_by_img    = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    pt_files = sorted(glob.glob(os.path.join(args.debug_dir, "*.pt")))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {args.debug_dir}")
    if args.max_images:
        pt_files = pt_files[:args.max_images]
    print(f"[INFO] Found {len(pt_files)} .pt files")

    all_results, skipped = [], 0

    for pt_path in tqdm(pt_files, desc="Images"):
        data         = torch.load(pt_path, map_location="cpu")
        img_basename = os.path.basename(str(data.get("img_path", "")))

        if img_basename in img_info:
            iinfo = img_info[img_basename]
        else:
            matched = [v for k, v in img_info.items() if img_basename in k]
            if matched:
                iinfo = matched[0]
            else:
                skipped += 1
                continue

        gt_anns = anns_by_img.get(iinfo["id"], [])
        if not gt_anns:
            continue

        img_results = analyse_image(
            data, gt_anns, iinfo["width"], iinfo["height"],
            last_layer_only=args.last_layer_only
        )
        all_results.extend(img_results)

    print(f"[INFO] Processed {len(all_results)} GT objects | skipped {skipped} images")

    if not all_results:
        print("[ERROR] No results. Check img_path in .pt files matches COCO filenames.")
        return

    aggregate_and_plot(all_results, args.out_dir, args.last_layer_only)


if __name__ == "__main__":
    main()