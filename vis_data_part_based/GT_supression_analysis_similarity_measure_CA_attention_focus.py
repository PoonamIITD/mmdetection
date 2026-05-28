"""
gt_suppression_analysis.py  (v3 — sampled_features similarity)
===============================================================
All original functionality preserved. Feature similarity now uses
sampled_features [decoder_L, Q, C] directly — the weighted sum of
image features at each query's sampling locations. This is the
correct representation for testing the feature orthogonality
hypothesis, as it captures exactly what each query extracted from
the image at each decoder layer.

Three similarity measures are computed per query pair per layer:

  1. sampled_feat_sim  — cosine similarity between sampled_features
                         vectors. DIRECTLY tests whether two queries
                         extracted the same image content.
                         THIS is the primary metric.

  2. attn_dist_sim     — dot product of normalised attention weight
                         distributions. Tests whether queries attend
                         to the same sampling points with the same
                         weights (HOW they look, not WHAT they see).

  3. attn_kl_div       — KL divergence of attention distributions.
                         High = distributions are different = queries
                         attend to different points.

  4. cls_score_sim     — cosine similarity of classification logits.
                         Kept for completeness but least informative.

Hypothesis under test
---------------------
  Group A (suppressed) should have HIGHER sampled_feat_sim than
  Group B (not suppressed) at the same spatial overlap level.

  If true  → both queries extract the same image features from
             shared regions → one dominates → suppression occurs
           → Feature orthogonality loss is directly motivated

  If false → suppression is not caused by feature co-adaptation
           → Focus on reference point anchoring instead

Usage
-----
python gt_suppression_analysis_v3.py \
    --gt_json        val_annotations.json \
    --pairs_json     occlusion_results.json \
    [--fn_json       fn_detections.json]   \
    --pt_dir         debug_outputs/        \
    [--occlusion_ratio  0.3] \
    [--output_json   gt_group_analysis.json]
"""

import json
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path
import math

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] torch not available — feature similarity will be skipped.")


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
# SUPPRESSION CONDITIONS
# ============================================================

def occluded_not_maintained(scores, eps=1e-6):
    end  = scores[-1]
    peak = max(scores)
    return (peak > end + eps) or (end < scores[0] - eps)


def occluder_increasing(scores, eps=1e-6):
    return scores[-1] > scores[0] + eps


# ============================================================
# BUILD FOUR-WAY INDEX FROM PAIRS JSON
# ============================================================

def build_pair_indices(pairs):
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
# FEATURE SIMILARITY  (v3 — uses sampled_features directly)
# ============================================================

def cosine_sim(a, b):
    """Cosine similarity between two 1-D float tensors."""
    a = a.float()
    b = b.float()
    return (F.normalize(a, dim=0) * F.normalize(b, dim=0)).sum().item()


def kl_div_sym(p, q, eps=1e-9):
    """
    Symmetric KL divergence between two distributions.
    Both p and q are normalised to sum to 1 before computation.
    """
    p = p.float(); q = q.float()
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    kl_pq = (p * ((p + eps) / (q + eps)).log()).sum().item()
    kl_qp = (q * ((q + eps) / (p + eps)).log()).sum().item()
    return (kl_pq + kl_qp) / 2.0

def compute_reference_drift(pt_dir, suppressed_ids, not_suppressed_ids, pairs_by_image, images, anns_by_id):
    """
    Compute reference point drift metrics for occluded queries.
    Returns raw metrics for Group A (suppressed) and Group B (unsuppressed).
    """
    drift_A = {"dist_to_gt": defaultdict(list), "dist_to_init": defaultdict(list), "step_displacement": defaultdict(list)}
    drift_B = {"dist_to_gt": defaultdict(list), "dist_to_init": defaultdict(list), "step_displacement": defaultdict(list)}
    pt_dir = Path(pt_dir)
    for pt_file in pt_dir.glob("*.pt"):
        data = torch.load(pt_file, map_location="cpu")
        img_name = Path(data["img_path"]).name
        # Determine image dimensions for absolute coordinates:
        # Find image entry by matching name (COCO images typically have unique file names):
        # If multiple images share name, might use image_id instead if available in pairs_by_image.
        img_info_list = [img for img in images.values() if img["file_name"] == img_name]
        if not img_info_list:
            continue  # no image info (shouldn't happen if file names match)
        img_info = img_info_list[0]
        img_w, img_h = img_info.get("width", 1), img_info.get("height", 1)
        # If image name not in pairs_by_image or no queries for this image, skip:
        if img_name not in pairs_by_image:
            continue
        # For each pair in this image (to get occluded queries interested):
        for p in pairs_by_image[img_name]:
            aid = p["occluded_gt_ann_id"]
            if aid not in suppressed_ids and aid not in not_suppressed_ids:
                continue  # skip queries not in our Group A or B sets
            group = drift_A if aid in suppressed_ids else drift_B
            q_idx = p["occluded_query"]
            # Get ground-truth center for occluded object:
            ann = anns_by_id.get(aid)
            if not ann:
                continue
            x, y, w, h = ann["bbox"]
            gt_cx, gt_cy = x + 0.5 * w, y + 0.5 * h
            # Calculate reference point positions per layer:
            ref_pts = data["reference_points"]  # shape [L+1, Q, 4]
            Lp1 = ref_pts.shape[0]  # number of reference point sets (layers + initial)
            # Reference points likely normalized within [0,1] for spatial coordinates:
            for l in range(Lp1):
                cx_norm = float(ref_pts[l, q_idx, 0])  # normalized center x
                cy_norm = float(ref_pts[l, q_idx, 1])  # normalized center y
                cx = cx_norm * img_w
                cy = cy_norm * img_h
                # Distance to GT center (in pixel units):
                dist_gt = math.hypot(cx - gt_cx, cy - gt_cy)
                # Distance to initial reference (layer 0):
                if l == 0:
                    dist_init = 0.0
                else:
                    init_cx_norm = float(ref_pts[0, q_idx, 0])
                    init_cy_norm = float(ref_pts[0, q_idx, 1])
                    init_cx = init_cx_norm * img_w
                    init_cy = init_cy_norm * img_h
                    dist_init = math.hypot(cx - init_cx, cy - init_cy)
                group["dist_to_gt"][l].append(dist_gt)
                group["dist_to_init"][l].append(dist_init)
                # For step displacement: if not initial layer:
                if l > 0:
                    prev_cx = float(ref_pts[l-1, q_idx, 0]) * img_w
                    prev_cy = float(ref_pts[l-1, q_idx, 1]) * img_h
                    step_d = math.hypot(cx - prev_cx, cy - prev_cy)
                    group["step_displacement"][l].append(step_d)
    return drift_A, drift_B

def compute_attention_focus(pt_dir, pairs_by_image, suppressed_ids, not_suppressed_ids, images, anns_by_id):
    """
    Compute attention focus fractions for occluded queries:
    Fraction of cross-attention weight inside target (occluded GT) bbox, inside occluder GT bbox, and outside both.
    Returns raw metrics for Group A and Group B.
    """
    focus_A = defaultdict(lambda: defaultdict(list))
    focus_B = defaultdict(lambda: defaultdict(list))
    pt_dir = Path(pt_dir)
    for pt_file in pt_dir.glob("*.pt"):
        data = torch.load(pt_file, map_location="cpu")
        img_name = Path(data["img_path"]).name
        if img_name not in pairs_by_image:
            continue
        # Prepare for quick coordinate -> absolute conversion:
        # We will assume sampling_locations are normalized [0,1] in image coordinates
        # Derive image width and height similarly as in compute_reference_drift:
        img_info_list = [img for img in images.values() if img["file_name"] == img_name]
        if not img_info_list:
            continue
        img_info = img_info_list[0]
        img_w, img_h = img_info.get("width", 1), img_info.get("height", 1)
        # For each occlusion pair in this image:
        for p in pairs_by_image[img_name]:
            aid = p["occluded_gt_ann_id"]
            if aid not in suppressed_ids and aid not in not_suppressed_ids:
                continue
            group = focus_A if aid in suppressed_ids else focus_B
            q_occ = p["occluded_query"]
            # Retrieve GT boxes for occluded target and occluder:
            ann_occ = anns_by_id.get(aid)
            occ_box = xywh_to_xyxy(ann_occ["bbox"]) if ann_occ else None
            occluder_ann = None
            if "occluder_gt_ann_id" in p:
                occluder_ann = anns_by_id.get(p["occluder_gt_ann_id"])
            # If occluder GT missing (which could happen if occluder not labeled or missed in GT):
            # we skip since we can't compute fractions properly.
            if occ_box is None or occluder_ann is None:
                continue
            occder_box = xywh_to_xyxy(occluder_ann["bbox"])
            # Coordinates expected (x1,y1,x2,y2)
            # Set up cross-attention data:
            attn_w = data["attention_weights"]  # [L, Q, H, Lv, P]
            samp_loc = data["sampling_locations"]  # [L, Q, H, Lv, P, 2]
            L, _, H, Lv, P = attn_w.shape
            # Flatten head and level dims for easier iteration or sum:
            # We will sum weights for each layer over all heads and levels:
            attn_w = attn_w[:, q_occ]  # shape [L, H, Lv, P]
            samp_loc = samp_loc[:, q_occ]  # shape [L, H, Lv, P, 2]
            for l in range(L):
                total_w = 0.0
                target_w = 0.0
                occluder_w = 0.0
                # iterate over all heads, levels, points:
                for h in range(H):
                    for lv in range(Lv):
                        for pidx in range(P):
                            w = float(attn_w[l, h, lv, pidx])
                            total_w += w
                            # Coordinates (assuming normalized relative to original image):
                            x_norm = float(samp_loc[l, h, lv, pidx, 0])
                            y_norm = float(samp_loc[l, h, lv, pidx, 1])
                            px = x_norm * img_w
                            py = y_norm * img_h
                            inside_occ = (px >= occ_box[0] and px <= occ_box[2] and py >= occ_box[1] and py <= occ_box[3])
                            inside_occder = (px >= occder_box[0] and px <= occder_box[2] and py >= occder_box[1] and py <= occder_box[3])
                            # Classify attention point: if inside occluder (including overlap) -> occluder; elif inside target only -> target; else -> background
                            if inside_occder:
                                occluder_w += w
                            elif inside_occ:
                                target_w += w
                            # else: do nothing for background (background_w = total_w - target_w - occluder_w)
                # Normalize fractions by total weight (should be ~1 across heads/levels for that query if fully normalized)
                if total_w <= 0:
                    continue
                target_frac = target_w / total_w
                occluder_frac = occluder_w / total_w
                bg_frac = 1.0 - target_frac - occluder_frac
                group[l]["target_frac"].append(target_frac)
                group[l]["occluder_frac"].append(occluder_frac)
                group[l]["background_frac"].append(bg_frac)
    return dict(focus_A), dict(focus_B)

def summarise_focus(focus_group_data):
    """Aggregate focus fractions per group into means per layer."""
    summary = {}
    if not focus_group_data:
        return summary
    layers = sorted(focus_group_data.keys(), key=int)
    summary["mean_target_frac_per_layer"] = [round(float(np.mean(focus_group_data[l]["target_frac"])), 6) for l in layers]
    summary["mean_occluder_frac_per_layer"] = [round(float(np.mean(focus_group_data[l]["occluder_frac"])), 6) for l in layers]
    summary["mean_background_frac_per_layer"] = [round(float(np.mean(focus_group_data[l]["background_frac"])), 6) for l in layers]
    # We can also include final-layer fractions and initial-layer fractions for quick comparison:
    last_layer = layers[-1]
    first_layer = layers[0]
    summary["initial_target_frac"] = round(float(np.mean(focus_group_data[first_layer]["target_frac"])), 6)
    summary["final_target_frac"] = round(float(np.mean(focus_group_data[last_layer]["target_frac"])), 6)
    summary["initial_occluder_frac"] = round(float(np.mean(focus_group_data[first_layer]["occluder_frac"])), 6)
    summary["final_occluder_frac"] = round(float(np.mean(focus_group_data[last_layer]["occluder_frac"])), 6)
    return summary

def summarise_drift(drift_group_data):
    """Aggregate reference drift metrics per group into mean and std values."""
    summary = {}
    if not drift_group_data:
        return summary
    # For each metric, compute mean per layer and overall statistics.
    # Distance to ground truth center:
    if "dist_to_gt" in drift_group_data:
        layers = sorted(drift_group_data["dist_to_gt"].keys(), key=int)
        summary["mean_dist_to_gt_per_layer"] = [round(float(np.mean(drift_group_data["dist_to_gt"][l])), 4) for l in layers]
        # Particularly highlight final distance:
        final_layer = layers[-1]
        initial_layer = layers[0]
        summary["final_dist_to_gt_mean"] = round(float(np.mean(drift_group_data["dist_to_gt"][final_layer])), 4)
        summary["initial_dist_to_gt_mean"] = round(float(np.mean(drift_group_data["dist_to_gt"][initial_layer])), 4)
    if "dist_to_init" in drift_group_data:
        layers = sorted(drift_group_data["dist_to_init"].keys(), key=int)
        summary["mean_dist_to_init_per_layer"] = [round(float(np.mean(drift_group_data["dist_to_init"][l])), 4) for l in layers]
        summary["final_dist_from_initial_mean"] = round(float(np.mean(drift_group_data["dist_to_init"][layers[-1]])), 4)
    if "step_displacement" in drift_group_data:
        steps = sorted(drift_group_data["step_displacement"].keys(), key=int)
        summary["mean_step_displacement"] = [round(float(np.mean(drift_group_data["step_displacement"][s])), 4) for s in steps]
    return summary

def compute_feature_similarity_for_pairs(pt_dir, pairs,
                                          suppressed_ids,
                                          not_suppressed_ids):
    """
    For each pair (occluded_query, occluder_query), loads the
    corresponding .pt file and computes four similarity measures
    at every decoder layer:

    PRIMARY
    -------
    sampled_feat_sim  : cosine similarity between sampled_features[l, qi]
                        and sampled_features[l, qj].
                        sampled_features[l, q] is the weighted sum of
                        image feature values at query q's sampling
                        locations at layer l. Shape per query: [C].
                        This is the DIRECT measurement of whether two
                        queries extracted the same image content.

    SECONDARY
    ---------
    attn_dist_sim     : dot product of normalised attention weight
                        distributions [H*Lv*P]. Measures whether
                        queries attend to the same points with the
                        same weights (pattern similarity).

    attn_kl_div       : symmetric KL divergence of attention
                        distributions. Higher = more different =
                        more discriminative attention patterns.

    cls_score_sim     : cosine similarity of classification logit
                        vectors. Measures class-prediction similarity.
                        Least informative for feature analysis but
                        kept for completeness.

    Returns
    -------
    results_A  : dict  layer_idx -> dict of lists  (Group A)
    results_B  : dict  layer_idx -> dict of lists  (Group B)
    per_pair_A : list of per-pair dicts  (Group A)
    per_pair_B : list of per-pair dicts  (Group B)
    """
    if not TORCH_AVAILABLE:
        print("[WARN] torch not available.")
        return {}, {}, [], []

    pt_dir = Path(pt_dir)

    # build image_name -> pairs lookup
    pairs_by_image = defaultdict(list)
    for p in pairs:
        pairs_by_image[p["image_name"]].append(p)

    # per-layer accumulators
    # each entry: layer -> {metric_name: [values]}
    def new_acc():
        return defaultdict(lambda: defaultdict(list))

    results_A = new_acc()
    results_B = new_acc()
    per_pair_A = []
    per_pair_B = []

    n_processed = 0
    n_skipped   = 0

    for pt_file in sorted(pt_dir.glob("*.pt")):
        data     = torch.load(pt_file, map_location="cpu")
        img_name = Path(data["img_path"]).name

        image_pairs = pairs_by_image.get(img_name, [])
        if not image_pairs:
            continue

        # check required tensors
        if "sampled_features" not in data:
            print(f"[WARN] sampled_features missing in {pt_file.name} — skipping.")
            n_skipped += 1
            continue

        sf   = data["sampled_features"].float()   # [L, Q, C]
        aw   = data["attention_weights"].float()  # [L, Q, H, Lv, P]
        cs   = data["cls_scores"].float()         # [L, Q, C_cls]

        L, Q, C_sf   = sf.shape
        _, _, H, Lv, P = aw.shape

        n_processed += 1

        for p in image_pairs:
            qi  = p["occluded_query"]
            qj  = p["occluder_query"]
            aid = p["occluded_gt_ann_id"]

            if qi >= Q or qj >= Q:
                continue

            # route to correct group
            if aid in suppressed_ids:
                target     = results_A
                pair_store = per_pair_A
            elif aid in not_suppressed_ids:
                target     = results_B
                pair_store = per_pair_B
            else:
                continue

            sf_sims    = []
            attn_sims  = []
            attn_kls   = []
            cls_sims   = []

            for l in range(L):

                # ── PRIMARY: sampled feature similarity ──────
                feat_i = sf[l, qi]   # [C]
                feat_j = sf[l, qj]   # [C]
                sf_sim = cosine_sim(feat_i, feat_j)

                # ── attention distribution similarity ─────────
                wi = aw[l, qi].reshape(-1)   # [H*Lv*P]
                wj = aw[l, qj].reshape(-1)

                wi_norm = wi / (wi.sum() + 1e-9)
                wj_norm = wj / (wj.sum() + 1e-9)
                attn_dot = (wi_norm * wj_norm).sum().item()
                attn_kl  = kl_div_sym(wi, wj)

                # ── cls score similarity ──────────────────────
                cls_sim = cosine_sim(cs[l, qi], cs[l, qj])

                sf_sims.append(round(sf_sim,   6))
                attn_sims.append(round(attn_dot, 6))
                attn_kls.append(round(attn_kl,  6))
                cls_sims.append(round(cls_sim,  6))

                target[l]["sampled_feat_sim"].append(sf_sim)
                target[l]["attn_dist_sim"].append(attn_dot)
                target[l]["attn_kl_div"].append(attn_kl)
                target[l]["cls_score_sim"].append(cls_sim)

            pair_store.append({
                "image":              img_name,
                "occluded_query":     qi,
                "occluder_query":     qj,
                "ann_id":             aid,
                "sampled_feat_sim":   sf_sims,
                "attn_dist_sim":      attn_sims,
                "attn_kl_div":        attn_kls,
                "cls_score_sim":      cls_sims,
                "mean_sf_sim":        round(float(np.mean(sf_sims)), 6),
                "mean_attn_sim":      round(float(np.mean(attn_sims)), 6),
                "sf_sim_delta":       round(sf_sims[-1] - sf_sims[0], 6),
            })

    print(f"  [Feature sim] pt files processed : {n_processed}")
    if n_skipped:
        print(f"  [Feature sim] pt files skipped   : {n_skipped} "
              f"(sampled_features missing)")

    return dict(results_A), dict(results_B), per_pair_A, per_pair_B


def summarise_similarity(results_by_layer):
    """
    Aggregate per-layer similarity statistics.

    Returns dict with per-layer means/stds and overall summary
    for each metric (sampled_feat_sim, attn_dist_sim, etc.)
    """
    if not results_by_layer:
        return {}

    layers  = sorted(results_by_layer.keys())
    metrics = list(results_by_layer[layers[0]].keys())
    n_pairs = len(results_by_layer[layers[0]][metrics[0]])

    summary = {"n_pairs": n_pairs}

    for metric in metrics:
        per_layer_means = []
        per_layer_stds  = []

        for l in layers:
            vals = results_by_layer[l][metric]
            per_layer_means.append(round(float(np.mean(vals)), 6))
            per_layer_stds.append(round(float(np.std(vals)),  6))

        all_vals = [v for l in layers for v in results_by_layer[l][metric]]

        summary[metric] = {
            "mean_per_layer":     per_layer_means,
            "std_per_layer":      per_layer_stds,
            "overall_mean":       round(float(np.mean(all_vals)), 6),
            "overall_std":        round(float(np.std(all_vals)),  6),
            "delta_L0_to_last":   round(per_layer_means[-1] - per_layer_means[0], 6),
            "pct_high_sim":       round(float((np.array(all_vals) > 0.7).mean() * 100), 2)
                                  if "sim" in metric else None,
            "pct_low_sim":        round(float((np.array(all_vals) < 0.3).mean() * 100), 2)
                                  if "sim" in metric else None,
        }

    return summary


def print_similarity_report(stats_A, stats_B):
    """
    Print the feature similarity comparison and interpret the result.
    """
    dash = "-" * 64
    print(f"\n{dash}")
    print(f"  FEATURE SIMILARITY ANALYSIS")
    print(f"  Primary metric: sampled_features cosine similarity")
    print(f"  (direct measurement of extracted image content per query)")
    print(f"{dash}")

    if not stats_A or not stats_B:
        print("  [No data — supply --pt_dir to enable]")
        return

    # ── per-metric report ─────────────────────────────────
    metric_labels = {
        "sampled_feat_sim": "Sampled feature sim  [PRIMARY]",
        "attn_dist_sim":    "Attention dist sim   [secondary]",
        "attn_kl_div":      "Attention KL div     [secondary — higher=more different]",
        "cls_score_sim":    "Cls score sim        [secondary]",
    }

    gaps = {}

    for metric, label in metric_labels.items():
        if metric not in stats_A or metric not in stats_B:
            continue

        mA = stats_A[metric]
        mB = stats_B[metric]
        gap = round(mA["overall_mean"] - mB["overall_mean"], 6)
        gaps[metric] = gap

        print(f"\n  [{label}]")
        print(f"    Group A  overall mean : {mA['overall_mean']:.4f}  "
              f"(std={mA['overall_std']:.4f})  n={stats_A['n_pairs']}")
        print(f"    Group B  overall mean : {mB['overall_mean']:.4f}  "
              f"(std={mB['overall_std']:.4f})  n={stats_B['n_pairs']}")
        print(f"    Gap (A − B)           : {gap:+.4f}")

        if mA.get("pct_high_sim") is not None:
            print(f"    Group A  % high (>0.7): {mA['pct_high_sim']:.1f}%  "
                  f"% low (<0.3): {mA['pct_low_sim']:.1f}%")
            print(f"    Group B  % high (>0.7): {mB['pct_high_sim']:.1f}%  "
                  f"% low (<0.3): {mB['pct_low_sim']:.1f}%")

        # per-layer bar
        print(f"    Per-layer gap (A − B):")
        for l, (a, b) in enumerate(zip(mA["mean_per_layer"],
                                        mB["mean_per_layer"])):
            bar = "+" * max(0, int((a - b) * 100))
            bar = bar if a >= b else "-" * max(0, int((b - a) * 100))
            print(f"      L{l}: A={a:.4f}  B={b:.4f}  "
                  f"gap={a-b:+.4f}  {bar}")

    # ── interpretation of PRIMARY metric ─────────────────
    sf_gap = gaps.get("sampled_feat_sim", 0)
    kl_gap = gaps.get("attn_kl_div", 0)    # negative gap = A has LOWER KL = more similar attention

    print(f"\n{dash}")
    print(f"  INTERPRETATION  (based on sampled_feat_sim gap = {sf_gap:+.4f})")
    print(f"{dash}")

    if sf_gap > 0.05:
        print(f"  CONFIRMS feature orthogonality hypothesis.")
        print(f"  Group A queries extract MORE similar image content ({sf_gap:+.4f} gap).")
        print(f"  Both queries are attending to and extracting the same features")
        print(f"  from the shared spatial region — the occluder's features dominate")
        print(f"  both queries, causing one to suppress the other.")
        print(f"")
        print(f"  RECOMMENDED LOSSES:")
        print(f"  1. Feature orthogonality loss (PRIMARY)")
        print(f"     Penalise cosine similarity between sampled_features[l,qi]")
        print(f"     and sampled_features[l,qj] for spatially overlapping pairs.")
        print(f"     Push them toward orthogonal feature extraction.")
        print(f"  2. Attention diversity loss (SECONDARY)")
        print(f"     KL divergence between attention distributions.")
        print(f"     Encourages queries to attend to different sampling points.")
        print(f"  3. Reference anchoring loss (TERTIARY)")
        print(f"     Keep sampling points close to own reference point.")

    elif sf_gap > 0.01:
        print(f"  WEAK support for feature orthogonality hypothesis.")
        print(f"  Small gap ({sf_gap:+.4f}) — feature similarity slightly higher for A.")
        print(f"  Feature co-adaptation is a contributing factor but not the")
        print(f"  primary cause of suppression.")
        print(f"")
        print(f"  RECOMMENDED LOSSES:")
        print(f"  1. Reference anchoring loss (PRIMARY)")
        print(f"     Spatial drift is likely the main problem.")
        print(f"  2. Feature orthogonality loss (SECONDARY — mild weight)")

    else:
        print(f"  DOES NOT confirm feature orthogonality hypothesis.")
        print(f"  Gap ({sf_gap:+.4f}) is negligible — suppressed and non-suppressed")
        print(f"  queries extract equally similar/different features.")
        print(f"  Feature co-adaptation is NOT the cause of suppression.")
        print(f"")
        print(f"  RECOMMENDED LOSSES:")
        print(f"  1. Reference anchoring loss (PRIMARY)")
        print(f"     Sampling point drift is the likely mechanism.")
        print(f"  2. Monotonicity regularization (SECONDARY)")
        print(f"     Force confidence to not decrease across decoder layers.")
        print(f"  Do NOT implement feature orthogonality loss — not motivated.")


# ============================================================
# TRAJECTORY STATISTICS  (unchanged)
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
        "std_net_delta":           round(float(deltas.std()),  4),
        "pct_net_decrease":        round(float((deltas < 0).mean() * 100), 2),
        "pct_net_increase":        round(float((deltas > 0).mean() * 100), 2),
        "peak_layer_distribution": {
            str(l): int((peak_layers == l).sum()) for l in range(arr.shape[1])
        },
        "pct_peak_at_L0":          round(float((peak_layers == 0).mean() * 100), 2),
        "pct_peak_at_last":        round(float((peak_layers == arr.shape[1]-1).mean() * 100), 2),
        "pct_recovered":           round(
            float(((arr[:, -1] - arr.min(axis=1)) /
                   (arr.max(axis=1) - arr.min(axis=1) + 1e-6) > 0.5).mean() * 100), 2
        ),
    }


# ============================================================
# SAMPLING OVERLAP HELPERS  (unchanged)
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
# PRINT HELPERS  (unchanged)
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
    if not ov or ov.get("mean_unweighted") is None:
        return
    print(f"\n  [{label}]")
    print(f"    Mean unweighted       : {ov['mean_unweighted']:.4f}")
    if ov.get("mean_attn_weighted") is not None:
        print(f"    Mean attn-weighted    : {ov['mean_attn_weighted']:.6f}")
    if ov["per_layer_unweighted"]:
        row = "  ".join(f"L{i}:{v:.4f}"
                        for i, v in enumerate(ov["per_layer_unweighted"]))
        print(f"    Per-layer unweighted  : {row}")
    if ov.get("per_layer_attn_weighted"):
        row = "  ".join(f"L{i}:{v:.6f}"
                        for i, v in enumerate(ov["per_layer_attn_weighted"]))
        print(f"    Per-layer attn-wtd    : {row}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_json",         required=True)
    parser.add_argument("--pairs_json",      required=True)
    parser.add_argument("--fn_json",         default=None)
    parser.add_argument("--pt_dir",          default=None,
                        help="Directory of .pt files containing sampled_features "
                             "[L, Q, C], attention_weights [L, Q, H, Lv, P], "
                             "and cls_scores [L, Q, C].")
    parser.add_argument("--occlusion_ratio", type=float, default=0.3)
    parser.add_argument("--output_json",     default="gt_group_analysis.json")
    args = parser.parse_args()

    # ── load ─────────────────────────────────────────────
    print("[INFO] Loading files...")
    gt_images, gt_anns_by_id, gt_anns_per_image, _ = load_coco(args.gt_json)
    pairs = load_pairs(args.pairs_json)
    pairs_by_image = defaultdict(list)
    for p in pairs:
        if "image_name" in p:
            pairs_by_image[p["image_name"]].append(p)
        else:
            print("Error occured while reading the pairs_by_image")
        
    print(f"  GT annotations           : {len(gt_anns_by_id)}")
    print(f"  Occlusion pairs loaded   : {len(pairs)}")

    fn_ann_ids = set()
    if args.fn_json:
        _, fn_anns_by_id, _, _ = load_coco(args.fn_json)
        fn_ann_ids = set(fn_anns_by_id.keys())
        print(f"  FN annotations (optional): {len(fn_ann_ids)}")

    # ── build indices ─────────────────────────────────────
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

    print(f"\n[INFO] Finding geometrically occluded GT objects (for B2)...")
    gt_occluded_ids = find_gt_occluded_ann_ids(
        gt_anns_per_image, args.occlusion_ratio
    )
    b2_gt_ids  = gt_occluded_ids - all_in_pairs_ann_ids
    group_C_gt = (set(gt_anns_by_id.keys())
                  - gt_occluded_ids - all_in_pairs_ann_ids)

    print(f"  GT-occluded (ratio>={args.occlusion_ratio}) : {len(gt_occluded_ids)}")
    print(f"  Of which in pairs list         : "
          f"{len(gt_occluded_ids & all_in_pairs_ann_ids)}")
    print(f"  Group B2 pool                  : {len(b2_gt_ids)}")
    print(f"  Group C  pool                  : {len(group_C_gt)}")

    A_tp_ids = suppressed_ann_ids - fn_ann_ids
    A_fn_ids = suppressed_ann_ids & fn_ann_ids

    def get_trajs(ann_id_iter, key="occluded_scores_per_layer"):
        result = []
        for aid in ann_id_iter:
            p = pair_by_ann_id.get(aid)
            if p and p.get(key) is not None:
                result.append(p[key])
        return result

    stats = {
        "A":           traj_stats(get_trajs(suppressed_ann_ids)),
        "A_tp":        traj_stats(get_trajs(A_tp_ids)) if fn_ann_ids else {},
        "A_fn":        traj_stats(get_trajs(A_fn_ids)) if fn_ann_ids else {},
        "B":           traj_stats(get_trajs(not_suppressed_ann_ids)),
        "occluder_A":  traj_stats(get_trajs(suppressed_ann_ids,
                                             "occluder_scores_per_layer")),
        "occluder_B":  traj_stats(get_trajs(not_suppressed_ann_ids,
                                             "occluder_scores_per_layer")),
        "occluder_all": traj_stats([
            p["occluder_scores_per_layer"] for p in pairs
            if p.get("occluder_scores_per_layer")
        ]),
    }

    pairs_A    = [pair_by_ann_id[a] for a in suppressed_ann_ids
                  if a in pair_by_ann_id]
    pairs_A_tp = [pair_by_ann_id[a] for a in A_tp_ids
                  if a in pair_by_ann_id]
    pairs_A_fn = [pair_by_ann_id[a] for a in A_fn_ids
                  if a in pair_by_ann_id]
    pairs_B    = [pair_by_ann_id[a] for a in not_suppressed_ann_ids
                  if a in pair_by_ann_id]
    ov = {
        "A":    overlap_stats(pairs_A),
        "A_tp": overlap_stats(pairs_A_tp) if fn_ann_ids else {},
        "A_fn": overlap_stats(pairs_A_fn) if fn_ann_ids else {},
        "B":    overlap_stats(pairs_B),
        "all":  overlap_stats(list(pair_by_ann_id.values())),
    }

    # ── feature similarity ────────────────────────────────
    sim_stats_A, sim_stats_B = {}, {}
    per_pair_A_sim, per_pair_B_sim = [], []
    focus_stats_A = focus_stats_B = {}
    drift_stats_A = drift_stats_B = {}

    if args.pt_dir:
        print(f"\n[INFO] Computing feature similarity from {args.pt_dir} ...")
        raw_A, raw_B, per_pair_A_sim, per_pair_B_sim = \
            compute_feature_similarity_for_pairs(
                args.pt_dir, pairs,
                suppressed_ann_ids,
                not_suppressed_ann_ids,
            )
        sim_stats_A = summarise_similarity(raw_A)
        sim_stats_B = summarise_similarity(raw_B)
        print("\n[INFO] Computing additional diagnostic metrics...")
        raw_focus_A, raw_focus_B = {}, {}
        raw_drift_A = raw_drift_B = {}
        # Compute attention focus fractions:
        focus_data_A, focus_data_B = compute_attention_focus(args.pt_dir, pairs_by_image, suppressed_ann_ids, not_suppressed_ann_ids, gt_images, gt_anns_by_id)
        focus_stats_A = summarise_focus(focus_data_A)
        focus_stats_B = summarise_focus(focus_data_B)
        # Compute reference point drift:
        drift_data_A, drift_data_B = compute_reference_drift(args.pt_dir, suppressed_ann_ids, not_suppressed_ann_ids, pairs_by_image, gt_images, gt_anns_by_id)
        drift_stats_A = summarise_drift(drift_data_A)
        drift_stats_B = summarise_drift(drift_data_B)

    else:
        print("\n[INFO] --pt_dir not supplied — skipping feature similarity.")
        print("       Add --pt_dir <path> to enable.")

    # ── print report ─────────────────────────────────────
    total_gt = len(gt_anns_by_id)
    sep  = "=" * 64
    dash = "-" * 64

    print(f"\n{sep}")
    print(f"  GT-WIDE SUPPRESSION ANALYSIS  (v3)")
    print(sep)

    print(f"\n  Total GT objects               : {total_gt:>6}")

    print(f"\n{dash}")
    print(f"  GROUP SIZES  (GT denominators)")
    print(f"{dash}")
    print(f"  A   suppressed (pairs, cond MET)     : "
          f"{len(suppressed_ann_ids):>6}")
    if fn_ann_ids:
        print(f"    A_tp  suppressed + TP (recovered)  : {len(A_tp_ids):>6}")
        print(f"    A_fn  suppressed + FN (missed)     : {len(A_fn_ids):>6}")
    print(f"  B   in pairs, cond NOT met           : "
          f"{len(not_suppressed_ann_ids):>6}")
    print(f"  B2  geom. occluded, not in pairs     : {len(b2_gt_ids):>6}")
    print(f"  C   not occluded (clean baseline)    : {len(group_C_gt):>6}")
    accounted = (len(suppressed_ann_ids) + len(not_suppressed_ann_ids)
                 + len(b2_gt_ids) + len(group_C_gt))
    print(f"  Unaccounted (?)                      : {total_gt - accounted:>6}")

    print(f"\n{dash}")
    print(f"  CONFIDENCE TRAJECTORY STATS  (occluded query)")
    print(f"{dash}")
    print_traj("Group A  — all suppressed",                stats["A"])
    if fn_ann_ids:
        print_traj("Group A_tp — suppressed + TP (recovered)", stats["A_tp"])
        print_traj("Group A_fn — suppressed + FN (missed)",    stats["A_fn"])
    print_traj("Group B  — in pairs, not suppressed",      stats["B"])

    print(f"\n{dash}")
    print(f"  CONFIDENCE TRAJECTORY STATS  (occluder query)")
    print(f"{dash}")
    print_traj("Occluder in Group A pairs", stats["occluder_A"])
    print_traj("Occluder in Group B pairs", stats["occluder_B"])
    print_traj("Occluder all pairs",        stats["occluder_all"])

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
              f"({'A drops more' if d_AB < 0 else 'B drops more'})")

    if fn_ann_ids and stats["A_tp"] and stats["A_fn"]:
        d_tp_fn = safe_delta(stats["A_tp"], stats["A_fn"], "mean_net_delta")
        if d_tp_fn is not None:
            print(f"  A_tp net_delta − A_fn net_delta     : {d_tp_fn:+.4f}")
        tp_rec = stats["A_tp"].get("pct_recovered", 0)
        fn_rec = stats["A_fn"].get("pct_recovered", 0)
        print(f"  % recovered — A_tp: {tp_rec:.1f}%  "
              f"A_fn: {fn_rec:.1f}%  (gap={tp_rec-fn_rec:+.1f}pp)")
        print(f"  Peak at L0  — A_tp: "
              f"{stats['A_tp'].get('pct_peak_at_L0', 0):.1f}%  "
              f"A_fn: {stats['A_fn'].get('pct_peak_at_L0', 0):.1f}%")

    print(f"\n{dash}")
    print(f"  SAMPLING OVERLAP")
    print(f"{dash}")
    print_overlap("Group A  — all suppressed",   ov["A"])
    if fn_ann_ids:
        print_overlap("Group A_tp — supp + TP",  ov["A_tp"])
        print_overlap("Group A_fn — supp + FN",  ov["A_fn"])
    print_overlap("Group B  — not suppressed",   ov["B"])
    print_overlap("All pairs",                   ov["all"])

    # feature similarity report
    print_similarity_report(sim_stats_A, sim_stats_B)

    print(f"\n{sep}\n")

    # Extend printed output to include new metrics:
    print("\n----------------------------------------------------------------")
    print("  ADDITIONAL DIAGNOSTIC METRICS")
    print("----------------------------------------------------------------")
    # Reference drift summary:
    if drift_stats_A and drift_stats_B:
        print(f"\n  Reference-Point Drift (avg distance in pixels):")
        # Example output: At initial layer and final layer for each group:
        initA = drift_stats_A.get("initial_dist_to_gt_mean"); finalA = drift_stats_A.get("final_dist_to_gt_mean")
        initB = drift_stats_B.get("initial_dist_to_gt_mean"); finalB = drift_stats_B.get("final_dist_to_gt_mean")
        if initA is not None and finalA is not None and initB is not None and finalB is not None:
            print(f"    Initial vs GT - Group A: {initA:.1f}px, Group B: {initB:.1f}px")
            print(f"    Final vs GT    - Group A: {finalA:.1f}px, Group B: {finalB:.1f}px")
        # Also print if final drift is larger for suppressed:
        if finalA is not None and finalB is not None:
            diff_final = finalA - finalB
            print(f"    (Group A final drift is {diff_final:+.1f}px vs Group B)")
    # Attention focus summary:
    if focus_stats_A and focus_stats_B:
        print(f"\n  Attention Focus Fractions (occluded query):")
        # Show final-layer fractions inside target vs occluder:
        final_tgt_A = focus_stats_A.get("final_target_frac"); final_occder_A = focus_stats_A.get("final_occluder_frac")
        final_tgt_B = focus_stats_B.get("final_target_frac"); final_occder_B = focus_stats_B.get("final_occluder_frac")
        if final_tgt_A is not None and final_occder_A is not None:
            print(f"    Group A (suppressed) final-layer: target={final_tgt_A:.3f}, occluder={final_occder_A:.3f}")
        if final_tgt_B is not None and final_occder_B is not None:
            print(f"    Group B (unsuppressed) final-layer: target={final_tgt_B:.3f}, occluder={final_occder_B:.3f}")
    # Query coupling (cross-attention overlap):
    print(f"\n  Query Coupling (Cross-attention Overlap):")
    # We use initial sampling overlap as a proxy (higher = more co-attention).
    if ov["A"].get("per_layer_unweighted") and ov["B"].get("per_layer_unweighted"):
        initial_overlap_A = ov["A"]["per_layer_unweighted"][0]
        initial_overlap_B = ov["B"]["per_layer_unweighted"][0]
        print(f"    Initial Layer Sampling Overlap: Group A = {initial_overlap_A:.3f}, Group B = {initial_overlap_B:.3f}")
        final_overlap_A = ov["A"]["per_layer_unweighted"][-1]
        final_overlap_B = ov["B"]["per_layer_unweighted"][-1]
        print(f"    Final Layer Sampling Overlap: Group A = {final_overlap_A:.3f}, Group B = {final_overlap_B:.3f}")

    sf_gap = None
    if sim_stats_A and sim_stats_B:
        sf_gap = round(
            sim_stats_A.get("sampled_feat_sim", {}).get("overall_mean", 0) -
            sim_stats_B.get("sampled_feat_sim", {}).get("overall_mean", 0), 6
        )

    output = {
        "config": {
            "occlusion_ratio_threshold": args.occlusion_ratio,
            "suppression_condition": (
                "occluded: peak_before_end OR end < start "
                "AND occluder: end > start"
            ),
            "fn_json_provided": args.fn_json is not None,
            "pt_dir_provided":  args.pt_dir is not None,
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
            "A": ov["A"], "A_tp": ov["A_tp"],
            "A_fn": ov["A_fn"], "B": ov["B"], "all": ov["all"],
        },
        "feature_similarity": {
            "group_A":     sim_stats_A,
            "group_B":     sim_stats_B,
            "gap_A_minus_B_sampled_feat": sf_gap,
            "interpretation": (
                "gap > 0.05 -> confirms feature orthogonality hypothesis. "
                "gap < 0.01 -> spatial/reference problem is primary cause."
            ),
        },
    }
    output["attention_focus"] = {
    "group_A": focus_stats_A,
    "group_B": focus_stats_B
    }
    output["reference_drift"] = {
        "group_A": drift_stats_A,
        "group_B": drift_stats_B
    }
    # 'coupling metrics' are essentially covered by sampling_overlap in output; we might explicitly add initial vs final overlap:
    if ov["A"].get("per_layer_unweighted"):
        output["query_coupling"] = {
            "initial_sampling_overlap_A": ov["A"]["per_layer_unweighted"][0],
            "initial_sampling_overlap_B": ov["B"]["per_layer_unweighted"][0],
            "final_sampling_overlap_A": ov["A"]["per_layer_unweighted"][-1],
            "final_sampling_overlap_B": ov["B"]["per_layer_unweighted"][-1]
        }
    # ... (save output JSON) ...
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[SAVED] {args.output_json}")


if __name__ == "__main__":
    main()
