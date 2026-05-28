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


def attn_to_distribution(attn_flat, temperature=1.0):
    """
    Convert a flat attention weight vector to a proper probability
    distribution using softmax.

    Why softmax even though deformable attention weights already sum to 1?
    -----------------------------------------------------------------------
    1. Half-precision storage introduces rounding errors. After .float()
       conversion the weights may not exactly sum to 1, making naive
       normalisation ( / sum ) numerically unstable for small weights.
    2. Softmax with log-space computation (F.log_softmax / F.softmax)
       is numerically stable and handles near-zero values correctly.
    3. Temperature allows controlling distribution sharpness:
         temperature < 1.0 → sharper distribution (emphasises top weights)
         temperature = 1.0 → identity (equivalent to re-normalising)
         temperature > 1.0 → softer distribution (spreads probability)
       Default 1.0 is neutral.

    Parameters
    ----------
    attn_flat   : 1-D float tensor  [H*Lv*P]
    temperature : float  (default 1.0)

    Returns
    -------
    prob : 1-D float tensor  [H*Lv*P]  — proper probability distribution
    """
    attn_flat = attn_flat.float()
    # softmax( logits / T ) where logits = attention weights
    # Note: since weights are already in [0,1] and sum to ~1,
    # dividing by temperature before softmax is equivalent to
    # sharpening/softening the peakiness of the distribution
    logits = attn_flat / max(temperature, 1e-6)
    prob   = F.softmax(logits, dim=0)
    return prob


def kl_div_symmetric(p, q, eps=1e-9):
    """
    Symmetric KL divergence (Jensen-Shannon style) between two
    probability distributions p and q.

    KL(p||q) = sum(p * log(p / q))
    Symmetric KL = (KL(p||q) + KL(q||p)) / 2

    Both p and q must be valid probability distributions (sum to 1,
    non-negative). Small eps is added inside log for numerical safety.

    Parameters
    ----------
    p, q : 1-D float tensors  [N]  — must sum to 1

    Returns
    -------
    float  — symmetric KL divergence, always >= 0
             0.0 = identical distributions
             higher = more different
    """
    p = p.float().clamp(min=eps)
    q = q.float().clamp(min=eps)

    # re-normalise after clamping to ensure sum=1
    p = p / p.sum()
    q = q / q.sum()

    kl_pq = (p * (p / q).log()).sum().item()
    kl_qp = (q * (q / p).log()).sum().item()

    return (kl_pq + kl_qp) / 2.0



# ============================================================
# FEATURE SIMILARITY  (v4 — softmax-corrected KL divergence)
# ============================================================
 
def compute_feature_similarity_for_pairs(pt_dir, pairs,
                                          suppressed_ids,
                                          not_suppressed_ids,
                                          attn_temperature=1.0):
    """
    Computes four similarity measures between query pairs per decoder layer.
 
    PRIMARY
    -------
    sampled_feat_sim
        Cosine similarity between sampled_features[l, qi] and
        sampled_features[l, qj].
        sampled_features[l, q] is the weighted sum of image feature
        values at query q's deformable attention sampling locations
        at decoder layer l.  Shape: [C].
        Direct measurement of whether two queries extracted the
        same image content from their attended regions.
 
    SECONDARY
    ---------
    attn_kl_div  (v4 CORRECTED)
        Symmetric KL divergence between the attention weight
        distributions of qi and qj at layer l.
 
        Computation:
          1. Flatten aw[l, qi]:  [H, Lv, P] → [H*Lv*P]
          2. Apply softmax with temperature T:
             wi = softmax( aw_flat_i / T )
             wj = softmax( aw_flat_j / T )
          3. Compute symmetric KL: (KL(wi||wj) + KL(wj||wi)) / 2
 
        Interpretation:
          High KL → queries attend to DIFFERENT sampling points
                    with DIFFERENT weights → diverse attention patterns
          Low  KL → queries attend to SAME sampling points with
                    SAME weights → identical attention patterns
 
        For suppression hypothesis:
          Group A (suppressed) should have LOWER KL than Group B
          if the cause is that both queries attend to the same
          sampling locations (spatial entanglement).
 
          Your results showed Group A has HIGHER KL than Group B,
          which means suppressed queries have MORE diverse attention
          patterns — consistent with attention collapse (occluded
          query's attention becomes random/unfocused) rather than
          co-location with the occluder.
 
    attn_dist_sim
        Dot product of softmax-normalised attention distributions.
        attn_dist_sim = (wi · wj) where wi, wj are the softmax
        distributions computed above.
        Complementary to KL: high dot product = similar distributions.
 
    cls_score_sim
        Cosine similarity of classification logit vectors [C_cls].
        Measures whether both queries predict similar class distributions.
        Group A > Group B in your data (+0.065 gap) confirming that
        classification-level convergence is part of the suppression
        mechanism even when feature-level representations diverge.
 
    Parameters
    ----------
    pt_dir           : str or Path
    pairs            : list of pair dicts from occlusion_results.json
    suppressed_ids   : set of occluded_gt_ann_ids (Group A)
    not_suppressed_ids : set of occluded_gt_ann_ids (Group B)
    attn_temperature : float (default 1.0)
        Temperature for softmax applied to attention weights before
        KL divergence computation.
        1.0 = neutral (re-normalises with softmax)
        < 1.0 = sharper (emphasises dominant sampling points)
        > 1.0 = softer (spreads probability more evenly)
    """
    if not TORCH_AVAILABLE:
        print("[WARN] torch not available.")
        return {}, {}, [], []
 
    pt_dir = Path(pt_dir)
 
    pairs_by_image = defaultdict(list)
    for p in pairs:
        pairs_by_image[p["image_name"]].append(p)
 
    def new_acc():
        return defaultdict(lambda: defaultdict(list))
 
    results_A  = new_acc()
    results_B  = new_acc()
    per_pair_A = []
    per_pair_B = []
    n_processed = 0
    n_skipped   = 0
 
    for pt_file in sorted(pt_dir.glob("*.pt")):
        data     = torch.load(pt_file, map_location="cpu",
                              weights_only=False)
        img_name = Path(data["img_path"]).name
 
        image_pairs = pairs_by_image.get(img_name, [])
        if not image_pairs:
            continue
 
        if "sampled_features" not in data:
            print(f"[WARN] sampled_features missing in {pt_file.name}")
            n_skipped += 1
            continue
 
        # convert all to float32 immediately — stored in half
        sf = data["sampled_features"].float()   # [L, Q, C_sf]
        aw = data["attention_weights"].float()  # [L, Q, H, Lv, P]
        cs = data["cls_scores"].float()         # [L, Q, C_cls]
 
        L, Q, C_sf     = sf.shape
        _, _, H, Lv, P = aw.shape
        N_pts          = H * Lv * P             # total sampling points per query
 
        n_processed += 1
 
        for p in image_pairs:
            qi  = p["occluded_query"]   # suppressed/hidden object
            qj  = p["occluder_query"]   # dominant/visible object
            aid = p["occluded_gt_ann_id"]
 
            if qi >= Q or qj >= Q:
                n_skipped += 1
                continue
 
            if aid in suppressed_ids:
                target     = results_A
                pair_store = per_pair_A
            elif aid in not_suppressed_ids:
                target     = results_B
                pair_store = per_pair_B
            else:
                continue
 
            sf_sims   = []
            attn_sims = []
            attn_kls  = []
            cls_sims  = []
 
            for l in range(L):
 
                # ── PRIMARY: sampled feature cosine similarity ────
                # sampled_features[l, q] is the content extracted
                # by query q's deformable attention at layer l
                feat_qi = sf[l, qi]   # [C_sf]
                feat_qj = sf[l, qj]   # [C_sf]
                sf_sim  = cosine_sim(feat_qi, feat_qj)
 
                # ── CORRECTED: attention KL divergence ────────────
                # Step 1: flatten attention weights to 1-D
                aw_qi_flat = aw[l, qi].reshape(N_pts)   # [H*Lv*P]
                aw_qj_flat = aw[l, qj].reshape(N_pts)   # [H*Lv*P]
 
                # Step 2: softmax-normalise with temperature
                # This converts to a proper probability distribution
                # and eliminates half-precision rounding artefacts
                wi = attn_to_distribution(aw_qi_flat, attn_temperature)
                wj = attn_to_distribution(aw_qj_flat, attn_temperature)
 
                # Step 3: symmetric KL divergence
                attn_kl = kl_div_symmetric(wi, wj)
 
                # Step 4: dot-product similarity (complementary metric)
                # high = similar attention patterns
                attn_dot = (wi * wj).sum().item()
 
                # ── cls score cosine similarity ───────────────────
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
                "image":            img_name,
                "occluded_query":   qi,
                "occluder_query":   qj,
                "ann_id":           aid,
                "sampled_feat_sim": sf_sims,
                "attn_dist_sim":    attn_sims,
                "attn_kl_div":      attn_kls,
                "cls_score_sim":    cls_sims,
                "mean_sf_sim":      round(float(np.mean(sf_sims)),   6),
                "mean_attn_kl":     round(float(np.mean(attn_kls)),  6),
                "mean_attn_sim":    round(float(np.mean(attn_sims)), 6),
                "sf_sim_delta":     round(sf_sims[-1] - sf_sims[0],  6),
                "attn_kl_delta":    round(attn_kls[-1] - attn_kls[0], 6),
            })
 
    print(f"  [Feature sim] pt files processed : {n_processed}")
    print(f"  [Feature sim] attn temperature   : {attn_temperature}")
    if n_skipped:
        print(f"  [Feature sim] skipped            : {n_skipped}")
 
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


def compute_self_attention_between_pairs(pt_dir, pairs,
                                          suppressed_ids,
                                          not_suppressed_ids):
    """
    Measures self-attention coupling between query pairs.

    VARIABLE CONVENTION (fixed throughout):
      qi  = occluded_query   → suppressed/hidden object, confidence drops
      qj  = occluder_query   → dominant/visible object,  confidence rises

    METRICS:
      a_i_to_j : how much qi (occluded) attends TO qj (occluder)
                 = topk_vals[l, qi] at position where topk_idx[l, qi] == qj
                 Interpretation: occluded query pulling information
                 from the occluder query in self-attention

      a_j_to_i : how much qj (occluder) attends TO qi (occluded)
                 = topk_vals[l, qj] at position where topk_idx[l, qj] == qi
                 Interpretation: occluder query pulling information
                 from the occluded query in self-attention

      asymmetry : a_i_to_j - a_j_to_i
                  > 0 : occluded pulls from occluder more than vice versa
                        (one-sided coupling → suppression signature)
                  ≈ 0 : balanced mutual coupling
                        (healthy → Group B pattern)

    GROUP ASSIGNMENT:
      Group A (results_A) : suppression condition MET
                            occluded_not_maintained AND occluder_increasing
      Group B (results_B) : in pairs but suppression condition NOT met

    ASSUMPTION about topk storage:
      topk_idx[l, q, k] = index of the k-th query that query q attends TO
      topk_vals[l, q, k] = corresponding attention weight
      i.e. row q of the self-attention matrix, top-k entries
      VERIFY this matches your saving code before trusting results.
    """
    pairs_by_image = defaultdict(list)
    for p in pairs:
        pairs_by_image[p["image_name"]].append(p)

    results_A = defaultdict(lambda: {
        "attn_i_to_j": [],   # occluded → occluder attention weights
        "attn_j_to_i": [],   # occluder → occluded attention weights
        "asymmetry":    []   # i_to_j - j_to_i
    })
    results_B = defaultdict(lambda: {
        "attn_i_to_j": [],
        "attn_j_to_i": [],
        "asymmetry":    []
    })

    per_pair_A = []
    per_pair_B = []
    n_processed = 0
    n_skipped   = 0

    for pt_file in sorted(Path(pt_dir).glob("*.pt")):
        data     = torch.load(pt_file, map_location="cpu")
        img_name = Path(data["img_path"]).name

        image_pairs = pairs_by_image.get(img_name, [])
        if not image_pairs:
            continue

        if "self_attn_topk_vals" not in data:
            print(f"[WARN] self_attn_topk_vals missing in {pt_file.name}")
            n_skipped += 1
            continue

        # topk_vals : [L, Q, topk]  — attention weights row q attends TO others
        # topk_idx  : [L, Q, topk]  — which query indices those weights go to
        topk_vals = data["self_attn_topk_vals"].float()      # [L, Q, topk]
        topk_idx  = data["self_attn_topk_idx"].long()        # [L, Q, topk] ← safe int

        L, Q, K = topk_vals.shape
        n_processed += 1

        for p in image_pairs:

            # qi = occluded object query (suppressed, confidence drops)
            qi  = p["occluded_query"]

            # qj = occluder object query (dominant, confidence rises)
            qj  = p["occluder_query"]

            # ann_id used for group routing
            aid = p["occluded_gt_ann_id"]

            if qi >= Q or qj >= Q:
                n_skipped += 1
                continue

            # route to Group A (suppressed) or Group B (not suppressed)
            if aid in suppressed_ids:
                target     = results_A
                pair_store = per_pair_A
            elif aid in not_suppressed_ids:
                target     = results_B
                pair_store = per_pair_B
            else:
                continue

            attn_i_to_j_per_layer = []   # occluded→occluder per layer
            attn_j_to_i_per_layer = []   # occluder→occluded per layer
            asymmetry_per_layer   = []

            for l in range(L):

                # ── occluded (qi) attending TO occluder (qj) ─────
                # topk_idx[l, qi] = indices of queries that qi attends to
                # topk_vals[l, qi] = corresponding attention weights
                qi_topk_indices = topk_idx[l, qi].tolist()  # which queries qi attends TO
                qi_topk_weights = topk_vals[l, qi]          # weights for those queries

                if qj in qi_topk_indices:
                    # occluder qj is in the top-k that occluded qi attends to
                    pos_of_qj   = qi_topk_indices.index(qj)
                    a_i_to_j    = qi_topk_weights[pos_of_qj].item()
                    # a_i_to_j > 0: occluded is pulling from occluder
                else:
                    # occluder not in qi's top-k → attention is negligible
                    a_i_to_j = 0.0

                # ── occluder (qj) attending TO occluded (qi) ─────
                # topk_idx[l, qj] = indices of queries that qj attends to
                # topk_vals[l, qj] = corresponding attention weights
                qj_topk_indices = topk_idx[l, qj].tolist()  # which queries qj attends TO
                qj_topk_weights = topk_vals[l, qj]          # weights for those queries

                if qi in qj_topk_indices:
                    # occluded qi is in the top-k that occluder qj attends to
                    pos_of_qi   = qj_topk_indices.index(qi)
                    a_j_to_i    = qj_topk_weights[pos_of_qi].item()
                    # a_j_to_i > 0: occluder is pulling from occluded
                else:
                    # occluded not in qj's top-k → attention is negligible
                    a_j_to_i = 0.0

                # ── asymmetry ────────────────────────────────────
                # positive: occluded pulls more from occluder than vice versa
                # negative: occluder pulls more from occluded (rare)
                asym = a_i_to_j - a_j_to_i

                attn_i_to_j_per_layer.append(round(a_i_to_j, 6))
                attn_j_to_i_per_layer.append(round(a_j_to_i, 6))
                asymmetry_per_layer.append(round(asym, 6))

                target[l]["attn_i_to_j"].append(a_i_to_j)
                target[l]["attn_j_to_i"].append(a_j_to_i)
                target[l]["asymmetry"].append(asym)

            pair_store.append({
                "image":               img_name,
                # query indices
                "occluded_query":      qi,   # suppressed object
                "occluder_query":      qj,   # dominant object
                "ann_id":              aid,
                # per-layer metrics
                "attn_occluded_to_occluder":  attn_i_to_j_per_layer,
                "attn_occluder_to_occluded":  attn_j_to_i_per_layer,
                "asymmetry":                  asymmetry_per_layer,
                # scalar summaries
                "mean_attn_occluded_to_occluder": round(
                    float(np.mean(attn_i_to_j_per_layer)), 6),
                "mean_attn_occluder_to_occluded": round(
                    float(np.mean(attn_j_to_i_per_layer)), 6),
                "mean_asymmetry": round(
                    float(np.mean(asymmetry_per_layer)), 6),
                # diagnostic: what fraction of layers has qj in qi's top-k?
                "pct_layers_occluder_in_occluded_topk": round(
                    sum(1 for v in attn_i_to_j_per_layer if v > 0) / L * 100, 1),
                # diagnostic: what fraction of layers has qi in qj's top-k?
                "pct_layers_occluded_in_occluder_topk": round(
                    sum(1 for v in attn_j_to_i_per_layer if v > 0) / L * 100, 1),
            })

    print(f"  [Self-attn] pt files processed : {n_processed}")
    if n_skipped:
        print(f"  [Self-attn] pairs skipped      : {n_skipped}")

    return dict(results_A), dict(results_B), per_pair_A, per_pair_B


def summarise_self_attention(results_by_layer):
    if not results_by_layer:
        return {}

    layers  = sorted(results_by_layer.keys())
    summary = {
        "n_pairs": len(results_by_layer[layers[0]]["attn_i_to_j"])
    }

    for metric in ["attn_i_to_j", "attn_j_to_i", "asymmetry"]:
        means = [round(float(np.mean(results_by_layer[l][metric])), 6)
                 for l in layers]
        stds  = [round(float(np.std(results_by_layer[l][metric])),  6)
                 for l in layers]
        all_v = [v for l in layers for v in results_by_layer[l][metric]]

        # fraction of pairs where qj appears in qi's top-k at this layer
        if metric == "attn_i_to_j":
            pct_nonzero_i_to_j = [
                round(float((np.array(results_by_layer[l][metric]) > 0).mean() * 100), 2)
                for l in layers
            ]
        else:
            pct_nonzero_i_to_j = None
        
        if metric == "attn_j_to_i":
            pct_nonzero_j_to_i = [
                round(float((np.array(results_by_layer[l][metric]) > 0).mean() * 100), 2)
                for l in layers
            ]
        else:
            pct_nonzero_j_to_i = None


        summary[metric] = {
            "mean_per_layer": means,
            "std_per_layer":  stds,
            "overall_mean":   round(float(np.mean(all_v)), 6),
            "overall_std":    round(float(np.std(all_v)),  6),
            "delta_L0_to_last": round(means[-1] - means[0], 6),
        }
        if pct_nonzero_i_to_j:
            summary[metric]["pct_qj_in_topk_per_layer"] = pct_nonzero_i_to_j
        if pct_nonzero_j_to_i:
            summary[metric]["pct_qi_in_topk_per_layer"] = pct_nonzero_j_to_i

    return summary


def print_self_attention_report(stats_A, stats_B):
    dash = "-" * 64
    print(f"\n{dash}")
    print(f"  SELF-ATTENTION ANALYSIS BETWEEN QUERY PAIRS")
    print(f"  attn_i_to_j : occluded query attending TO occluder query")
    print(f"  attn_j_to_i : occluder query attending TO occluded query")
    print(f"  asymmetry   : i_to_j - j_to_i  (+ means occluded pulls from occluder)")
    print(f"{dash}")

    if not stats_A or not stats_B:
        print("  [No data — supply --pt_dir]")
        return

    for metric, label in [
        ("attn_i_to_j", "Occluded → Occluder  (qi attends to qj)"),
        ("attn_j_to_i", "Occluder → Occluded  (qj attends to qi)"),
        ("asymmetry",   "Asymmetry            (i_to_j − j_to_i)"),
    ]:
        mA = stats_A.get(metric, {}); mB = stats_B.get(metric, {})
        if not mA or not mB:
            continue

        gap = round(mA["overall_mean"] - mB["overall_mean"], 6)
        print(f"\n  [{label}]")
        print(f"    Group A mean : {mA['overall_mean']:.6f}  "
              f"(std={mA['overall_std']:.6f})  n={stats_A['n_pairs']}")
        print(f"    Group B mean : {mB['overall_mean']:.6f}  "
              f"(std={mB['overall_std']:.6f})  n={stats_B['n_pairs']}")
        print(f"    Gap (A − B)  : {gap:+.6f}")

        if "pct_qj_in_topk_per_layer" in mA:
            print(f"    % pairs where qj in qi top-k:")
            for l, pct in enumerate(mA["pct_qj_in_topk_per_layer"]):
                pct_b = stats_B.get(metric, {}).get(
                    "pct_qj_in_topk_per_layer", [0]*6)[l]
                print(f"      L{l}: A={pct:.1f}%  B={pct_b:.1f}%")
        
        if "pct_qi_in_topk_per_layer" in mA:
            print(f"    % pairs where qi in qj top-k:")
            for l, pct in enumerate(mA["pct_qi_in_topk_per_layer"]):
                pct_b = stats_B.get(metric, {}).get(
                    "pct_qi_in_topk_per_layer", [0]*6)[l]
                print(f"      L{l}: A={pct:.1f}%  B={pct_b:.1f}%")

        print(f"    Per-layer (A / B):")
        for l, (a, b) in enumerate(zip(mA["mean_per_layer"],
                                        mB["mean_per_layer"])):
            print(f"      L{l}: A={a:.6f}  B={b:.6f}  gap={a-b:+.6f}")

    # interpretation
    i_to_j_gap = round(
        stats_A.get("attn_i_to_j", {}).get("overall_mean", 0) -
        stats_B.get("attn_i_to_j", {}).get("overall_mean", 0), 6
    )
    asym_A = stats_A.get("asymmetry", {}).get("overall_mean", 0)
    asym_B = stats_B.get("asymmetry", {}).get("overall_mean", 0)

    print(f"\n{dash}")
    print(f"  INTERPRETATION")
    print(f"{dash}")

    # if i_to_j_gap > 0.001:
    #     print(f"  CONFIRMS self-attention coupling hypothesis.")
    #     print(f"  Suppressed (A) queries attend MORE to their occluder")
    #     print(f"  in self-attention (gap={i_to_j_gap:+.6f}).")
    #     print(f"  The occluded query copies the occluder's representation")
    #     print(f"  before cross-attention — this is the ROOT CAUSE of")
    #     print(f"  feature co-adaptation and subsequent confidence suppression.")
    #     print(f"")
    #     print(f"  DIRECTLY MOTIVATED FIX:")
    #     print(f"  Self-attention masking between competing queries.")
    #     print(f"  For pairs with high spatial overlap and confidence gap,")
    #     print(f"  mask or down-weight the attention from qi to qj.")
    #     print(f"  This prevents the occluded query from copying the")
    #     print(f"  occluder's representation at the self-attention stage.")
    # elif i_to_j_gap > 0.0001:
    #     print(f"  WEAK self-attention coupling (gap={i_to_j_gap:+.6f}).")
    #     print(f"  Some coupling via self-attention but not the primary driver.")
    #     print(f"  Cross-attention spatial overlap remains primary mechanism.")
    # else:
    #     print(f"  No self-attention coupling detected (gap={i_to_j_gap:+.6f}).")
    #     print(f"  Suppression occurs purely through cross-attention competition.")
    #     print(f"  Self-attention masking would NOT help.")

    # if asym_A > 0 and asym_A > asym_B:
    #     print(f"")
    #     print(f"  Asymmetry confirms direction: occluded query pulls from occluder")
    #     print(f"  more than occluder pulls from occluded.")
    #     print(f"  Group A asymmetry: {asym_A:+.6f}")
    #     print(f"  Group B asymmetry: {asym_B:+.6f}")

    # Primary signal is asymmetry gap, not i_to_j gap
    asym_gap    = stats_A["asymmetry"]["overall_mean"] - \
                stats_B["asymmetry"]["overall_mean"]
    j_to_i_gap  = stats_A["attn_j_to_i"]["overall_mean"] - \
                stats_B["attn_j_to_i"]["overall_mean"]

    if asym_gap > 0.003 and j_to_i_gap < -0.005:
        print("CONFIRMS asymmetric coupling hypothesis.")
        print("Suppressed pairs show ONE-SIDED self-attention:")
        print("  occluded pulls from occluder but occluder ignores occluded.")
        print("  This prevents complementary specialisation.")
        print("Fix: symmetrisation loss — encourage attn(qj→qi) for competing pairs.")
    elif asym_gap > 0.003:
        print("PARTIAL: asymmetry present but occluder attention gap unclear.")
    elif i_to_j_gap > 0.003:
        print("CONFIRMS coupling: occluded strongly attends to occluder.")
        print("Fix: self-attention masking between competing queries.")
    else:
        print("Suppression not driven by self-attention.")
        print("Focus on cross-attention spatial overlap fixes.")

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
        print(f"\n[INFO] Computing self-attention scores between pairs...")
        raw_sa_A, raw_sa_B, pp_sa_A, pp_sa_B = \
            compute_self_attention_between_pairs(
                args.pt_dir, pairs,
                suppressed_ann_ids,
                not_suppressed_ann_ids,
            )
        sa_stats_A = summarise_self_attention(raw_sa_A)
        sa_stats_B = summarise_self_attention(raw_sa_B)
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

    print_self_attention_report(sa_stats_A, sa_stats_B)

    print(f"\n{sep}\n")

    # ── save JSON ─────────────────────────────────────────
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

    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[SAVED] {args.output_json}")


if __name__ == "__main__":
    main()


# INITIALIZATION (L0)
# ├── Both groups: asymmetry ≈ 0  (symmetric, neither pulls from the other)
# ├── Group B: HIGHER mutual coupling overall
# │   attn(i→j) B=0.0088 > A=0.0052
# │   attn(j→i) B=0.0101 > A=0.0053
# │   → Both queries are already more mutually aware in healthy pairs
# └── Group A: LOWER mutual coupling at start
#     → Queries begin more isolated from each other

# AFTER FIRST CROSS-ATTENTION (L1 onward)
# ├── Cross-attention updates both queries based on image features
# ├── In Group B: occluder attends strongly to occluded (j→i stays high)
# │   → Occluder adapts its representation knowing the occluded exists
# │   → Both queries develop COMPLEMENTARY representations
# │   → "Head" query and "t-shirt" query specialise differently
# │
# └── In Group A: occluder stops attending to occluded (j→i drops low)
#     → Occluder develops INDEPENDENTLY without awareness of occluded
#     → Occluded continues pulling from occluder (i→j stays moderate)
#     → One-sided information flow: occluded copies occluder
#     → No complementary specialisation emerges
#     → Both queries converge to similar representations
#     → Occluder dominates → occluded confidence collapses


# Healthy (Group B) attn(j→i) / attn(i→j) ratio:
#   Overall: 0.02152 / 0.02680 = 0.803  ← occluder attends ~80% as much as occluded attends to it

# Suppressed (Group A) attn(j→i) / attn(i→j) ratio:
#   Overall: 0.01162 / 0.02280 = 0.510  ← occluder attends only ~51% as much

# Target ratio for fix: push Group A from 0.51 toward Group B's 0.80

# For competing query pairs (qi=occluded, qj=occluder),
#     encourage the occluder to attend to the occluded query
#     in proportion to how much the occluded attends to it.


# Self-attn symmetrisation loss     ✅ STRONGLY  Direct fix for j→i deficit
#   Encourage attn(qj→qi) to be    PRIMARY      Group B has 1.85× higher j→i
#   ~80% of attn(qi→qj)                         Target ratio = 0.80
