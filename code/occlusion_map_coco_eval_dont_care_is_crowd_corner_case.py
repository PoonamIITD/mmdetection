"""
Occlusion-wise mAP using COCOeval — with full "don't-care" for other levels.

Problem with plain iscrowd=1
─────────────────────────────
iscrowd=1 suppresses a detection only when it *overlaps* the crowd box above
the IoU threshold.  A detection that fires inside another-level region but
whose IoU with that region's box is below the threshold still gets counted
as FP against the current level.  This contaminates precision.

Fix: OcclusionCOCOeval (subclass of COCOeval)
──────────────────────────────────────────────
We override evaluateImg() to add a pre-pass that marks every detection as
"dt_ignore" if it overlaps *sufficiently* with any other-level annotation,
regardless of the IoU threshold used for TP matching.

Specifically, a detection d is flagged as "don't care" when:

    IoU(d, any_other_level_gt) ≥ DONT_CARE_IOU_THRESH   (default 0.1)

The threshold is intentionally low (0.1) so that a detection whose box
overlaps even slightly with another-level region is not penalised.
You can tighten it (e.g. 0.5) if you want stricter spatial separation.

After the pre-pass the parent evaluateImg() runs normally, but those
detections are already marked ignored so they are skipped in TP/FP scoring.

Occlusion levels : light, moderate, severe   (no synthetic "all")
Custom maxDets   : [100, 300, 1000]

Requirements:
    pip install pycocotools matplotlib tabulate

Usage:
    python occlusion_map_cocoeval.py \
        --gt        instances_validation_merged_mannual.json \
        --baseline  results_baseline_GDINO_final_val.json \
        --updated   results_sampling_loss_final_val.json \
        --output    occlusion_map_report \
        --dont-care-iou 0.1 \
        --per-category
"""

import json
import copy
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import pycocotools.mask as maskUtils
except ImportError:
    raise ImportError("pip install pycocotools")

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# ── Config ────────────────────────────────────────────────────────────────────
OCCLUSION_LEVELS   = ["light", "moderate", "severe"]
MAX_DETS           = [100, 300, 1000]
DONT_CARE_IOU_DEFAULT = 0.1   # overlap with other-level GT → detection ignored

LEVEL_COLORS = {
    "light":    "#3498DB",
    "moderate": "#F39C12",
    "severe":   "#8E44AD",
}
MODEL_STYLES = {
    "baseline": {"ls": "--", "lw": 2.0, "alpha": 0.85},
    "updated":  {"ls": "-",  "lw": 2.5, "alpha": 0.95},
}

# ── I/O ───────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ── GT preparation ────────────────────────────────────────────────────────────

def build_gt_for_level(gt_full, target_level):
    """
    Tag every annotation with iscrowd:
        target_level  → iscrowd = 0   (evaluated normally)
        other levels  → iscrowd = 1   (used as don't-care regions by
                                       OcclusionCOCOeval; also suppresses
                                       detections that overlap them via the
                                       standard COCOeval iscrowd path)

    We also inject a custom field  _is_other_level = True/False  so that
    OcclusionCOCOeval can build both the per-image spatial lookup (boxes)
    and the other-level GT id set — used to distinguish Cases A/B/C in
    evaluateImg without re-parsing annotation attributes.
    """
    gt_subset = copy.deepcopy(gt_full)
    for ann in gt_subset["annotations"]:
        lvl = (ann.get("attributes", {})
                   .get("occlusion_level", "unknown")
                   .strip().lower())
        is_other = (lvl != target_level)
        ann["iscrowd"]       = 1 if is_other else 0
        ann["_is_other_level"] = is_other   # custom flag for our subclass
    return gt_subset

# ── Custom COCOeval subclass ──────────────────────────────────────────────────

class OcclusionCOCOeval(COCOeval):
    """
    Extends COCOeval with full spatial don't-care for other-level GT regions.

    ┌─────────────────────────────────────────────────────────────────────┐
    │  Three detection cases and how each is handled                      │
    ├──────┬───────────────────────────────┬────────────────────────────  │
    │ Case │ Situation                     │ Decision                     │
    ├──────┼───────────────────────────────┼────────────────────────────  │
    │  A   │ D matched target-level GT     │ TP — never touched           │
    │      │ (iscrowd=0) at current IoU    │                              │
    ├──────┼───────────────────────────────┼────────────────────────────  │
    │  B   │ D unmatched AND overlaps      │ Ignored (don't-care)         │
    │      │ other-level GT ≥ dont_care_   │ Not counted as FP            │
    │      │ iou (pure other-level FP)     │                              │
    ├──────┼───────────────────────────────┼────────────────────────────  │
    │  C   │ D overlaps BOTH a target-     │ Per-threshold decision:      │
    │      │ level GT AND an other-level   │  • threshold t where D was   │
    │      │ GT, but D lost the target-    │    beaten to target GT by    │
    │      │ level GT to a better-scoring  │    a rival → Ignored         │
    │      │ rival detection               │  • threshold t where D had   │
    │      │                               │    no target match at all    │
    │      │                               │    → Ignored (other-level    │
    │      │                               │    overlap justifies it)     │
    └──────┴───────────────────────────────┴────────────────────────────  ┘

    Case C reasoning
    ─────────────────
    When a detection D straddles a target-level and an other-level object:
    - If a better detection already claimed the target-level GT, D failing
      to match it is not a meaningful error — D genuinely sits over an
      ambiguous region containing an other-level object.
    - We therefore ignore D per-threshold only when:
        (a) D did NOT match the target-level GT at that threshold, AND
        (b) D overlaps any other-level GT ≥ dont_care_iou.
    - If D DID match the target-level GT at some threshold → TP at that
      threshold, regardless of other-level overlap (Case A, not touched).

    This means dtIgnore is set per-threshold for Case C, not globally,
    preserving any threshold where the detection was genuinely a TP.

    iscrowd=1 (set by build_gt_for_level) additionally suppresses D when
    IoU(D, other-level GT) ≥ the *current* iouThrs[t] — our post-pass
    extends that to the lower dont_care_iou at all thresholds simultaneously.

    Parameters
    ----------
    dont_care_iou : float
        Minimum IoU with any other-level GT box for a detection to be
        treated as don't-care.  Default 0.1 (lenient).  Set lower to
        ignore more detections near other-level regions.
    """

    def __init__(self, coco_gt, coco_dt, iou_type="bbox",
                 dont_care_iou=DONT_CARE_IOU_DEFAULT):
        super().__init__(coco_gt, coco_dt, iou_type)
        self.dont_care_iou = dont_care_iou

        # image_id → list of other-level GT boxes  [x, y, w, h]
        self._other_level_gts = defaultdict(list)
        # image_id → set of other-level GT annotation ids (for fast lookup)
        self._other_level_ids = defaultdict(set)

        for ann in coco_gt.dataset["annotations"]:
            if ann.get("_is_other_level", False):
                self._other_level_gts[ann["image_id"]].append(
                    np.array(ann["bbox"], dtype=np.float64)
                )
                self._other_level_ids[ann["image_id"]].add(ann["id"])

    # ------------------------------------------------------------------ #
    # Vectorised IoU: one detection box vs many GT boxes                  #
    # All boxes [x, y, w, h] (COCO convention).                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _iou_one_vs_many(dt_box, gt_boxes):
        """
        dt_box   : array-like [x, y, w, h]
        gt_boxes : list of array-like [x, y, w, h]
        Returns  : np.ndarray shape [len(gt_boxes)]
        """
        if not gt_boxes:
            return np.array([], dtype=np.float64)

        dx, dy, dw, dh = dt_box
        dt_area = dw * dh
        if dt_area <= 0:
            return np.zeros(len(gt_boxes))

        gt_arr = np.array(gt_boxes, dtype=np.float64)   # [N, 4]
        gx, gy, gw, gh = gt_arr[:, 0], gt_arr[:, 1], gt_arr[:, 2], gt_arr[:, 3]

        ix = np.maximum(0.0, np.minimum(dx + dw, gx + gw) - np.maximum(dx, gx))
        iy = np.maximum(0.0, np.minimum(dy + dh, gy + gh) - np.maximum(dy, gy))
        inter  = ix * iy
        union  = dt_area + gw * gh - inter
        return np.where(union > 0, inter / union, 0.0)

    # ------------------------------------------------------------------ #
    # Override evaluateImg                                                 #
    # ------------------------------------------------------------------ #
    def evaluateImg(self, imgId, catId, aRng, maxDet):
        """
        Post-processes the parent result to handle Cases B and C.

        Post-pass logic (per detection D, per threshold t):

            matched_to_target[t]  = dt_match[t, d] != 0
                                     AND matched GT is NOT an other-level id

            overlaps_other        = max IoU(D, other-level GTs) ≥ dont_care_iou

            Decision:
              if matched_to_target[t]:          → keep as TP  (Case A)
              elif overlaps_other:              → dtIgnore[t, d] = True
                                                   (Cases B and C)
              else:                             → leave as FP (genuine FP,
                                                   no other-level proximity)
        """
        result = super().evaluateImg(imgId, catId, aRng, maxDet)
        if result is None:
            return result

        other_boxes = self._other_level_gts.get(imgId, [])
        other_ids   = self._other_level_ids.get(imgId, set())
        if not other_boxes:
            return result   # no other-level GTs in this image

        dt_ids  = result["dtIds"]
        if len(dt_ids) == 0:
            return result

        dt_match  = result["dtMatches"]   # [T, D]  gt_id (0 = unmatched)
        dt_ignore = result["dtIgnore"]    # [T, D]  bool

        dt_anns = self.cocoDt.loadAnns(dt_ids)
        n_thr   = dt_match.shape[0]

        for d_idx, dt_ann in enumerate(dt_anns):
            # Already fully ignored at every threshold → skip
            if np.all(dt_ignore[:, d_idx]):
                continue

            dt_box = dt_ann.get("bbox")
            if dt_box is None:
                continue

            # Does this detection spatially overlap any other-level region?
            ious_vs_other = self._iou_one_vs_many(dt_box, other_boxes)
            overlaps_other = (len(ious_vs_other) > 0 and
                              ious_vs_other.max() >= self.dont_care_iou)

            if not overlaps_other:
                # No meaningful other-level overlap → leave everything as-is.
                # Pure FPs here are real FPs; pure TPs are real TPs.
                continue

            # Detection overlaps an other-level region.
            # Now decide per-threshold: was it a genuine TP at this threshold?
            for t in range(n_thr):
                if dt_ignore[t, d_idx]:
                    continue  # already ignored at this threshold

                matched_gt_id = dt_match[t, d_idx]

                if matched_gt_id != 0 and matched_gt_id not in other_ids:
                    # Case A: matched a real target-level GT → TP, keep it.
                    pass
                else:
                    # Case B: purely unmatched, other-level only.
                    # Case C: overlaps both but lost target-level GT to a rival,
                    #         OR matched an other-level GT (parent sets ignore
                    #         anyway, but we make it explicit here too).
                    # → Don't-care: not a FP, not a TP.
                    dt_ignore[t, d_idx] = True

        result["dtIgnore"] = dt_ignore
        return result

# ── COCOeval runner ───────────────────────────────────────────────────────────

def build_cocoeval(gt_dict, pred_list, iou_type="bbox",
                   dont_care_iou=DONT_CARE_IOU_DEFAULT):
    """
    Build and run OcclusionCOCOeval with custom maxDets.

    iscrowd=1 flags (set by build_gt_for_level) handle the case where a
    detection *matches* another-level GT at the current IoU threshold.

    OcclusionCOCOeval.evaluateImg() additionally catches detections that
    partially overlap another-level region but fall below the main IoU
    threshold — they are also marked don't-care (not FP).
    """
    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()

    gt_image_ids   = {img["id"] for img in gt_dict["images"]}
    filtered_preds = [p for p in pred_list if p["image_id"] in gt_image_ids]

    if not filtered_preds:
        return None

    coco_dt   = coco_gt.loadRes(filtered_preds)
    coco_eval = OcclusionCOCOeval(coco_gt, coco_dt, iou_type,
                                   dont_care_iou=dont_care_iou)

    coco_eval.params.maxDets = MAX_DETS

    coco_eval.evaluate()
    coco_eval.accumulate()
    return coco_eval

# ── Stats extraction ──────────────────────────────────────────────────────────

def get_stats(coco_eval):
    """
    Compute summary stats matching COCOeval.summarize() for custom maxDets.

    COCOeval.eval["precision"] shape: [T, R, K, A, M]
        T = IoU thresholds (10: 0.50…0.95)
        R = recall points  (101: 0…1)
        K = categories
        A = area ranges    (all, small, medium, large)
        M = maxDets levels

    COCOeval.eval["recall"] shape: [T, K, A, M]
    """
    if coco_eval is None:
        return {}

    p  = coco_eval.params
    ev = coco_eval.eval

    iou_thrs = np.linspace(0.5, 0.95, 10)

    def _t_range(lo, hi):
        t_lo = np.where(np.isclose(iou_thrs, lo))[0]
        t_hi = np.where(np.isclose(iou_thrs, hi))[0]
        if not len(t_lo) or not len(t_hi):
            return None, None
        return int(t_lo[0]), int(t_hi[0]) + 1

    def _aind(area):
        idx = [i for i, ar in enumerate(p.areaRngLbl) if ar == area]
        return idx[0] if idx else None

    def _ap(iou_lo, iou_hi, area="all", max_det_idx=-1):
        t0, t1 = _t_range(iou_lo, iou_hi)
        ai     = _aind(area)
        if t0 is None or ai is None:
            return float("nan")
        prec = ev["precision"][t0:t1, :, :, ai, max_det_idx]
        prec = prec[prec > -1]
        return float(np.mean(prec)) if len(prec) else float("nan")

    def _ar(iou_lo, iou_hi, area="all", max_det_idx=-1):
        t0, t1 = _t_range(iou_lo, iou_hi)
        ai     = _aind(area)
        if t0 is None or ai is None:
            return float("nan")
        rec = ev["recall"][t0:t1, :, ai, max_det_idx]
        rec = rec[rec > -1]
        return float(np.mean(rec)) if len(rec) else float("nan")

    stats = {
        "AP@[.5:.95]":                          _ap(0.50, 0.95, "all",    2),
        "AP@.50":                               _ap(0.50, 0.50, "all",    2),
        "AP@.75":                               _ap(0.75, 0.75, "all",    2),
        "AP_small":                             _ap(0.50, 0.95, "small",  2),
        "AP_medium":                            _ap(0.50, 0.95, "medium", 2),
        "AP_large":                             _ap(0.50, 0.95, "large",  2),
        f"AR@[.5:.95]_maxDets{MAX_DETS[0]}":   _ar(0.50, 0.95, "all",    0),
        f"AR@[.5:.95]_maxDets{MAX_DETS[1]}":   _ar(0.50, 0.95, "all",    1),
        f"AR@[.5:.95]_maxDets{MAX_DETS[2]}":   _ar(0.50, 0.95, "all",    2),
        "AR_small":                             _ar(0.50, 0.95, "small",  2),
        "AR_medium":                            _ar(0.50, 0.95, "medium", 2),
        "AR_large":                             _ar(0.50, 0.95, "large",  2),
    }
    return stats


def get_pr_curve(coco_eval, iou_thresh=0.5, area="all", max_det_idx=-1):
    """
    Extract a PR curve from COCOeval.eval["precision"].

    precision array: [T, R, K, A, M]
    Returns (recall_thrs [101], prec_mean [101]).
    """
    if coco_eval is None:
        return np.linspace(0, 1, 101), np.zeros(101)

    p  = coco_eval.params
    ev = coco_eval.eval

    iou_thrs = np.linspace(0.5, 0.95, 10)
    t_idx    = np.where(np.isclose(iou_thrs, iou_thresh))[0]
    if not len(t_idx):
        return np.linspace(0, 1, 101), np.zeros(101)
    t_idx = int(t_idx[0])

    aind = [i for i, ar in enumerate(p.areaRngLbl) if ar == area]
    if not aind:
        return np.linspace(0, 1, 101), np.zeros(101)

    prec = ev["precision"][t_idx, :, :, aind[0], max_det_idx]   # [101, K]
    prec_mean = np.array([
        float(np.mean(row[row > -1])) if np.any(row > -1) else 0.0
        for row in prec
    ])
    return np.linspace(0, 1, 101), prec_mean

# ── Plots (unchanged logic, same functions as before) ─────────────────────────

def plot_pr_per_occlusion(evals, output_dir, iou_thresh=0.5, max_det_idx=2):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"PR Curves by Occlusion Level  (IoU={iou_thresh}, "
        f"maxDets={MAX_DETS[max_det_idx]})\n"
        "Dashed = Baseline  |  Solid = Updated  |  "
        "Other-level regions fully ignored (don't-care)",
        fontsize=13, y=1.02
    )
    for ax, level in zip(axes, OCCLUSION_LEVELS):
        color = LEVEL_COLORS[level]
        aps   = {}
        for model_name in ["baseline", "updated"]:
            r, p_vals = get_pr_curve(evals[level][model_name],
                                     iou_thresh=iou_thresh,
                                     area="all", max_det_idx=max_det_idx)
            ap = float(np.mean(p_vals))
            aps[model_name] = (r, p_vals, ap)
            style = MODEL_STYLES[model_name]
            ax.plot(r, p_vals, color=color,
                    ls=style["ls"], lw=style["lw"], alpha=style["alpha"],
                    label=f"{model_name.capitalize()}  AP={ap:.4f}")

        r_b, p_b, ap_b = aps["baseline"]
        r_u, p_u, ap_u = aps["updated"]
        diff = p_u - p_b
        ax.fill_between(r_u, p_b, p_u, where=(diff > 0),
                        alpha=0.15, color="green", label="Updated better")
        ax.fill_between(r_u, p_b, p_u, where=(diff < 0),
                        alpha=0.15, color="red",   label="Baseline better")
        ax.text(0.04, 0.06,
                f"ΔAP = {ap_u - ap_b:+.4f}",
                transform=ax.transAxes, fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.9))
        ax.set_title(f"{level.capitalize()} Occlusion",
                     color=color, fontsize=13, fontweight="bold")
        ax.set_xlabel("Recall",    fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out = Path(output_dir) / (
        f"pr_per_occlusion_iou{iou_thresh}_maxDets{MAX_DETS[max_det_idx]}.png"
    )
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_pr_overlay(evals, output_dir, iou_thresh=0.5, max_det_idx=2):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(
        f"PR Curves — All Occlusion Levels  (IoU={iou_thresh}, "
        f"maxDets={MAX_DETS[max_det_idx]})\n"
        "Colour = Occlusion Level  |  Dashed = Baseline  |  Solid = Updated",
        fontsize=12
    )
    for level in OCCLUSION_LEVELS:
        color = LEVEL_COLORS[level]
        for model_name in ["baseline", "updated"]:
            r, p_vals = get_pr_curve(evals[level][model_name],
                                     iou_thresh=iou_thresh,
                                     area="all", max_det_idx=max_det_idx)
            ap    = float(np.mean(p_vals))
            style = MODEL_STYLES[model_name]
            ax.plot(r, p_vals, color=color,
                    ls=style["ls"], lw=style["lw"], alpha=style["alpha"],
                    label=f"{level.capitalize()} {model_name.capitalize()} "
                          f"(AP={ap:.3f})")
    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = Path(output_dir) / (
        f"pr_overlay_iou{iou_thresh}_maxDets{MAX_DETS[max_det_idx]}.png"
    )
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_pr_maxdets_comparison(evals, output_dir, iou_thresh=0.5):
    maxdet_colors = ["#1ABC9C", "#E67E22", "#E74C3C"]
    for model_name in ["baseline", "updated"]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            f"PR Curves by maxDets — {model_name.capitalize()}  "
            f"(IoU={iou_thresh})\nmaxDets compared: {MAX_DETS}",
            fontsize=13, y=1.02
        )
        for ax, level in zip(axes, OCCLUSION_LEVELS):
            for md_idx, (md, md_color) in enumerate(zip(MAX_DETS, maxdet_colors)):
                r, p_vals = get_pr_curve(evals[level][model_name],
                                         iou_thresh=iou_thresh,
                                         area="all", max_det_idx=md_idx)
                ap = float(np.mean(p_vals))
                ax.plot(r, p_vals, color=md_color, lw=2.0,
                        label=f"maxDets={md}  AP={ap:.4f}")
            ax.set_title(f"{level.capitalize()} Occlusion",
                         color=LEVEL_COLORS[level], fontsize=13, fontweight="bold")
            ax.set_xlabel("Recall",    fontsize=11)
            ax.set_ylabel("Precision", fontsize=11)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.25)
        plt.tight_layout()
        out = Path(output_dir) / f"pr_maxdets_{model_name}_iou{iou_thresh}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")


def plot_pr_iou_comparison(evals, output_dir, max_det_idx=2):
    iou_settings = [0.50, 0.75]
    iou_colors   = ["#2980B9", "#8E44AD"]
    for model_name in ["baseline", "updated"]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            f"PR Curves by IoU Threshold — {model_name.capitalize()}\n"
            f"maxDets={MAX_DETS[max_det_idx]}",
            fontsize=13, y=1.02
        )
        for ax, level in zip(axes, OCCLUSION_LEVELS):
            for iou_thresh, iou_color in zip(iou_settings, iou_colors):
                r, p_vals = get_pr_curve(evals[level][model_name],
                                         iou_thresh=iou_thresh,
                                         area="all", max_det_idx=max_det_idx)
                ap = float(np.mean(p_vals))
                ax.plot(r, p_vals, color=iou_color, lw=2.0,
                        label=f"IoU={iou_thresh}  AP={ap:.4f}")
            ax.set_title(f"{level.capitalize()} Occlusion",
                         color=LEVEL_COLORS[level], fontsize=13, fontweight="bold")
            ax.set_xlabel("Recall",    fontsize=11)
            ax.set_ylabel("Precision", fontsize=11)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.25)
        plt.tight_layout()
        out = Path(output_dir) / (
            f"pr_iou_{model_name}_maxDets{MAX_DETS[max_det_idx]}.png"
        )
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")


def plot_pr_grid(evals, output_dir, max_det_idx=2):
    iou_settings = [0.50, 0.75]
    iou_colors   = ["#2980B9", "#8E44AD"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"PR Curves Grid — Rows: Model  |  Cols: Occlusion Level\n"
        f"maxDets={MAX_DETS[max_det_idx]}  |  Blue=IoU@.50  |  Purple=IoU@.75",
        fontsize=13
    )
    for row, model_name in enumerate(["baseline", "updated"]):
        for col, level in enumerate(OCCLUSION_LEVELS):
            ax = axes[row][col]
            for iou_thresh, iou_color in zip(iou_settings, iou_colors):
                r, p_vals = get_pr_curve(evals[level][model_name],
                                         iou_thresh=iou_thresh,
                                         area="all", max_det_idx=max_det_idx)
                ap = float(np.mean(p_vals))
                ax.plot(r, p_vals, color=iou_color, lw=2.0,
                        label=f"IoU={iou_thresh}  AP={ap:.4f}")
            ax.set_title(
                f"{model_name.capitalize()} — {level.capitalize()}",
                color=LEVEL_COLORS[level], fontsize=11, fontweight="bold"
            )
            ax.set_xlabel("Recall",    fontsize=10)
            ax.set_ylabel("Precision", fontsize=10)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out = Path(output_dir) / f"pr_grid_maxDets{MAX_DETS[max_det_idx]}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

# ── Summary tables ────────────────────────────────────────────────────────────

def print_summary_table(all_stats):
    metric_keys = list(next(iter(
        next(iter(all_stats.values())).values()
    )).keys())

    header = ["Metric"]
    for level in OCCLUSION_LEVELS:
        short = level[:3].upper()
        header += [f"{short} Base", f"{short} Upd", f"{short} Δ"]

    rows = []
    for metric in metric_keys:
        row = [metric]
        for level in OCCLUSION_LEVELS:
            b = all_stats[level]["baseline"].get(metric, float("nan"))
            u = all_stats[level]["updated"].get(metric,  float("nan"))
            d = u - b if not (np.isnan(b) or np.isnan(u)) else float("nan")
            row += [
                f"{b:.4f}"  if not np.isnan(b) else "  —  ",
                f"{u:.4f}"  if not np.isnan(u) else "  —  ",
                f"{d:+.4f}" if not np.isnan(d) else "  —  ",
            ]
        rows.append(row)

    print(f"\n{'='*130}")
    print(
        f"  OCCLUSION-WISE mAP — OcclusionCOCOeval  |  maxDets={MAX_DETS}  |  "
        "Full don't-care for other-level regions"
    )
    print(f"{'='*130}")
    if HAS_TABULATE:
        print(tabulate(rows, headers=header, tablefmt="rounded_outline"))
    else:
        col_w = [max(len(h), max(len(str(r[i])) for r in rows))
                 for i, h in enumerate(header)]
        fmt   = "  ".join(f"{{:<{w}}}" for w in col_w)
        print(fmt.format(*header))
        print("  ".join("-" * w for w in col_w))
        for row in rows:
            print(fmt.format(*[str(c) for c in row]))


def print_per_category_table(b_cat_stats, u_cat_stats, level, metric="AP@.50"):
    all_cats = sorted(set(list(b_cat_stats) + list(u_cat_stats)))
    rows = []
    for cat in all_cats:
        b = b_cat_stats.get(cat, {}).get(metric, float("nan"))
        u = u_cat_stats.get(cat, {}).get(metric, float("nan"))
        d = u - b if not (np.isnan(b) or np.isnan(u)) else float("nan")
        verdict = (
            ("✅" if d > 0.001 else ("❌" if d < -0.001 else "≈"))
            if not np.isnan(d) else "—"
        )
        rows.append([
            cat,
            f"{b:.4f}" if not np.isnan(b) else "—",
            f"{u:.4f}" if not np.isnan(u) else "—",
            f"{d:+.4f}" if not np.isnan(d) else "—",
            verdict,
        ])

    headers = ["Category", f"{metric} Base", f"{metric} Upd", "Δ", ""]
    print(f"\n── Per-Category {metric} — {level.capitalize()} Occlusion ──")
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    else:
        col_w = [max(len(h), max(len(str(r[i])) for r in rows))
                 for i, h in enumerate(headers)]
        fmt   = "  ".join(f"{{:<{w}}}" for w in col_w)
        print(fmt.format(*headers))
        print("  ".join("-" * w for w in col_w))
        for row in rows:
            print(fmt.format(*[str(c) for c in row]))


def run_per_category(gt_dict, pred_list, dont_care_iou=DONT_CARE_IOU_DEFAULT):
    """Run OcclusionCOCOeval per category."""
    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()

    gt_image_ids   = {img["id"] for img in gt_dict["images"]}
    filtered_preds = [p for p in pred_list if p["image_id"] in gt_image_ids]
    if not filtered_preds:
        return {}

    coco_dt   = coco_gt.loadRes(filtered_preds)
    cat_names = {c["id"]: c["name"] for c in gt_dict["categories"]}
    results   = {}

    for cat_id, cat_name in cat_names.items():
        coco_eval = OcclusionCOCOeval(coco_gt, coco_dt, "bbox",
                                       dont_care_iou=dont_care_iou)
        coco_eval.params.catIds  = [cat_id]
        coco_eval.params.maxDets = MAX_DETS
        coco_eval.evaluate()
        coco_eval.accumulate()
        results[cat_name] = get_stats(coco_eval)

    return results

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Occlusion-wise mAP + PR curves.\n"
            "Other-level regions are full don't-care: detections there are\n"
            "neither TP nor FP, regardless of IoU with the crowd box."
        )
    )
    parser.add_argument("--gt",           required=True)
    parser.add_argument("--baseline",     required=True)
    parser.add_argument("--updated",      required=True)
    parser.add_argument("--output",       default="occlusion_map_report")
    parser.add_argument("--dont-care-iou", type=float,
                        default=DONT_CARE_IOU_DEFAULT,
                        help=(
                            "IoU overlap with any other-level GT box above which "
                            "a detection is treated as don't-care (not FP). "
                            f"Default: {DONT_CARE_IOU_DEFAULT}. "
                            "Lower = more lenient (more detections ignored). "
                            "Set to 0.0 to ignore every detection that overlaps "
                            "any other-level region at all."
                        ))
    parser.add_argument("--per-category", action="store_true",
                        help="Per-category breakdown per occlusion level")
    args = parser.parse_args()

    for p in (args.gt, args.baseline, args.updated):
        if not Path(p).exists():
            raise FileNotFoundError(f"Not found: {p}")

    Path(args.output).mkdir(parents=True, exist_ok=True)

    dont_care_iou = args.dont_care_iou

    # ── Load ────────────────────────────────────────────────────────────────
    print(f"\nLoading GT        : {args.gt}")
    gt_full   = load_json(args.gt)
    cat_names = {c["id"]: c["name"] for c in gt_full.get("categories", [])}
    print("  Categories: " +
          ", ".join(f"{k}={v}" for k, v in sorted(cat_names.items())))

    level_counts = defaultdict(int)
    for ann in gt_full["annotations"]:
        lvl = (ann.get("attributes", {})
                   .get("occlusion_level", "unknown")
                   .strip().lower())
        level_counts[lvl] += 1
    print("  GT distribution: " +
          "  ".join(f"{lvl}={level_counts[lvl]:,}" for lvl in OCCLUSION_LEVELS))

    print(f"\nLoading Baseline  : {args.baseline}")
    b_preds = load_json(args.baseline)
    print(f"  {len(b_preds):,} predictions")

    print(f"Loading Updated   : {args.updated}")
    u_preds = load_json(args.updated)
    print(f"  {len(u_preds):,} predictions")

    print(f"\nmaxDets setting   : {MAX_DETS}")
    print(f"Don't-care IoU    : {dont_care_iou}  "
          f"(detections overlapping other-level GTs above this are not FP)")

    # ── Evaluate per occlusion level ─────────────────────────────────────────
    evals     = {}
    all_stats = {}

    for level in OCCLUSION_LEVELS:
        print(f"\n{'─'*60}")
        print(f"Evaluating: {level.upper()}")
        print(f"{'─'*60}")

        gt_subset = build_gt_for_level(gt_full, level)

        active  = sum(1 for a in gt_subset["annotations"] if a["iscrowd"] == 0)
        ignored = sum(1 for a in gt_subset["annotations"] if a["iscrowd"] == 1)
        print(f"  Active  (iscrowd=0, evaluated)    : {active:,}")
        print(f"  Ignored (iscrowd=1, don't-care)   : {ignored:,}")
        print(f"  Don't-care IoU threshold          : {dont_care_iou}")

        evals[level]     = {}
        all_stats[level] = {}

        for model_name, preds in [("baseline", b_preds), ("updated", u_preds)]:
            print(f"\n  >> {model_name.capitalize()}")
            coco_eval = build_cocoeval(gt_subset, preds,
                                       dont_care_iou=dont_care_iou)
            evals[level][model_name]     = coco_eval
            all_stats[level][model_name] = get_stats(coco_eval)

    # ── Summary table ────────────────────────────────────────────────────────
    print_summary_table(all_stats)

    # ── PR curve plots ───────────────────────────────────────────────────────
    print(f"\nGenerating plots → {args.output}/")
    for iou_thresh in [0.5, 0.75]:
        plot_pr_per_occlusion(evals, args.output,
                              iou_thresh=iou_thresh, max_det_idx=2)
        plot_pr_overlay(evals, args.output,
                        iou_thresh=iou_thresh, max_det_idx=2)

    plot_pr_maxdets_comparison(evals, args.output, iou_thresh=0.5)
    plot_pr_iou_comparison(evals, args.output, max_det_idx=2)
    plot_pr_grid(evals, args.output, max_det_idx=2)

    # ── Per-category breakdown ───────────────────────────────────────────────
    if args.per_category:
        for level in OCCLUSION_LEVELS:
            print(f"\n{'='*60}")
            print(f"Per-Category: {level.upper()}")
            print(f"{'='*60}")
            gt_subset = build_gt_for_level(gt_full, level)

            print("  >> Baseline ...")
            b_cat = run_per_category(gt_subset, b_preds,
                                     dont_care_iou=dont_care_iou)
            print("  >> Updated ...")
            u_cat = run_per_category(gt_subset, u_preds,
                                     dont_care_iou=dont_care_iou)

            for metric in ["AP@.50", "AP@[.5:.95]", "AP@.75"]:
                print_per_category_table(b_cat, u_cat, level, metric)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    save_data = {}
    for level, models in all_stats.items():
        save_data[level] = {
            m: {k: (v if not np.isnan(v) else None) for k, v in stats.items()}
            for m, stats in models.items()
        }
    out_path = Path(args.output) / "occlusion_map_results.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to : {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()