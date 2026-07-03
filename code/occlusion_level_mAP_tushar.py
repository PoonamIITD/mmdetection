"""
Occlusion-wise mAP using COCOeval — Strict Target-Potential Matching

Logic:
1. iscrowd is completely removed from the pipeline.
2. The evaluator natively evaluates ONLY the target-level ground truths.
3. A prediction is flagged as "don't care" (dtIgnore=True) IF AND ONLY IF:
    a) Its max IoU with ANY target-level GT is < 0.5 (It has no potential to be a TP).
    b) Its max IoU with ANY other-level GT is >= DONT_CARE_IOU_THRESH.
4. If it has >= 0.5 IoU with a target, it is left entirely to standard COCO 
   matching (rewarded for TP, penalized as FP for poor localization at higher thresholds).

Requirements:
    pip install pycocotools matplotlib tabulate
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import copy

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
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
DONT_CARE_IOU_DEFAULT = 0.1 

LEVEL_COLORS = {
    "light":    "#3498DB",
    "moderate": "#F39C12",
    "severe":   "#8E44AD",
}
MODEL_STYLES = {
    "baseline": {"ls": "--", "lw": 2.0, "alpha": 0.85},
    "updated":  {"ls": "-",  "lw": 2.5, "alpha": 0.95},
}

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ── GT preparation ────────────────────────────────────────────────────────────

def build_gt_for_level(gt_full, target_level):
    """
    Strips 'iscrowd' manipulation. 
    Returns:
      1. A COCO-compatible dict containing ONLY target-level annotations.
      2. A dictionary of other-level bounding boxes keyed by (image_id, category_id)
         for the custom ignore pass.
    """
    gt_subset = copy.deepcopy(gt_full)
    target_anns = []
    
    # (image_id, category_id) -> list of [x, y, w, h]
    other_gts = defaultdict(list)

    for ann in gt_subset["annotations"]:
        lvl = (ann.get("attributes", {})
                  .get("occlusion_level", "unknown")
                  .strip().lower())
        
        if lvl == target_level:
            target_anns.append(ann)
        else:
            key = (ann["image_id"], ann["category_id"])
            other_gts[key].append(np.array(ann["bbox"], dtype=np.float64))
            
    gt_subset["annotations"] = target_anns
    return gt_subset, other_gts

# ── Custom COCOeval subclass ──────────────────────────────────────────────────

class OcclusionCOCOeval(COCOeval):
    def __init__(self, coco_gt, coco_dt, other_gts, iou_type="bbox",
                 dont_care_iou=DONT_CARE_IOU_DEFAULT):
        super().__init__(coco_gt, coco_dt, iou_type)
        self.dont_care_iou = dont_care_iou
        self._other_gts = other_gts

        # Pre-calculate target GT boxes for fast "potential match" checking
        self._target_gts = defaultdict(list)
        for ann in coco_gt.dataset["annotations"]:
            key = (ann["image_id"], ann["category_id"])
            self._target_gts[key].append(np.array(ann["bbox"], dtype=np.float64))

    @staticmethod
    def _iou_one_vs_many(dt_box, gt_boxes):
        if not gt_boxes:
            return np.array([], dtype=np.float64)

        dx, dy, dw, dh = dt_box
        dt_area = dw * dh
        if dt_area <= 0:
            return np.zeros(len(gt_boxes))

        gt_arr = np.array(gt_boxes, dtype=np.float64)
        gx, gy, gw, gh = gt_arr[:, 0], gt_arr[:, 1], gt_arr[:, 2], gt_arr[:, 3]

        ix = np.maximum(0.0, np.minimum(dx + dw, gx + gw) - np.maximum(dx, gx))
        iy = np.maximum(0.0, np.minimum(dy + dh, gy + gh) - np.maximum(dy, gy))
        inter  = ix * iy
        union  = dt_area + gw * gh - inter
        return np.where(union > 0, inter / union, 0.0)

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        """
        Executes standard COCO matching first. 
        Then filters pure False Positives that overlap other-level GTs.
        """
        result = super().evaluateImg(imgId, catId, aRng, maxDet)
        if result is None:
            return result

        dt_ids = result["dtIds"]
        if len(dt_ids) == 0:
            return result

        other_boxes = self._other_gts.get((imgId, catId), [])
        target_boxes = self._target_gts.get((imgId, catId), [])

        if not other_boxes:
            return result 

        dt_ignore = result["dtIgnore"] 
        dt_anns = self.cocoDt.loadAnns(dt_ids)

        for d_idx, dt_ann in enumerate(dt_anns):
            dt_box = dt_ann.get("bbox")
            if dt_box is None:
                continue

            # Check 1: Does it have potential to match a target GT?
            ious_vs_target = self._iou_one_vs_many(dt_box, target_boxes)
            max_target_iou = ious_vs_target.max() if len(ious_vs_target) > 0 else 0.0
            
            if max_target_iou >= 0.5:
                # Potential match -> Let standard COCO math handle it (TP or FP).
                continue

            # Check 2: Does it overlap another-level GT enough to be forgiven?
            ious_vs_other = self._iou_one_vs_many(dt_box, other_boxes)
            max_other_iou = ious_vs_other.max() if len(ious_vs_other) > 0 else 0.0
            
            if max_other_iou >= self.dont_care_iou:
                # Completely missed targets, but hit an other-level object. -> Ignore.
                dt_ignore[:, d_idx] = True

        result["dtIgnore"] = dt_ignore
        return result

# ── COCOeval runner ───────────────────────────────────────────────────────────

def build_cocoeval(gt_dict, other_gts, pred_list, iou_type="bbox",
                   dont_care_iou=DONT_CARE_IOU_DEFAULT):
    coco_gt = COCO()
    coco_gt.dataset = gt_dict
    coco_gt.createIndex()

    gt_image_ids   = {img["id"] for img in gt_dict["images"]}
    filtered_preds = [p for p in pred_list if p["image_id"] in gt_image_ids]

    if not filtered_preds:
        return None

    coco_dt   = coco_gt.loadRes(filtered_preds)
    coco_eval = OcclusionCOCOeval(coco_gt, coco_dt, other_gts, iou_type,
                                   dont_care_iou=dont_care_iou)

    coco_eval.params.maxDets = MAX_DETS
    coco_eval.evaluate()
    coco_eval.accumulate()
    return coco_eval

# ── Stats extraction ──────────────────────────────────────────────────────────

def get_stats(coco_eval):
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

    return {
        "AP@[.5:.95]":                  _ap(0.50, 0.95, "all",    2),
        "AP@.50":                       _ap(0.50, 0.50, "all",    2),
        "AP@.75":                       _ap(0.75, 0.75, "all",    2),
        "AP_small":                     _ap(0.50, 0.95, "small",  2),
        "AP_medium":                    _ap(0.50, 0.95, "medium", 2),
        "AP_large":                     _ap(0.50, 0.95, "large",  2),
        f"AR@[.5:.95]_maxDets{MAX_DETS[0]}":   _ar(0.50, 0.95, "all",    0),
        f"AR@[.5:.95]_maxDets{MAX_DETS[1]}":   _ar(0.50, 0.95, "all",    1),
        f"AR@[.5:.95]_maxDets{MAX_DETS[2]}":   _ar(0.50, 0.95, "all",    2),
        "AR_small":                     _ar(0.50, 0.95, "small",  2),
        "AR_medium":                    _ar(0.50, 0.95, "medium", 2),
        "AR_large":                     _ar(0.50, 0.95, "large",  2),
    }

def get_pr_curve(coco_eval, iou_thresh=0.5, area="all", max_det_idx=-1):
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

    prec = ev["precision"][t_idx, :, :, aind[0], max_det_idx]
    prec_mean = np.array([
        float(np.mean(row[row > -1])) if np.any(row > -1) else 0.0
        for row in prec
    ])
    return np.linspace(0, 1, 101), prec_mean

# ── Plots ─────────────────────────

def plot_pr_per_occlusion(evals, output_dir, iou_thresh=0.5, max_det_idx=2):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"PR Curves by Occlusion Level  (IoU={iou_thresh}, "
        f"maxDets={MAX_DETS[max_det_idx]})\n"
        "Dashed = Baseline  |  Solid = Updated  |  Strict Potential Matching",
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
        ax.text(0.04, 0.06, f"ΔAP = {ap_u - ap_b:+.4f}",
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
    out = Path(output_dir) / f"pr_per_occlusion_iou{iou_thresh}_maxDets{MAX_DETS[max_det_idx]}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

# (Other plotting functions: plot_pr_overlay, plot_pr_maxdets_comparison, 
#  plot_pr_iou_comparison, plot_pr_grid remain identical in structure to original)

def print_summary_table(all_stats):
    metric_keys = list(next(iter(next(iter(all_stats.values())).values())).keys())
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
    print(f"  OCCLUSION-WISE mAP | maxDets={MAX_DETS} | Strict Potential Matching")
    print(f"{'='*130}")
    if HAS_TABULATE:
        print(tabulate(rows, headers=header, tablefmt="rounded_outline"))
    else:
        col_w = [max(len(h), max(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
        fmt   = "  ".join(f"{{:<{w}}}" for w in col_w)
        print(fmt.format(*header))
        print("  ".join("-" * w for w in col_w))
        for row in rows:
            print(fmt.format(*[str(c) for c in row]))

def run_per_category(gt_subset, other_gts, pred_list, dont_care_iou=DONT_CARE_IOU_DEFAULT):
    coco_gt = COCO()
    coco_gt.dataset = gt_subset
    coco_gt.createIndex()

    gt_image_ids   = {img["id"] for img in gt_subset["images"]}
    filtered_preds = [p for p in pred_list if p["image_id"] in gt_image_ids]
    if not filtered_preds:
        return {}

    coco_dt   = coco_gt.loadRes(filtered_preds)
    cat_names = {c["id"]: c["name"] for c in gt_subset["categories"]}
    results   = {}

    for cat_id, cat_name in cat_names.items():
        coco_eval = OcclusionCOCOeval(coco_gt, coco_dt, other_gts, "bbox",
                                       dont_care_iou=dont_care_iou)
        coco_eval.params.catIds  = [cat_id]
        coco_eval.params.maxDets = MAX_DETS
        coco_eval.evaluate()
        coco_eval.accumulate()
        results[cat_name] = get_stats(coco_eval)

    return results

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Occlusion-wise mAP - Strict Rules")
    parser.add_argument("--gt",           required=True)
    parser.add_argument("--baseline",     required=True)
    parser.add_argument("--updated",      required=True)
    parser.add_argument("--output",       default="occlusion_map_report")
    parser.add_argument("--dont-care-iou", type=float, default=DONT_CARE_IOU_DEFAULT)
    parser.add_argument("--per-category", action="store_true")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    dont_care_iou = args.dont_care_iou

    print(f"\nLoading GT        : {args.gt}")
    gt_full = load_json(args.gt)
    b_preds = load_json(args.baseline)
    u_preds = load_json(args.updated)

    evals     = {}
    all_stats = {}

    for level in OCCLUSION_LEVELS:
        print(f"\n{'─'*60}\nEvaluating: {level.upper()}\n{'─'*60}")
        gt_subset, other_gts = build_gt_for_level(gt_full, level)

        evals[level]     = {}
        all_stats[level] = {}

        for model_name, preds in [("baseline", b_preds), ("updated", u_preds)]:
            print(f"  >> {model_name.capitalize()}")
            coco_eval = build_cocoeval(gt_subset, other_gts, preds,
                                       dont_care_iou=dont_care_iou)
            evals[level][model_name]     = coco_eval
            all_stats[level][model_name] = get_stats(coco_eval)

    print_summary_table(all_stats)

    print(f"\nGenerating plots → {args.output}/")
    for iou_thresh in [0.5, 0.75]:
        plot_pr_per_occlusion(evals, args.output, iou_thresh=iou_thresh, max_det_idx=2)

    # ── Per-category breakdown ───────────────────────────────────────────────
    if args.per_category:
        for level in OCCLUSION_LEVELS:
            gt_subset, other_gts = build_gt_for_level(gt_full, level)
            b_cat = run_per_category(gt_subset, other_gts, b_preds, dont_care_iou)
            u_cat = run_per_category(gt_subset, other_gts, u_preds, dont_care_iou)
            # (Print tables function call here)

    save_data = {}
    for level, models in all_stats.items():
        save_data[level] = {
            m: {k: (v if not np.isnan(v) else None) for k, v in stats.items()}
            for m, stats in models.items()
        }
    out_path = Path(args.output) / "occlusion_map_results.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    main()