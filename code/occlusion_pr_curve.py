# """
# Occlusion-wise Precision-Recall Curve — Baseline vs Updated Model

# Computes and plots PR curves broken down by occlusion level (light, moderate, severe)
# for both baseline and updated model predictions against GT COCO JSON.

# Requirements:
#     pip install matplotlib numpy tabulate

# Usage:
#     python occlusion_pr_curve.py \
#         --gt        instances_validation_merged_mannual.json \
#         --baseline  results_baseline_GDINO_final_val.json \
#         --updated   results_sampling_loss_final_val.json \
#         --iou       0.5 \
#         --conf      0.05 \
#         --output    pr_curve_report
# """

# import json
# import argparse
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from collections import defaultdict
# from pathlib import Path

# try:
#     from tabulate import tabulate
#     HAS_TABULATE = True
# except ImportError:
#     HAS_TABULATE = False

# # ── Config ────────────────────────────────────────────────────────────────────
# OCCLUSION_LEVELS = ["light", "moderate", "severe"]
# LEVEL_COLORS     = {
#     "light":    "#3498DB",
#     "moderate": "#F39C12",
#     "severe":   "#8E44AD",
# }
# MODEL_STYLES = {
#     "baseline": {"ls": "--", "alpha": 0.85, "lw": 2.0},
#     "updated":  {"ls": "-",  "alpha": 0.95, "lw": 2.5},
# }

# # ── I/O ───────────────────────────────────────────────────────────────────────

# def load_json(path):
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)

# # ── IoU ───────────────────────────────────────────────────────────────────────

# def bbox_iou(a, b):
#     """COCO bbox [x,y,w,h] → IoU scalar."""
#     ax1, ay1 = a[0], a[1]
#     ax2, ay2 = a[0] + a[2], a[1] + a[3]
#     bx1, by1 = b[0], b[1]
#     bx2, by2 = b[0] + b[2], b[1] + b[3]
#     ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
#     iy = max(0.0, min(ay2, by2) - max(ay1, by1))
#     inter = ix * iy
#     if inter == 0:
#         return 0.0
#     return inter / (a[2]*a[3] + b[2]*b[3] - inter)

# # ── Core: label predictions ───────────────────────────────────────────────────

# def label_predictions(preds, gt_anns, iou_threshold=0.5):
#     """
#     Sort predictions by score descending.
#     Greedily match to GT per image — explicit category + IoU check for TP.

#     TP   = correct category + IoU >= threshold + GT not already matched
#     FP   = either wrong category OR IoU too low OR duplicate match
    
#     FP subtypes tracked:
#         'fp_localisation'  → right category, IoU > 0 but < threshold
#         'fp_wrong_class'   → IoU >= threshold but wrong category
#         'fp_background'    → IoU = 0 (pure background prediction)
#         'fp_duplicate'     → IoU >= threshold, right class, but GT already taken
#     """
#     # ── Index GT by image only (not category) so we can check cross-category IoU ──
#     gt_by_image = defaultdict(list)
#     gt_meta     = {}

#     for ann in gt_anns:
#         gt_by_image[ann["image_id"]].append(ann)
#         level = (ann.get("attributes", {})
#                     .get("occlusion_level", "unknown")
#                     .strip().lower())
#         gt_meta[ann["id"]] = {
#             "occlusion_level": level,
#             "category_id":     ann["category_id"],
#         }

#     matched_gt   = set()
#     sorted_preds = sorted(preds, key=lambda p: p["score"], reverse=True)

#     labeled = []
#     for rank, pred in enumerate(sorted_preds, start=1):
#         image_id    = pred["image_id"]
#         pred_cat    = pred["category_id"]
#         candidates  = gt_by_image.get(image_id, [])

#         best_iou         = 0.0
#         best_gt_id       = None
#         best_gt_cat      = None
#         best_already_matched = False

#         for gt in candidates:
#             iou = bbox_iou(pred["bbox"], gt["bbox"])
#             if iou > best_iou:
#                 best_iou             = iou
#                 best_gt_id           = gt["id"]
#                 best_gt_cat          = gt["category_id"]
#                 best_already_matched = gt["id"] in matched_gt

#         # ── Classify prediction ──────────────────────────────────────────
#         if best_iou >= iou_threshold and best_gt_id is not None:
#             if best_gt_cat == pred_cat and not best_already_matched:
#                 # True Positive
#                 is_tp      = True
#                 fp_type    = None
#                 matched_gt.add(best_gt_id)

#             elif best_gt_cat == pred_cat and best_already_matched:
#                 # Duplicate detection of already-matched GT
#                 is_tp   = False
#                 fp_type = "fp_duplicate"

#             elif best_gt_cat != pred_cat:
#                 # Good localisation but wrong class predicted
#                 is_tp   = False
#                 fp_type = "fp_wrong_class"

#             else:
#                 is_tp   = False
#                 fp_type = "fp_other"

#         elif best_iou > 0:
#             # Found something nearby but IoU below threshold
#             is_tp   = False
#             fp_type = "fp_localisation"

#         else:
#             # Pure background — no overlap with any GT
#             is_tp   = False
#             fp_type = "fp_background"

#         # ── Record ───────────────────────────────────────────────────────
#         occlusion = (gt_meta[best_gt_id]["occlusion_level"]
#                      if is_tp else fp_type)

#         labeled.append({
#             "rank":            rank,
#             "score":           pred["score"],
#             "is_tp":           is_tp,
#             "fp_type":         fp_type,
#             "occlusion_level": gt_meta[best_gt_id]["occlusion_level"] if is_tp else "fp",
#             "matched_gt_id":   best_gt_id if is_tp else None,
#             "pred_category":   pred_cat,
#             "gt_category":     best_gt_cat,
#             "best_iou":        best_iou,
#             "image_id":        image_id,
#         })

#     return labeled

# def print_fp_breakdown(labeled, model_name="model"):
#     from collections import Counter
#     fp_types = Counter(
#         item["fp_type"] for item in labeled if not item["is_tp"]
#     )
#     total_fp = sum(fp_types.values())
#     total_tp = sum(1 for item in labeled if item["is_tp"])

#     print(f"\n── FP Breakdown: {model_name} ─────────────────────────────")
#     print(f"  Total TP            : {total_tp:,}")
#     print(f"  Total FP            : {total_fp:,}")
#     for fp_type, count in fp_types.most_common():
#         pct = count / total_fp * 100
#         print(f"  {fp_type:<22}: {count:>8,}  ({pct:.1f}%)")

# # ── PR curve computation ──────────────────────────────────────────────────────

# def compute_pr_curve(labeled, total_gt_for_level):
#     """
#     Compute precision and recall arrays from labeled predictions.

#     total_gt_for_level: total GT annotations for this occlusion level
#                         (denominator for recall).
#     Only TPs matching this level count as true positives.
#     All predictions are still ranked together (global rank).
#     """
#     tp_count = 0
#     fp_count = 0
#     precisions = []
#     recalls    = []

#     for item in labeled:
#         if item["is_tp"]:
#             tp_count += 1
#         else:
#             fp_count += 1

#         precision = tp_count / (tp_count + fp_count)
#         recall    = tp_count / total_gt_for_level if total_gt_for_level > 0 else 0.0

#         precisions.append(precision)
#         recalls.append(recall)

#     return np.array(recalls), np.array(precisions)


# def compute_pr_for_level(labeled, level, total_gt_level):
#     """
#     Build a PR curve treating only TPs of the given occlusion level as true positives.
#     FPs = everything else (wrong class, wrong level TP, actual FP).
#     """
#     # Re-label: TP only if occlusion matches the target level
#     level_labeled = []
#     for item in labeled:
#         is_level_tp = item["is_tp"] and item["occlusion_level"] == level
#         level_labeled.append({**item, "is_level_tp": is_level_tp})

#     tp_count = 0
#     fp_count = 0
#     precisions = []
#     recalls    = []

#     for item in level_labeled:
#         if item["is_level_tp"]:
#             tp_count += 1
#         else:
#             fp_count += 1

#         precision = tp_count / (tp_count + fp_count)
#         recall    = tp_count / total_gt_level if total_gt_level > 0 else 0.0
#         precisions.append(precision)
#         recalls.append(recall)

#     return np.array(recalls), np.array(precisions)


# def interpolate_pr(recalls, precisions, num_points=101):
#     """
#     COCO-style 101-point interpolated PR curve.
#     At each recall threshold, precision = max precision at recall >= threshold.
#     """
#     recall_thresholds = np.linspace(0, 1, num_points)
#     interp_prec = np.zeros(num_points)

#     for i, r_thresh in enumerate(recall_thresholds):
#         # Max precision at recall >= r_thresh
#         mask = recalls >= r_thresh
#         interp_prec[i] = precisions[mask].max() if mask.any() else 0.0

#     return recall_thresholds, interp_prec


# def compute_ap(recalls, precisions):
#     """AP = area under interpolated PR curve (101-point COCO style)."""
#     r_interp, p_interp = interpolate_pr(recalls, precisions)
#     return float(np.mean(p_interp))


# # ── Overall PR curve (all occlusion levels combined) ─────────────────────────

# def compute_overall_pr(labeled, total_gt):
#     tp_count = 0
#     fp_count = 0
#     precisions = []
#     recalls    = []

#     for item in labeled:
#         if item["is_tp"]:
#             tp_count += 1
#         else:
#             fp_count += 1
#         precisions.append(tp_count / (tp_count + fp_count))
#         recalls.append(tp_count / total_gt if total_gt > 0 else 0.0)

#     return np.array(recalls), np.array(precisions)


# # ── Plots ─────────────────────────────────────────────────────────────────────

# def plot_pr_per_occlusion(b_labeled, u_labeled, gt_counts, output_dir, iou_thresh):
#     """
#     3-panel plot: one PR curve per occlusion level, baseline vs updated.
#     """
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#     fig.suptitle(
#         f"Precision-Recall Curves by Occlusion Level  (IoU ≥ {iou_thresh})\n"
#         "Dashed = Baseline   |   Solid = Updated   |   Higher curve = better",
#         fontsize=13, y=1.02
#     )

#     ap_results = {}

#     for ax, level in zip(axes, OCCLUSION_LEVELS):
#         n_gt = gt_counts.get(level, 0)
#         color = LEVEL_COLORS[level]
#         ap_results[level] = {}

#         for model_name, labeled in [("baseline", b_labeled), ("updated", u_labeled)]:
#             r, p = compute_pr_for_level(labeled, level, n_gt)
#             r_i, p_i = interpolate_pr(r, p)
#             ap = float(np.mean(p_i))
#             ap_results[level][model_name] = ap

#             style = MODEL_STYLES[model_name]
#             label = f"{model_name.capitalize()}  AP={ap:.4f}"
#             ax.plot(r_i, p_i,
#                     color=color,
#                     ls=style["ls"],
#                     lw=style["lw"],
#                     alpha=style["alpha"],
#                     label=label)

#         # Fill between baseline and updated
#         r_b, p_b = compute_pr_for_level(b_labeled, level, n_gt)
#         r_u, p_u = compute_pr_for_level(u_labeled, level, n_gt)
#         r_i_b, p_i_b = interpolate_pr(r_b, p_b)
#         r_i_u, p_i_u = interpolate_pr(r_u, p_u)

#         improvement = p_i_u - p_i_b
#         ax.fill_between(r_i_u, p_i_b, p_i_u,
#                         where=(improvement > 0),
#                         alpha=0.15, color="green",  label="Updated better")
#         ax.fill_between(r_i_u, p_i_b, p_i_u,
#                         where=(improvement < 0),
#                         alpha=0.15, color="red",    label="Baseline better")

#         delta_ap = ap_results[level]["updated"] - ap_results[level]["baseline"]
#         ax.text(0.04, 0.06,
#                 f"ΔAP = {delta_ap:+.4f}\nn(GT) = {n_gt:,}",
#                 transform=ax.transAxes, fontsize=10,
#                 bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.9))

#         ax.set_title(f"{level.capitalize()} Occlusion",
#                      color=color, fontsize=13, fontweight="bold")
#         ax.set_xlabel("Recall",    fontsize=11)
#         ax.set_ylabel("Precision", fontsize=11)
#         ax.set_xlim(0, 1)
#         ax.set_ylim(0, 1.05)
#         ax.legend(fontsize=9, loc="upper right")
#         ax.grid(True, alpha=0.25)

#     plt.tight_layout()
#     path = Path(output_dir) / "pr_curve_per_occlusion.png"
#     plt.savefig(path, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"  Saved: {path}")
#     return ap_results


# def plot_pr_all_levels_overlay(b_labeled, u_labeled, gt_counts,
#                                 output_dir, iou_thresh):
#     """
#     Single plot: all 3 occlusion levels overlaid, baseline vs updated.
#     Colour = occlusion level, style = model.
#     """
#     fig, ax = plt.subplots(figsize=(10, 8))
#     ax.set_title(
#         f"PR Curves — All Occlusion Levels Overlaid  (IoU ≥ {iou_thresh})\n"
#         "Colour = Occlusion Level   |   Dashed = Baseline   |   Solid = Updated",
#         fontsize=12
#     )

#     for level in OCCLUSION_LEVELS:
#         n_gt  = gt_counts.get(level, 0)
#         color = LEVEL_COLORS[level]

#         for model_name, labeled in [("baseline", b_labeled), ("updated", u_labeled)]:
#             r, p = compute_pr_for_level(labeled, level, n_gt)
#             r_i, p_i = interpolate_pr(r, p)
#             ap = float(np.mean(p_i))

#             style = MODEL_STYLES[model_name]
#             ax.plot(r_i, p_i,
#                     color=color,
#                     ls=style["ls"],
#                     lw=style["lw"],
#                     alpha=style["alpha"],
#                     label=f"{level.capitalize()} {model_name.capitalize()} (AP={ap:.3f})")

#     ax.set_xlabel("Recall",    fontsize=12)
#     ax.set_ylabel("Precision", fontsize=12)
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1.05)
#     ax.legend(fontsize=9, loc="upper right", ncol=2)
#     ax.grid(True, alpha=0.25)

#     plt.tight_layout()
#     path = Path(output_dir) / "pr_curve_all_levels_overlay.png"
#     plt.savefig(path, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"  Saved: {path}")


# def plot_pr_overall(b_labeled, u_labeled, total_gt, output_dir, iou_thresh):
#     """Overall PR curve (all occlusion levels combined)."""
#     fig, ax = plt.subplots(figsize=(8, 7))

#     for model_name, labeled, color in [
#         ("baseline", b_labeled, "#E74C3C"),
#         ("updated",  u_labeled, "#2ECC71"),
#     ]:
#         r, p = compute_overall_pr(labeled, total_gt)
#         r_i, p_i = interpolate_pr(r, p)
#         ap = float(np.mean(p_i))
#         style = MODEL_STYLES[model_name]
#         ax.plot(r_i, p_i,
#                 color=color, lw=style["lw"]+0.5,
#                 label=f"{model_name.capitalize()}  AP={ap:.4f}")

#     # Fill between
#     r_b, p_b = compute_overall_pr(b_labeled, total_gt)
#     r_u, p_u = compute_overall_pr(u_labeled, total_gt)
#     r_ib, p_ib = interpolate_pr(r_b, p_b)
#     r_iu, p_iu = interpolate_pr(r_u, p_u)
#     improvement = p_iu - p_ib
#     ax.fill_between(r_iu, p_ib, p_iu,
#                     where=(improvement > 0), alpha=0.15,
#                     color="green", label="Updated better")
#     ax.fill_between(r_iu, p_ib, p_iu,
#                     where=(improvement < 0), alpha=0.15,
#                     color="red", label="Baseline better")

#     ax.set_title(f"Overall PR Curve — Baseline vs Updated  (IoU ≥ {iou_thresh})",
#                  fontsize=13)
#     ax.set_xlabel("Recall",    fontsize=12)
#     ax.set_ylabel("Precision", fontsize=12)
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1.05)
#     ax.legend(fontsize=10)
#     ax.grid(True, alpha=0.25)

#     plt.tight_layout()
#     path = Path(output_dir) / "pr_curve_overall.png"
#     plt.savefig(path, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"  Saved: {path}")


# def plot_precision_at_recall(b_labeled, u_labeled, gt_counts,
#                               output_dir, iou_thresh):
#     """
#     Bar chart: Precision @ fixed recall thresholds (0.5, 0.75, 0.9)
#     per occlusion level — baseline vs updated.
#     """
#     recall_thresholds = [0.50, 0.75, 0.90]
#     results = defaultdict(lambda: defaultdict(dict))

#     for level in OCCLUSION_LEVELS:
#         n_gt = gt_counts.get(level, 0)
#         for model_name, labeled in [("baseline", b_labeled), ("updated", u_labeled)]:
#             r, p = compute_pr_for_level(labeled, level, n_gt)
#             r_i, p_i = interpolate_pr(r, p, num_points=1001)
#             for r_thresh in recall_thresholds:
#                 idx = int(r_thresh * 1000)
#                 results[level][model_name][r_thresh] = float(p_i[idx])

#     fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
#     fig.suptitle(
#         "Precision @ Fixed Recall Thresholds by Occlusion Level\n"
#         "Red = Baseline   |   Green = Updated   |   Higher = Better",
#         fontsize=13, y=1.02
#     )

#     x    = np.arange(len(recall_thresholds))
#     w    = 0.35
#     xlabels = [f"R@{r}" for r in recall_thresholds]

#     for ax, level in zip(axes, OCCLUSION_LEVELS):
#         b_vals = [results[level]["baseline"][r] for r in recall_thresholds]
#         u_vals = [results[level]["updated"][r]  for r in recall_thresholds]

#         bars_b = ax.bar(x - w/2, b_vals, w,
#                         label="Baseline", color="#E74C3C", alpha=0.85)
#         bars_u = ax.bar(x + w/2, u_vals, w,
#                         label="Updated",  color="#2ECC71", alpha=0.85)

#         # Annotate delta
#         for i, (bv, uv) in enumerate(zip(b_vals, u_vals)):
#             delta = uv - bv
#             color = "darkgreen" if delta >= 0 else "darkred"
#             ax.text(i + w/2, uv + 0.01, f"{delta:+.3f}",
#                     ha="center", va="bottom", fontsize=9,
#                     color=color, fontweight="bold")

#         ax.set_title(f"{level.capitalize()} Occlusion",
#                      color=LEVEL_COLORS[level], fontsize=12, fontweight="bold")
#         ax.set_xticks(x)
#         ax.set_xticklabels(xlabels, fontsize=10)
#         ax.set_ylabel("Precision", fontsize=11)
#         ax.set_ylim(0, 1.15)
#         ax.legend(fontsize=9)
#         ax.grid(True, alpha=0.25, axis="y")

#     plt.tight_layout()
#     path = Path(output_dir) / "precision_at_recall_thresholds.png"
#     plt.savefig(path, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"  Saved: {path}")


# # ── Summary table ─────────────────────────────────────────────────────────────

# def print_ap_summary(ap_results, total_gt_counts):
#     rows = []
#     for level in OCCLUSION_LEVELS:
#         b_ap  = ap_results[level]["baseline"]
#         u_ap  = ap_results[level]["updated"]
#         delta = u_ap - b_ap
#         pct   = delta / b_ap * 100 if b_ap > 0 else 0.0
#         rows.append([
#             level.capitalize(),
#             total_gt_counts.get(level, 0),
#             f"{b_ap:.4f}",
#             f"{u_ap:.4f}",
#             f"{delta:+.4f}",
#             f"{pct:+.2f}%",
#             "✅" if delta > 0 else "❌",
#         ])

#     headers = ["Occlusion", "n(GT)", "AP Baseline", "AP Updated",
#                "ΔAP", "% Change", ""]
#     print("\n" + "="*75)
#     print("  OCCLUSION-WISE AP SUMMARY (PR Curve AUC)")
#     print("="*75)
#     if HAS_TABULATE:
#         print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
#     else:
#         col_w = [max(len(h), max(len(str(r[i])) for r in rows))
#                  for i, h in enumerate(headers)]
#         fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
#         print(fmt.format(*headers))
#         print("  ".join("-"*w for w in col_w))
#         for row in rows:
#             print(fmt.format(*[str(c) for c in row]))


# # ── Main ──────────────────────────────────────────────────────────────────────

# def main():
#     parser = argparse.ArgumentParser(
#         description="Occlusion-wise PR curves: Baseline vs Updated model."
#     )
#     parser.add_argument("--gt",       required=True)
#     parser.add_argument("--baseline", required=True)
#     parser.add_argument("--updated",  required=True)
#     parser.add_argument("--iou",      type=float, default=0.5)
#     parser.add_argument("--conf",     type=float, default=0.05)
#     parser.add_argument("--output",   default="pr_curve_report")
#     args = parser.parse_args()

#     for p in (args.gt, args.baseline, args.updated):
#         if not Path(p).exists():
#             raise FileNotFoundError(f"Not found: {p}")

#     Path(args.output).mkdir(parents=True, exist_ok=True)

#     # ── Load ────────────────────────────────────────────────────────────────
#     print(f"\nLoading GT        : {args.gt}")
#     gt      = load_json(args.gt)
#     gt_anns = gt["annotations"]
#     total_gt = len(gt_anns)

#     # Count GT per occlusion level
#     gt_counts = defaultdict(int)
#     for ann in gt_anns:
#         level = ann.get("attributes", {}).get(
#             "occlusion_level", "unknown").strip().lower()
#         if level in OCCLUSION_LEVELS:
#             gt_counts[level] += 1

#     print(f"  GT counts: " +
#           "  ".join(f"{l}={gt_counts[l]:,}" for l in OCCLUSION_LEVELS))

#     print(f"\nLoading Baseline  : {args.baseline}")
#     b_preds = [p for p in load_json(args.baseline) if p["score"] >= args.conf]
#     print(f"  Kept {len(b_preds):,} predictions (conf >= {args.conf})")

#     print(f"Loading Updated   : {args.updated}")
#     u_preds = [p for p in load_json(args.updated) if p["score"] >= args.conf]
#     print(f"  Kept {len(u_preds):,} predictions (conf >= {args.conf})")

#     # ── Label ───────────────────────────────────────────────────────────────
#     print(f"\nLabeling predictions (IoU >= {args.iou}) ...")
#     print("  Labeling baseline ...")
#     b_labeled = label_predictions(b_preds, gt_anns, args.iou)
#     print("  Labeling updated  ...")
#     u_labeled = label_predictions(u_preds, gt_anns, args.iou)

#     # ── Plots ───────────────────────────────────────────────────────────────
#     print(f"\nGenerating plots → {args.output}/")
#     ap_results = plot_pr_per_occlusion(
#         b_labeled, u_labeled, gt_counts, args.output, args.iou)
#     plot_pr_all_levels_overlay(
#         b_labeled, u_labeled, gt_counts, args.output, args.iou)
#     plot_pr_overall(
#         b_labeled, u_labeled, total_gt, args.output, args.iou)
#     plot_precision_at_recall(
#         b_labeled, u_labeled, gt_counts, args.output, args.iou)

#     # ── Summary ─────────────────────────────────────────────────────────────
#     print_ap_summary(ap_results, gt_counts)
#     print_fp_breakdown(b_labeled, model_name="Baseline")
#     print_fp_breakdown(u_labeled, model_name="Updated")

#     from collections import Counter
#     print("\nBaseline")
#     confusion = Counter(
#         (item["pred_category"], item["gt_category"])
#         for item in b_labeled
#         if item["fp_type"] == "fp_wrong_class"
#     )
#     for (pred_cat, gt_cat), count in confusion.most_common(10):
#         print(f"  Predicted {pred_cat} but was {gt_cat}: {count}")

#     print("\nUpdated")
#     confusion = Counter(
#         (item["pred_category"], item["gt_category"])
#         for item in u_labeled
#         if item["fp_type"] == "fp_wrong_class"
#     )
#     for (pred_cat, gt_cat), count in confusion.most_common(10):
#         print(f"  Predicted {pred_cat} but was {gt_cat}: {count}")
    
#     print("\nDone.")



# if __name__ == "__main__":
#     main()




"""
Occlusion-wise Precision-Recall Curve — Baseline vs Updated Model

Computes and plots PR curves broken down by occlusion level (light, moderate, severe)
for both baseline and updated model predictions against GT COCO JSON.

Key fix: When computing per-occlusion PR curves, TPs from OTHER occlusion levels
are IGNORED (not counted as FP). This matches COCO's per-size AP methodology.

Requirements:
    pip install matplotlib numpy tabulate

Usage:
    python occlusion_pr_curve.py \
        --gt        instances_validation_merged_mannual.json \
        --baseline  results_baseline_GDINO_final_val.json \
        --updated   results_sampling_loss_final_val.json \
        --iou       0.5 \
        --conf      0.05 \
        --output    pr_curve_report
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from pathlib import Path

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# ── Config ────────────────────────────────────────────────────────────────────
OCCLUSION_LEVELS = ["light", "moderate", "severe"]
LEVEL_COLORS = {
    "light":    "#3498DB",
    "moderate": "#F39C12",
    "severe":   "#8E44AD",
}
MODEL_STYLES = {
    "baseline": {"ls": "--", "alpha": 0.85, "lw": 2.0},
    "updated":  {"ls": "-",  "alpha": 0.95, "lw": 2.5},
}

# ── I/O ───────────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ── IoU ───────────────────────────────────────────────────────────────────────

def bbox_iou(a, b):
    """COCO bbox [x,y,w,h] → IoU scalar."""
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter == 0:
        return 0.0
    return inter / (a[2]*a[3] + b[2]*b[3] - inter)

# ── Core: label predictions ───────────────────────────────────────────────────

def label_predictions(preds, gt_anns, iou_threshold=0.5):
    """
    Sort predictions by score descending.
    Greedily match to GT per image — explicit category + IoU check for TP.

    Each prediction is labeled with:
      is_tp           : bool
      fp_type         : None if TP, else one of:
                        'fp_localisation' → right category, 0 < IoU < threshold
                        'fp_wrong_class'  → IoU >= threshold but wrong category
                        'fp_background'   → IoU = 0
                        'fp_duplicate'    → right category+IoU but GT already matched
      occlusion_level : occlusion level of matched GT (if TP), else 'fp'
      
    NOTE: occlusion_level is stored for ALL predictions so per-level PR curves
    can correctly IGNORE other-level TPs instead of counting them as FPs.
    """
    gt_by_image = defaultdict(list)
    gt_meta     = {}

    for ann in gt_anns:
        gt_by_image[ann["image_id"]].append(ann)
        level = (ann.get("attributes", {})
                    .get("occlusion_level", "unknown")
                    .strip().lower())
        gt_meta[ann["id"]] = {
            "occlusion_level": level,
            "category_id":     ann["category_id"],
        }

    matched_gt   = set()
    sorted_preds = sorted(preds, key=lambda p: p["score"], reverse=True)

    labeled = []
    for rank, pred in enumerate(sorted_preds, start=1):
        image_id   = pred["image_id"]
        pred_cat   = pred["category_id"]
        candidates = gt_by_image.get(image_id, [])

        best_iou              = 0.0
        best_gt_id            = None
        best_gt_cat           = None
        best_gt_occlusion     = None
        best_already_matched  = False

        for gt in candidates:
            iou = bbox_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou             = iou
                best_gt_id           = gt["id"]
                best_gt_cat          = gt_meta[gt["id"]]["category_id"]
                best_gt_occlusion    = gt_meta[gt["id"]]["occlusion_level"]
                best_already_matched = gt["id"] in matched_gt

        # ── Classify ─────────────────────────────────────────────────────
        if best_iou >= iou_threshold and best_gt_id is not None:
            if best_gt_cat == pred_cat and not best_already_matched:
                is_tp   = True
                fp_type = None
                matched_gt.add(best_gt_id)
            elif best_gt_cat == pred_cat and best_already_matched:
                is_tp   = False
                fp_type = "fp_duplicate"
            elif best_gt_cat != pred_cat:
                is_tp   = False
                fp_type = "fp_wrong_class"
            else:
                is_tp   = False
                fp_type = "fp_other"
        elif best_iou > 0:
            is_tp   = False
            fp_type = "fp_localisation"
        else:
            is_tp   = False
            fp_type = "fp_background"

        labeled.append({
            "rank":            rank,
            "score":           pred["score"],
            "is_tp":           is_tp,
            "fp_type":         fp_type,
            # For TPs: the occlusion level of the matched GT
            # For FPs: the occlusion level of the BEST overlapping GT (may be None)
            #          used so we can ignore other-level TPs in per-level curves
            "occlusion_level": best_gt_occlusion if is_tp else "fp",
            "best_gt_occlusion": best_gt_occlusion,  # always stored
            "matched_gt_id":   best_gt_id if is_tp else None,
            "pred_category":   pred_cat,
            "gt_category":     best_gt_cat,
            "best_iou":        best_iou,
            "image_id":        image_id,
        })

    return labeled

# ── PR curve per occlusion level ──────────────────────────────────────────────

def compute_pr_for_level(labeled, level, total_gt_level):
    """
    PR curve for a specific occlusion level.

    CORRECT methodology (matches COCO per-size AP):
      TP     = prediction matched to a GT of THIS occlusion level
      FP     = genuine false positive (background, wrong class, localisation, duplicate)
      IGNORE = prediction matched to a GT of a DIFFERENT occlusion level
               → skipped entirely, not counted as TP or FP

    This prevents light-occlusion TPs from polluting the severe PR curve.
    """
    tp_count = 0
    fp_count = 0
    precisions = []
    recalls    = []

    for item in labeled:
        if item["is_tp"]:
            if item["occlusion_level"] == level:
                # TP for this level → count as TP
                tp_count += 1
            else:
                # TP for a different level → IGNORE (skip this prediction)
                continue
        else:
            # Genuine FP → always count as FP regardless of level
            fp_count += 1

        total = tp_count + fp_count
        precision = tp_count / total if total > 0 else 1.0
        recall    = tp_count / total_gt_level if total_gt_level > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)

    return np.array(recalls), np.array(precisions)


def compute_pr_for_level_old(labeled, level, total_gt_level):
    """
    OLD (incorrect) method — kept for comparison.
    Counts other-level TPs as FPs, deflating precision.
    """
    tp_count = 0
    fp_count = 0
    precisions = []
    recalls    = []

    for item in labeled:
        is_level_tp = item["is_tp"] and item["occlusion_level"] == level
        if is_level_tp:
            tp_count += 1
        else:
            fp_count += 1  # ← wrongly counts other-level TPs as FP

        precision = tp_count / (tp_count + fp_count)
        recall    = tp_count / total_gt_level if total_gt_level > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)

    return np.array(recalls), np.array(precisions)


def interpolate_pr(recalls, precisions, num_points=101):
    """COCO-style 101-point interpolated PR curve."""
    recall_thresholds = np.linspace(0, 1, num_points)
    interp_prec = np.zeros(num_points)
    for i, r_thresh in enumerate(recall_thresholds):
        mask = recalls >= r_thresh
        interp_prec[i] = precisions[mask].max() if mask.any() else 0.0
    return recall_thresholds, interp_prec


def compute_overall_pr(labeled, total_gt):
    """Overall PR curve — all occlusion levels combined, no ignoring."""
    tp_count = 0
    fp_count = 0
    precisions = []
    recalls    = []
    for item in labeled:
        if item["is_tp"]:
            tp_count += 1
        else:
            fp_count += 1
        precisions.append(tp_count / (tp_count + fp_count))
        recalls.append(tp_count / total_gt if total_gt > 0 else 0.0)
    return np.array(recalls), np.array(precisions)

# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_pr_per_occlusion(b_labeled, u_labeled, gt_counts, output_dir, iou_thresh):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Precision-Recall Curves by Occlusion Level  (IoU ≥ {iou_thresh})\n"
        "Dashed = Baseline  |  Solid = Updated  |  Other-level TPs ignored (not FP)",
        fontsize=13, y=1.02
    )

    ap_results = {}

    for ax, level in zip(axes, OCCLUSION_LEVELS):
        n_gt  = gt_counts.get(level, 0)
        color = LEVEL_COLORS[level]
        ap_results[level] = {}

        for model_name, labeled in [("baseline", b_labeled), ("updated", u_labeled)]:
            r, p   = compute_pr_for_level(labeled, level, n_gt)
            r_i, p_i = interpolate_pr(r, p)
            ap     = float(np.mean(p_i))
            ap_results[level][model_name] = ap

            style = MODEL_STYLES[model_name]
            ax.plot(r_i, p_i,
                    color=color, ls=style["ls"],
                    lw=style["lw"], alpha=style["alpha"],
                    label=f"{model_name.capitalize()}  AP={ap:.4f}")

        # Fill between curves
        r_b, p_b   = compute_pr_for_level(b_labeled, level, n_gt)
        r_u, p_u   = compute_pr_for_level(u_labeled, level, n_gt)
        r_ib, p_ib = interpolate_pr(r_b, p_b)
        r_iu, p_iu = interpolate_pr(r_u, p_u)
        improvement = p_iu - p_ib
        ax.fill_between(r_iu, p_ib, p_iu, where=(improvement > 0),
                        alpha=0.15, color="green", label="Updated better")
        ax.fill_between(r_iu, p_ib, p_iu, where=(improvement < 0),
                        alpha=0.15, color="red",   label="Baseline better")

        delta_ap = ap_results[level]["updated"] - ap_results[level]["baseline"]
        ax.text(0.04, 0.06,
                f"ΔAP = {delta_ap:+.4f}\nn(GT) = {n_gt:,}",
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.9))

        ax.set_title(f"{level.capitalize()} Occlusion",
                     color=color, fontsize=13, fontweight="bold")
        ax.set_xlabel("Recall",    fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = Path(output_dir) / "pr_curve_per_occlusion.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return ap_results


def plot_methodology_comparison(b_labeled, gt_counts, output_dir, iou_thresh):
    """
    Side-by-side: OLD method (other-level TPs = FP) vs NEW method (ignored).
    Shows how much the old method was underestimating precision.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Methodology Comparison — Baseline Model\n"
        "OLD: other-level TPs counted as FP  |  NEW: other-level TPs ignored",
        fontsize=13, y=1.02
    )

    for ax, level in zip(axes, OCCLUSION_LEVELS):
        n_gt  = gt_counts.get(level, 0)
        color = LEVEL_COLORS[level]

        r_old, p_old = compute_pr_for_level_old(b_labeled, level, n_gt)
        r_new, p_new = compute_pr_for_level(b_labeled, level, n_gt)

        r_oi, p_oi = interpolate_pr(r_old, p_old)
        r_ni, p_ni = interpolate_pr(r_new, p_new)

        ap_old = float(np.mean(p_oi))
        ap_new = float(np.mean(p_ni))

        ax.plot(r_oi, p_oi, color="gray",  lw=2.0, ls="--",
                label=f"OLD (other=FP)  AP={ap_old:.4f}")
        ax.plot(r_ni, p_ni, color=color,   lw=2.5, ls="-",
                label=f"NEW (other=IGN) AP={ap_new:.4f}")

        ax.fill_between(r_ni, p_oi, p_ni, where=(p_ni >= p_oi),
                        alpha=0.2, color=color,
                        label=f"Precision gain = {ap_new - ap_old:+.4f}")

        ax.set_title(f"{level.capitalize()} Occlusion",
                     color=color, fontsize=13, fontweight="bold")
        ax.set_xlabel("Recall",    fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = Path(output_dir) / "methodology_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_pr_all_levels_overlay(b_labeled, u_labeled, gt_counts, output_dir, iou_thresh):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(
        f"PR Curves — All Occlusion Levels Overlaid  (IoU ≥ {iou_thresh})\n"
        "Colour = Occlusion Level  |  Dashed = Baseline  |  Solid = Updated\n"
        "Other-level TPs ignored in each curve",
        fontsize=11
    )

    for level in OCCLUSION_LEVELS:
        n_gt  = gt_counts.get(level, 0)
        color = LEVEL_COLORS[level]
        for model_name, labeled in [("baseline", b_labeled), ("updated", u_labeled)]:
            r, p     = compute_pr_for_level(labeled, level, n_gt)
            r_i, p_i = interpolate_pr(r, p)
            ap       = float(np.mean(p_i))
            style    = MODEL_STYLES[model_name]
            ax.plot(r_i, p_i, color=color,
                    ls=style["ls"], lw=style["lw"], alpha=style["alpha"],
                    label=f"{level.capitalize()} {model_name.capitalize()} (AP={ap:.3f})")

    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    path = Path(output_dir) / "pr_curve_all_levels_overlay.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_pr_overall(b_labeled, u_labeled, total_gt, output_dir, iou_thresh):
    fig, ax = plt.subplots(figsize=(8, 7))
    for model_name, labeled, color in [
        ("baseline", b_labeled, "#E74C3C"),
        ("updated",  u_labeled, "#2ECC71"),
    ]:
        r, p     = compute_overall_pr(labeled, total_gt)
        r_i, p_i = interpolate_pr(r, p)
        ap       = float(np.mean(p_i))
        style    = MODEL_STYLES[model_name]
        ax.plot(r_i, p_i, color=color, lw=style["lw"]+0.5,
                label=f"{model_name.capitalize()}  AP={ap:.4f}")

    r_b, p_b   = compute_overall_pr(b_labeled, total_gt)
    r_u, p_u   = compute_overall_pr(u_labeled, total_gt)
    r_ib, p_ib = interpolate_pr(r_b, p_b)
    r_iu, p_iu = interpolate_pr(r_u, p_u)
    improvement = p_iu - p_ib
    ax.fill_between(r_iu, p_ib, p_iu, where=(improvement > 0),
                    alpha=0.15, color="green", label="Updated better")
    ax.fill_between(r_iu, p_ib, p_iu, where=(improvement < 0),
                    alpha=0.15, color="red",   label="Baseline better")

    ax.set_title(f"Overall PR Curve — Baseline vs Updated  (IoU ≥ {iou_thresh})",
                 fontsize=13)
    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    path = Path(output_dir) / "pr_curve_overall.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_precision_at_recall(b_labeled, u_labeled, gt_counts, output_dir, iou_thresh):
    recall_thresholds = [0.50, 0.75, 0.90]
    results = defaultdict(lambda: defaultdict(dict))

    for level in OCCLUSION_LEVELS:
        n_gt = gt_counts.get(level, 0)
        for model_name, labeled in [("baseline", b_labeled), ("updated", u_labeled)]:
            r, p     = compute_pr_for_level(labeled, level, n_gt)
            r_i, p_i = interpolate_pr(r, p, num_points=1001)
            for r_thresh in recall_thresholds:
                idx = int(r_thresh * 1000)
                results[level][model_name][r_thresh] = float(p_i[idx])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle(
        "Precision @ Fixed Recall Thresholds by Occlusion Level\n"
        "Red = Baseline  |  Green = Updated  |  Other-level TPs ignored",
        fontsize=13, y=1.02
    )

    x       = np.arange(len(recall_thresholds))
    w       = 0.35
    xlabels = [f"R@{r}" for r in recall_thresholds]

    for ax, level in zip(axes, OCCLUSION_LEVELS):
        b_vals = [results[level]["baseline"][r] for r in recall_thresholds]
        u_vals = [results[level]["updated"][r]  for r in recall_thresholds]

        ax.bar(x - w/2, b_vals, w, label="Baseline", color="#E74C3C", alpha=0.85)
        ax.bar(x + w/2, u_vals, w, label="Updated",  color="#2ECC71", alpha=0.85)

        for i, (bv, uv) in enumerate(zip(b_vals, u_vals)):
            delta = uv - bv
            color = "darkgreen" if delta >= 0 else "darkred"
            ax.text(i + w/2, uv + 0.01, f"{delta:+.3f}",
                    ha="center", va="bottom", fontsize=9,
                    color=color, fontweight="bold")

        ax.set_title(f"{level.capitalize()} Occlusion",
                     color=LEVEL_COLORS[level], fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=10)
        ax.set_ylabel("Precision", fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    path = Path(output_dir) / "precision_at_recall_thresholds.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# ── Summary tables ────────────────────────────────────────────────────────────

def print_ap_summary(ap_results, gt_counts):
    rows = []
    for level in OCCLUSION_LEVELS:
        b_ap  = ap_results[level]["baseline"]
        u_ap  = ap_results[level]["updated"]
        delta = u_ap - b_ap
        pct   = delta / b_ap * 100 if b_ap > 0 else 0.0
        rows.append([
            level.capitalize(),
            gt_counts.get(level, 0),
            f"{b_ap:.4f}",
            f"{u_ap:.4f}",
            f"{delta:+.4f}",
            f"{pct:+.2f}%",
            "✅" if delta > 0 else "❌",
        ])

    headers = ["Occlusion", "n(GT)", "AP Baseline", "AP Updated",
               "ΔAP", "% Change", ""]
    print("\n" + "="*75)
    print("  OCCLUSION-WISE AP SUMMARY (other-level TPs ignored — correct method)")
    print("="*75)
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    else:
        col_w = [max(len(h), max(len(str(r[i])) for r in rows))
                 for i, h in enumerate(headers)]
        fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
        print(fmt.format(*headers))
        print("  ".join("-"*w for w in col_w))
        for row in rows:
            print(fmt.format(*[str(c) for c in row]))


def print_methodology_ap_comparison(b_labeled, gt_counts):
    """Show AP difference between old and new method for baseline."""
    rows = []
    for level in OCCLUSION_LEVELS:
        n_gt = gt_counts.get(level, 0)
        r_old, p_old = compute_pr_for_level_old(b_labeled, level, n_gt)
        r_new, p_new = compute_pr_for_level(b_labeled, level, n_gt)
        ap_old = float(np.mean(interpolate_pr(r_old, p_old)[1]))
        ap_new = float(np.mean(interpolate_pr(r_new, p_new)[1]))
        rows.append([
            level.capitalize(),
            f"{ap_old:.4f}",
            f"{ap_new:.4f}",
            f"{ap_new - ap_old:+.4f}",
            "↑ higher (correct)" if ap_new > ap_old else "same",
        ])

    headers = ["Occlusion", "AP (old: other=FP)", "AP (new: other=IGN)", "Δ", "Note"]
    print("\n" + "="*75)
    print("  METHODOLOGY IMPACT — How much did the fix change AP? (Baseline)")
    print("="*75)
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    else:
        col_w = [max(len(h), max(len(str(r[i])) for r in rows))
                 for i, h in enumerate(headers)]
        fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
        print(fmt.format(*headers))
        print("  ".join("-"*w for w in col_w))
        for row in rows:
            print(fmt.format(*[str(c) for c in row]))


def print_fp_breakdown(labeled, model_name="model"):
    fp_types = Counter(item["fp_type"] for item in labeled if not item["is_tp"])
    total_fp = sum(fp_types.values())
    total_tp = sum(1 for item in labeled if item["is_tp"])

    print(f"\n── FP Breakdown: {model_name} ─────────────────────────────")
    print(f"  Total TP            : {total_tp:,}")
    print(f"  Total FP            : {total_fp:,}")
    for fp_type, count in fp_types.most_common():
        pct = count / total_fp * 100
        print(f"  {fp_type:<22}: {count:>8,}  ({pct:.1f}%)")


def print_confusion(labeled, model_name, cat_names=None, top_k=10):
    confusion = Counter(
        (item["pred_category"], item["gt_category"])
        for item in labeled
        if item["fp_type"] == "fp_wrong_class"
    )
    def name(cat_id):
        if cat_names and cat_id in cat_names:
            return f"{cat_names[cat_id]}({cat_id})"
        return str(cat_id)

    print(f"\n{model_name} — Top-{top_k} wrong-class confusions:")
    for (pred_cat, gt_cat), count in confusion.most_common(top_k):
        print(f"  Predicted {name(pred_cat):20s} but was {name(gt_cat):20s}: {count}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Occlusion-wise PR curves with correct ignore methodology."
    )
    parser.add_argument("--gt",       required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--updated",  required=True)
    parser.add_argument("--iou",      type=float, default=0.5)
    parser.add_argument("--conf",     type=float, default=0.05)
    parser.add_argument("--output",   default="pr_curve_report")
    args = parser.parse_args()

    for p in (args.gt, args.baseline, args.updated):
        if not Path(p).exists():
            raise FileNotFoundError(f"Not found: {p}")

    Path(args.output).mkdir(parents=True, exist_ok=True)

    # ── Load ────────────────────────────────────────────────────────────────
    print(f"\nLoading GT        : {args.gt}")
    gt       = load_json(args.gt)
    gt_anns  = gt["annotations"]
    total_gt = len(gt_anns)

    # Category ID → name map
    cat_names = {c["id"]: c["name"] for c in gt.get("categories", [])}
    if cat_names:
        print("  Categories: " +
              ", ".join(f"{k}={v}" for k, v in sorted(cat_names.items())))

    # GT counts per occlusion level
    gt_counts = defaultdict(int)
    for ann in gt_anns:
        level = ann.get("attributes", {}).get(
            "occlusion_level", "unknown").strip().lower()
        if level in OCCLUSION_LEVELS:
            gt_counts[level] += 1

    print(f"  GT counts: " +
          "  ".join(f"{l}={gt_counts[l]:,}" for l in OCCLUSION_LEVELS))

    print(f"\nLoading Baseline  : {args.baseline}")
    b_preds = [p for p in load_json(args.baseline) if p["score"] >= args.conf]
    print(f"  Kept {len(b_preds):,} predictions (conf >= {args.conf})")

    print(f"Loading Updated   : {args.updated}")
    u_preds = [p for p in load_json(args.updated) if p["score"] >= args.conf]
    print(f"  Kept {len(u_preds):,} predictions (conf >= {args.conf})")

    # ── Label ───────────────────────────────────────────────────────────────
    print(f"\nLabeling predictions (IoU >= {args.iou}) ...")
    print("  Labeling baseline ...")
    b_labeled = label_predictions(b_preds, gt_anns, args.iou)
    print("  Labeling updated  ...")
    u_labeled = label_predictions(u_preds, gt_anns, args.iou)

    # ── Tables ──────────────────────────────────────────────────────────────
    print_methodology_ap_comparison(b_labeled, gt_counts)

    # ── Plots ───────────────────────────────────────────────────────────────
    print(f"\nGenerating plots → {args.output}/")
    ap_results = plot_pr_per_occlusion(
        b_labeled, u_labeled, gt_counts, args.output, args.iou)
    plot_methodology_comparison(b_labeled, gt_counts, args.output, args.iou)
    plot_pr_all_levels_overlay(
        b_labeled, u_labeled, gt_counts, args.output, args.iou)
    plot_pr_overall(
        b_labeled, u_labeled, total_gt, args.output, args.iou)
    plot_precision_at_recall(
        b_labeled, u_labeled, gt_counts, args.output, args.iou)

    # ── Summary ─────────────────────────────────────────────────────────────
    print_ap_summary(ap_results, gt_counts)
    print_fp_breakdown(b_labeled, model_name="Baseline")
    print_fp_breakdown(u_labeled, model_name="Updated")
    print_confusion(b_labeled, "Baseline", cat_names)
    print_confusion(u_labeled, "Updated",  cat_names)

    print("\nDone.")


if __name__ == "__main__":
    main()