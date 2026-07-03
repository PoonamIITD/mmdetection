# """
# CityPersons log-average miss rate (MR^-2) evaluation.

# This is a faithful port of the OFFICIAL evaluation algorithm from
# cvgroup-njust/CityPersons, evaluation/eval_script/eval_MR_multisetup.py
# (itself a Python re-implementation of the original Caltech-pedestrian
# Matlab benchmark, patched into the pycocotools COCOeval class).

# Adapted to:
#   - read a single COCO-format GT json with visibility stored at
#     ann["attributes"]["visibility_ratio"] (instead of requiring a
#     pre-merged single-category GT json with a precomputed 'ignore' field)
#   - accept a list of target category ids + a subset height/vis window
#     directly, instead of hardcoding category_id == 1

# Everything else -- the matching rule, the DT height cull, the FPPI/MR
# sampling -- mirrors the official file section-by-section (see comments).

# Requirements:
#     pip install matplotlib tabulate numpy

# Usage: same CLI as before.
#     python citypersons_mr_eval_official.py \
#         --gt        instances_validation_merged_mannual.json \
#         --baseline  results_baseline_GDINO_final_val.json \
#         --updated   results_sampling_loss_final_val.json \
#         --output    citypersons_mr_report \
#         --target-categories pedestrian rider "sitting person" \
#         --subsets standard
# """

# import json
# import argparse
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from pathlib import Path
# from collections import defaultdict

# try:
#     from tabulate import tabulate
#     HAS_TABULATE = True
# except ImportError:
#     HAS_TABULATE = False

# # ── Config ────────────────────────────────────────────────────────────────────

# # Official CityPersons subset definitions -- verbatim from
# # eval_MR_multisetup.py Params.setDetParams():
# #   self.HtRng  = [[50,1e5**2], [50,75],       [50,1e5**2],  [20,1e5**2]]
# #   self.VisRng = [[0.65,1e5**2],[0.65,1e5**2], [0.2,0.65],   [0.2,1e5**2]]
# #   self.SetupLbl = ['Reasonable','Reasonable_small','Reasonable_occ=heavy','All']
# STANDARD_SUBSETS = {
#     "Reasonable":       {"height_range": (50,  1e5), "vis_range": (0.65, 1.01)},
#     "Reasonable_small": {"height_range": (50,  75),  "vis_range": (0.65, 1.01)},
#     "Heavy_Occlusion":  {"height_range": (50,  1e5), "vis_range": (0.20, 0.65)},
#     "All":              {"height_range": (20,  1e5), "vis_range": (0.20, 1.01)},
# }

# CUSTOM_SUBSETS = {
#     "Light_custom":    {"height_range": (0, 1e5), "vis_range": (0.80, 1.01)},
#     "Moderate_custom": {"height_range": (0, 1e5), "vis_range": (0.20, 0.80)},
#     "Severe_custom":   {"height_range": (0, 1e5), "vis_range": (0.00, 0.20)},
# }

# IOU_THRESHOLD = 0.5          # official: p.iouThrs = [0.5]
# EXP_FILTER = 1.25            # official: p.expFilter = 1.25 (DT height cull margin)
# FPPI_REF_POINTS = np.array(  # official: p.fppiThrs, verbatim
#     [0.0100, 0.0178, 0.0316, 0.0562, 0.1000, 0.1778, 0.3162, 0.5623, 1.0000]
# )

# SUBSET_COLORS = {
#     "Reasonable":       "#2ECC71",
#     "Reasonable_small": "#3498DB",
#     "Heavy_Occlusion":  "#8E44AD",
#     "All":              "#7F8C8D",
#     "Light_custom":     "#3498DB",
#     "Moderate_custom":  "#F39C12",
#     "Severe_custom":    "#8E44AD",
# }
# MODEL_STYLES = {
#     "baseline": {"ls": "--", "lw": 2.0, "alpha": 0.85},
#     "updated":  {"ls": "-",  "lw": 2.5, "alpha": 0.95},
# }

# # ── I/O ───────────────────────────────────────────────────────────────────────

# def load_json(path):
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)

# # ── Geometry ──────────────────────────────────────────────────────────────────
# # These correspond exactly to the official COCOeval.iou() method, which is a
# # SINGLE overloaded function: for a real ("non-crowd") GT box it computes
# #     unionarea = det_area + gt_area - intersection      -> true IoU
# # for an ignore ("crowd") GT box it computes
# #     unionarea = det_area                                -> IoA (intersection/det_area)
# # We keep them split into iou()/ioa() for readability; evaluate_mr() below
# # calls the right one depending on the GT's ignore flag, same net effect.

# def box_area(box):
#     w = max(0.0, box[2])
#     h = max(0.0, box[3])
#     return w * h


# def iou(box_a, box_b):
#     ax1, ay1, aw, ah = box_a
#     ax2, ay2 = ax1 + aw, ay1 + ah
#     bx1, by1, bw, bh = box_b
#     bx2, by2 = bx1 + bw, by1 + bh
#     ix1, iy1 = max(ax1, bx1), max(ay1, by1)
#     ix2, iy2 = min(ax2, bx2), min(ay2, by2)
#     iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
#     inter = iw * ih
#     union = box_area(box_a) + box_area(box_b) - inter
#     return inter / union if union > 0 else 0.0


# def ioa(det_box, gt_box):
#     """Intersection over the DETECTION's area (official: crowd-GT branch of iou())."""
#     dx1, dy1, dw, dh = det_box
#     dx2, dy2 = dx1 + dw, dy1 + dh
#     gx1, gy1, gw, gh = gt_box
#     gx2, gy2 = gx1 + gw, gy1 + gh
#     ix1, iy1 = max(dx1, gx1), max(dy1, gy1)
#     ix2, iy2 = min(dx2, gx2), min(dy2, gy2)
#     iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
#     inter = iw * ih
#     det_area = box_area(det_box)
#     return inter / det_area if det_area > 0 else 0.0

# # ── Category helper ──────────────────────────────────────────────────────────

# def get_target_category_ids(gt_full, target_names):
#     target_names = {n.strip().lower() for n in target_names}
#     return {
#         c["id"] for c in gt_full.get("categories", [])
#         if c.get("name", "").strip().lower() in target_names
#     }

# # ── Core evaluation (port of evaluateImg + accumulate + summarize) ──────────

# def evaluate_mr(gt_full, pred_list, target_category_ids, subset_def,
#                  iou_thr=IOU_THRESHOLD, exp_filter=EXP_FILTER):
#     """
#     Faithful port of:
#       - COCOeval._prepare()   -> the per-annotation 'ignore' flag construction
#       - COCOeval.evaluateImg()-> per-image greedy matching, DT height cull
#       - COCOeval.accumulate() -> global score-sort, drop dismissed dets, cumsum
#       - COCOeval.summarize()  -> MR = 1-recall @ 9 FPPI points, MR^-2 = geo-mean
#     """
#     h_lo, h_hi = subset_def["height_range"]
#     v_lo, v_hi = subset_def["vis_range"]

#     # ---- _prepare(): build per-image GT with an 'ignore' flag ----
#     # Official: gt['ignore'] starts at a category-derived base value (0 for
#     # pedestrian/rider/sitting-person, 1 for every other CityPersons class,
#     # since those are folded into the same evaluated category permanently
#     # ignored). It is THEN overridden to 1 if height/vis falls outside this
#     # subset's window -- but a box that was already ignore stays ignore.
#     gt_by_image = {img["id"]: [] for img in gt_full["images"]}
#     for ann in gt_full["annotations"]:
#         img_id = ann["image_id"]
#         if img_id not in gt_by_image:
#             continue
#         is_target_cat = ann["category_id"] in target_category_ids
#         h = ann["bbox"][3]
#         vis = ann.get("attributes", {}).get("visibility_ratio")

#         if (not is_target_cat) or (vis is None):
#             ignore = 1  # permanent ignore region (or missing vis info)
#         else:
#             in_range = (h_lo <= h <= h_hi) and (v_lo <= vis <= v_hi)
#             ignore = 0 if in_range else 1

#         gt_by_image[img_id].append({"bbox": ann["bbox"], "ignore": ignore})

#     total_positives = sum(1 for anns in gt_by_image.values() for a in anns if a["ignore"] == 0)
#     num_images = len(gt_by_image)
#     if total_positives == 0:
#         return None

#     # official: "sort gt ignore last" -> np.argsort(ignore flags), stable
#     for anns in gt_by_image.values():
#         anns.sort(key=lambda a: a["ignore"])

#     # ---- evaluateImg(): DT pool, height-filtered by expFilter per subset ----
#     valid_image_ids = set(gt_by_image.keys())
#     dt_by_image = defaultdict(list)
#     for p in pred_list:
#         if p["image_id"] not in valid_image_ids or p["category_id"] not in target_category_ids:
#             continue
#         dh = p["bbox"][3]
#         if not (h_lo / exp_filter <= dh < h_hi * exp_filter):
#             continue  # official: dt = [d for d in dt if hRng[0]/expFilter <= d.height < hRng[1]*expFilter]
#         dt_by_image[p["image_id"]].append(p)

#     # ---- evaluateImg(): per-image greedy single-pass matching ----
#     # official bstOa/bstg/bstm loop, faithfully reproduced:
#     #   - dt processed in score-descending order (within image)
#     #   - for each dt, scan gt (real-first, ignore-last); once a real match
#     #     is already locked in (best_kind != -2) and we reach the ignore
#     #     section, STOP scanning (break) -- ignore GT can only be used as a
#     #     fallback when no real GT qualifies.
#     #   - best-so-far overlap must strictly improve on the current best,
#     #     starting from the threshold itself (so nothing below 0.5 ever wins)
#     all_dt_records = []  # (score, is_tp, is_dismissed)
#     for img_id, gts in gt_by_image.items():
#         dts = sorted(dt_by_image.get(img_id, []), key=lambda p: -p["score"])
#         matched_gt = [False] * len(gts)
#         for d in dts:
#             best_ov, best_gi, best_kind = iou_thr, -2, -2  # kind: 1=real, -1=ignore
#             for gi, g in enumerate(gts):
#                 if matched_gt[gi]:
#                     continue
#                 if best_kind != -2 and g["ignore"] == 1:
#                     break  # already matched a real GT -> never fall back to ignore
#                 ov = iou(d["bbox"], g["bbox"]) if g["ignore"] == 0 else ioa(d["bbox"], g["bbox"])
#                 if ov < best_ov:
#                     continue
#                 best_ov, best_gi = ov, gi
#                 best_kind = 1 if g["ignore"] == 0 else -1

#             if best_gi == -2:
#                 all_dt_records.append((d["score"], False, False))   # unmatched -> FP candidate
#             elif best_kind == 1:
#                 matched_gt[best_gi] = True
#                 all_dt_records.append((d["score"], True, False))    # TP
#             else:
#                 all_dt_records.append((d["score"], False, True))    # matched ignore -> dismissed

#     # ---- accumulate(): global sort by score, drop dismissed dets, cumsum ----
#     # official: dtm/dtIg concatenated across all images, globally re-sorted
#     # by score (mergesort, matches Matlab tie-breaking), then
#     #   inds = where(dtIg == 0); tps = tps[inds]; fps = fps[inds]
#     # i.e. ignore-matched detections are removed from the ranked sequence
#     # entirely before cumsum -- they never even occupy a FPPI "slot".
#     all_dt_records.sort(key=lambda r: -r[0])
#     kept_tp_flags = [is_tp for _, is_tp, is_ignored in all_dt_records if not is_ignored]

#     fppi_list = [0.0]
#     mr_list = [1.0]
#     cum_tp = cum_fp = 0
#     for is_tp in kept_tp_flags:
#         if is_tp:
#             cum_tp += 1
#         else:
#             cum_fp += 1
#         fppi_list.append(cum_fp / num_images)
#         mr_list.append(1.0 - cum_tp / total_positives)

#     fppi_arr = np.array(fppi_list)
#     mr_arr = np.array(mr_list)
#     mr2 = log_average_miss_rate(fppi_arr, mr_arr)

#     return {
#         "fppi": fppi_arr,
#         "miss_rate": mr_arr,
#         "total_positives": total_positives,
#         "num_images": num_images,
#         "num_predictions_considered": len(kept_tp_flags),
#         "mr2": mr2,
#     }


# def log_average_miss_rate(fppi, mr, ref_points=FPPI_REF_POINTS):
#     """
#     Official summarize(): for each of the 9 FPPI reference points, take the
#     recall at the rightmost curve index whose fppi <= ref (searchsorted
#     side='right') - 1), i.e. as many detections kept as possible without
#     exceeding the reference FPPI, then MR = 1 - recall, MR^-2 = geometric
#     mean over the 9 points. Points with no detection at/under the smallest
#     ref default to the leading virtual point (fppi=0, mr=1.0), matching the
#     official q[]=0 default (-> mr=1) rather than the official code's raw
#     negative-index wraparound quirk, which is not meaningfully reachable
#     once num_images is reasonably large.
#     """
#     sampled = []
#     for ref in ref_points:
#         idxs = np.where(fppi <= ref)[0]
#         idx = idxs[-1] if len(idxs) else 0
#         sampled.append(mr[idx])
#     sampled = np.maximum(np.array(sampled), 1e-10)
#     return float(np.exp(np.mean(np.log(sampled))))

# # ── Plots ─────────────────────────────────────────────────────────────────────

# def plot_mr_fppi(evals, output_dir, subset_names, filename_suffix=""):
#     n = len(subset_names)
#     ncols = min(n, 3)
#     nrows = (n + ncols - 1) // ncols
#     fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows), squeeze=False)

#     for idx, subset_name in enumerate(subset_names):
#         ax = axes[idx // ncols][idx % ncols]
#         color = SUBSET_COLORS.get(subset_name, "#333333")
#         plotted_any = False
#         for model_name in ["baseline", "updated"]:
#             res = evals[subset_name][model_name]
#             style = MODEL_STYLES[model_name]
#             if res is None:
#                 ax.text(0.5, 0.5, f"{model_name}: no positive GT\nfor this subset",
#                         ha="center", va="center", transform=ax.transAxes, fontsize=9)
#                 continue
#             ax.plot(res["fppi"], res["miss_rate"], color=color,
#                      ls=style["ls"], lw=style["lw"], alpha=style["alpha"],
#                      label=f"{model_name.capitalize()}  MR\u207b\u00b2={res['mr2']*100:.2f}%")
#             plotted_any = True
#         ax.set_xscale("log")
#         ax.set_yscale("log")
#         ax.set_xlim(1e-2, 1e1)
#         ax.set_ylim(1e-2, 1.0)
#         ax.set_xlabel("False Positives Per Image (FPPI)", fontsize=10)
#         ax.set_ylabel("Miss Rate", fontsize=10)
#         ax.set_title(subset_name, color=color, fontsize=12, fontweight="bold")
#         if plotted_any:
#             ax.legend(fontsize=9, loc="lower left")
#         ax.grid(True, which="both", alpha=0.25)

#     for idx in range(n, nrows * ncols):
#         axes[idx // ncols][idx % ncols].axis("off")

#     plt.tight_layout()
#     out = Path(output_dir) / f"mr_fppi_curves{filename_suffix}.png"
#     plt.savefig(out, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"  Saved: {out}")

# # ── Summary table ─────────────────────────────────────────────────────────────

# def print_summary_table(all_results, subset_names):
#     header = ["Subset", "Base MR⁻²", "Upd MR⁻²", "Δ (pp)", "#Positives", "#Images"]
#     rows = []
#     for subset_name in subset_names:
#         b = all_results[subset_name]["baseline"]
#         u = all_results[subset_name]["updated"]
#         b_mr = b["mr2"] if b else float("nan")
#         u_mr = u["mr2"] if u else float("nan")
#         delta = (u_mr - b_mr) * 100 if not (np.isnan(b_mr) or np.isnan(u_mr)) else float("nan")
#         n_pos = b["total_positives"] if b else (u["total_positives"] if u else 0)
#         n_img = b["num_images"] if b else (u["num_images"] if u else 0)
#         rows.append([
#             subset_name,
#             f"{b_mr*100:.2f}%" if not np.isnan(b_mr) else "—",
#             f"{u_mr*100:.2f}%" if not np.isnan(u_mr) else "—",
#             f"{delta:+.2f}" if not np.isnan(delta) else "—",
#             f"{n_pos:,}",
#             f"{n_img:,}",
#         ])

#     print(f"\n{'='*100}")
#     print("  CityPersons MR⁻² (official algorithm port)  |  lower is better  |  "
#           "IoU≥0.5 real match, IoA≥0.5 ignore, expFilter=1.25 DT height cull")
#     print(f"{'='*100}")
#     if HAS_TABULATE:
#         print(tabulate(rows, headers=header, tablefmt="rounded_outline"))
#     else:
#         col_w = [max(len(h), max(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
#         fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
#         print(fmt.format(*header))
#         print("  ".join("-" * w for w in col_w))
#         for row in rows:
#             print(fmt.format(*[str(c) for c in row]))

# # ── Main ──────────────────────────────────────────────────────────────────────

# def main():
#     parser = argparse.ArgumentParser(
#         description="Official-algorithm CityPersons MR^-2 evaluation (log-average miss rate)."
#     )
#     parser.add_argument("--gt", required=True)
#     parser.add_argument("--baseline", required=True)
#     parser.add_argument("--updated", required=True)
#     parser.add_argument("--output", default="citypersons_mr_report")
#     parser.add_argument(
#         "--target-categories", nargs="+",
#         default=["pedestrian", "rider", "sitting person"],
#         help="Category names counted as the evaluated 'person' target class."
#     )
#     parser.add_argument(
#         "--subsets", choices=["standard", "custom", "both"], default="standard",
#     )
#     args = parser.parse_args()

#     for p in (args.gt, args.baseline, args.updated):
#         if not Path(p).exists():
#             raise FileNotFoundError(f"Not found: {p}")
#     Path(args.output).mkdir(parents=True, exist_ok=True)

#     print(f"\nLoading GT        : {args.gt}")
#     gt_full = load_json(args.gt)
#     cat_names = {c["id"]: c["name"] for c in gt_full.get("categories", [])}
#     print("  Categories: " + ", ".join(f"{k}={v}" for k, v in sorted(cat_names.items())))

#     target_category_ids = get_target_category_ids(gt_full, args.target_categories)
#     if not target_category_ids:
#         raise ValueError(
#             f"None of {args.target_categories} matched a category name in GT: "
#             f"{list(cat_names.values())}"
#         )
#     print("  Target category ids (must-detect class): " +
#           ", ".join(f"{cid}={cat_names[cid]}" for cid in sorted(target_category_ids)))

#     n_missing_vis = sum(
#         1 for a in gt_full["annotations"]
#         if a["category_id"] in target_category_ids
#         and a.get("attributes", {}).get("visibility_ratio") is None
#     )
#     if n_missing_vis:
#         print(f"  WARNING: {n_missing_vis} target-category annotations have no "
#               f"visibility_ratio and will be treated as ignore in every subset.")

#     print(f"\nLoading Baseline  : {args.baseline}")
#     b_preds = load_json(args.baseline)
#     print(f"  {len(b_preds):,} predictions")

#     print(f"Loading Updated   : {args.updated}")
#     u_preds = load_json(args.updated)
#     print(f"  {len(u_preds):,} predictions")

#     subset_defs = {}
#     if args.subsets in ("standard", "both"):
#         subset_defs.update(STANDARD_SUBSETS)
#     if args.subsets in ("custom", "both"):
#         subset_defs.update(CUSTOM_SUBSETS)

#     subset_names = list(subset_defs.keys())
#     all_results = {}

#     for subset_name, subset_def in subset_defs.items():
#         print(f"\n{'─'*60}")
#         print(f"Evaluating subset: {subset_name}  "
#               f"(height {subset_def['height_range']}, vis {subset_def['vis_range']})")
#         print(f"{'─'*60}")

#         all_results[subset_name] = {}
#         for model_name, preds in [("baseline", b_preds), ("updated", u_preds)]:
#             res = evaluate_mr(gt_full, preds, target_category_ids, subset_def)
#             all_results[subset_name][model_name] = res
#             if res is None:
#                 print(f"  {model_name}: no positive GT boxes in this subset — skipped")
#             else:
#                 print(f"  {model_name}: MR⁻²={res['mr2']*100:.2f}%  "
#                       f"(#positives={res['total_positives']:,}, "
#                       f"#images={res['num_images']:,}, "
#                       f"#preds_used={res['num_predictions_considered']:,})")

#     print_summary_table(all_results, subset_names)

#     print(f"\nGenerating plots → {args.output}/")
#     if args.subsets in ("standard", "both"):
#         plot_mr_fppi(all_results, args.output,
#                      [n for n in STANDARD_SUBSETS if n in subset_names],
#                      filename_suffix="_standard")
#     if args.subsets in ("custom", "both"):
#         plot_mr_fppi(all_results, args.output,
#                      [n for n in CUSTOM_SUBSETS if n in subset_names],
#                      filename_suffix="_custom")

#     save_data = {}
#     for subset_name, models in all_results.items():
#         save_data[subset_name] = {}
#         for model_name, res in models.items():
#             save_data[subset_name][model_name] = None if res is None else {
#                 "mr2": res["mr2"],
#                 "total_positives": res["total_positives"],
#                 "num_images": res["num_images"],
#                 "num_predictions_considered": res["num_predictions_considered"],
#             }
#     out_path = Path(args.output) / "citypersons_mr_results.json"
#     with open(out_path, "w") as f:
#         json.dump(save_data, f, indent=2)
#     print(f"\nResults saved to : {out_path}")
#     print("Done.")


# if __name__ == "__main__":
#     main()




"""
CityPersons log-average miss rate (MR^-2) evaluation using the ACTUAL
official code (coco.py / eval_MR_multisetup.py from cvgroup-njust/CityPersons,
copied verbatim, patched only for Python-3/modern-numpy syntax -- see
eval_script/citypersons_official_wrapper.py for the exact diffs and the
GT/prediction format conversion this requires).

Usage: identical CLI to the earlier versions.
    python citypersons_mr_eval_real_official.py \
        --gt        instances_validation_merged_mannual.json \
        --baseline  results_baseline_GDINO_final_val.json \
        --updated   results_sampling_loss_final_val.json \
        --output    citypersons_mr_report \
        --target-categories pedestrian rider "sitting person" \
        --subsets standard
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import io
import tempfile
import contextlib

from pycocotools.coco import COCO
from citypersons_official_wrapper import (  # noqa: E402
    convert_gt_to_official_format, convert_preds_to_official_format, run_subset,
)

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# Official CityPersons subset definitions -- verbatim from
# eval_MR_multisetup.py Params.setDetParams()
STANDARD_SUBSETS = {
    "Reasonable":       {"height_range": (50,  1e5 ** 2), "vis_range": (0.65, 1e5 ** 2)},
    "Reasonable_small": {"height_range": (50,  75),       "vis_range": (0.65, 1e5 ** 2)},
    "Heavy_Occlusion":  {"height_range": (50,  1e5 ** 2), "vis_range": (0.20, 0.65)},
    "All":              {"height_range": (20,  1e5 ** 2), "vis_range": (0.20, 1e5 ** 2)},
}
CUSTOM_SUBSETS = {
    "Light_custom":    {"height_range": (0, 1e5 ** 2), "vis_range": (0.80, 1e5 ** 2)},
    "Moderate_custom": {"height_range": (0, 1e5 ** 2), "vis_range": (0.20, 0.80)},
    "Severe_custom":   {"height_range": (0, 1e5 ** 2), "vis_range": (0.00, 0.20)},
}

SUBSET_COLORS = {
    "Reasonable": "#2ECC71", "Reasonable_small": "#3498DB",
    "Heavy_Occlusion": "#8E44AD", "All": "#7F8C8D",
    "Light_custom": "#3498DB", "Moderate_custom": "#F39C12", "Severe_custom": "#8E44AD",
}
MODEL_STYLES = {
    "baseline":    {"ls": "--", "lw": 2.0, "alpha": 0.85},
    "updated":     {"ls": "-",  "lw": 2.5, "alpha": 0.95},
    "updated_nms": {"ls": ":",  "lw": 2.5, "alpha": 0.95},
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_target_category_ids(gt_full, target_names):
    target_names = {n.strip().lower() for n in target_names}
    return {c["id"] for c in gt_full.get("categories", [])
            if c.get("name", "").strip().lower() in target_names}

def apply_nms(preds, iou_thresh, class_agnostic=True):
    """
    Greedy NMS over a COCO-format prediction list [{image_id, category_id,
    bbox: [x,y,w,h], score}, ...]. Returns a new list with suppressed boxes
    removed.

    class_agnostic=True: NMS is applied across ALL target categories
    together within an image (so a 'pedestrian' box and a 'rider' box
    that overlap heavily still suppress each other) -- this matches the
    duplicate-in-crowd hypothesis, since query duplication isn't
    necessarily confined to one category. Set False for standard
    per-category NMS.
    """
    from collections import defaultdict

    def iou(a, b):
        ax1, ay1, aw, ah = a
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx1, by1, bw, bh = b
        bx2, by2 = bx1 + bw, by1 + bh
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        a_area = max(0.0, aw) * max(0.0, ah)
        b_area = max(0.0, bw) * max(0.0, bh)
        union = a_area + b_area - inter
        return inter / union if union > 0 else 0.0

    grouped = defaultdict(list)
    for p in preds:
        key = p["image_id"] if class_agnostic else (p["image_id"], p["category_id"])
        grouped[key].append(p)

    kept = []
    for key, group in grouped.items():
        group_sorted = sorted(group, key=lambda p: -p["score"])
        suppressed = [False] * len(group_sorted)
        for i in range(len(group_sorted)):
            if suppressed[i]:
                continue
            kept.append(group_sorted[i])
            for j in range(i + 1, len(group_sorted)):
                if suppressed[j]:
                    continue
                if iou(group_sorted[i]["bbox"], group_sorted[j]["bbox"]) >= iou_thresh:
                    suppressed[j] = True
    return kept

def plot_mr_fppi(evals, output_dir, subset_names, filename_suffix=""):
    n = len(subset_names)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows), squeeze=False)
    for idx, subset_name in enumerate(subset_names):
        ax = axes[idx // ncols][idx % ncols]
        color = SUBSET_COLORS.get(subset_name, "#333333")
        plotted_any = False
        for model_name in ["baseline", "updated", "updated_nms"]:
            res = evals[subset_name][model_name]
            style = MODEL_STYLES[model_name]
            if res is None:
                ax.text(0.5, 0.5, f"{model_name}: no positive GT\nfor this subset",
                        ha="center", va="center", transform=ax.transAxes, fontsize=9)
                continue
            ax.plot(res["fppi"], res["miss_rate"], color=color,
                     ls=style["ls"], lw=style["lw"], alpha=style["alpha"],
                     label=f"{model_name.capitalize()}  MR\u207b\u00b2={res['mr2']*100:.2f}%")
            plotted_any = True
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-2, 1e1)
        ax.set_ylim(1e-2, 1.0)
        ax.set_xlabel("False Positives Per Image (FPPI)", fontsize=10)
        ax.set_ylabel("Miss Rate", fontsize=10)
        ax.set_title(subset_name, color=color, fontsize=12, fontweight="bold")
        if plotted_any:
            ax.legend(fontsize=9, loc="lower left")
        ax.grid(True, which="both", alpha=0.25)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    plt.tight_layout()
    out = Path(output_dir) / f"mr_fppi_curves{filename_suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def print_summary_table(all_results, subset_names):
    header = ["Subset", "Base MR⁻²", "Upd MR⁻²", "Upd+NMS MR⁻²",
              "Δ Upd (pp)", "Δ Upd+NMS (pp)", "#Positives", "#Images"]
    rows = []
    for subset_name in subset_names:
        b = all_results[subset_name]["baseline"]
        u = all_results[subset_name]["updated"]
        un = all_results[subset_name]["updated_nms"]
        b_mr = b["mr2"] if b else float("nan")
        u_mr = u["mr2"] if u else float("nan")
        un_mr = un["mr2"] if un else float("nan")
        delta_u = (u_mr - b_mr) * 100 if not (np.isnan(b_mr) or np.isnan(u_mr)) else float("nan")
        delta_un = (un_mr - b_mr) * 100 if not (np.isnan(b_mr) or np.isnan(un_mr)) else float("nan")
        n_pos = b["total_positives"] if b else (u["total_positives"] if u else 0)
        n_img = b["num_images"] if b else (u["num_images"] if u else 0)
        rows.append([subset_name,
                     f"{b_mr*100:.2f}%" if not np.isnan(b_mr) else "—",
                     f"{u_mr*100:.2f}%" if not np.isnan(u_mr) else "—",
                     f"{un_mr*100:.2f}%" if not np.isnan(un_mr) else "—",
                     f"{delta_u:+.2f}" if not np.isnan(delta_u) else "—",
                     f"{delta_un:+.2f}" if not np.isnan(delta_un) else "—",
                     f"{n_pos:,}", f"{n_img:,}"])
    print(f"\n{'='*110}")
    print("  CityPersons MR⁻² (REAL official code)  |  lower is better  |  "
          "Upd+NMS = updated with post-hoc NMS applied")
    print(f"{'='*110}")
    if HAS_TABULATE:
        print(tabulate(rows, headers=header, tablefmt="rounded_outline"))
    else:
        col_w = [max(len(h), max(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
        fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
        print(fmt.format(*header))
        print("  ".join("-" * w for w in col_w))
        for row in rows:
            print(fmt.format(*[str(c) for c in row]))


def main():
    parser = argparse.ArgumentParser(description="CityPersons MR^-2 eval using the real official code.")
    parser.add_argument("--gt", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--updated", required=True)
    parser.add_argument("--output", default="citypersons_mr_report")
    parser.add_argument("--target-categories", nargs="+",
                         default=["pedestrian", "rider", "sitting person"])
    parser.add_argument("--subsets", choices=["standard", "custom", "both"], default="standard")
    parser.add_argument("--nms-iou-thresh", type=float, default=0.5,
                         help="IoU threshold for post-hoc NMS applied to 'updated' predictions.")
    parser.add_argument("--nms-class-agnostic", action="store_true", default=True,
                         help="Apply NMS across all target categories jointly (default: True).")
    args = parser.parse_args()

    for p in (args.gt, args.baseline, args.updated):
        if not Path(p).exists():
            raise FileNotFoundError(f"Not found: {p}")
    Path(args.output).mkdir(parents=True, exist_ok=True)

    print(f"\nLoading GT        : {args.gt}")
    gt_full = load_json(args.gt)
    cat_names = {c["id"]: c["name"] for c in gt_full.get("categories", [])}
    print("  Categories: " + ", ".join(f"{k}={v}" for k, v in sorted(cat_names.items())))

    target_category_ids = get_target_category_ids(gt_full, args.target_categories)
    if not target_category_ids:
        raise ValueError(f"None of {args.target_categories} matched a category name in GT: "
                          f"{list(cat_names.values())}")
    print("  Target category ids: " +
          ", ".join(f"{cid}={cat_names[cid]}" for cid in sorted(target_category_ids)))

    print(f"\nLoading Baseline  : {args.baseline}")
    b_preds = load_json(args.baseline)
    print(f"Loading Updated   : {args.updated}")
    u_preds = load_json(args.updated)

    # ---- convert to the flat single-category format the official code needs ----
    tmp_dir = Path(tempfile.mkdtemp(prefix="citypersons_official_"))
    gt_official_path = tmp_dir / "gt_official.json"
    convert_gt_to_official_format(gt_full, target_category_ids, gt_official_path)
    # NOTE: the official _prepare() mutates ann['ignore'] IN PLACE on the
    # shared annotation dicts held by the COCO object. The official
    # eval_demo.py works around this by reloading COCO(annFile) fresh for
    # every id_setup in its loop -- we do the same below (a fresh COCO(...)
    # per subset per model) rather than reusing one cocoGt object, otherwise
    # ignore flags from a stricter subset (e.g. Reasonable) leak into a
    # later, looser one (e.g. All) and silently zero out its positive count.

    b_preds_official = convert_preds_to_official_format(b_preds, target_category_ids)
    u_preds_official = convert_preds_to_official_format(u_preds, target_category_ids)
    print(f"\n  {len(b_preds_official):,} baseline / {len(u_preds_official):,} updated "
          f"target-category predictions after conversion")

    print(f"\nApplying post-hoc NMS to updated predictions (IoU >= {args.nms_iou_thresh}, "
          f"class_agnostic={args.nms_class_agnostic})...")
    u_preds_nms = apply_nms(u_preds, args.nms_iou_thresh, class_agnostic=args.nms_class_agnostic)
    print(f"  updated: {len(u_preds):,} -> {len(u_preds_nms):,} predictions after NMS "
          f"(-{len(u_preds) - len(u_preds_nms):,})")
    u_preds_nms_official = convert_preds_to_official_format(u_preds_nms, target_category_ids)

    subset_defs = {}
    if args.subsets in ("standard", "both"):
        subset_defs.update(STANDARD_SUBSETS)
    if args.subsets in ("custom", "both"):
        subset_defs.update(CUSTOM_SUBSETS)
    subset_names = list(subset_defs.keys())

    all_results = {}
    for subset_name, subset_def in subset_defs.items():
        print(f"\n{'─'*60}")
        print(f"Evaluating subset: {subset_name}  "
              f"(height {subset_def['height_range']}, vis {subset_def['vis_range']})")
        print(f"{'─'*60}")
        all_results[subset_name] = {}
        for model_name, preds_official in [
            ("baseline", b_preds_official),
            ("updated", u_preds_official),
            ("updated_nms", u_preds_nms_official),
        ]:
            with contextlib.redirect_stdout(io.StringIO()):
                cocoGt = COCO(str(gt_official_path))  # fresh load -- see NOTE above
                res = run_subset(cocoGt, preds_official,
                                  subset_def["height_range"], subset_def["vis_range"], subset_name)
            all_results[subset_name][model_name] = res
            if res is None:
                print(f"  {model_name}: no positive GT boxes in this subset — skipped")
            else:
                print(f"  {model_name}: MR⁻²={res['mr2']*100:.2f}%  "
                      f"(#positives={res['total_positives']:,}, #images={res['num_images']:,}, "
                      f"#preds_used={res['num_predictions_considered']:,})")

    print_summary_table(all_results, subset_names)

    print(f"\nGenerating plots → {args.output}/")
    if args.subsets in ("standard", "both"):
        plot_mr_fppi(all_results, args.output,
                     [n for n in STANDARD_SUBSETS if n in subset_names], filename_suffix="_standard")
    if args.subsets in ("custom", "both"):
        plot_mr_fppi(all_results, args.output,
                     [n for n in CUSTOM_SUBSETS if n in subset_names], filename_suffix="_custom")

    save_data = {}
    for subset_name, models in all_results.items():
        save_data[subset_name] = {}
        for model_name, res in models.items():
            save_data[subset_name][model_name] = None if res is None else {
                "mr2": res["mr2"], "total_positives": res["total_positives"],
                "num_images": res["num_images"],
                "num_predictions_considered": res["num_predictions_considered"],
            }
    out_path = Path(args.output) / "citypersons_mr_results.json"
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to : {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()