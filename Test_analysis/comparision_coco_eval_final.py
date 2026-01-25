import numpy as np
import json
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import random
from PIL import Image, ImageDraw, ImageFont


# ---------------- STATS FUNCTIONS ---------------- #

def coco_classwise_stats_with_fp_dump(
    gt_json,
    pred_json,
    iou_thr=0.5,
    score_thr=0.28,
    fp_out_json=None,
    dump_fp=False,
    class_thresholds=None,
    return_eval=False
):
    """
    If class_thresholds is a dict {class_id: threshold}, apply per-class thresholds.
    Otherwise, use score_thr globally.
    If return_eval=True, returns (stats, cocoEval) tuple.
    """
    cocoGt = COCO(gt_json)

    with open(pred_json, "r") as f:
        preds = json.load(f)

    # Filter predictions based on class-wise or global threshold
    if class_thresholds is not None:
        preds = [p for p in preds 
                 if p["score"] >= class_thresholds.get(p["category_id"], score_thr)]
    else:
        preds = [p for p in preds if p["score"] >= score_thr]

    cocoDt = cocoGt.loadRes(preds)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType="bbox")

    cocoEval.params.iouThrs = np.array([iou_thr])
    cocoEval.params.maxDets = [100]
    cocoEval.params.areaRng = [[0**2, 1e5**2]]
    cocoEval.evaluate()

    stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    false_positives = []

    iou_idx = 0

    for e in cocoEval.evalImgs:
        if e is None:
            continue

        cls = int(e["category_id"])
        dtMatches = e["dtMatches"][iou_idx]
        dtIgnore  = e["dtIgnore"][iou_idx]
        gtMatches = e["gtMatches"][iou_idx]
        gtIgnore  = e["gtIgnore"]
        dtIds     = e["dtIds"]

        # --- TP / FP ---
        for m, ign, dt_id in zip(dtMatches, dtIgnore, dtIds):
            if ign:
                continue
            if m > 0:
                stats[cls]["TP"] += 1
            else:
                stats[cls]["FP"] += 1
                if dump_fp:
                    pred = cocoDt.anns[dt_id]
                    false_positives.append(pred)

        # --- FN ---
        for m, ign in zip(gtMatches, gtIgnore):
            if ign:
                continue
            if m == 0:
                stats[cls]["FN"] += 1

    if dump_fp and fp_out_json is not None:
        with open(fp_out_json, "w") as f:
            json.dump(false_positives, f, indent=2)

    if return_eval:
        return stats, cocoEval
    return stats


def summarize(stats):
    rows = {}
    for cls, v in stats.items():
        rows[cls] = v
    return rows


def fp_distance_per_class(stats_a, stats_b, class_id):
    """L1 distance of FP counts for a single class"""
    fp_a = stats_a.get(class_id, {}).get("FP", 0)
    fp_b = stats_b.get(class_id, {}).get("FP", 0)
    return abs(fp_a - fp_b)


def fp_distance_total(stats_a, stats_b):
    """L1 distance of FP counts across all classes"""
    all_classes = set(stats_a.keys()) | set(stats_b.keys())
    dist = 0
    for c in all_classes:
        dist += abs(stats_a.get(c, {}).get("FP", 0) -
                    stats_b.get(c, {}).get("FP", 0))
    return dist


def print_table(stats, title, class_thresholds=None):
    print("\n" + "=" * 85)
    print(title)
    print("=" * 85)
    if class_thresholds:
        print(f"{'Class':<8} {'Thr':<7} {'TP':<6} {'FP':<6} {'FN':<6} {'Prec':<8} {'Recall':<8}")
    else:
        print(f"{'Class':<8} {'TP':<6} {'FP':<6} {'FN':<6} {'Prec':<8} {'Recall':<8}")
    
    for cls in sorted(stats):
        TP = stats[cls]["TP"]
        FP = stats[cls]["FP"]
        FN = stats[cls]["FN"]
        prec = TP / (TP + FP + 1e-6)
        rec  = TP / (TP + FN + 1e-6)
        
        if class_thresholds:
            thr = class_thresholds.get(cls, 0.0)
            print(f"{cls:<8} {thr:<7.3f} {TP:<6} {FP:<6} {FN:<6} {prec:.3f}   {rec:.3f}")
        else:
            print(f"{cls:<8} {TP:<6} {FP:<6} {FN:<6} {prec:.3f}   {rec:.3f}")


# ---------------- IMAGE PLOTTING FUNCTIONS ---------------- #

def extract_gt_match_map(cocoEval):
    """
    Returns:
      gt_match_map[class_id][image_id][gt_id] = matched_dt_id or 0
    """
    gt_match_map = defaultdict(lambda: defaultdict(dict))

    iou_idx = 0
    for e in cocoEval.evalImgs:
        if e is None:
            continue

        cls = int(e["category_id"])
        img_id = int(e["image_id"])

        gtIds = e["gtIds"]
        gtMatches = e["gtMatches"][iou_idx]
        gtIgnore = e["gtIgnore"]

        for gt_id, m, ign in zip(gtIds, gtMatches, gtIgnore):
            if ign:
                continue
            gt_match_map[cls][img_id][gt_id] = int(m)  # 0 = FN, >0 = TP

    return gt_match_map


def collect_codino_tp_gdino_fn(
    cocoGt,
    codino_eval,
    gdino_eval,
    max_per_class=3,
    seed=42
):
    """Collect cases where CODINO has TP but GDINO has FN"""
    random.seed(seed)

    codino_map = extract_gt_match_map(codino_eval)
    gdino_map = extract_gt_match_map(gdino_eval)

    selected = defaultdict(list)

    for cls in codino_map:
        for img_id in codino_map[cls]:
            for gt_id, codino_match in codino_map[cls][img_id].items():
                gdino_match = gdino_map.get(cls, {}).get(img_id, {}).get(gt_id, 0)

                # CODINO TP & GDINO FN
                if codino_match > 0 and gdino_match == 0:
                    ann = cocoGt.anns[gt_id]
                    selected[cls].append(ann)

    # random sample per class
    final = {}
    for cls, anns in selected.items():
        if len(anns) > max_per_class:
            final[cls] = random.sample(anns, max_per_class)
        else:
            final[cls] = anns

    return final


def collect_fn_cases_per_class(
    cocoGt,
    codino_eval,
    gdino_eval,
    max_per_class=3,
    seed=42
):
    """
    Returns:
      {
        "gdino_fn": {cls: [gt_anns]},   # GDINO FN but CODINO TP
        "codino_fn": {cls: [gt_anns]},  # HDINO FN but GDINO TP
        "both_fn": {cls: [gt_anns]},    # Both FN
      }
    """
    random.seed(seed)

    codino_map = extract_gt_match_map(codino_eval)
    gdino_map = extract_gt_match_map(gdino_eval)

    cases = {
        "gdino_fn": defaultdict(list),
        "codino_fn": defaultdict(list),
        "both_fn": defaultdict(list),
    }

    # Get all classes and images
    all_classes = set(codino_map.keys()) | set(gdino_map.keys())
    
    for cls in all_classes:
        all_images = set(codino_map.get(cls, {}).keys()) | set(gdino_map.get(cls, {}).keys())
        
        for img_id in all_images:
            codino_gts = codino_map.get(cls, {}).get(img_id, {})
            gdino_gts = gdino_map.get(cls, {}).get(img_id, {})
            
            all_gt_ids = set(codino_gts.keys()) | set(gdino_gts.keys())
            
            for gt_id in all_gt_ids:
                codino_match = codino_gts.get(gt_id, 0)
                gdino_match = gdino_gts.get(gt_id, 0)

                if codino_match > 0 and gdino_match == 0:
                    cases["gdino_fn"][cls].append(cocoGt.anns[gt_id])

                elif codino_match == 0 and gdino_match > 0:
                    cases["codino_fn"][cls].append(cocoGt.anns[gt_id])

                elif codino_match == 0 and gdino_match == 0:
                    cases["both_fn"][cls].append(cocoGt.anns[gt_id])

    # sample max_per_class
    final = {}
    for k, per_cls in cases.items():
        final[k] = {}
        for cls, anns in per_cls.items():
            if len(anns) > max_per_class:
                final[k][cls] = random.sample(anns, max_per_class)
            else:
                final[k][cls] = anns

    return final


def plot_annotations_on_image(image_path, annotations, cocoGt, output_path, label_suffix):
    """Plot bounding boxes with labels on image"""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

    colors = [
        "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
        "#FFA500", "#800080", "#FFC0CB", "#A52A2A", "#808080", "#000080", "#008080"
    ]

    for ann in annotations:
        bbox = ann["bbox"]
        cat_id = ann["category_id"]
        cat_name = cocoGt.cats[cat_id]["name"] if cat_id in cocoGt.cats else f"Class {cat_id}"
        color = colors[cat_id % len(colors)]

        x, y, w, h = bbox
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)

        label = f"{cat_name} ({label_suffix})"
        text_bbox = draw.textbbox((x, y), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        draw.rectangle([x, y - text_height - 4, x + text_width + 4, y], fill=color)
        draw.text((x + 2, y - text_height - 2), label, fill="white", font=font)

    img.save(output_path)
    print(f"  Saved: {output_path}")


def save_fn_images(
    fn_cases,
    cocoGt,
    image_dir,
    output_root
):
    """Save FN case visualizations organized by case type and class"""
    for case_name, per_cls in fn_cases.items():
        if not per_cls:
            continue
            
        for cls, anns in per_cls.items():
            if not anns:
                continue
                
            cls_name = cocoGt.cats[cls]["name"]
            out_dir = os.path.join(output_root, case_name, f"{cls}_{cls_name}")
            os.makedirs(out_dir, exist_ok=True)

            img_group = defaultdict(list)
            for ann in anns:
                img_group[ann["image_id"]].append(ann)

            for img_id, ann_list in img_group.items():
                img_info = cocoGt.imgs[img_id]
                src = os.path.join(image_dir, img_info["file_name"])
                if not os.path.exists(src):
                    print(f"  Warning: Image not found: {src}")
                    continue

                dst = os.path.join(out_dir, img_info["file_name"])
                plot_annotations_on_image(
                    src,
                    ann_list,
                    cocoGt,
                    dst,
                    label_suffix=case_name.replace("_", " ").upper()
                )

    print(f"✅ FN visualizations saved to {output_root}")


def save_classwise_difference_images(
    classwise_diffs,
    cocoGt,
    image_dir,
    output_root
):
    """Save CODINO-TP / GDINO-FN images organized by class"""
    os.makedirs(output_root, exist_ok=True)

    for cls, anns in classwise_diffs.items():
        if not anns:
            continue
            
        cls_name = cocoGt.cats[cls]["name"]
        cls_dir = os.path.join(output_root, f"{cls}_{cls_name}")
        os.makedirs(cls_dir, exist_ok=True)

        img_group = defaultdict(list)
        for ann in anns:
            img_group[ann["image_id"]].append(ann)

        for img_id, ann_list in img_group.items():
            img_info = cocoGt.imgs[img_id]
            src = os.path.join(image_dir, img_info["file_name"])
            if not os.path.exists(src):
                print(f"  Warning: Image not found: {src}")
                continue

            out = os.path.join(cls_dir, img_info["file_name"])
            plot_annotations_on_image(src, ann_list, cocoGt, out, label_suffix="HDINO TP / GDINO FN")

    print(f"✅ Saved class-wise HDINO-TP / GDINO-FN images to: {output_root}")


# ---------------- MAIN ---------------- #

GT_JSON     = "merged_instances_val2017_codetr_final.json"
# CODINO_JSON = "codetr_results_val_set_coco_fixed.json"
HDINO_JSON = "hdino_results_val_set_coco_fixed.json"
GDINO_JSON  = "gdino_results_val_set_coco_fixed.json"
IMAGE_DIR   = "/home/poonam_rajput/scratch/dataset/RSUD_dataset/images/val"  # Update this to your COCO images directory
OUTPUT_DIR_DIFF = "hdino_tp_gdino_fn_visualizations"
OUTPUT_DIR_FN = "hdino_fn_cases_visualizations"

# ---- COMPUTE HDINO STATS ----
print("Computing HDINO stats...")
codino_stats, codino_eval = coco_classwise_stats_with_fp_dump(
    GT_JSON, HDINO_JSON, score_thr=0.31, 
    fp_out_json="hdino_false_positive.json", dump_fp=True,
    return_eval=True
)
codino_stats = summarize(codino_stats)

# Get all classes present in CODINO
all_classes = sorted(codino_stats.keys())

# ---- SWEEP GDINO THRESHOLD CLASS-WISE ----
print("\nFinding optimal threshold for each class...")
best_class_thresholds = {}
class_sweep_results = {}

for cls in all_classes:
    print(f"\nOptimizing class {cls}...")
    best_thr = None
    best_dist = float("inf")
    
    for thr in np.arange(0.20, 0.40, 0.01):
        # Test this threshold for the current class
        test_thresholds = best_class_thresholds.copy()
        test_thresholds[cls] = float(thr)
        
        gdino_stats = coco_classwise_stats_with_fp_dump(
            GT_JSON, GDINO_JSON, class_thresholds=test_thresholds
        )
        gdino_stats = summarize(gdino_stats)
        
        # Compute FP distance for this class only
        dist = fp_distance_per_class(codino_stats, gdino_stats, cls)
        
        if dist < best_dist:
            best_dist = dist
            best_thr = thr
    
    best_class_thresholds[cls] = best_thr
    class_sweep_results[cls] = (best_thr, best_dist)
    print(f"  Class {cls}: best_thr={best_thr:.3f}, FP_dist={best_dist}")

# ---- COMPUTE FINAL GDINO STATS WITH CLASS-WISE THRESHOLDS ----
print("\n" + "="*85)
print("Computing final GDINO stats with class-wise thresholds...")
gdino_stats_final, gdino_eval = coco_classwise_stats_with_fp_dump(
    GT_JSON, GDINO_JSON, 
    class_thresholds=best_class_thresholds,
    fp_out_json="for_hdino_gdino_false_positive_classwise.json", 
    dump_fp=True,
    return_eval=True
)
gdino_stats_final = summarize(gdino_stats_final)

# Calculate total FP distance
total_fp_dist = fp_distance_total(codino_stats, gdino_stats_final)

# ---- PRINT RESULTS ----
print("\n" + "="*85)
print("OPTIMAL CLASS-WISE THRESHOLDS FOR GDINO")
print("="*85)
for cls in sorted(best_class_thresholds.keys()):
    thr, dist = class_sweep_results[cls]
    print(f"Class {cls:<3}: threshold={thr:.3f}, FP_distance={dist}")

print(f"\n✅ Total FP distance (class-wise) = {total_fp_dist}")

# Save thresholds to JSON
with open("gdino_classwise_thresholds.json", "w") as f:
    json.dump(best_class_thresholds, f, indent=2)
print("✅ Class-wise thresholds saved to: gdino_classwise_thresholds.json")


# ---- VISUALIZATION 1: CODINO-TP / GDINO-FN ----
print("\n" + "="*85)
print("EXTRACTING HDINO-TP & GDINO-FN EXAMPLES...")
print("="*85)

cocoGt = COCO(GT_JSON)

classwise_diffs = collect_codino_tp_gdino_fn(
    cocoGt,
    codino_eval,
    gdino_eval,
    max_per_class=3
)

print(f"\nFound {sum(len(anns) for anns in classwise_diffs.values())} CODINO-TP / GDINO-FN cases")
save_classwise_difference_images(
    classwise_diffs,
    cocoGt,
    IMAGE_DIR,
    OUTPUT_DIR_DIFF
)

# ---- VISUALIZATION 2: ALL FN CASES ----
print("\n" + "="*85)
print("COLLECTING FN CASES (GDINO FN / HDINO FN / BOTH FN)...")
print("="*85)

fn_cases = collect_fn_cases_per_class(
    cocoGt,
    codino_eval=codino_eval,
    gdino_eval=gdino_eval,
    max_per_class=3
)

# Print summary
for case_name, per_cls in fn_cases.items():
    total = sum(len(anns) for anns in per_cls.values())
    print(f"{case_name}: {total} cases across {len(per_cls)} classes")

save_fn_images(
    fn_cases,
    cocoGt,
    IMAGE_DIR,
    output_root=OUTPUT_DIR_FN
)

print("\n" + "="*85)
print("ALL VISUALIZATIONS COMPLETE!")
print("="*85)
print(f"1. HDINO-TP / GDINO-FN images: {OUTPUT_DIR_DIFF}/")
print(f"2. FN cases (organized by type): {OUTPUT_DIR_FN}/")
print("="*85)


# ---- COMPARISON TABLES ----
print_table(codino_stats,
            "CODINO — COCO Confusion Matrix (IoU=0.5, Scr_thr=0.3)")
print_table(gdino_stats_final,
            "GDINO — COCO Confusion Matrix (IoU=0.5, Class-wise Thresholds)",
            class_thresholds=best_class_thresholds)