import numpy as np
import json
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def coco_classwise_stats_with_fp_dump(
    gt_json,
    pred_json,
    iou_thr=0.5,
    score_thr=0.28,
    fp_out_json=None,
    dump_fp=False,
    class_thresholds=None
):
    """
    If class_thresholds is a dict {class_id: threshold}, apply per-class thresholds.
    Otherwise, use score_thr globally.
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


# ---------------- MAIN ---------------- #

GT_JSON     = "merged_instances_val2017_codetr_final.json"
CODINO_JSON = "codetr_results_val_set_coco_fixed.json"
GDINO_JSON  = "gdino_results_val_set_coco_fixed.json"

# ---- COMPUTE CODINO STATS ----
print("Computing CODINO stats...")
codino_stats = coco_classwise_stats_with_fp_dump(
    GT_JSON, CODINO_JSON, score_thr=0.3, 
    fp_out_json="codino_false_positive.json", dump_fp=True
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
gdino_stats_final = coco_classwise_stats_with_fp_dump(
    GT_JSON, GDINO_JSON, 
    class_thresholds=best_class_thresholds,
    fp_out_json="gdino_false_positive_classwise.json", 
    dump_fp=True
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

# ---- COMPARISON TABLES ----
print_table(codino_stats,
            "CODINO — COCO Confusion Matrix (IoU=0.5, Scr_thr=0.3)")
print_table(gdino_stats_final,
            "GDINO — COCO Confusion Matrix (IoU=0.5, Class-wise Thresholds)",
            class_thresholds=best_class_thresholds)