import numpy as np
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def gt_match_analysis(gt_json, pred_json, iou_thrs=(0.5, 0.75, 0.9), max_dets=100, score_thr=0.3):
    """
    Returns GT match statistics using COCO's own matching logic
    """

    cocoGt = COCO(gt_json)
    import json
    with open(pred_json, "r") as f:
        preds = json.load(f)

    preds = [p for p in preds if p["score"] >= score_thr]

    cocoDt = cocoGt.loadRes(preds)

    cocoEval = COCOeval(cocoGt, cocoDt, iouType="bbox")
    cocoEval.params.iouThrs = np.array(iou_thrs)
    cocoEval.params.maxDets = [max_dets]
    cocoEval.params.areaRng = [[0**2, 1e5**2]]  # all areas

    cocoEval.evaluate()

    # stats[iou][class] = matched GT count
    matched_gt = {iou: defaultdict(int) for iou in iou_thrs}
    total_gt = defaultdict(int)

    for e in cocoEval.evalImgs:
        if e is None:
            continue

        cls = e["category_id"]

        gtIgnore = e["gtIgnore"]
        for ign in gtIgnore:
            if not ign:
                total_gt[cls] += 1

        for i, iou in enumerate(iou_thrs):
            gtm = e["gtMatches"][i]
            for m, ign in zip(gtm, gtIgnore):
                if ign:
                    continue
                if m > 0:
                    matched_gt[iou][cls] += 1

    return matched_gt, total_gt


def summarize(matched_gt, total_gt, model_name):
    print("\n" + "=" * 80)
    print(f"{model_name} — GT MATCH QUALITY (COCO matching)")
    print("=" * 80)

    for iou in sorted(matched_gt.keys()):
        matched = sum(matched_gt[iou].values())
        total = sum(total_gt.values())
        recall = matched / (total + 1e-6)

        print(f"IoU ≥ {iou:.2f} : "
              f"Matched GT = {matched}/{total} "
              f"({recall:.3f})")

    print("\nPer-class GT coverage at IoU ≥ 0.75:")
    print(f"{'Class':<8} {'Matched':<10} {'Total':<10} {'Recall':<8}")

    for cls in sorted(total_gt.keys()):
        m = matched_gt[0.75].get(cls, 0)
        t = total_gt[cls]
        r = m / (t + 1e-6)
        print(f"{cls:<8} {m:<10} {t:<10} {r:.3f}")


# ---------------- MAIN ---------------- #

GT_JSON = "instances_val2017.json"
CODINO_JSON = "codetr_results_val_set_coco_fixed.json"
GDINO_JSON = "gdino_results_val_set_coco_fixed.json"

codino_matched, total_gt = gt_match_analysis(GT_JSON, CODINO_JSON)
gdino_matched, _        = gt_match_analysis(GT_JSON, GDINO_JSON)

summarize(codino_matched, total_gt, "CODINO")
summarize(gdino_matched,  total_gt, "GDINO")
