import numpy as np
import json
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
from PIL import Image, ImageDraw, ImageFont


# -------------------------------------------------
# GDINO: High-IoU but Low-Confidence True Positives
# -------------------------------------------------

def collect_gdino_high_iou_low_conf(
    gt_json,
    gdino_json,
    iou_thr=0.5,
    low_conf_thr=0.3,
    max_per_class=10
):
    """
    Collect GDINO detections that:
      - match a GT with IoU >= iou_thr
      - have confidence < low_conf_thr

    Returns:
      underconfident[class_id] = list of dicts:
        {
          "ann": dt_ann,
          "iou": float,
          "score": float,
          "image_id": int
        }
    """

    cocoGt = COCO(gt_json)

    with open(gdino_json, "r") as f:
        preds = json.load(f)

    # IMPORTANT: do NOT threshold by score
    cocoDt = cocoGt.loadRes(preds)

    cocoEval = COCOeval(cocoGt, cocoDt, iouType="bbox")
    cocoEval.params.iouThrs = np.array([iou_thr])
    cocoEval.params.maxDets = [100]
    cocoEval.evaluate()

    iou_idx = 0
    underconfident = defaultdict(list)

    for e in cocoEval.evalImgs:
        if e is None:
            continue

        cls = int(e["category_id"])
        img_id = int(e["image_id"])

        dtIds = e["dtIds"]
        dtScores = e["dtScores"]
        dtMatches = e["dtMatches"][iou_idx]
        gtIds = e["gtIds"]

        # IoU matrix for this (image, class)
        ious = cocoEval.ious.get((img_id, cls), None)
        if ious is None:
            continue

        for d_idx, (dt_id, score, match) in enumerate(zip(dtIds, dtScores, dtMatches)):
            if match == 0:
                continue  # not matched to GT

            # safe GT index lookup
            try:
                gt_idx = gtIds.index(int(match))
            except ValueError:
                continue

            iou = ious[d_idx, gt_idx]

            if iou >= iou_thr and score < low_conf_thr:
                underconfident[cls].append({
                    "bbox": cocoDt.anns[dt_id]["bbox"],
                    "category_id": cls,
                    "iou": float(iou),
                    "score": float(score),
                    "image_id": img_id
                })

    # Limit samples per class
    final = {}
    for cls, items in underconfident.items():
        final[cls] = items[:max_per_class]

    return final

def collect_gdino_low_iou_high_conf(
    gt_json,
    gdino_json,
    match_iou_thr=0.5,
    low_loc_thr=0.7,
    high_conf_thr=0.7,
    max_per_class=10
):
    """
    Collects OVER-CONFIDENT detections:
    - matched TP at IoU >= match_iou_thr
    - but IoU < low_loc_thr (poor localization)
    - high confidence score
    """

    cocoGt = COCO(gt_json)

    with open(gdino_json, "r") as f:
        preds = json.load(f)

    cocoDt = cocoGt.loadRes(preds)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType="bbox")

    # IMPORTANT: matching threshold
    cocoEval.params.iouThrs = np.array([match_iou_thr])
    cocoEval.params.maxDets = [100]
    cocoEval.evaluate()

    iou_idx = 0
    overconfident = defaultdict(list)

    for e in cocoEval.evalImgs:
        if e is None:
            continue

        cls = int(e["category_id"])
        img_id = int(e["image_id"])

        dtIds = e["dtIds"]
        dtScores = e["dtScores"]
        dtMatches = e["dtMatches"][iou_idx]
        gtIds = e["gtIds"]

        ious = cocoEval.ious.get((img_id, cls), None)
        if ious is None:
            continue

        for d_idx, (dt_id, score, match) in enumerate(
            zip(dtIds, dtScores, dtMatches)
        ):
            if match == 0:
                continue  # not a TP

            gt_idx = gtIds.index(int(match))
            iou = ious[d_idx, gt_idx]

            if (
                iou < low_loc_thr
                and score >= high_conf_thr
            ):
                overconfident[cls].append({
                    "bbox": cocoDt.anns[dt_id]["bbox"],
                    "category_id": cls,
                    "iou": float(iou),
                    "score": float(score),
                    "image_id": img_id
                })

    return {
        cls: items[:max_per_class]
        for cls, items in overconfident.items()
    }


# -------------------------------------------------
# Visualization Utilities
# -------------------------------------------------

def plot_annotations_on_image(image_path, annotations, cocoGt, output_path, label_suffix):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
        )
    except:
        font = ImageFont.load_default()

    colors = [
        "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF",
        "#00FFFF", "#FFA500", "#800080", "#808080"
    ]

    for ann in annotations:
        x, y, w, h = ann["bbox"]
        cls = ann["category_id"]
        cls_name = cocoGt.cats[cls]["name"]
        color = colors[cls % len(colors)]

        # Draw bbox
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)

        # ---- label with IoU + score ----
        iou = ann.get("iou", -1)
        score = ann.get("score", -1)

        label = (
            f"{cls_name} | "
            f"IoU={iou:.2f} | "
            f"score={score:.2f} | "
            f"{label_suffix}"
        )

        # Compute text size SAFELY
        tb = draw.textbbox((0, 0), label, font=font)
        text_w = tb[2] - tb[0]
        text_h = tb[3] - tb[1]

        # Draw label BELOW the box (safe)
        tx = x
        ty = y + h + 4

        draw.rectangle(
            [tx, ty, tx + text_w + 6, ty + text_h + 6],
            fill=color
        )
        draw.text(
            (tx + 3, ty + 3),
            label,
            fill="white",
            font=font
        )

    img.save(output_path)


def save_calibration_images(
    cases,
    cocoGt,
    image_dir,
    output_root,
    label_suffix
):
    os.makedirs(output_root, exist_ok=True)

    for cls, items in cases.items():
        if not items:
            continue

        cls_name = cocoGt.cats[cls]["name"]
        cls_dir = os.path.join(output_root, f"{cls}_{cls_name}")
        os.makedirs(cls_dir, exist_ok=True)

        # group detections per image
        img_group = defaultdict(list)
        for x in items:
            img_group[x["image_id"]].append({
                "bbox": x["bbox"],
                "category_id": x["category_id"],
                "iou": x["iou"],
                "score": x["score"]
            })

        for img_id, anns in img_group.items():
            img_info = cocoGt.imgs[img_id]
            src = os.path.join(image_dir, img_info["file_name"])
            if not os.path.exists(src):
                continue

            dst = os.path.join(cls_dir, img_info["file_name"])

            plot_annotations_on_image(
                src,
                anns,
                cocoGt,
                dst,
                label_suffix
            )

    print(f"✅ Saved visualizations to {output_root}")



# -------------------------------------------------
# MAIN
# -------------------------------------------------

GT_JSON = "merged_instances_val2017_codetr_final.json"
GDINO_JSON = "gdino_results_val_set_coco_fixed.json"
IMAGE_DIR = "/home/poonam_rajput/scratch/dataset/RSUD_dataset/images/val"

print("\n" + "=" * 85)
print("GDINO CALIBRATION ANALYSIS: High IoU but Low Confidence")
print("=" * 85)

cocoGt = COCO(GT_JSON)

gdino_underconf = collect_gdino_high_iou_low_conf(
    gt_json=GT_JSON,
    gdino_json=GDINO_JSON,
    iou_thr=0.5,
    low_conf_thr=0.30,
    max_per_class=5
)

# gdino_underconf = collect_gdino_low_iou_high_conf(
#     gt_json=GT_JSON,
#     gdino_json=GDINO_JSON,
#     max_per_class=5
# )
total = sum(len(v) for v in gdino_underconf.values())
print(f"Found {total} GDINO under-confident true detections\n")

for cls, items in gdino_underconf.items():
    scores = [x["score"] for x in items]
    cls_name = cocoGt.cats[cls]["name"]
    print(f"Class {cls} ({cls_name}): {len(items)} cases | "
          f"avg score={np.mean(scores):.3f}")

save_calibration_images(
    gdino_underconf,
    cocoGt,
    IMAGE_DIR,
    output_root="gdino_underconfident_tp",
    label_suffix="HIGH IoU · LOW score"
)

# save_calibration_images(
#     gdino_underconf,
#     cocoGt,
#     IMAGE_DIR,
#     output_root="gdino_overconfident_fp",
#     label_suffix="LOW IoU · HIGH score"
# )
