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

def collect_gdino_underconfident_tp(
    gt_json,
    gdino_json,
    low_conf_thr=0.3,
    max_per_class=10
):
    """
    Under-confident True Positives:
      - matched to a GT (TP)
      - low confidence score
      - NO IoU reasoning beyond COCO matching
    """

    cocoGt = COCO(gt_json)
    with open(gdino_json, "r") as f:
        preds = json.load(f)

    cocoDt = cocoGt.loadRes(preds)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType="bbox")

    # Default COCO matching (IoU >= 0.5 internally)
    cocoEval.evaluate()

    underconf = defaultdict(list)
    iou_idx = 0  # first IoU threshold

    for e in cocoEval.evalImgs:
        if e is None:
            continue

        cls = int(e["category_id"])
        img_id = int(e["image_id"])

        dtIds = e["dtIds"]
        dtScores = e["dtScores"]
        dtMatches = e["dtMatches"][iou_idx]

        for dt_id, score, match in zip(dtIds, dtScores, dtMatches):
            if match > 0 and score < low_conf_thr:
                underconf[cls].append({
                    "bbox": cocoDt.anns[dt_id]["bbox"],
                    "category_id": cls,
                    "score": float(score),
                    "image_id": img_id
                })

    return {
        cls: items[:max_per_class]
        for cls, items in underconf.items()
    }

def collect_gdino_overconfident_fp(
    gt_json,
    gdino_json,
    high_conf_thr=0.7,
    max_per_class=10
):
    """
    Over-confident False Positives:
      - NOT matched to any GT (FP)
      - high confidence score
      - NO IoU involvement
    """

    cocoGt = COCO(gt_json)
    with open(gdino_json, "r") as f:
        preds = json.load(f)

    cocoDt = cocoGt.loadRes(preds)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType="bbox")
    cocoEval.evaluate()

    overconf = defaultdict(list)
    iou_idx = 0

    for e in cocoEval.evalImgs:
        if e is None:
            continue

        cls = int(e["category_id"])
        img_id = int(e["image_id"])

        dtIds = e["dtIds"]
        dtScores = e["dtScores"]
        dtMatches = e["dtMatches"][iou_idx]

        for dt_id, score, match in zip(dtIds, dtScores, dtMatches):
            if match == 0 and score >= high_conf_thr:
                overconf[cls].append({
                    "bbox": cocoDt.anns[dt_id]["bbox"],
                    "category_id": cls,
                    "score": float(score),
                    "image_id": img_id
                })

    return {
        cls: items[:max_per_class]
        for cls, items in overconf.items()
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
            f"score={score:.2f} | "
            f"{label_suffix}")
        

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

        img_group = defaultdict(list)
        for x in items:
            img_group[x["image_id"]].append({
                "bbox": x["bbox"],
                "category_id": x["category_id"],
                "score": x["score"]   # ✅ ONLY score
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

gdino_underconf = collect_gdino_underconfident_tp(
    GT_JSON,
    GDINO_JSON,
    low_conf_thr=0.3,
    max_per_class=5
)

gdino_overconf = collect_gdino_overconfident_fp(
    GT_JSON,
    GDINO_JSON,
    high_conf_thr=0.7,
    max_per_class=5
)

save_calibration_images(
    gdino_underconf,
    cocoGt,
    IMAGE_DIR,
    output_root="gdino_underconfident_tp",
    label_suffix="TP · LOW confidence"
)

save_calibration_images(
    gdino_overconf,
    cocoGt,
    IMAGE_DIR,
    output_root="gdino_overconfident_fp",
    label_suffix="FP · HIGH confidence"
)
