import os
import json
import cv2
import argparse
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------
# COLORS  (BGR)
# ---------------------------------------------------

RED    = (0,   0,   255)
GREEN  = (0,   255, 0  )
BLUE   = (255, 0,   0  )
YELLOW = (0,   255, 255)
WHITE  = (255, 255, 255)

RSUD_CLASSES = [
    'person','rickshaw','rickshaw van','auto rickshaw',
    'truck','pickup truck','private car','motorcycle',
    'bicycle','bus','micro bus','covered van','human hauler'
]

# ---------------------------------------------------
# LOAD COCO GT
# ---------------------------------------------------

def load_coco(gt_json):

    with open(gt_json, "r") as f:
        coco = json.load(f)

    images         = {}
    anns_per_image = defaultdict(list)
    categories     = {}
    anns_by_id     = {}           # NEW: fast ann_id lookup

    for img in coco["images"]:
        images[img["file_name"]] = img

    for ann in coco["annotations"]:
        anns_per_image[ann["image_id"]].append(ann)
        anns_by_id[ann["id"]] = ann  # NEW

    for cat in coco["categories"]:
        categories[cat["id"] - 1] = cat["name"]

    return images, anns_per_image, categories, anns_by_id


# ---------------------------------------------------
# BOX UTILS
# ---------------------------------------------------

def xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def find_gt_box_by_ann_id(anns_by_id, ann_id):
    """O(1) lookup via dict instead of linear scan."""
    ann = anns_by_id.get(ann_id)
    if ann is None:
        return None
    return xywh_to_xyxy(ann["bbox"])


def draw_box(img, bbox, color, label):
    x1, y1, x2, y2 = [int(v) for v in bbox]

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

    # background chip for readability
    (tw, th), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
    )
    ty = max(th + 6, y1 - 4)
    cv2.rectangle(
        img,
        (x1, ty - th - 6),
        (x1 + tw + 4, ty),
        color,
        -1
    )
    cv2.putText(
        img, label,
        (x1 + 2, ty - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65, WHITE, 2,
        cv2.LINE_AA
    )


# ---------------------------------------------------
# VISUALIZE ONE PAIR
# ---------------------------------------------------

def visualize_pair(
    pair,
    coco_images,
    coco_anns,
    categories,
    anns_by_id,
    image_root,
    output_dir,
    idx
):
    image_name = pair["image_name"]

    if image_name not in coco_images:
        print(f"[WARN] Missing image info: {image_name}")
        return False

    img_info = coco_images[image_name]
    img_path = os.path.join(image_root, image_name)

    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Cannot read: {img_path}")
        return False

    # ── fetch GT boxes ──────────────────────────────
    occ_box = find_gt_box_by_ann_id(
        anns_by_id,
        pair["occluded_gt_ann_id"]    # FIXED (was using occluded id for both)
    )
    occr_box = find_gt_box_by_ann_id(
        anns_by_id,
        pair["occluder_gt_ann_id"]    # FIXED
    )

    if occ_box is None or occr_box is None:
        print(
            f"[WARN] GT boxes not found for pair {idx} "
            f"(ann_ids {pair['occluded_gt_ann_id']} / "
            f"{pair['occluder_gt_ann_id']})"
        )
        return False

    occ_cls  = pair["occluded_class"]
    occr_cls = pair["occluder_class"]

    occ_cls_name  = categories.get(occ_cls,  RSUD_CLASSES[occ_cls]  if occ_cls  < len(RSUD_CLASSES) else str(occ_cls))
    occr_cls_name = categories.get(occr_cls, RSUD_CLASSES[occr_cls] if occr_cls < len(RSUD_CLASSES) else str(occr_cls))

    # ── draw occluded (RED) ─────────────────────────
    draw_box(
        img, occ_box, RED,
        f"OCCLUDED | {occ_cls_name} | {pair['occluded_score']:.2f}"
    )

    # ── draw occluder (GREEN) ───────────────────────
    draw_box(
        img, occr_box, GREEN,
        f"OCCLUDER | {occr_cls_name} | {pair['occluder_score']:.2f}"
    )

    # ── overlay summary text ────────────────────────
    summary = (
        f"Pair#{idx:04d}  |  "
        f"Overlap={pair['occlusion_ratio']:.2f}  |  "
        f"SameClass={pair['same_class']}  |  "
        f"OccDelta={pair['occluded_delta']:.2f}"
    )
    cv2.putText(
        img, summary,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85, YELLOW, 2,
        cv2.LINE_AA
    )

    # ── save ────────────────────────────────────────
    out_name = f"{idx:04d}_{Path(image_name).stem}.jpg"
    out_path = os.path.join(output_dir, out_name)
    cv2.imwrite(out_path, img)
    print(f"[SAVED] {out_path}")
    return True


# ---------------------------------------------------
# ENTRY
# ---------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Visualize occlusion pairs from occlusion_results.json"
    )

    parser.add_argument("--pairs_json",  required=True,
                        help="Path to occlusion_results.json")
    parser.add_argument("--gt_json",     required=True,
                        help="Path to COCO ground-truth JSON")
    parser.add_argument("--image_root",  required=True,
                        help="Root directory containing the images")
    parser.add_argument("--output_dir",  default="vis_pairs",
                        help="Where to save visualized images")

    # optional filters
    parser.add_argument("--max_pairs",   type=int, default=None,
                        help="Limit number of pairs to visualize (default: all)")
    parser.add_argument("--same_class_only", action="store_true",
                        help="Only visualize same-class pairs")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── load pairs ──────────────────────────────────
    with open(args.pairs_json, "r") as f:
        pairs = json.load(f)

    print(f"Loaded {len(pairs)} pairs from {args.pairs_json}")

    # optional same-class filter
    if args.same_class_only:
        pairs = [p for p in pairs if p["same_class"]]
        print(f"After same-class filter: {len(pairs)} pairs")

    if args.max_pairs is not None:
        pairs = pairs[:args.max_pairs]
        print(f"Capped at {len(pairs)} pairs")

    # ── load COCO ───────────────────────────────────
    coco_images, coco_anns, categories, anns_by_id = load_coco(args.gt_json)

    # ── visualize ───────────────────────────────────
    saved = 0
    skipped = 0

    for idx, pair in enumerate(pairs):
        ok = visualize_pair(
            pair,
            coco_images,
            coco_anns,
            categories,
            anns_by_id,
            args.image_root,
            args.output_dir,
            idx
        )
        if ok:
            saved += 1
        else:
            skipped += 1

    print(f"\nDone — saved: {saved}  |  skipped: {skipped}")


if __name__ == "__main__":
    main()