"""
visualize_new_fps.py
=====================
Crops and saves the top-K highest-confidence "new" FPs from
new_fp_diagnosis.json (produced by fp_clustering_diagnosis.py) for
manual visual inspection.

Draws:
  - FP box in RED
  - all target-category GT boxes in the same image in GREEN
    (thicker outline for the GT box the FP is nearest to, if any)

Usage:
    python3 visualize_new_fps.py \
        --gt val_instances_with_occlusion_visibility_ratio.json \
        --images-root /path/to/citypersons/leftImg8bit/val \
        --detail-json new_fp_diagnosis.json \
        --top-k 20 \
        --pad 80 \
        --out-dir top_fp_visualizations
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from citypersons_fp_clustering_diagnosis import (
    load_json, get_target_category_ids, build_gt_structures, iou,
)


def draw_box(draw, bbox, color, width=3):
    x, y, w, h = bbox
    draw.rectangle([x, y, x + w, y + h], outline=color, width=width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--images-root", required=True,
                     help="Root dir that gt['images'][i]['file_name'] is relative to.")
    ap.add_argument("--detail-json", default="new_fp_diagnosis.json")
    ap.add_argument("--target-categories", nargs="+",
                     default=["pedestrian", "rider", "sitting person"])
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--pad", type=int, default=80,
                     help="Padding in px around the FP box for the crop.")
    ap.add_argument("--out-dir", default="top_fp_visualizations")
    args = ap.parse_args()

    gt_full = load_json(args.gt)
    target_category_ids = get_target_category_ids(gt_full, args.target_categories)
    _, gt_all_by_image = build_gt_structures(gt_full, target_category_ids)

    id_to_filename = {img["id"]: img["file_name"] for img in gt_full["images"]}

    detailed = load_json(args.detail_json)
    detailed = sorted(detailed, key=lambda r: -r["score"])[:args.top_k]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images_root = Path(args.images_root)

    for rank, d in enumerate(detailed, start=1):
        img_id = d["img_id"]
        fname = id_to_filename.get(img_id)
        if fname is None:
            print(f"[skip] no filename for img_id={img_id}")
            continue

        img_path = images_root / fname
        if not img_path.exists():
            print(f"[skip] missing file: {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        fp_bbox = d["bbox"]
        gt_boxes = gt_all_by_image.get(img_id, [])

        # nearest GT (if any) gets a thicker green box, rest get thin green
        best_gi, best_iou = -1, 0.0
        for gi, g in enumerate(gt_boxes):
            ov = iou(fp_bbox, g)
            if ov > best_iou:
                best_iou, best_gi = ov, gi

        for gi, g in enumerate(gt_boxes):
            draw_box(draw, g, "lime", width=5 if gi == best_gi else 2)

        draw_box(draw, fp_bbox, "red", width=3)

        # crop around FP box with padding, clamped to image bounds
        x, y, w, h = fp_bbox
        left = max(0, int(x - args.pad))
        top = max(0, int(y - args.pad))
        right = min(img.width, int(x + w + args.pad))
        bottom = min(img.height, int(y + h + args.pad))
        crop = img.crop((left, top, right, bottom))

        out_name = (f"{rank:02d}_img{img_id}_score{d['score']:.3f}"
                    f"_{d['label']}_cl{d['cluster_size']}.png")
        crop.save(out_dir / out_name)
        print(f"saved {out_name}")

    print(f"\nDone. {len(detailed)} crops saved to {out_dir}/")


if __name__ == "__main__":
    main()