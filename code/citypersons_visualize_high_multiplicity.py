"""
visualize_high_multiplicity.py
================================
Crops and saves the top-K highest-multiplicity persistent-location FPs
from high_multiplicity_locations.json (produced by
fp_multiplicity_and_hallucination_analysis.py) for manual visual
inspection.

Draws:
  - baseline FP location in ORANGE (thick, dashed-look outline)
  - each updated FP matched to that location in RED (thin outline each)
  - all target-category GT boxes in the image in GREEN

Usage:
    python3 visualize_high_multiplicity.py \
        --gt val_instances_with_occlusion_visibility_ratio.json \
        --images-root /path/to/citypersons/leftImg8bit/val \
        --detail-json high_multiplicity_locations.json \
        --top-k 20 \
        --pad 80 \
        --out-dir high_multiplicity_visualizations
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from citypersons_fp_clustering_diagnosis import load_json, get_target_category_ids, build_gt_structures


def draw_dashed_rect(draw, bbox, color, width=4, dash=10, gap=6):
    x, y, w, h = bbox
    x2, y2 = x + w, y + h
    # top & bottom edges
    for (xa, ya, xb) in [(x, y, x2), (x, y2, x2)]:
        pos = xa
        while pos < xb:
            end = min(pos + dash, xb)
            draw.line([(pos, ya), (end, ya)], fill=color, width=width)
            pos += dash + gap
    # left & right edges
    for (xa, ya, yb) in [(x, y, y2), (x2, y, y2)]:
        pos = ya
        while pos < yb:
            end = min(pos + dash, yb)
            draw.line([(xa, pos), (xa, end)], fill=color, width=width)
            pos += dash + gap


def draw_box(draw, bbox, color, width=3):
    x, y, w, h = bbox
    draw.rectangle([x, y, x + w, y + h], outline=color, width=width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--detail-json", default="high_multiplicity_locations.json")
    ap.add_argument("--target-categories", nargs="+",
                     default=["pedestrian", "rider", "sitting person"])
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--pad", type=int, default=80)
    ap.add_argument("--out-dir", default="high_multiplicity_visualizations")
    args = ap.parse_args()

    gt_full = load_json(args.gt)
    target_category_ids = get_target_category_ids(gt_full, args.target_categories)
    _, gt_all_by_image = build_gt_structures(gt_full, target_category_ids)

    id_to_filename = {img["id"]: img["file_name"] for img in gt_full["images"]}

    detail = load_json(args.detail_json)[:args.top_k]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_root = Path(args.images_root)

    for rank, d in enumerate(detail, start=1):
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

        # GT boxes (green)
        for g in gt_all_by_image.get(img_id, []):
            draw_box(draw, g, "lime", width=2)

        # baseline FP location (orange, dashed)
        draw_dashed_rect(draw, d["baseline_bbox"], "orange", width=4)

        # all updated FPs matched to this location (red)
        for u in d["updated_fps"]:
            draw_box(draw, u["bbox"], "red", width=2)

        # crop around the baseline box, padded to fit all updated boxes too
        all_boxes = [d["baseline_bbox"]] + [u["bbox"] for u in d["updated_fps"]]
        xs = [b[0] for b in all_boxes] + [b[0] + b[2] for b in all_boxes]
        ys = [b[1] for b in all_boxes] + [b[1] + b[3] for b in all_boxes]
        left = max(0, int(min(xs) - args.pad))
        top = max(0, int(min(ys) - args.pad))
        right = min(img.width, int(max(xs) + args.pad))
        bottom = min(img.height, int(max(ys) + args.pad))
        crop = img.crop((left, top, right, bottom))

        out_name = f"{rank:02d}_img{img_id}_mult{d['multiplicity']}.png"
        crop.save(out_dir / out_name)
        print(f"saved {out_name}  (multiplicity={d['multiplicity']})")

    print(f"\nDone. {len(detail)} crops saved to {out_dir}/")


if __name__ == "__main__":
    main()