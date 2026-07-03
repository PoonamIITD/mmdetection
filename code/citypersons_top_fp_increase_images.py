# """
# visualize_top_fp_increase_images.py
# =====================================
# Visualizes the images with the LARGEST per-image FP-count increase
# (updated - baseline) in the Heavy_Occlusion subset -- the concrete
# images driving the paired significance test in
# solidify_heavy_occlusion_fp_comparison.py.

# Draws, on the FULL image (not a crop, since these images often have many
# scattered FPs):
#   - GT boxes (Heavy_Occlusion window, non-ignore) in GREEN, each tagged
#     with "B:<n> U:<n>" = how many baseline / updated FPs overlap that GT
#     region (IoU >= --near-gt-iou), so you don't have to eyeball dense
#     clutter to see WHERE the extra updated FPs are concentrating.
#   - baseline FPs in BLUE
#   - updated FPs in RED
#   - a corner summary tallying FPs that aren't near ANY GT box
#     (background/occluder hallucinations) for both models.

# Usage:
#     python3 visualize_top_fp_increase_images.py \
#         --gt val_instances_with_occlusion_visibility_ratio.json \
#         --baseline results_citypersons_baseline_val.json \
#         --updated results_citypersons_sampling_loss_val.json \
#         --images-root ../../dataset/CityPersons_dataset/images/val \
#         --score-thresh 0.1 \
#         --near-gt-iou 0.1 \
#         --top-k 12 \
#         --out-dir top_fp_increase_visualizations
# """

# import argparse
# from pathlib import Path
# from collections import defaultdict

# from PIL import Image, ImageDraw, ImageFont

# from citypersons_fp_clustering_diagnosis import (
#     load_json, get_target_category_ids, build_gt_structures, match_and_collect_fps, iou,
# )


# def draw_box(draw, bbox, color, width=3, label=None, label_pos="above", font=None):
#     x, y, w, h = bbox
#     draw.rectangle([x, y, x + w, y + h], outline=color, width=width)
#     if label:
#         ty = max(0, y - 16) if label_pos == "above" else (y + h + 2)
#         # small dark background behind text for readability over busy images
#         tw = draw.textlength(label, font=font) if font else len(label) * 7
#         draw.rectangle([x, ty, x + tw + 4, ty + 14], fill=(0, 0, 0))
#         draw.text((x + 2, ty), label, fill=color, font=font)


# def count_nearby_fps(gt_bbox, fp_list, iou_thresh):
#     return sum(1 for r in fp_list if iou(gt_bbox, r["bbox"]) >= iou_thresh)


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--gt", required=True)
#     ap.add_argument("--baseline", required=True)
#     ap.add_argument("--updated", required=True)
#     ap.add_argument("--images-root", required=True)
#     ap.add_argument("--target-categories", nargs="+",
#                      default=["pedestrian", "rider", "sitting person"])
#     ap.add_argument("--score-thresh", type=float, default=0.1)
#     ap.add_argument("--near-gt-iou", type=float, default=0.1,
#                      help="IoU threshold for counting an FP as 'near' a GT box for tagging.")
#     ap.add_argument("--top-k", type=int, default=12)
#     ap.add_argument("--out-dir", default="top_fp_increase_visualizations")
#     args = ap.parse_args()

#     try:
#         font = ImageFont.load_default(size=13)
#     except TypeError:
#         font = ImageFont.load_default()

#     gt_full = load_json(args.gt)
#     target_category_ids = get_target_category_ids(gt_full, args.target_categories)
#     b_preds = load_json(args.baseline)
#     u_preds = load_json(args.updated)

#     gt_by_image_subset, _ = build_gt_structures(gt_full, target_category_ids)
#     id_to_filename = {img["id"]: img["file_name"] for img in gt_full["images"]}

#     print("Matching baseline FPs...")
#     b_fps = match_and_collect_fps(gt_by_image_subset, b_preds, target_category_ids, args.score_thresh)
#     print("Matching updated FPs...")
#     u_fps = match_and_collect_fps(gt_by_image_subset, u_preds, target_category_ids, args.score_thresh)

#     b_by_img = defaultdict(list)
#     for r in b_fps:
#         b_by_img[r["img_id"]].append(r)
#     u_by_img = defaultdict(list)
#     for r in u_fps:
#         u_by_img[r["img_id"]].append(r)

#     # per-image diff, ranked descending
#     all_img_ids = set(gt_by_image_subset.keys())
#     diffs = []
#     for img_id in all_img_ids:
#         b_n = len(b_by_img.get(img_id, []))
#         u_n = len(u_by_img.get(img_id, []))
#         diffs.append((img_id, u_n - b_n, b_n, u_n))
#     diffs.sort(key=lambda d: -d[1])
#     top = diffs[:args.top_k]

#     out_dir = Path(args.out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)
#     images_root = Path(args.images_root)

#     print(f"\nTop {args.top_k} images by FP-count increase (updated - baseline):")
#     for rank, (img_id, diff, b_n, u_n) in enumerate(top, start=1):
#         fname = id_to_filename.get(img_id)
#         if fname is None:
#             print(f"[skip] no filename for img_id={img_id}")
#             continue
#         img_path = images_root / fname
#         if not img_path.exists():
#             print(f"[skip] missing file: {img_path}")
#             continue

#         img = Image.open(img_path).convert("RGB")
#         draw = ImageDraw.Draw(img)

#         b_list = b_by_img.get(img_id, [])
#         u_list = u_by_img.get(img_id, [])

#         # GT (non-ignore, Heavy_Occlusion window) in green, tagged with
#         # per-region FP counts so density is readable at a glance
#         gt_boxes_shown = [g["bbox"] for g in gt_by_image_subset[img_id] if g["ignore"] == 0]
#         for gt_bbox in gt_boxes_shown:
#             b_near = count_nearby_fps(gt_bbox, b_list, args.near_gt_iou)
#             u_near = count_nearby_fps(gt_bbox, u_list, args.near_gt_iou)
#             tag = f"B:{b_near} U:{u_near}"
#             draw_box(draw, gt_bbox, "lime", width=2, label=tag, label_pos="above", font=font)

#         # baseline FPs in blue (thin, since GT tags already summarize density)
#         for r in b_list:
#             draw_box(draw, r["bbox"], "deepskyblue", width=2)

#         # updated FPs in red
#         for r in u_list:
#             draw_box(draw, r["bbox"], "red", width=2)

#         # background-only FPs (not near any GT box at all) for both models
#         b_bg = sum(1 for r in b_list
#                    if not any(iou(r["bbox"], gt) >= args.near_gt_iou for gt in gt_boxes_shown))
#         u_bg = sum(1 for r in u_list
#                    if not any(iou(r["bbox"], gt) >= args.near_gt_iou for gt in gt_boxes_shown))
#         summary = f"Background FPs (no nearby GT):  Baseline={b_bg}  Updated={u_bg}"
#         draw.rectangle([5, 5, 5 + draw.textlength(summary, font=font) + 10, 24], fill=(0, 0, 0))
#         draw.text((8, 6), summary, fill="yellow", font=font)

#         out_name = f"{rank:02d}_img{img_id}_diff{diff:+d}_base{b_n}_upd{u_n}.png"
#         img.save(out_dir / out_name)
#         print(f"  {rank:2d}. img_id={img_id}  baseline_FP={b_n}  updated_FP={u_n}  "
#               f"diff={diff:+d}  background(B/U)={b_bg}/{u_bg}  -> saved {out_name}")

#     print(f"\nDone. {len(top)} images saved to {out_dir}/")
#     print("Legend: GREEN = GT box, tagged 'B:x U:y' (baseline/updated FPs overlapping that GT)")
#     print("        BLUE = baseline FP box | RED = updated FP box")
#     print("        Yellow corner text = FPs with no nearby GT at all (background hallucinations)")


# if __name__ == "__main__":
#     main()




"""
visualize_top_fp_increase_images.py
=====================================
Visualizes the images with the LARGEST per-image FP-count increase
(updated - baseline) in the Heavy_Occlusion subset -- the concrete
images driving the paired significance test in
solidify_heavy_occlusion_fp_comparison.py.

Draws, on the FULL image (not a crop, since these images often have many
scattered FPs):
  - GT boxes (Heavy_Occlusion window, non-ignore) in GREEN, each tagged
    with "B:<n> U:<n>" = how many baseline / updated FPs overlap that GT
    region (IoU >= --near-gt-iou), so you don't have to eyeball dense
    clutter to see WHERE the extra updated FPs are concentrating.
  - baseline FPs in BLUE, updated FPs in RED
  - SOLID box  = this FP is near a real GT box (IoU >= --near-gt-iou) --
    part of the "duplication at real people" story, already counted in
    that GT's B:x U:y tag.
  - DASHED box = this FP has NO nearby GT at all -- a background/occluder
    hallucination, individually traceable now instead of only being a
    corner-count number.
  - a corner summary tallying the dashed (background) FPs for both models.

Usage:
    python3 visualize_top_fp_increase_images.py \
        --gt val_instances_with_occlusion_visibility_ratio.json \
        --baseline results_citypersons_baseline_val.json \
        --updated results_citypersons_sampling_loss_val.json \
        --images-root ../../dataset/CityPersons_dataset/images/val \
        --score-thresh 0.1 \
        --near-gt-iou 0.1 \
        --top-k 12 \
        --out-dir top_fp_increase_visualizations
"""

import argparse
from pathlib import Path
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

from citypersons_fp_clustering_diagnosis import (
    load_json, get_target_category_ids, build_gt_structures, match_and_collect_fps, iou,
)


def draw_box(draw, bbox, color, width=3, label=None, label_pos="above", font=None):
    x, y, w, h = bbox
    draw.rectangle([x, y, x + w, y + h], outline=color, width=width)
    if label:
        ty = max(0, y - 16) if label_pos == "above" else (y + h + 2)
        # small dark background behind text for readability over busy images
        tw = draw.textlength(label, font=font) if font else len(label) * 7
        draw.rectangle([x, ty, x + tw + 4, ty + 14], fill=(0, 0, 0))
        draw.text((x + 2, ty), label, fill=color, font=font)


def draw_dashed_box(draw, bbox, color, width=2, dash=6, gap=4):
    """Dashed rectangle -- used to mark FPs with NO nearby GT (background
    hallucinations), visually distinct from solid boxes (near-GT FPs)."""
    x, y, w, h = bbox
    x2, y2 = x + w, y + h
    for (xa, ya, xb) in [(x, y, x2), (x, y2, x2)]:
        pos = xa
        while pos < xb:
            end = min(pos + dash, xb)
            draw.line([(pos, ya), (end, ya)], fill=color, width=width)
            pos += dash + gap
    for (xa, ya, yb) in [(x, y, y2), (x2, y, y2)]:
        pos = ya
        while pos < yb:
            end = min(pos + dash, yb)
            draw.line([(xa, pos), (xa, end)], fill=color, width=width)
            pos += dash + gap


def count_nearby_fps(gt_bbox, fp_list, iou_thresh):
    return sum(1 for r in fp_list if iou(gt_bbox, r["bbox"]) >= iou_thresh)


def is_near_any_gt(fp_bbox, gt_boxes, iou_thresh):
    return any(iou(fp_bbox, gt) >= iou_thresh for gt in gt_boxes)


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _expand(bbox, pad_px):
    x, y, w, h = bbox
    return [x - pad_px, y - pad_px, w + 2 * pad_px, h + 2 * pad_px]


def _rects_overlap(a, b):
    ax1, ay1, aw, ah = a
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = b
    bx2, by2 = bx1 + bw, by1 + bh
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def cluster_background_fps(items, pad_px):
    """Spatially cluster background FPs (no nearby GT) using padded-box
    overlap as the connectivity rule -- these boxes rarely overlap each
    other directly (they're scattered, not literal duplicates), so a raw
    IoU-based cluster (like cluster_gt for GT boxes) would under-merge.
    Padding each box outward before checking overlap groups FPs that sit
    close together in the same hotspot region (e.g. all along one railing
    or storefront) even if their own boxes don't touch.

    items: list of dicts {"bbox":, "source": "baseline"/"updated"}
    Returns: list of clusters, each a dict {"enclosing_bbox":, "b_count":, "u_count":}
    """
    n = len(items)
    if n == 0:
        return []
    uf = _UnionFind(n)
    padded = [_expand(it["bbox"], pad_px) for it in items]
    for i in range(n):
        for j in range(i + 1, n):
            if _rects_overlap(padded[i], padded[j]):
                uf.union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)

    clusters = []
    for idx_list in groups.values():
        xs1, ys1, xs2, ys2 = [], [], [], []
        b_count, u_count = 0, 0
        for i in idx_list:
            x, y, w, h = items[i]["bbox"]
            xs1.append(x); ys1.append(y); xs2.append(x + w); ys2.append(y + h)
            if items[i]["source"] == "baseline":
                b_count += 1
            else:
                u_count += 1
        enclosing = [min(xs1), min(ys1), max(xs2) - min(xs1), max(ys2) - min(ys1)]
        clusters.append({"enclosing_bbox": enclosing, "b_count": b_count, "u_count": u_count,
                          "size": len(idx_list)})
    return clusters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--updated", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--target-categories", nargs="+",
                     default=["pedestrian", "rider", "sitting person"])
    ap.add_argument("--score-thresh", type=float, default=0.1)
    ap.add_argument("--near-gt-iou", type=float, default=0.1,
                     help="IoU threshold for counting an FP as 'near' a GT box for tagging.")
    ap.add_argument("--bg-cluster-pad", type=float, default=20.0,
                     help="Pixel padding used to spatially cluster background "
                          "(no-nearby-GT) FPs into hotspot regions for tagging.")
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--out-dir", default="top_fp_increase_visualizations")
    args = ap.parse_args()

    try:
        font = ImageFont.load_default(size=13)
    except TypeError:
        font = ImageFont.load_default()

    gt_full = load_json(args.gt)
    target_category_ids = get_target_category_ids(gt_full, args.target_categories)
    b_preds = load_json(args.baseline)
    u_preds = load_json(args.updated)

    gt_by_image_subset, _ = build_gt_structures(gt_full, target_category_ids)
    id_to_filename = {img["id"]: img["file_name"] for img in gt_full["images"]}

    print("Matching baseline FPs...")
    b_fps = match_and_collect_fps(gt_by_image_subset, b_preds, target_category_ids, args.score_thresh)
    print("Matching updated FPs...")
    u_fps = match_and_collect_fps(gt_by_image_subset, u_preds, target_category_ids, args.score_thresh)

    b_by_img = defaultdict(list)
    for r in b_fps:
        b_by_img[r["img_id"]].append(r)
    u_by_img = defaultdict(list)
    for r in u_fps:
        u_by_img[r["img_id"]].append(r)

    # per-image diff, ranked descending
    all_img_ids = set(gt_by_image_subset.keys())
    diffs = []
    for img_id in all_img_ids:
        b_n = len(b_by_img.get(img_id, []))
        u_n = len(u_by_img.get(img_id, []))
        diffs.append((img_id, u_n - b_n, b_n, u_n))
    diffs.sort(key=lambda d: -d[1])
    top = diffs[:args.top_k]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_root = Path(args.images_root)

    print(f"\nTop {args.top_k} images by FP-count increase (updated - baseline):")
    for rank, (img_id, diff, b_n, u_n) in enumerate(top, start=1):
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

        b_list = b_by_img.get(img_id, [])
        u_list = u_by_img.get(img_id, [])

        # GT (non-ignore, Heavy_Occlusion window) in green, tagged with
        # per-region FP counts so density is readable at a glance
        gt_boxes_shown = [g["bbox"] for g in gt_by_image_subset[img_id] if g["ignore"] == 0]
        for gt_bbox in gt_boxes_shown:
            b_near = count_nearby_fps(gt_bbox, b_list, args.near_gt_iou)
            u_near = count_nearby_fps(gt_bbox, u_list, args.near_gt_iou)
            tag = f"B:{b_near} U:{u_near}"
            draw_box(draw, gt_bbox, "lime", width=2, label=tag, label_pos="above", font=font)

        # baseline FPs: SOLID = near a real GT, DASHED = background hallucination
        b_bg_boxes = []
        for r in b_list:
            if is_near_any_gt(r["bbox"], gt_boxes_shown, args.near_gt_iou):
                draw_box(draw, r["bbox"], "deepskyblue", width=2)
            else:
                draw_dashed_box(draw, r["bbox"], "deepskyblue", width=2)
                b_bg_boxes.append(r)

        # updated FPs: SOLID = near a real GT, DASHED = background hallucination
        u_bg_boxes = []
        for r in u_list:
            if is_near_any_gt(r["bbox"], gt_boxes_shown, args.near_gt_iou):
                draw_box(draw, r["bbox"], "red", width=2)
            else:
                draw_dashed_box(draw, r["bbox"], "red", width=2)
                u_bg_boxes.append(r)

        b_bg, u_bg = len(b_bg_boxes), len(u_bg_boxes)

        # cluster background (no-nearby-GT) FPs into spatial hotspots and
        # tag each one with "BG B:x U:y", same convention as GT tags
        bg_items = ([{"bbox": r["bbox"], "source": "baseline"} for r in b_bg_boxes] +
                    [{"bbox": r["bbox"], "source": "updated"} for r in u_bg_boxes])
        bg_clusters = cluster_background_fps(bg_items, pad_px=args.bg_cluster_pad)
        for cl in bg_clusters:
            tag = f"BG B:{cl['b_count']} U:{cl['u_count']}"
            # draw a loose orange rectangle around the whole hotspot region,
            # slightly padded so it visibly encloses the scattered boxes
            ex, ey, ew, eh = cl["enclosing_bbox"]
            pad = 6
            draw.rectangle([ex - pad, ey - pad, ex + ew + pad, ey + eh + pad],
                            outline="orange", width=2)
            tw = draw.textlength(tag, font=font)
            ty = max(0, ey - pad - 16)
            draw.rectangle([ex - pad, ty, ex - pad + tw + 4, ty + 14], fill=(0, 0, 0))
            draw.text((ex - pad + 2, ty), tag, fill="orange", font=font)

        summary = f"Background FPs (dashed, no nearby GT):  Baseline={b_bg}  Updated={u_bg}"
        draw.rectangle([5, 5, 5 + draw.textlength(summary, font=font) + 10, 24], fill=(0, 0, 0))
        draw.text((8, 6), summary, fill="yellow", font=font)

        out_name = f"{rank:02d}_img{img_id}_diff{diff:+d}_base{b_n}_upd{u_n}.png"
        img.save(out_dir / out_name)
        print(f"  {rank:2d}. img_id={img_id}  baseline_FP={b_n}  updated_FP={u_n}  "
              f"diff={diff:+d}  background(B/U)={b_bg}/{u_bg}  -> saved {out_name}")

    print(f"\nDone. {len(top)} images saved to {out_dir}/")
    print("Legend: GREEN  = GT box, tagged 'B:x U:y' (baseline/updated FPs overlapping that GT)")
    print("        BLUE   = baseline FP | RED = updated FP")
    print("        SOLID box  = FP is near a real GT (counted in that GT's B:x U:y tag)")
    print("        DASHED box = FP has NO nearby GT -- background/occluder hallucination")
    print("        ORANGE box = spatial hotspot cluster of background FPs, tagged")
    print("                     'BG B:x U:y' (baseline/updated FP count in that hotspot)")
    print("        Yellow corner text = total background FP count per model, whole image")


if __name__ == "__main__":
    main()