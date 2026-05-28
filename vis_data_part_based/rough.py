import torch
import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2

RSUD_CLASSES = ['person','rickshaw','rickshaw van','auto rickshaw',
                'truck','pickup truck','private car','motorcycle','bicycle','bus','micro bus','covered van','human hauler'
]

# --------------------------------------------------
# 🔵 Pick best query
# --------------------------------------------------
def select_best_query(cls_scores):
    # cls_scores: [L, Q, C]
    final_scores = cls_scores[-1]  # [Q, C]
    max_scores, _ = final_scores.max(dim=1)
    q_idx = torch.argmax(max_scores).item()
    return q_idx


#----------------------------------------------------------------------------------------------------------------
# select low confidence query from the initial decoder layer class confidence to see how its confidence improve 
#----------------------------------------------------------------------------------------------------------------
def select_low_conf_query(cls_scores, layer=-1, target_score=0.1):
    """
    cls_scores: [L, Q, C]
    layer: which decoder layer to inspect
    target_score: pick query closest to this score
    """

    scores_l = cls_scores[layer]  # [Q, C]

    # max class score per query
    max_scores, _ = scores_l.max(dim=1)  # [Q]

    # find query closest to target_score
    diff = torch.abs(max_scores - target_score)

    q_idx = torch.argmin(diff).item()

    return q_idx, max_scores[q_idx].item()

import math

def get_gt_box_norm(gt_bbox, image_path):
    """
    gt_bbox: [x, y, w, h] in pixel coordinates
    returns normalized GT center and box size
    """
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    H, W = img.shape[:2]
    x, y, w, h = gt_bbox

    cx = (x + w / 2.0) / W
    cy = (y + h / 2.0) / H
    bw = w / W
    bh = h / H

    return cx, cy, bw, bh


def find_query_ids_close_to_gt(
    data,
    image_path,
    gt_bbox,
    layer=-1,
    margin_ratio=0.15,
    dist_ratio=0.35,
    topk=None,
):
    """
    Return query ids whose reference points are close to the GT box.

    A query is selected if:
      1) its reference point lies inside an expanded GT box, OR
      2) its distance to GT center is within a threshold.

    Args:
        data: dict loaded from .pt
        image_path: path to image
        gt_bbox: [x, y, w, h] in pixel coordinates
        layer: decoder layer to inspect (default: last)
        margin_ratio: expands GT box on each side
        dist_ratio: distance threshold relative to GT diagonal
        topk: optionally keep only top-k closest queries

    Returns:
        query_ids: list[int]
        distances: 1D tensor of distances for all queries
    """
    refs = data["reference_points"]   # expected [L, Q, 2] or [L, Q, 4]
    if refs.dim() == 3:
        refs_l = refs[layer]          # [Q, 2] or [Q, 4]
    else:
        raise ValueError(f"Unexpected reference_points shape: {refs.shape}")

    # use only x,y
    refs_xy = refs_l[:, :2]

    cx, cy, bw, bh = get_gt_box_norm(gt_bbox, image_path)

    # expanded GT box bounds in normalized coordinates
    x1 = cx - (bw / 2.0) * (1.0 + margin_ratio)
    y1 = cy - (bh / 2.0) * (1.0 + margin_ratio)
    x2 = cx + (bw / 2.0) * (1.0 + margin_ratio)
    y2 = cy + (bh / 2.0) * (1.0 + margin_ratio)

    # clamp to valid range
    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(1.0, x2)
    y2 = min(1.0, y2)

    rx = refs_xy[:, 0]
    ry = refs_xy[:, 1]

    inside = (rx >= x1) & (rx <= x2) & (ry >= y1) & (ry <= y2)

    # normalized distance to GT center
    d = torch.sqrt((rx - cx) ** 2 + (ry - cy) ** 2)
    gt_diag = math.sqrt(bw ** 2 + bh ** 2)
    near = d <= (dist_ratio * gt_diag)

    mask = inside | near
    query_ids = torch.where(mask)[0]

    # optionally keep only closest queries
    if topk is not None and query_ids.numel() > topk:
        qd = d[query_ids]
        order = torch.argsort(qd)
        query_ids = query_ids[order[:topk]]

    return query_ids.tolist(), d

def plot_score_curve(cls_scores, q_idx, save_dir, class_names=None):
    """
    cls_scores: [L, Q, C]
    """

    L = cls_scores.shape[0]

    scores_per_layer = cls_scores[:, q_idx]  # [L, C]

    # best class + score per layer
    best_scores, best_classes = scores_per_layer.max(dim=1)

    score_curve = best_scores.cpu().numpy()
    class_curve = best_classes.cpu().numpy()

    # 🔥 overall maximum confidence
    max_conf = best_scores.max().item()

    # 🔥 layer where max occurs
    max_layer = best_scores.argmax().item()

    # 🔵 Plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(L), score_curve, marker='o')

    # annotate class name
    for l in range(L):

        cls_id = class_curve[l]

        if class_names is not None:
            cls_name = class_names[cls_id]
        else:
            cls_name = str(cls_id)

        plt.text(l, score_curve[l], cls_name, fontsize=8)

    plt.xlabel("Decoder Layer")
    plt.ylabel("Confidence")

    # 🔥 title includes max score
    plt.title(
        f"Query {q_idx} | Max Conf: {max_conf:.3f} @ Layer {max_layer}"
    )

    plt.grid()

    # 🔥 filename also contains max confidence
    save_name = f"score_curve_max_{max_conf:.3f}.png"

    plt.savefig(os.path.join(save_dir, save_name))
    plt.close()

# --------------------------------------------------
# 🔴 Overlay sampling on image
# --------------------------------------------------
def overlay_sampling(data, image_path, q_idx, save_dir):
    img = cv2.imread(image_path)

    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return

    H_img, W_img = img.shape[:2]

    sampling = data["sampling_locations"]   # [L, Q, H, Lv, P, 2]
    attn = data["attention_weights"]        # [L, Q, H, Lv, P]
    refs = data["reference_points"]

    # print(sampling[0][8][0][0])
    # 🎨 Colors for 4 feature levels (BGR)
    level_colors = [
    (0, 0, 255),    # Level 0 → Red (finest)
    (0, 255, 255),  # Level 1 → Yellow
    (0, 255, 0),    # Level 2 → Green
    (255, 255, 0),  # Level 3 → Cyan
    (255, 0, 0),    # Level 4 → Blue (coarsest)
    ]

    L = sampling.shape[0]
    H = sampling.shape[2]
    Lv = sampling.shape[3]

    # 👉 Only visualize FIRST layer (coarse)

    for l in range(L):
        canvas = img.copy()

        for lv in range(Lv):
            color = level_colors[lv]

            for h in range(H):

                pts = sampling[l, q_idx, h, lv].cpu().numpy()   # [P, 2]
                weights = attn[l, q_idx, h, lv].cpu().numpy()   # [P]

                # normalize weights for visualization
                w_max = weights.max() + 1e-6

                for p in range(len(pts)):

                    # floating-point image coordinates
                    x = pts[p, 0] * W_img
                    y = pts[p, 1] * H_img

                    # clamp
                    x = max(0, min(W_img - 1, x))
                    y = max(0, min(H_img - 1, y))

                    # attention strength
                    alpha = weights[p] / w_max

                    # radius reflects attention
                    r = int(2 + 10 * alpha)

                    # draw point
                    cv2.circle(
                        canvas,
                        (int(x), int(y)),
                        r,
                        color,
                        -1
                    )

                    # optional: connect to reference point
                    ref = refs[l, q_idx][:2].cpu().numpy()

                    x_ref = int(ref[0] * W_img)
                    y_ref = int(ref[1] * H_img)

                    cv2.line(
                        canvas,
                        (x_ref, y_ref),
                        (int(x), int(y)),
                        color,
                        1
                    )

        # 🔵 reference point
        ref = refs[l, q_idx][:2].cpu().numpy()
        x_ref = int(ref[0] * W_img)
        y_ref = int(ref[1] * H_img)

        x_ref = max(0, min(W_img - 1, x_ref))
        y_ref = max(0, min(H_img - 1, y_ref))

        cv2.circle(canvas, (x_ref, y_ref), 7, (255, 255, 255), -1)

        out_path = os.path.join(save_dir, f"layer_{l}_coarse_to_fine.png")
        cv2.imwrite(out_path, canvas)

# --------------------------------------------------
# 🔥 Main processing
# --------------------------------------------------
def process_file(data, image_path, save_dir, idx):

    cls_scores = data["cls_scores"]  # [L, Q, C]

    # --------------------------------------------------
    # Fixed target-related query ids
    # obtained from last-layer GT matching
    # --------------------------------------------------

    TARGET_QUERY_IDS = [
        10, 11, 12, 13, 15, 22, 28, 29, 32, 35,
        41, 49, 55, 76, 82, 100, 117, 127, 129,
        299, 355, 376, 405, 428, 436, 469, 486,
        508, 624, 636, 656, 684, 709, 711, 722,
        791, 805, 806, 822, 829, 837, 852, 887,
        891
    ]

    # impainted
    # TARGET_QUERY_IDS = [
    #     8, 13, 25, 32, 36, 42, 70, 77, 87, 120, 128, 156, 
    #     158, 225, 241, 378, 432, 444, 474, 476, 482, 498, 
    #     541, 565, 572, 607, 626, 630, 637, 680, 683, 695, 
    #     709, 711, 771, 799, 816, 829, 835, 836, 837, 851, 883
    # ]
    print(
        f"[INFO] Visualizing {len(TARGET_QUERY_IDS)} "
        f"fixed target queries"
    )

    # --------------------------------------------------
    # Plot + sampling visualization
    # --------------------------------------------------

    for q_idx in TARGET_QUERY_IDS:

        # skip invalid ids
        if q_idx >= cls_scores.shape[1]:
            print(f"[WARN] Invalid query id {q_idx}")
            continue

        q_dir = os.path.join(
            save_dir,
            f"query_{q_idx}"
        )

        os.makedirs(q_dir, exist_ok=True)

        # ----------------------------------------------
        # score curve across decoder layers
        # ----------------------------------------------

        plot_score_curve(
            cls_scores,
            q_idx,
            q_dir,
            class_names=RSUD_CLASSES
        )

        # ----------------------------------------------
        # sampling shift visualization
        # ----------------------------------------------

        overlay_sampling(
            data,
            image_path,
            q_idx,
            q_dir
        )

# --------------------------------------------------
# 🧪 Main
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pt_dir", type=str, required=True)
    parser.add_argument("--image_name", type=str, required=True)
    parser.add_argument("--output", type=str, default="rough")

    args = parser.parse_args()

    pt_files = sorted(Path(args.pt_dir).glob("*.pt"))[:50]

    os.makedirs(args.output, exist_ok=True)

    found = False

    for idx, pt_file in enumerate(pt_files):

        data = torch.load(pt_file)

        image_path = data["img_path"]

        if not os.path.exists(image_path):
            print(f"[WARN] Missing image: {image_path}")
            continue

        img_filename = Path(image_path).name   # 🔥 full filename

        # process only requested image
        if img_filename != args.image_name:
            continue

        found = True

        save_dir = os.path.join(
            args.output,
            Path(image_path).stem
        )
        os.makedirs(save_dir, exist_ok=True)

        print(f"[INFO] Processing ONLY: {img_filename}")

        process_file(data, image_path, save_dir, idx)

        break

    if not found:
        print(f"[ERROR] Image {args.image_name} not found.")


if __name__ == "__main__":
    main()