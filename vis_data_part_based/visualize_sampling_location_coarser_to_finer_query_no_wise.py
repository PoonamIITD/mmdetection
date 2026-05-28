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
# 🔵 Pick best query (UNCHANGED, just unused)
# --------------------------------------------------
def select_best_query(cls_scores):
    final_scores = cls_scores[-1]  # [Q, C]
    max_scores, _ = final_scores.max(dim=1)
    q_idx = torch.argmax(max_scores).item()
    return q_idx


# --------------------------------------------------
# 🔵 Low confidence query (UNCHANGED, unused)
# --------------------------------------------------
def select_low_conf_query(cls_scores, layer=-1, target_score=0.1):
    scores_l = cls_scores[layer]
    max_scores, _ = scores_l.max(dim=1)
    diff = torch.abs(max_scores - target_score)
    q_idx = torch.argmin(diff).item()
    return q_idx, max_scores[q_idx].item()


# --------------------------------------------------
# 🔵 Plot score evolution (UNCHANGED)
# --------------------------------------------------
def plot_score_curve(cls_scores, q_idx, save_dir, class_names=None):
    L = cls_scores.shape[0]

    scores_per_layer = cls_scores[:, q_idx]  # [L, C]

    best_scores, best_classes = scores_per_layer.max(dim=1)

    score_curve = best_scores.cpu().numpy()
    class_curve = best_classes.cpu().numpy()

    plt.figure(figsize=(8, 5))
    plt.plot(range(L), score_curve, marker='o')

    for l in range(L):
        cls_id = class_curve[l]
        cls_name = class_names[cls_id] if class_names else str(cls_id)
        plt.text(l, score_curve[l], cls_name, fontsize=8)

    plt.xlabel("Decoder Layer")
    plt.ylabel("Confidence")
    plt.title(f"Query {q_idx} evolution")
    plt.grid()

    plt.savefig(os.path.join(save_dir, "score_curve.png"))
    plt.close()


# --------------------------------------------------
# 🔴 Overlay sampling (UNCHANGED)
# --------------------------------------------------
def overlay_sampling(data, image_path, q_idx, save_dir):
    img = cv2.imread(image_path)

    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return

    H_img, W_img = img.shape[:2]

    sampling = data["sampling_locations"]
    attn = data["attention_weights"]
    refs = data["reference_points"]

    level_colors = [
        (0, 0, 255),
        (0, 255, 255),
        (0, 255, 0),
        (255, 255, 0),
        (255, 0, 0),
    ]

    L = sampling.shape[0]
    H = sampling.shape[2]
    Lv = sampling.shape[3]

    for l in range(L):
        canvas = img.copy()

        for lv in range(Lv):
            color = level_colors[lv]

            for h in range(H):
                pts = sampling[l, q_idx, h, lv].cpu().numpy()
                weights = attn[l, q_idx, h, lv].cpu().numpy()

                p_idx = np.argmax(weights)

                x = int(pts[p_idx, 0] * W_img)
                y = int(pts[p_idx, 1] * H_img)

                x = max(0, min(W_img - 1, x))
                y = max(0, min(H_img - 1, y))

                r = int(4 + 12 * weights[p_idx])

                cv2.circle(canvas, (x, y), r, color, -1)

        ref = refs[l, q_idx][:2].cpu().numpy()
        x_ref = int(ref[0] * W_img)
        y_ref = int(ref[1] * H_img)

        x_ref = max(0, min(W_img - 1, x_ref))
        y_ref = max(0, min(H_img - 1, y_ref))

        cv2.circle(canvas, (x_ref, y_ref), 7, (255, 255, 255), -1)

        out_path = os.path.join(save_dir, f"layer_{l}_coarse_to_fine.png")
        cv2.imwrite(out_path, canvas)


# --------------------------------------------------
# 🔥 Main processing (ONLY CHANGE HERE)
# --------------------------------------------------
def process_file(data, image_path, save_dir, idx, q_idx):

    cls_scores = data["cls_scores"]

    # 🔥 Use user-provided query index
    print(f"[INFO] Processing {image_path} → Using Query {q_idx}")

    # 🔵 Score curve
    plot_score_curve(cls_scores, q_idx, save_dir, class_names=RSUD_CLASSES)

    # 🔴 Sampling overlays
    overlay_sampling(data, image_path, q_idx, save_dir)


# --------------------------------------------------
# 🧪 Main
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pt_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="vis_outputs_across_decoder_layers_query_3")

    # 🔥 NEW ARGUMENT (only addition)
    parser.add_argument("--query_idx", type=int, default=3,
                        help="Query index to visualize (e.g., 3 for q3)")

    args = parser.parse_args()

    pt_files = sorted(Path(args.pt_dir).glob("*.pt"))[:50]

    os.makedirs(args.output, exist_ok=True)

    for idx, pt_file in enumerate(pt_files):

        data = torch.load(pt_file)
        image_path = data["img_path"]

        if not os.path.exists(image_path):
            print(f"[WARN] Missing image: {image_path}")
            continue

        img_name = Path(image_path).stem
        save_dir = os.path.join(args.output, img_name)
        os.makedirs(save_dir, exist_ok=True)

        print(f"[INFO] Processing {pt_file} → {image_path}")

        # 🔥 pass query index
        process_file(data, image_path, save_dir, idx, args.query_idx)


if __name__ == "__main__":
    main()