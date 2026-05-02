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

# --------------------------------------------------
# 🔵 Plot score evolution
# --------------------------------------------------
# def plot_score_curve(cls_scores, q_idx, save_dir):
#     L = cls_scores.shape[0]

#     scores_per_layer = cls_scores[:, q_idx]  # [L, C]
#     best_class = scores_per_layer[-1].argmax().item()
#     score_curve = scores_per_layer[:, best_class].cpu().numpy()

#     plt.figure()
#     plt.plot(range(L), score_curve, marker='o')
#     plt.xlabel("Layer")
#     plt.ylabel("Score")
#     plt.title(f"Query {q_idx} | Class {best_class}")
#     plt.grid()

#     plt.savefig(os.path.join(save_dir, "score_curve.png"))
#     plt.close()

def plot_score_curve(cls_scores, q_idx, save_dir, class_names=None):
    """
    cls_scores: [L, Q, C]
    """
    L = cls_scores.shape[0]

    scores_per_layer = cls_scores[:, q_idx]  # [L, C]

    # 🔥 per-layer best class + score
    best_scores, best_classes = scores_per_layer.max(dim=1)  # [L]

    score_curve = best_scores.cpu().numpy()
    class_curve = best_classes.cpu().numpy()

    # 🔵 Plot score curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(L), score_curve, marker='o')

    # 🔥 annotate class at each layer
    for l in range(L):
        cls_id = class_curve[l]

        if class_names is not None:
            cls_name = class_names[cls_id]
        else:
            cls_name = str(cls_id)

        plt.text(l, score_curve[l], cls_name, fontsize=8)

    plt.xlabel("Decoder Layer")
    plt.ylabel("Confidence")
    plt.title(f"Query {q_idx} evolution")
    plt.grid()

    plt.savefig(os.path.join(save_dir, "score_curve.png"))
    plt.close()

# --------------------------------------------------
# 🔴 Overlay sampling on image
# --------------------------------------------------
def overlay_sampling(data, image_path, q_idx, save_dir):
    img = cv2.imread(image_path)  # BGR

    if img is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return

    H_img, W_img = img.shape[:2]

    sampling = data["sampling_locations"]   # [L, Q, H, Lv, P, 2]
    attn = data["attention_weights"]
    refs = data["reference_points"]

    L = sampling.shape[0]
    H = sampling.shape[2]
    Lv = sampling.shape[3]

    for l in range(L):
        canvas = img.copy()  # stay in BGR for OpenCV

        for h in range(H):
            for lv in range(Lv):
                pts = sampling[l, q_idx, h, lv].cpu().numpy()
                weights = attn[l, q_idx, h, lv].cpu().numpy()

                for p in range(len(pts)):
                    x = int(pts[p, 0] * W_img)
                    y = int(pts[p, 1] * H_img)

                    # clamp to image bounds
                    x = max(0, min(W_img - 1, x))
                    y = max(0, min(H_img - 1, y))

                    # better radius scaling
                    r = int(3 + 10 * weights[p])

                    cv2.circle(canvas, (x, y), r, (0, 0, 255), -1)  # RED (BGR)

        # reference point (GREEN)
        ref = refs[l, q_idx][:2].cpu().numpy()
        x_ref = int(ref[0] * W_img)
        y_ref = int(ref[1] * H_img)

        x_ref = max(0, min(W_img - 1, x_ref))
        y_ref = max(0, min(H_img - 1, y_ref))

        cv2.circle(canvas, (x_ref, y_ref), 6, (0, 255, 0), -1)

        # ✅ SAVE DIRECTLY WITH OPENCV (NO QUALITY LOSS)
        out_path = os.path.join(save_dir, f"layer_{l}.png")
        cv2.imwrite(out_path, canvas)

# --------------------------------------------------
# 🔥 Main processing
# --------------------------------------------------
def process_file(data, image_path, save_dir, idx):
    # data = torch.load(pt_path)

    cls_scores = data["cls_scores"]  # [L, Q, C]

    # 🔵 Select best query
    # q_idx = select_best_query(cls_scores)
    # print(f"[INFO] Processing {image_path} → Query {q_idx}")

    q_idx, score = select_low_conf_query(cls_scores, layer=-1, target_score=0.03)
    print(f"[INFO] Selected query {q_idx} with initial score {score:.3f}")

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
    parser.add_argument("--output", type=str, default="vis_outputs_across_decoder_layers")

    args = parser.parse_args()

    pt_files = sorted(Path(args.pt_dir).glob("*.pt"))[:50]

    os.makedirs(args.output, exist_ok=True)

    for idx, pt_file in enumerate(pt_files):

        # 🔥 Load pt file FIRST
        data = torch.load(pt_file)

        # 🔥 Get image path directly (no guessing)
        image_path = data["img_path"]

        if not os.path.exists(image_path):
            print(f"[WARN] Missing image: {image_path}")
            continue

        # 🔥 Better folder naming (use actual image name)
        img_name = Path(image_path).stem
        save_dir = os.path.join(args.output, img_name)
        os.makedirs(save_dir, exist_ok=True)

        print(f"[INFO] Processing {pt_file} → {image_path}")

        # 🔥 Pass data directly (no re-load inside)
        process_file(data, image_path, save_dir, idx)


if __name__ == "__main__":
    main()