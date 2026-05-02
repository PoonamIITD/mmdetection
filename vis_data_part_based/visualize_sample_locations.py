import torch
import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
PT_FOLDER = "./pts/"
IMG_FOLDER = "/home/poonam_rajput/scratch/dataset/RSUD_dataset/images/val"
SAVE_FOLDER = "./vis_output"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# 🔥 Define your 13 class names here
CLASS_NAMES = [
    "class_0", "class_1", "class_2", "class_3",
    "class_4", "class_5", "class_6", "class_7",
    "class_8", "class_9", "class_10", "class_11",
    "class_12"
]

NUM_CLASSES = len(CLASS_NAMES)

# =========================
# STEP 1: LOAD FILES
# =========================
files = sorted(glob.glob(os.path.join(PT_FOLDER, "*.pt")))

# =========================
# STEP 2: FIND BEST 5 SAMPLE PER CLASS
# =========================
TOP_K = 5

best_samples = {c: [] for c in range(NUM_CLASSES)}
# best_scores = {c: -1 for c in range(NUM_CLASSES)}

import torch
import glob

files = sorted(glob.glob(os.path.join(PT_FOLDER, "*.pt")))

for file in files:
    data = torch.load(file)

    det_labels = data["det_labels"]
    det_scores = data["det_scores"]

    for i in range(len(det_labels)):
        cls = int(det_labels[i])
        score = float(det_scores[i])

        if cls >= NUM_CLASSES:
            continue

        best_samples[cls].append((score, file, i))

# 🔥 keep only top 10 per class
for cls in range(NUM_CLASSES):
    best_samples[cls] = sorted(best_samples[cls], key=lambda x: -x[0])[:TOP_K]

print("Best samples selected per class.")

# =========================
# STEP 3: VISUALIZATION
# =========================
def visualize(file, det_idx, class_id, rank, score):

    data = torch.load(file)

    # -------- Image loading --------
    img_path = data.get("img_path", None)

    if img_path is None or not os.path.exists(img_path):
        # fallback using filename
        base_name = os.path.basename(file).replace(".pt", ".jpg")
        img_path = os.path.join(IMG_FOLDER, base_name)

    if not os.path.exists(img_path):
        print(f"Image not found for {file}")
        return

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    H, W = img.shape[:2]

    dpi = 100
    plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)  # ✅ image-native size
    plt.imshow(img)

    # -------- Load sampling --------
    sampling_locations = data["sampling_locations"]   # [nq, heads, levels, points, 2]
    attention_weights = data["attention_weights"]

    # flatten
    points = sampling_locations.reshape(-1, 2)
    weights = attention_weights.reshape(-1)

    # top-k filtering
    topk = min(300, len(weights))
    idx = torch.topk(weights, topk).indices

    points = points[idx]

    # clip to valid range
    points = points.clamp(0, 1)

    x = (points[:, 0] * W).numpy()
    y = (points[:, 1] * H).numpy()  

    # -------- Draw bbox --------
    det_bbox = data["det_bboxes"][det_idx]
    x1, y1, x2, y2 = det_bbox.int().tolist()

    # bbox
    plt.gca().add_patch(
        plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                      edgecolor='green', facecolor='none', linewidth=2)
    )

    # sampling points
    # all points
    plt.scatter(x, y, s=5, c='blue', alpha=0.4)

    # inside GT
    mask = (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)
    plt.scatter(x[mask], y[mask], s=8, c='red')

    class_name = CLASS_NAMES[class_id]
    img_name = os.path.basename(img_path).split('.')[0]

    plt.title(f"{class_name}")
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    save_path = os.path.join(SAVE_FOLDER, f"{img_name}_{class_name}.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Saved: {save_path}")

# =========================
# STEP 4: RUN FOR ALL CLASSES
# =========================
for cls in range(NUM_CLASSES):

    samples = best_samples[cls]

    if len(samples) == 0:
        print(f"No samples for class {cls}")
        continue

    for rank, (score, file, det_idx) in enumerate(samples):
        visualize(file, det_idx, cls, rank, score)

print("All visualizations saved.")