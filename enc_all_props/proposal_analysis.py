import h5py
import json
import numpy as np
import torch
from torchvision.ops import box_iou
from pycocotools.coco import COCO
import os

# -----------------------------
# Paths
# -----------------------------
H5_PATH = "hdino/enc_all_props_hdino.h5"
COCO_ANN = "/home/poonam_rajput/scratch/dataset/RSUD_dataset/annotations/merged_instances_val2017_codetr_final.json"
IMG_ROOT = "/home/poonam_rajput/scratch/dataset/RSUD_dataset/images/val"

# -----------------------------
# Utils
# -----------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.T
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)

# -----------------------------
# Load COCO GT
# -----------------------------
coco = COCO(COCO_ANN)

# Map filename → image_id
img_ids = coco.getImgIds()
fname_to_id = {
    coco.loadImgs(i)[0]["file_name"]: i for i in img_ids
}

# -----------------------------
# Load H5 proposals
# -----------------------------
f = h5py.File(H5_PATH, "r")
image_groups = f["images"]

all_gt_ious = []

# -----------------------------
# Main loop
# -----------------------------
for image_name in image_groups.keys():

    if image_name not in fname_to_id:
        continue

    image_id = fname_to_id[image_name]
    img_info = coco.loadImgs(image_id)[0]
    H, W = img_info["height"], img_info["width"]

    # ----- Load proposals -----
    props_unact = image_groups[image_name]["enc_outputs_coord_unact"][:]
    props = sigmoid(props_unact)
    props = cxcywh_to_xyxy(props)

    # scale to pixels
    props[:, [0, 2]] *= W
    props[:, [1, 3]] *= H

    props = torch.tensor(props, dtype=torch.float32)

    # ----- Load GT boxes -----
    ann_ids = coco.getAnnIds(imgIds=image_id, iscrowd=False)
    anns = coco.loadAnns(ann_ids)

    if len(anns) == 0:
        continue

    gt_boxes = []
    for a in anns:
        x, y, w, h = a["bbox"]
        gt_boxes.append([x, y, x + w, y + h])

    gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)

    # ----- IoU computation -----
    ious = box_iou(gt_boxes, props)   # [num_gt, num_props]
    max_iou_per_gt = ious.max(dim=1)[0]

    all_gt_ious.append(max_iou_per_gt)

# -----------------------------
# Final result
# -----------------------------
all_gt_ious = torch.cat(all_gt_ious)
print(f"Average GT IoU: {all_gt_ious.mean().item():.4f}")
print(f"Recall@0.5: {(all_gt_ious > 0.5).float().mean().item():.4f}")
print(f"Recall@0.65: {(all_gt_ious > 0.65).float().mean().item():.4f}")
print(f"Recall@0.75: {(all_gt_ious > 0.75).float().mean().item():.4f}")
print(f"Recall@0.85: {(all_gt_ious > 0.85).float().mean().item():.4f}")
print(f"Recall@0.95: {(all_gt_ious > 0.95).float().mean().item():.4f}")
