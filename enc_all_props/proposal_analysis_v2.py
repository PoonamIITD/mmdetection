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
H5_PATH = "codino/enc_all_props_codino.h5"
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
all_mean_gt_ious = []         # mean IoU per GT over proposals
all_top5_gt_ious = []         # mean of top-5 IoUs per GT
all_prop_counts = []          # number of proposals per image
all_props_above_thresh = {    # proposal density
    0.5: [],
    0.75: [],
    0.85: []
}

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
    # ----- IoU computation -----
    ious = box_iou(gt_boxes, props)   # [num_gt, num_props]

    # Max IoU per GT (existing metric)
    max_iou_per_gt = ious.max(dim=1)[0]
    all_gt_ious.append(max_iou_per_gt)

    # Mean IoU per GT (proposal quality spread)
    mean_iou_per_gt = ious.mean(dim=1)
    all_mean_gt_ious.append(mean_iou_per_gt)

    # Top-5 IoU per GT (are there multiple good proposals?)
    topk = min(5, ious.shape[1])
    top5_iou_per_gt = torch.topk(ious, k=topk, dim=1).values.mean(dim=1)
    all_top5_gt_ious.append(top5_iou_per_gt)

    # Proposal count per image
    all_prop_counts.append(props.shape[0])

    # Proposal density above thresholds (per GT, normalized)
    for t in all_props_above_thresh:
        frac = (ious > t).float().mean(dim=1)
        all_props_above_thresh[t].append(frac)


# -----------------------------
# Aggregate statistics
# -----------------------------
all_gt_ious = torch.cat(all_gt_ious)
all_mean_gt_ious = torch.cat(all_mean_gt_ious)
all_top5_gt_ious = torch.cat(all_top5_gt_ious)

print("==== MaxIoU-based metrics ====")
print(f"Average GT IoU: {all_gt_ious.mean().item():.4f}")
for t in [0.5, 0.65, 0.75, 0.85, 0.95]:
    print(f"Recall@{t}: {(all_gt_ious > t).float().mean().item():.4f}")

print("\n==== IoU distribution diagnostics ====")
print(f"Mean IoU per GT: {all_mean_gt_ious.mean().item():.4f}")
print(f"Median IoU per GT: {all_mean_gt_ious.median().item():.4f}")
print(f"Top-5 mean IoU per GT: {all_top5_gt_ious.mean().item():.4f}")

print("\n==== Proposal density ====")
for t in all_props_above_thresh:
    vals = torch.cat(all_props_above_thresh[t])
    print(f"Fraction of proposals with IoU > {t}: {vals.mean().item():.4f}")

print("\n==== Proposal count ====")
print(f"Avg proposals per image: {np.mean(all_prop_counts):.1f}")
print(f"Median proposals per image: {np.median(all_prop_counts):.1f}")
