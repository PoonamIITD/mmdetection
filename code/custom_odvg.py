import os
import json
import jsonlines
from collections import defaultdict

# === Paths ===
data_root = '../../dataset/RSUD_dataset/'
coco_json_path = os.path.join(data_root, 'annotations/instances_test2017.json')
out_path = os.path.join(data_root, 'annotations_open_vocab/instances_test2017_odvg.json')
label_map_path = os.path.join(data_root, 'annotations_open_vocab/instances_test2017_label_map.json')

# === Category Descriptions (VG caption per class) ===
category_descriptions = {
    "person": "A person is a living being with a complex physical form, including a head, torso, limbs, and varied appearance based on ethnicity and individual traits",
    "rickshaw": "A rickshaw is a human-powered or motorized vehicle with a simple frame, seating, and often two or three wheels",
    "rickshaw van": "A rickshaw van is a motorized three-wheeled vehicle with an enclosed cabin for passengers or goods, and typically a driver upfront",
    "auto rickshaw": "An auto rickshaw is a compact, three-wheeled motorized vehicle with a cabin for passengers, a driver upfront, and a rear engine",
    "truck": "A truck is a large, motorized vehicle with a driver’s cabin, cargo area, wheels, and often a distinct front grille",
    "pickup truck": "A pickup truck is a smaller motorized vehicle with a driver’s cabin and an open cargo bed in the rear",
    "private car": "A private car is a four-wheeled motor vehicle designed for personal transportation, typically with seating for passengers and an enclosed cabin",
    "motorcycle": "A motorcycle is a two-wheeled motor vehicle with a seat for a rider and often a pillion seat for a passenger",
    "bicycle": "A bicycle is a human-powered vehicle with two wheels, pedals, a frame, handlebars, and a seat for a rider",
    "bus": "A bus is a large motorized vehicle with a passenger cabin, typically featuring multiple seats, windows, and a distinctive elongated shape",
    "micro bus": "A micro bus is a smaller motorized vehicle, similar to a standard bus but more compact with seating for fewer passengers",
    "covered van": "A covered van is a motorized vehicle with a closed cargo area, often used for transporting goods, and may have a driver’s cabin upfront",
    "human hauler": "A human hauler is a motorized vehicle designed for transporting passengers, similar to an auto rickshaw or tuk-tuk, with a cabin and driver upfront",
}

# === Load COCO JSON ===
with open(coco_json_path, 'r') as f:
    coco = json.load(f)

# Map category id <-> name
cat_id_to_name = {cat['id']: cat['name'] for cat in coco['categories']}
cat_name_to_id = {v: k for k, v in cat_id_to_name.items()}

# Create label_map (string keys)
label_map = {str(idx): name for idx, name in cat_id_to_name.items()}

# Build image_id -> file info
img_id_to_info = {
    img['id']: {'filename': img['file_name'], 'width': img['width'], 'height': img['height']}
    for img in coco['images']
}

# Build image_id -> list of annotations
img_to_anns = defaultdict(list)
for ann in coco['annotations']:
    img_to_anns[ann['image_id']].append(ann)

# === Build ODVG JSON ===
metas = []
for img_id, anns in img_to_anns.items():
    img_info = img_id_to_info[img_id]
    instances = []
    phrase_to_boxes = defaultdict(list)
    used_phrases = set()

    for ann in anns:
        x, y, w, h = ann['bbox']
        bbox = [x, y, x + w, y + h]
        cat_id = ann['category_id']
        cat_name = cat_id_to_name[cat_id]

        # Add detection instance
        instances.append({
            "bbox": bbox,
            "label": cat_id,
            "category": cat_name
        })

        # Group bboxes by phrase
        phrase_to_boxes[cat_name].append(bbox)

    # Build caption and regions
    caption = ""
    current_offset = 0
    regions = []

    for phrase, boxes in phrase_to_boxes.items():
        phrase_desc = category_descriptions.get(phrase, f"A {phrase}.").strip()

        # Add to caption
        if caption:
            caption += ". "
            current_offset += 2  # Account for ". "
        start = current_offset
        caption += phrase_desc
        end = start + len(phrase_desc)
        current_offset = end

        # All boxes for this phrase share the same token span
        regions.append({
            "phrase": phrase,
            "bbox": boxes,
            "tokens_positive": [[start, end]]
        })

    metas.append({
        "filename": img_info['filename'],
        "height": img_info['height'],
        "width": img_info['width'],
        "detection": {
            "instances": instances
        },
        "grounding": {
            "caption": caption,
            "regions": regions
        }
    })

# === Save ODVG JSON ===
with jsonlines.open(out_path, mode='w') as writer:
    writer.write_all(metas)

# === Save label_map.json ===
with open(label_map_path, 'w') as f:
    json.dump(label_map, f, indent=4)
