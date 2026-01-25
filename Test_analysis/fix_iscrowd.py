import json

# gt_path = "updated_merged_instances_val2017_CODETR_round.json"
gt_path = "merged_instances_val2017_codetr_final.json"

with open(gt_path, "r") as f:
    coco = json.load(f)

fixed_area = 0
fixed_iscrowd = 0

for ann in coco["annotations"]:
    # ---- iscrowd ----
    if "iscrowd" not in ann:
        ann["iscrowd"] = 0
        fixed_iscrowd += 1

    # ---- area ----
    if "area" not in ann or ann["area"] is None:
        x, y, w, h = ann["bbox"]
        ann["area"] = float(w * h)
        fixed_area += 1

with open(gt_path, "w") as f:
    json.dump(coco, f, indent=2)

print(f"✅ fixed iscrowd: {fixed_iscrowd}")
print(f"✅ fixed area: {fixed_area}")