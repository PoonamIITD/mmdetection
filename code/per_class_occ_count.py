from collections import defaultdict
import json

gt = json.load(open("instances_validation_merged_mannual.json"))

# build category id → name map
cat_names = {c["id"]: c["name"] for c in gt["categories"]}

# count per class per occlusion level
counts = defaultdict(lambda: defaultdict(int))
for ann in gt["annotations"]:
    level = ann.get("attributes", {}).get("occlusion_level", "unknown")
    cat   = cat_names.get(ann["category_id"], str(ann["category_id"]))
    counts[cat][level] += 1

# print table
print(f"{'Category':<20} {'Light':>8} {'Moderate':>10} {'Severe':>8} {'Total':>8}")
print("-" * 56)
for cat in sorted(counts):
    l = counts[cat].get("light", 0)
    m = counts[cat].get("moderate", 0)
    s = counts[cat].get("severe", 0)
    print(f"{cat:<20} {l:>8} {m:>10} {s:>8} {l+m+s:>8}")