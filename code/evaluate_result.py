from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 1. Load your ground truth (The standard annotation file)
gt_json = "/home/poonam_rajput/scratch/dataset/RSUD_dataset/annotations/merged_instances_val2017_codetr_final.json"
coco_gt = COCO(gt_json)

# 2. Load the JSON you generated from your pickle
dt_json = "results_sampling_loss_final.json"
coco_dt = coco_gt.loadRes(dt_json)

# 3. Initialize the evaluator
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

# 4. Set the thresholds you want to investigate
# We include 0.1 and 0.2 to check your localization hypothesis
# coco_eval.params.iouThrs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
# coco_eval.params.max

# 5. Execute evaluation
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

# 6. Extract results for specific thresholds if needed
# stats[0] is AP at IoU=0.50:0.95
# stats[1] is AP at IoU=0.50
# To access specific ones, you can inspect coco_eval.stats
print(f"AP at IoU=0.10: {coco_eval.stats[0]}") # Example