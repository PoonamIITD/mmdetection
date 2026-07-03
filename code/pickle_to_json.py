import mmengine
import json
import numpy as np
import torch

def convert_gdino_pkl_to_coco(pkl_path, output_json):
    data = mmengine.load(pkl_path)
    coco_results = []
    
    print(f"Converting {len(data)} entries...")

    for entry in data:
        image_id = entry.get('img_id')
        pred = entry.get('pred_instances', {})
        
        bboxes = pred.get('bboxes')
        scores = pred.get('scores')
        labels = pred.get('labels')

        # Convert to numpy
        if isinstance(bboxes, torch.Tensor): bboxes = bboxes.cpu().numpy()
        if isinstance(scores, torch.Tensor): scores = scores.cpu().numpy()
        if isinstance(labels, torch.Tensor): labels = labels.cpu().numpy()

        for i in range(len(bboxes)):
            # CRITICAL FIX: Add 1 to convert 0-12 indices to 1-13 category IDs
            cat_id = int(labels[i]) + 1
            
            x1, y1, x2, y2 = bboxes[i].tolist()
            
            # Save in COCO format [x_min, y_min, width, height]
            coco_results.append({
                "image_id": int(image_id),
                "category_id": cat_id,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(scores[i])
            })

    with open(output_json, "w") as f:
        json.dump(coco_results, f, indent=4)
    print(f"✅ Saved to {output_json}")

# convert_gdino_pkl_to_coco("results_sampling_loss_val.pkl", "results_sampling_loss_final_val.json")  #val is for the new predictions using new GT with occlusion levels
# convert_gdino_pkl_to_coco("results_baseline_GDINO_val.pkl", "results_baseline_GDINO_final_val.json")
# convert_gdino_pkl_to_coco("results_repulsion_penalty_weights_val.pkl", "results_repulsion_penalty_weights_final_val.json")
# convert_gdino_pkl_to_coco("results_citypersons_val_updated.pkl", "results_citypersons_sampling_loss_val.json")
convert_gdino_pkl_to_coco("results_citypersons_val_baseline.pkl", "results_citypersons_baseline_val.json")