import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def compute_iou(box1, box2):
    """Compute IoU between two boxes in [x, y, w, h] format"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to [x1, y1, x2, y2]
    box1_coords = [x1, y1, x1 + w1, y1 + h1]
    box2_coords = [x2, y2, x2 + w2, y2 + h2]
    
    # Intersection coordinates
    xi1 = max(box1_coords[0], box2_coords[0])
    yi1 = max(box1_coords[1], box2_coords[1])
    xi2 = min(box1_coords[2], box2_coords[2])
    yi2 = min(box1_coords[3], box2_coords[3])
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def match_predictions_to_gt(preds, gt_anns, iou_threshold=0.5):
    """Match predictions to ground truth annotations"""
    matched_pairs = []
    
    # Group by image_id
    pred_by_img = defaultdict(list)
    gt_by_img = defaultdict(list)
    
    for pred in preds:
        pred_by_img[pred['image_id']].append(pred)
    
    for ann in gt_anns:
        gt_by_img[ann['image_id']].append(ann)
    
    # Match predictions to ground truth
    for img_id in pred_by_img.keys():
        img_preds = sorted(pred_by_img[img_id], key=lambda x: x['score'], reverse=True)
        img_gts = gt_by_img.get(img_id, [])
        
        matched_gt = set()
        
        for pred in img_preds:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(img_gts):
                if gt_idx in matched_gt:
                    continue
                
                iou = compute_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                matched_pairs.append({
                    'pred_cat': pred['category_id'],
                    'gt_cat': img_gts[best_gt_idx]['category_id'],
                    'iou': best_iou
                })
                matched_gt.add(best_gt_idx)
            else:
                # False positive - background class (0)
                matched_pairs.append({
                    'pred_cat': pred['category_id'],
                    'gt_cat': 0,  # Background
                    'iou': best_iou
                })
        
        # Unmatched ground truth (false negatives)
        for gt_idx, gt in enumerate(img_gts):
            if gt_idx not in matched_gt:
                matched_pairs.append({
                    'pred_cat': 0,  # Background (missed detection)
                    'gt_cat': gt['category_id'],
                    'iou': 0
                })
    
    return matched_pairs

def plot_confusion_matrix(preds_file, gt_file, iou_threshold=0.5, score_threshold=0.5,
                         category_names=None, save_path='confusion_matrix.png'):
    """
    Plot confusion matrix from COCO format predictions and ground truth
    
    Args:
        preds_file: Path to predictions JSON file
        gt_file: Path to ground truth COCO JSON file
        iou_threshold: IoU threshold for matching (default: 0.5)
        category_names: Dict mapping category_id to name (optional)
        save_path: Path to save the plot
    """
    # Load data
    with open(preds_file, 'r') as f:
        preds = json.load(f)
    
    preds = [p for p in preds if p['score'] >= score_threshold] 
    
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
        gt_anns = gt_data['annotations']
        
        # Extract category names if not provided
        if category_names is None and 'categories' in gt_data:
            category_names = {cat['id']: cat['name'] for cat in gt_data['categories']}
    
    # Match predictions to ground truth
    matched_pairs = match_predictions_to_gt(preds, gt_anns, iou_threshold)
    
    # Get all unique categories (including background=0)
    all_cats = set([0])  # Background
    for pair in matched_pairs:
        all_cats.add(pair['pred_cat'])
        all_cats.add(pair['gt_cat'])
    
    all_cats = sorted(list(all_cats))
    n_classes = len(all_cats)
    
    # Create confusion matrix
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    cat_to_idx = {cat: idx for idx, cat in enumerate(all_cats)}
    
    for pair in matched_pairs:
        gt_idx = cat_to_idx[pair['gt_cat']]
        pred_idx = cat_to_idx[pair['pred_cat']]
        conf_matrix[gt_idx, pred_idx] += 1
    
    # Create labels
    if category_names:
        labels = [category_names.get(cat, f'Class {cat}') if cat != 0 else 'Background' 
                  for cat in all_cats]
    else:
        labels = [f'Class {cat}' if cat != 0 else 'Background' for cat in all_cats]
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix (IoU threshold: {iou_threshold} , score_threshold: {score_threshold})', fontsize=16, pad=20)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Ground Truth', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    
    # Print metrics
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Calculate per-class accuracy (excluding background)
    print("\nPer-class metrics:")
    for idx, cat in enumerate(all_cats):
        if cat == 0:
            continue
        tp = conf_matrix[idx, idx]
        total_gt = conf_matrix[idx, :].sum()
        total_pred = conf_matrix[:, idx].sum()
        
        recall = tp / total_gt if total_gt > 0 else 0
        precision = tp / total_pred if total_pred > 0 else 0
        
        label = category_names.get(cat, f'Class {cat}') if category_names else f'Class {cat}'
        print(f"{label}: Precision={precision:.3f}, Recall={recall:.3f}")
    
    plt.show()
    return conf_matrix, labels

# Example usage:
if __name__ == "__main__":
    # Plot confusion matrix
    conf_matrix, labels = plot_confusion_matrix(
        preds_file='codetr_results_val_set_coco_fixed.json',
        # preds_file='codetr_results_val_set_coco_fixed.json',
        gt_file='instances_val2017.json',
        iou_threshold=0.9,
        score_threshold=0.3,
        save_path='codetr_confusion_matrix.png'
    )