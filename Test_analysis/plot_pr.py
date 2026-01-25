import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def compute_iou(box1, box2):
    """Compute IoU between two boxes in [x, y, w, h] format"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    box1_coords = [x1, y1, x1 + w1, y1 + h1]
    box2_coords = [x2, y2, x2 + w2, y2 + h2]
    
    xi1 = max(box1_coords[0], box2_coords[0])
    yi1 = max(box1_coords[1], box2_coords[1])
    xi2 = min(box1_coords[2], box2_coords[2])
    yi2 = min(box1_coords[3], box2_coords[3])
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def compute_pr_curve(preds, gt_anns, iou_threshold=0.5, category_id=None, score_threshold=0.0):
    """
    Compute precision-recall curve for predictions
    
    Args:
        preds: List of predictions
        gt_anns: List of ground truth annotations
        iou_threshold: IoU threshold for matching
        category_id: Specific category to evaluate (None for all categories)
        score_threshold: Minimum confidence score for predictions
    
    Returns:
        precisions, recalls, thresholds, ap
    """
    # Filter by score threshold
    preds = [p for p in preds if p['score'] >= score_threshold]
    
    # Filter by category if specified
    if category_id is not None:
        preds = [p for p in preds if p['category_id'] == category_id]
        gt_anns = [g for g in gt_anns if g['category_id'] == category_id]
    
    # Sort predictions by confidence score (descending)
    preds = sorted(preds, key=lambda x: x['score'], reverse=True)
    
    # Group ground truth by image
    gt_by_img = defaultdict(list)
    for gt in gt_anns:
        gt_by_img[gt['image_id']].append(gt)
    
    total_gt = len(gt_anns)
    
    # Track which GTs have been matched
    gt_matched = {img_id: [False] * len(gts) for img_id, gts in gt_by_img.items()}
    
    # Calculate TP and FP for each prediction
    tp = np.zeros(len(preds))
    fp = np.zeros(len(preds))
    
    for pred_idx, pred in enumerate(preds):
        img_id = pred['image_id']
        img_gts = gt_by_img.get(img_id, [])
        
        if len(img_gts) == 0:
            fp[pred_idx] = 1
            continue
        
        # Find best matching GT
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(img_gts):
            if category_id is None or gt['category_id'] == pred['category_id']:
                iou = compute_iou(pred['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
        
        # Check if match is valid
        if best_iou >= iou_threshold:
            if not gt_matched[img_id][best_gt_idx]:
                tp[pred_idx] = 1
                gt_matched[img_id][best_gt_idx] = True
            else:
                fp[pred_idx] = 1  # Already matched (duplicate detection)
        else:
            fp[pred_idx] = 1
    
    # Compute cumulative TP and FP
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # Compute precision and recall
    recalls = tp_cumsum / total_gt if total_gt > 0 else np.zeros_like(tp_cumsum)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    precisions = np.nan_to_num(precisions)
    
    # Get confidence thresholds
    thresholds = np.array([p['score'] for p in preds])
    
    # Compute AP using 11-point interpolation (COCO style)
    ap = compute_ap(recalls, precisions)
    
    return precisions, recalls, thresholds, ap

def compute_ap(recalls, precisions):
    """Compute Average Precision using 101-point interpolation (COCO style)"""
    # Add sentinel values at the beginning and end
    mrec = np.concatenate(([0.], recalls, [1.]))
    mpre = np.concatenate(([0.], precisions, [0.]))
    
    # Compute precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # Integrate area under curve using 101-point interpolation
    recall_thresholds = np.linspace(0, 1, 101)
    ap = 0.0
    
    for t in recall_thresholds:
        # Find precisions where recall >= t
        p_vals = mpre[mrec >= t]
        if len(p_vals) > 0:
            ap += p_vals[0]
    
    ap /= 101
    return ap

def plot_pr_curves(preds_files, gt_file, model_names, iou_threshold=0.5, 
                   category_id=None, score_threshold=0.0, save_path='pr_curves.png'):
    """
    Plot PR curves for multiple models
    
    Args:
        preds_files: List of prediction file paths
        gt_file: Path to ground truth COCO JSON
        model_names: List of model names corresponding to pred_files
        iou_threshold: IoU threshold for matching
        category_id: Specific category to plot (None for overall)
        score_threshold: Minimum confidence score for predictions
        save_path: Path to save the plot
    """
    # Load ground truth
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
        gt_anns = gt_data['annotations']
        category_names = {cat['id']: cat['name'] for cat in gt_data['categories']}
        all_categories = [cat['id'] for cat in gt_data['categories']]
    
    # Prepare figure
    if category_id is None:
        # Plot overall + per-class
        n_categories = len(all_categories)
        n_cols = 3
        n_rows = (n_categories + n_cols) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        axes = axes.flatten() if n_categories > 1 else [axes]
    else:
        # Single plot for specified category
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        axes = [ax]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    
    # Plot for each model
    results_summary = []
    
    if category_id is None:
        # Overall PR curve (all categories combined)
        ax = axes[0]
        ax.set_title(f'Overall PR Curve (All Categories)\nScore ≥ {score_threshold}', 
                    fontsize=14, fontweight='bold')
        
        for model_idx, (pred_file, model_name) in enumerate(zip(preds_files, model_names)):
            with open(pred_file, 'r') as f:
                preds = json.load(f)
            
            precisions, recalls, thresholds, ap = compute_pr_curve(
                preds, gt_anns, iou_threshold, category_id=None, score_threshold=score_threshold
            )
            
            ax.plot(recalls, precisions, '-', linewidth=2, 
                   color=colors[model_idx], label=f'{model_name} (AP={ap:.3f})')
            
            results_summary.append({
                'model': model_name,
                'category': 'Overall',
                'ap': ap,
                'precisions': precisions,
                'recalls': recalls,
                'thresholds': thresholds
            })
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        
        # Per-category PR curves
        for cat_idx, cat_id in enumerate(all_categories):
            ax = axes[cat_idx + 1]
            cat_name = category_names.get(cat_id, f'Class {cat_id}')
            ax.set_title(f'{cat_name}', fontsize=12, fontweight='bold')
            
            for model_idx, (pred_file, model_name) in enumerate(zip(preds_files, model_names)):
                with open(pred_file, 'r') as f:
                    preds = json.load(f)
                
                precisions, recalls, thresholds, ap = compute_pr_curve(
                    preds, gt_anns, iou_threshold, category_id=cat_id, score_threshold=score_threshold
                )
                
                ax.plot(recalls, precisions, '-', linewidth=2,
                       color=colors[model_idx], label=f'{model_name} (AP={ap:.3f})')
                
                results_summary.append({
                    'model': model_name,
                    'category': cat_name,
                    'category_id': cat_id,
                    'ap': ap,
                    'precisions': precisions,
                    'recalls': recalls,
                    'thresholds': thresholds
                })
            
            ax.set_xlabel('Recall', fontsize=10)
            ax.set_ylabel('Precision', fontsize=10)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.05])
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8)
        
        # Hide unused subplots
        for idx in range(len(all_categories) + 1, len(axes)):
            axes[idx].axis('off')
    
    else:
        # Single category plot
        ax = axes[0]
        cat_name = category_names.get(category_id, f'Class {category_id}')
        ax.set_title(f'PR Curve - {cat_name}\nScore ≥ {score_threshold}', 
                    fontsize=14, fontweight='bold')
        
        for model_idx, (pred_file, model_name) in enumerate(zip(preds_files, model_names)):
            with open(pred_file, 'r') as f:
                preds = json.load(f)
            
            precisions, recalls, thresholds, ap = compute_pr_curve(
                preds, gt_anns, iou_threshold, category_id=category_id, score_threshold=score_threshold
            )
            
            ax.plot(recalls, precisions, '-', linewidth=2,
                   color=colors[model_idx], label=f'{model_name} (AP={ap:.3f})')
            
            results_summary.append({
                'model': model_name,
                'category': cat_name,
                'category_id': category_id,
                'ap': ap,
                'precisions': precisions,
                'recalls': recalls,
                'thresholds': thresholds
            })
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPR curves saved to {save_path}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("PRECISION-RECALL CURVE SUMMARY")
    print("=" * 80)
    print(f"IoU Threshold: {iou_threshold}")
    print(f"Score Threshold: {score_threshold}")
    print("\n" + "-" * 80)
    print(f"{'Model':<20} {'Category':<25} {'AP':<10}")
    print("-" * 80)
    
    # Group by category for comparison
    by_category = defaultdict(list)
    for result in results_summary:
        by_category[result['category']].append(result)
    
    for category, results in by_category.items():
        for result in results:
            print(f"{result['model']:<20} {category:<25} {result['ap']:.4f}")
        print("-" * 80)
    
    # Overall comparison
    print("\n" + "=" * 80)
    print("MODEL COMPARISON (Average AP across all categories)")
    print("=" * 80)
    
    for model_name in model_names:
        model_results = [r for r in results_summary if r['model'] == model_name and r['category'] != 'Overall']
        if model_results:
            mean_ap = np.mean([r['ap'] for r in model_results])
            print(f"{model_name}: mAP = {mean_ap:.4f}")
    
    plt.show()
    
    return results_summary

def plot_pr_curves_single_figure(preds_files, gt_file, model_names, 
                                  iou_threshold=0.5, score_threshold=0.0, 
                                  save_path='pr_curves_single.png'):
    """
    Plot all PR curves in a single figure for easy comparison
    """
    # Load ground truth
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
        gt_anns = gt_data['annotations']
        category_names = {cat['id']: cat['name'] for cat in gt_data['categories']}
        all_categories = [cat['id'] for cat in gt_data['categories']]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    linestyles = ['-', '--', '-.', ':']
    
    # Plot overall curves
    print("\n" + "=" * 80)
    print("OVERALL PR CURVES COMPARISON")
    print("=" * 80)
    
    for model_idx, (pred_file, model_name) in enumerate(zip(preds_files, model_names)):
        with open(pred_file, 'r') as f:
            preds = json.load(f)
        
        precisions, recalls, thresholds, ap = compute_pr_curve(
            preds, gt_anns, iou_threshold, category_id=None, score_threshold=score_threshold
        )
        
        linestyle = linestyles[model_idx % len(linestyles)]
        ax.plot(recalls, precisions, linestyle, linewidth=3,
               color=colors[model_idx], label=f'{model_name} (AP={ap:.4f})', alpha=0.8)
        
        print(f"{model_name}: AP = {ap:.4f}")
        
        # Find precision at recall thresholds
        recall_points = [0.1, 0.3, 0.5, 0.7, 0.9]
        print(f"  Precision at recall thresholds:")
        for r_threshold in recall_points:
            # Find precision at this recall
            idx = np.searchsorted(recalls, r_threshold)
            if idx < len(precisions):
                p = precisions[idx]
                print(f"    R={r_threshold}: P={p:.4f}")
    
    ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
    ax.set_title(f'Precision-Recall Curves Comparison\n(IoU: {iou_threshold}, Score ≥ {score_threshold})', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    
    # Add diagonal line for reference
    ax.plot([0, 1], [1, 0], 'k--', alpha=0.3, linewidth=1, label='Random')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSingle PR curve plot saved to {save_path}")
    print("=" * 80)
    
    plt.show()

# Example usage
if __name__ == "__main__":
    # Option 1: Plot all categories in separate subplots
    print("Generating comprehensive PR curves (all categories)...")
    results = plot_pr_curves(
        preds_files=['codetr_results_val_set_coco_fixed.json', 'gdino_results_val_set_coco_fixed.json'],
        gt_file='instances_val2017.json',
        model_names=['CO-DETR', 'GDINO'],
        iou_threshold=0.9,
        score_threshold=0.3,  # Filter predictions with score >= 0.3
        save_path='pr_curves_all_categories.png'
    )
    
    # Option 2: Plot overall comparison in single figure
    print("\nGenerating single comparison PR curve...")
    plot_pr_curves_single_figure(
        preds_files=['codetr_results_val_set_coco_fixed.json', 'gdino_results_val_set_coco_fixed.json'],
        gt_file='instances_val2017.json',
        model_names=['CO-DETR', 'GDINO'],
        iou_threshold=0.9,
        score_threshold=0.3,  # Filter predictions with score >= 0.3
        save_path='pr_curves_comparison.png'
    )
    
    # # Option 3: Plot specific category
    # print("\nGenerating PR curve for specific category...")
    # results_person = plot_pr_curves(
    #     preds_files=['codetr_preds.json', 'gdino_preds.json'],
    #     gt_file='gt.json',
    #     model_names=['CO-DETR', 'GDINO'],
    #     iou_threshold=0.5,
    #     score_threshold=0.3,  # Filter predictions with score >= 0.3
    #     category_id=1,  # Change to your category ID
    #     save_path='pr_curve_person.png'
    # )