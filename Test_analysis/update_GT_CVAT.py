import json

def map_image_ids(original_gt_path, new_annotations_path, output_path):
    """
    Map image IDs from original GT to new CVAT annotations.
    
    Args:
        original_gt_path: Path to original ground truth JSON file
        new_annotations_path: Path to new CVAT annotations JSON file
        output_path: Path to save updated annotations
    """
    
    # Load both JSON files
    with open(original_gt_path, 'r') as f:
        original_gt = json.load(f)
    
    with open(new_annotations_path, 'r') as f:
        new_annotations = json.load(f)
    
    # Create mapping: filename -> original image_id
    filename_to_original_id = {}
    for img in original_gt['images']:
        filename_to_original_id[img['file_name']] = img['id']
    
    # Create mapping: new image_id -> original image_id
    new_id_to_original_id = {}
    for img in new_annotations['images']:
        filename = img['file_name']
        if filename in filename_to_original_id:
            new_id_to_original_id[img['id']] = filename_to_original_id[filename]
            # Update the image ID in the new annotations
            img['id'] = filename_to_original_id[filename]
        else:
            print(f"Warning: {filename} not found in original GT")
    
    # Update image_id in all annotations
    for ann in new_annotations['annotations']:
        old_image_id = ann['image_id']
        if old_image_id in new_id_to_original_id:
            ann['image_id'] = new_id_to_original_id[old_image_id]
        else:
            print(f"Warning: annotation with image_id {old_image_id} has no mapping")
    
    # Save the updated annotations
    with open(output_path, 'w') as f:
        json.dump(new_annotations, f, indent=2)
    
    print(f"Successfully mapped {len(new_id_to_original_id)} images")
    print(f"Updated annotations saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    original_gt_path = "instances_val2017.json"
    new_annotations_path = "instances_default_codetr.json"
    output_path = "instances_default_codetr_updated.json"
    
    map_image_ids(original_gt_path, new_annotations_path, output_path)