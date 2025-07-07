import os
import cv2

# Paths
input_folder = '/home/poonam_rajput/scratch/dataset/RSUD_dataset/images/test'
output_root = '/home/poonam_rajput/scratch/dataset/RSUD_dataset/images/resized_test_images'

# List of scales to resize to (width, height)
scales = [
    (1280, 720),
    (960, 540),
    (640, 360)
]

# Create output folders
for width, height in scales:
    out_dir = os.path.join(output_root, f'resized_{width}x{height}')
    os.makedirs(out_dir, exist_ok=True)

# Resize and save images
for img_name in os.listdir(input_folder):
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(input_folder, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Warning: Failed to load {img_name}")
        continue

    for width, height in scales:
        resized_img = cv2.resize(img, (width, height))
        save_path = os.path.join(output_root, f'resized_{width}x{height}', img_name)
        cv2.imwrite(save_path, resized_img)

print("✅ Image resizing completed.")
