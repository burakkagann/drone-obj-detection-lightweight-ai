"""
convert_visdrone_to_yolo.py
---------------------------
Converts VisDrone annotation format to YOLOv5 format.

Input:
- VisDrone annotations: data/my_dataset/visdrone/labels/train/annotations/train/
- Corresponding images: data/my_dataset/visdrone/images/train/

Output:
- YOLOv5 formatted labels: data/my_dataset/visdrone/labels/train/yolo-labels/
"""

import os
import cv2

# Define paths
input_labels_dir = 'data/my_dataset/visdrone/labels/train/annotations/train'
output_labels_dir = 'data/my_dataset/visdrone/labels/train/yolo-labels'
images_dir = 'data/my_dataset/visdrone/images/train'

# Ensure output directory exists
os.makedirs(output_labels_dir, exist_ok=True)

for file in os.listdir(input_labels_dir):
    if not file.endswith('.txt'):
        continue

    image_name = file.replace('.txt', '.jpg')
    image_path = os.path.join(images_dir, image_name)
    if not os.path.exists(image_path):
        print(f"Image not found for {file}, skipping.")
        continue

    # Read image to get dimensions
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    input_file_path = os.path.join(input_labels_dir, file)
    output_file_path = os.path.join(output_labels_dir, file)

    with open(input_file_path, 'r') as f_in, open(output_file_path, 'w') as f_out:
        for line in f_in:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue

            x_min, y_min, box_w, box_h = map(float, parts[:4])
            class_id = int(parts[5])

            # Filter out invalid/ignored classes
            if class_id in [0, 11]:
                continue

            # Convert to YOLO format
            x_center = (x_min + box_w / 2) / w
            y_center = (y_min + box_h / 2) / h
            box_w /= w
            box_h /= h

            # Write to output file
            f_out.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")

print("✅ Conversion complete. YOLO labels saved to:", output_labels_dir)
