"""Script to convert VisDrone annotations to NanoDet format"""

import os
import shutil
from pathlib import Path
import cv2

def convert_annotations(img_dir, ann_dir, output_dir):
    """Convert VisDrone annotations to NanoDet format"""
    # Create output directories
    output_dir = Path(output_dir)
    output_img_dir = output_dir / "images"
    output_ann_dir = output_dir / "labels"
    
    for split in ["train", "val", "test"]:
        (output_img_dir / split).mkdir(parents=True, exist_ok=True)
        (output_ann_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in ["train", "val", "test"]:
        img_path = Path(img_dir) / split
        ann_path = Path(ann_dir) / split
        
        if not img_path.exists() or not ann_path.exists():
            print(f"Skipping {split} split - directories not found")
            continue
        
        print(f"Processing {split} split...")
        
        # Process each image
        for img_file in img_path.glob("*.jpg"):
            # Copy image
            shutil.copy2(img_file, output_img_dir / split / img_file.name)
            
            # Convert annotation
            ann_file = ann_path / (img_file.stem + ".txt")
            if not ann_file.exists():
                print(f"Warning: No annotation file for {img_file.name}")
                continue
            
            # Read image dimensions
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"Warning: Could not read {img_file.name}")
                continue
            height, width = img.shape[:2]
            
            # Convert annotations
            output_anns = []
            with open(ann_file, "r") as f:
                for line in f:
                    data = line.strip().split(",")
                    if len(data) < 8:
                        continue
                    
                    # VisDrone format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
                    x, y, w, h = map(float, data[:4])
                    category = int(data[5]) - 1  # Convert 1-based to 0-based indexing
                    
                    # Skip ignored regions and invalid categories
                    if category < 0 or category >= 10:  # 10 classes in VisDrone
                        continue
                    
                    # Convert to normalized coordinates
                    x_center = (x + w/2) / width
                    y_center = (y + h/2) / height
                    w = w / width
                    h = h / height
                    
                    # Write in YOLO format: <category> <x_center> <y_center> <width> <height>
                    output_anns.append(f"{category} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            
            # Save converted annotations
            output_ann_file = output_ann_dir / split / (img_file.stem + ".txt")
            with open(output_ann_file, "w") as f:
                f.write("\n".join(output_anns))
        
        print(f"Finished processing {split} split")

def main():
    # Define paths
    data_root = Path("data/my_dataset/visdrone")
    original_img_dir = data_root / "images"
    original_ann_dir = data_root / "labels"
    output_dir = data_root / "nanodet_format"
    
    # Convert annotations
    convert_annotations(original_img_dir, original_ann_dir, output_dir)
    print("Conversion completed!")

if __name__ == "__main__":
    main() 