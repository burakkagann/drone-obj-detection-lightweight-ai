#!/usr/bin/env python3
"""
Convert DOTA dataset annotations to YOLO format
This script converts DOTA format annotations to YOLO format for training YOLOv5
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

class DOTAtoYOLOConverter:
    def __init__(self, src_path, dst_path):
        self.src_path = Path(src_path)
        self.dst_path = Path(dst_path)
        # DOTA uses numeric class IDs (0-14)
        self.num_classes = 15  # DOTA has 15 classes
        
    def create_folders(self):
        """Create the necessary folder structure"""
        print("Creating folder structure...")
        for split in ['train', 'val']:
            (self.dst_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.dst_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    def convert_coordinates(self, points, img_w, img_h):
        """Convert DOTA polygon coordinates to YOLO format (x_center, y_center, width, height)"""
        points = np.array(points, dtype=np.float32).reshape(4, 2)
        
        # Calculate bounding box
        x_min = np.min(points[:, 0])
        y_min = np.min(points[:, 1])
        x_max = np.max(points[:, 0])
        y_max = np.max(points[:, 1])
        
        # Convert to YOLO format
        x_center = (x_min + x_max) / (2 * img_w)
        y_center = (y_min + y_max) / (2 * img_h)
        width = (x_max - x_min) / img_w
        height = (y_max - y_min) / img_h
        
        return [x_center, y_center, width, height]
    
    def convert_annotations(self, split='train'):
        """Convert DOTA format annotations to YOLO format"""
        print(f"Converting {split} annotations...")
        src_label_dir = self.src_path / 'labels' / f'{split}_original'
        src_img_dir = self.src_path / 'images' / split
        dst_label_dir = self.dst_path / 'labels' / split
        dst_img_dir = self.dst_path / 'images' / split
        
        # Get list of image files
        img_files = list(src_img_dir.glob('*.png'))
        print(f"Found {len(img_files)} images in {split} set")
        
        for img_file in tqdm(img_files):
            # Get image dimensions
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"Warning: Could not read image {img_file}")
                continue
            img_h, img_w = img.shape[:2]
            
            # Convert label
            label_file = src_label_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                print(f"Warning: Label file {label_file} not found")
                continue
            
            yolo_labels = []
            with open(label_file, 'r') as f:
                for line in f:
                    try:
                        parts = line.strip().split()
                        if len(parts) < 10:  # DOTA format has 10 values per line
                            continue
                        
                        # Extract coordinates and class ID
                        coords = [float(x) for x in parts[:8]]
                        class_id = int(parts[8])  # Class ID is the 9th value
                        
                        # Skip if class_id is invalid
                        if class_id >= self.num_classes:
                            print(f"Warning: Invalid class ID {class_id} in {label_file}")
                            continue
                        
                        # Convert coordinates
                        yolo_coords = self.convert_coordinates(coords, img_w, img_h)
                        
                        # Create YOLO format line: class_id x_center y_center width height
                        yolo_line = f"{class_id} {' '.join([f'{x:.6f}' for x in yolo_coords])}"
                        yolo_labels.append(yolo_line)
                    except Exception as e:
                        print(f"Error processing line in {label_file}: {e}")
                        continue
            
            # Write YOLO format labels
            dst_label_file = dst_label_dir / f"{img_file.stem}.txt"
            with open(dst_label_file, 'w') as f:
                f.write('\n'.join(yolo_labels))

    def convert(self):
        """Convert both train and validation sets"""
        print("Starting DOTA to YOLO conversion...")
        self.create_folders()
        for split in ['train', 'val']:
            self.convert_annotations(split)
        print("Conversion completed!")

if __name__ == "__main__":
    src_path = "data/my_dataset/dota/dota-v1.0"
    dst_path = "data/my_dataset/dota/dota-v1.0/yolov5n"
    
    converter = DOTAtoYOLOConverter(src_path, dst_path)
    converter.convert() 