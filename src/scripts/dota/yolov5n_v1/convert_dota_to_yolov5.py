#!/usr/bin/env python3
"""
DOTA to YOLOv5 Conversion Script
This script converts DOTA dataset annotations to YOLOv5 format
and handles image splitting for large aerial images.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import math

class DOTAtoYOLOConverter:
    def __init__(self, src_path, dst_path):
        self.src_path = Path(src_path)
        self.dst_path = Path(dst_path)
        self.classes = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court',
                       'basketball-court', 'ground-track-field', 'harbor', 'bridge',
                       'large-vehicle', 'small-vehicle', 'helicopter', 'roundabout',
                       'soccer-ball-field', 'swimming-pool']
        
    def create_folders(self):
        """Create necessary folders for YOLOv5 format"""
        folders = ['images/train', 'images/val', 'images/test',
                  'labels/train', 'labels/val', 'labels/test']
        for folder in folders:
            (self.dst_path / folder).mkdir(parents=True, exist_ok=True)

    def convert_coordinates(self, points, img_size):
        """Convert DOTA coordinates to YOLO format"""
        points = np.array(points).reshape(4, 2)
        
        # Calculate center, width, height
        center_x = points[:, 0].mean() / img_size[0]
        center_y = points[:, 1].mean() / img_size[1]
        
        # Calculate width and height
        width = np.max(points[:, 0]) - np.min(points[:, 0])
        height = np.max(points[:, 1]) - np.min(points[:, 1])
        
        # Normalize
        width /= img_size[0]
        height /= img_size[1]
        
        # Calculate rotation angle
        angle = math.atan2(points[1][1] - points[0][1],
                          points[1][0] - points[0][0])
        angle = math.degrees(angle)
        if angle < 0:
            angle += 360
        angle = angle / 360.0  # Normalize angle
        
        return [center_x, center_y, width, height, angle]

    def convert_annotation(self, txt_file, img_file, split_type):
        """Convert single annotation file"""
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"Warning: Could not read image {img_file}")
            return
        
        img_size = img.shape[1], img.shape[0]  # width, height
        
        # Read annotations
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        
        yolo_anns = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9:  # DOTA format should have at least 9 parts
                continue
                
            class_name = parts[8]
            if class_name not in self.classes:
                continue
                
            class_id = self.classes.index(class_name)
            points = [float(x) for x in parts[:8]]  # 8 coordinates (x1,y1,x2,y2,x3,y3,x4,y4)
            
            # Convert to YOLO format
            yolo_coords = self.convert_coordinates(points, img_size)
            yolo_anns.append([class_id] + yolo_coords)
        
        # Save converted annotations
        dst_img_path = self.dst_path / f'images/{split_type}' / img_file.name
        dst_label_path = self.dst_path / f'labels/{split_type}' / (img_file.stem + '.txt')
        
        # Copy image
        shutil.copy2(img_file, dst_img_path)
        
        # Save labels
        with open(dst_label_path, 'w') as f:
            for ann in yolo_anns:
                f.write(' '.join(map(str, ann)) + '\n')

    def convert_dataset(self):
        """Convert entire dataset"""
        self.create_folders()
        
        # Process each split
        for split_type in ['train', 'val', 'test']:
            img_path = self.src_path / 'images' / split_type
            label_path = self.src_path / 'labels' / f'{split_type}_original'
            
            if not img_path.exists() or not label_path.exists():
                print(f"Warning: {split_type} path does not exist")
                continue
            
            print(f"Converting {split_type} split...")
            img_files = list(img_path.glob('*.png')) + list(img_path.glob('*.jpg'))
            
            for img_file in tqdm(img_files):
                txt_file = label_path / (img_file.stem + '.txt')
                if txt_file.exists():
                    self.convert_annotation(txt_file, img_file, split_type)

def main():
    src_path = Path("data/my_dataset/dota/dota-v1.0")
    dst_path = Path("data/my_dataset/dota/dota-v1.0/yolov5n")
    
    converter = DOTAtoYOLOConverter(src_path, dst_path)
    converter.convert_dataset()

if __name__ == "__main__":
    main() 