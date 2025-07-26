#!/usr/bin/env python3
"""
VisDrone to MobileNet-SSD Format Converter
Creates proper directory structure and symlinks for MobileNet-SSD training

Author: Burak Kağan Yılmazer
Project: Drone Object Detection - MobileNet-SSD
"""

import os
import shutil
from pathlib import Path

def create_mobilenet_ssd_structure():
    """
    Creates MobileNet-SSD compatible directory structure from existing VisDrone dataset
    Uses symbolic links to avoid duplicating large amounts of data
    """
    
    # Define paths
    base_path = Path("C:/Users/burak/OneDrive/Desktop/Git Repos/drone-obj-detection-lightweight-ai")
    visdrone_path = base_path / "data" / "my_dataset" / "visdrone"
    mobilenet_ssd_path = visdrone_path / "mobilenet-ssd"
    
    print(f"[INFO] Creating MobileNet-SSD structure at: {mobilenet_ssd_path}")
    
    # Create base directory
    mobilenet_ssd_path.mkdir(exist_ok=True)
    
    # Create subdirectories
    for subset in ['train', 'val', 'test']:
        # Create image and label directories
        (mobilenet_ssd_path / "images" / subset).mkdir(parents=True, exist_ok=True)
        (mobilenet_ssd_path / "labels" / subset).mkdir(parents=True, exist_ok=True)
        
        print(f"[INFO] Processing {subset} dataset...")
        
        # Source paths
        src_images = visdrone_path / subset / "images"
        src_labels = visdrone_path / subset / "labels"
        
        # Destination paths
        dst_images = mobilenet_ssd_path / "images" / subset
        dst_labels = mobilenet_ssd_path / "labels" / subset
        
        # Check if source directories exist
        if not src_images.exists():
            print(f"[WARNING] Source images directory not found: {src_images}")
            continue
        if not src_labels.exists():
            print(f"[WARNING] Source labels directory not found: {src_labels}")
            continue
        
        # Create symbolic links for images
        print(f"[INFO] Creating symlinks for {subset} images...")
        for img_file in src_images.glob("*.jpg"):
            dst_file = dst_images / img_file.name
            if not dst_file.exists():
                try:
                    # Use relative path for better portability
                    relative_src = os.path.relpath(img_file, dst_file.parent)
                    dst_file.symlink_to(relative_src)
                except OSError:
                    # If symlink fails, copy the file
                    shutil.copy2(img_file, dst_file)
        
        # Create symbolic links for labels
        print(f"[INFO] Creating symlinks for {subset} labels...")
        for label_file in src_labels.glob("*.txt"):
            dst_file = dst_labels / label_file.name
            if not dst_file.exists():
                try:
                    # Use relative path for better portability
                    relative_src = os.path.relpath(label_file, dst_file.parent)
                    dst_file.symlink_to(relative_src)
                except OSError:
                    # If symlink fails, copy the file
                    shutil.copy2(label_file, dst_file)
        
        # Count files
        img_count = len(list(dst_images.glob("*.jpg")))
        label_count = len(list(dst_labels.glob("*.txt")))
        print(f"[SUCCESS] {subset}: {img_count} images, {label_count} labels")
    
    # Create classes.txt file with VisDrone class names
    create_classes_file(mobilenet_ssd_path)
    
    print(f"[SUCCESS] MobileNet-SSD dataset structure created successfully!")
    print(f"[INFO] Dataset location: {mobilenet_ssd_path}")
    
    return mobilenet_ssd_path

def create_classes_file(mobilenet_ssd_path):
    """
    Creates classes.txt file with VisDrone class names
    """
    
    # VisDrone class mapping (10 classes)
    visdrone_classes = [
        "pedestrian",      # 0
        "people",          # 1  
        "bicycle",         # 2
        "car",             # 3
        "van",             # 4
        "truck",           # 5
        "tricycle",        # 6
        "awning-tricycle", # 7
        "bus",             # 8
        "motor"            # 9
    ]
    
    classes_file = mobilenet_ssd_path / "classes.txt"
    
    print(f"[INFO] Creating classes.txt with {len(visdrone_classes)} classes...")
    
    with open(classes_file, 'w') as f:
        for class_name in visdrone_classes:
            f.write(f"{class_name}\n")
    
    print(f"[SUCCESS] Classes file created: {classes_file}")
    
    # Also create a class mapping file for reference
    mapping_file = mobilenet_ssd_path / "class_mapping.txt"
    with open(mapping_file, 'w') as f:
        f.write("VisDrone Class Mapping:\n")
        f.write("=" * 25 + "\n")
        for i, class_name in enumerate(visdrone_classes):
            f.write(f"{i}: {class_name}\n")
    
    print(f"[INFO] Class mapping file created: {mapping_file}")

def validate_dataset_structure(mobilenet_ssd_path):
    """
    Validates the created dataset structure
    """
    
    print("\n[VALIDATION] Checking dataset structure...")
    
    # Check directory structure
    required_dirs = [
        "images/train", "images/val", "images/test",
        "labels/train", "labels/val", "labels/test"
    ]
    
    for dir_path in required_dirs:
        full_path = mobilenet_ssd_path / dir_path
        if full_path.exists():
            file_count = len(list(full_path.glob("*")))
            print(f"[OK] {dir_path}: {file_count} files")
        else:
            print(f"[ERROR] Missing directory: {dir_path}")
    
    # Check classes.txt
    classes_file = mobilenet_ssd_path / "classes.txt"
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            class_count = len(f.readlines())
        print(f"[OK] classes.txt: {class_count} classes")
    else:
        print(f"[ERROR] Missing classes.txt file")
    
    print("[VALIDATION] Dataset structure validation complete!")

if __name__ == "__main__":
    print("=" * 60)
    print("VisDrone to MobileNet-SSD Format Converter")
    print("=" * 60)
    
    try:
        # Create the structure
        dataset_path = create_mobilenet_ssd_structure()
        
        # Validate the structure
        validate_dataset_structure(dataset_path)
        
        print("\n" + "=" * 60)
        print("[SUCCESS] Conversion completed successfully!")
        print(f"[INFO] MobileNet-SSD dataset ready at: {dataset_path}")
        print("=" * 60)
        
    except Exception as e:
        print(f"[ERROR] Conversion failed: {e}")
        import traceback
        traceback.print_exc()