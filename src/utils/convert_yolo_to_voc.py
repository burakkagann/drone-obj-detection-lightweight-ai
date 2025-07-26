#!/usr/bin/env python3
"""
YOLO to VOC Format Converter for MobileNet-SSD Training
Converts VisDrone YOLO annotations to VOC XML format

Author: Burak Kağan Yılmazer
Project: Drone Object Detection - MobileNet-SSD
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
import argparse

# VisDrone class mapping (same as in training script)
VISDRONE_CLASSES = {
    0: "pedestrian",
    1: "people", 
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor"
}

def create_voc_xml(image_path, image_width, image_height, objects, output_path):
    """
    Create VOC XML annotation file
    
    Args:
        image_path: Path to the image file
        image_width: Image width in pixels
        image_height: Image height in pixels  
        objects: List of objects with class_id and bbox coordinates
        output_path: Output XML file path
    """
    
    # Create root element
    annotation = ET.Element('annotation')
    
    # Add folder
    folder = ET.SubElement(annotation, 'folder')
    folder.text = 'VisDrone'
    
    # Add filename
    filename = ET.SubElement(annotation, 'filename')
    filename.text = os.path.basename(image_path)
    
    # Add path
    path = ET.SubElement(annotation, 'path')
    path.text = str(image_path)
    
    # Add source
    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'VisDrone'
    
    # Add size
    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(image_width)
    height = ET.SubElement(size, 'height')
    height.text = str(image_height)
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'
    
    # Add segmented
    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'
    
    # Add objects
    for obj in objects:
        class_id = obj['class_id']
        bbox = obj['bbox']  # [x_center, y_center, width, height] normalized
        
        # Convert YOLO normalized coordinates to VOC pixel coordinates
        x_center = bbox[0] * image_width
        y_center = bbox[1] * image_height
        box_width = bbox[2] * image_width
        box_height = bbox[3] * image_height
        
        xmin = int(x_center - box_width / 2)
        ymin = int(y_center - box_height / 2)
        xmax = int(x_center + box_width / 2)
        ymax = int(y_center + box_height / 2)
        
        # Ensure coordinates are within image bounds
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(image_width, xmax)
        ymax = min(image_height, ymax)
        
        # Skip invalid boxes
        if xmax <= xmin or ymax <= ymin:
            continue
            
        # Create object element
        obj_elem = ET.SubElement(annotation, 'object')
        
        name = ET.SubElement(obj_elem, 'name')
        name.text = VISDRONE_CLASSES[class_id]
        
        pose = ET.SubElement(obj_elem, 'pose')
        pose.text = 'Unspecified'
        
        truncated = ET.SubElement(obj_elem, 'truncated')
        truncated.text = '0'
        
        difficult = ET.SubElement(obj_elem, 'difficult')
        difficult.text = '0'
        
        # Bounding box
        bndbox = ET.SubElement(obj_elem, 'bndbox')
        xmin_elem = ET.SubElement(bndbox, 'xmin')
        xmin_elem.text = str(xmin)
        ymin_elem = ET.SubElement(bndbox, 'ymin')
        ymin_elem.text = str(ymin)
        xmax_elem = ET.SubElement(bndbox, 'xmax')
        xmax_elem.text = str(xmax)
        ymax_elem = ET.SubElement(bndbox, 'ymax')
        ymax_elem.text = str(ymax)
    
    # Write XML file
    tree = ET.ElementTree(annotation)
    tree.write(output_path, encoding='utf-8', xml_declaration=True)

def parse_yolo_annotation(yolo_file):
    """
    Parse YOLO annotation file
    
    Args:
        yolo_file: Path to YOLO .txt file
        
    Returns:
        List of objects with class_id and bbox coordinates
    """
    objects = []
    
    if not os.path.exists(yolo_file):
        return objects
        
    with open(yolo_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) != 5:
                continue
                
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Validate class_id
            if class_id not in VISDRONE_CLASSES:
                print(f"[WARNING] Unknown class_id: {class_id}")
                continue
                
            objects.append({
                'class_id': class_id,
                'bbox': [x_center, y_center, width, height]
            })
    
    return objects

def get_image_dimensions(image_path):
    """Get image dimensions using PIL"""
    try:
        with Image.open(image_path) as img:
            return img.size  # Returns (width, height)
    except Exception as e:
        print(f"[ERROR] Could not get dimensions for {image_path}: {e}")
        return None

def convert_dataset_split(mobilenet_ssd_path, split_name):
    """
    Convert one dataset split (train/val/test) from YOLO to VOC format
    
    Args:
        mobilenet_ssd_path: Path to mobilenet-ssd dataset directory
        split_name: 'train', 'val', or 'test'
    """
    print(f"\n[INFO] Converting {split_name} split...")
    
    # Paths
    images_dir = mobilenet_ssd_path / "images" / split_name
    labels_dir = mobilenet_ssd_path / "labels" / split_name
    voc_output_dir = mobilenet_ssd_path / "voc_format" / split_name
    
    # Create output directory
    voc_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(images_dir.glob("*.jpg"))
    print(f"[INFO] Found {len(image_files)} images in {split_name}")
    
    converted_count = 0
    skipped_count = 0
    
    for i, image_path in enumerate(image_files):
        if i % 100 == 0:
            print(f"[PROGRESS] Processing {i}/{len(image_files)} images...")
        
        # Get corresponding YOLO annotation file
        label_file = labels_dir / f"{image_path.stem}.txt"
        
        # Get image dimensions
        dimensions = get_image_dimensions(image_path)
        if dimensions is None:
            skipped_count += 1
            continue
            
        width, height = dimensions
        
        # Parse YOLO annotations
        objects = parse_yolo_annotation(label_file)
        
        # Create output XML file path
        xml_output_path = voc_output_dir / f"{image_path.stem}.xml"
        
        # Convert to VOC format
        try:
            create_voc_xml(
                image_path=image_path,
                image_width=width,
                image_height=height,
                objects=objects,
                output_path=xml_output_path
            )
            converted_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to convert {image_path.name}: {e}")
            skipped_count += 1
    
    print(f"[SUCCESS] {split_name} conversion complete:")
    print(f"  • Converted: {converted_count} files")
    print(f"  • Skipped: {skipped_count} files")
    print(f"  • Output: {voc_output_dir}")

def main():
    """Main conversion function"""
    parser = argparse.ArgumentParser(description="Convert YOLO to VOC format for MobileNet-SSD training")
    parser.add_argument("--dataset-path", type=str, 
                       default="C:/Users/burak/OneDrive/Desktop/Git Repos/drone-obj-detection-lightweight-ai/data/my_dataset/visdrone/mobilenet-ssd",
                       help="Path to MobileNet-SSD dataset directory")
    
    args = parser.parse_args()
    
    print("="*70)
    print("YOLO to VOC Format Converter for MobileNet-SSD")
    print("="*70)
    
    mobilenet_ssd_path = Path(args.dataset_path)
    
    # Validate input directory
    if not mobilenet_ssd_path.exists():
        print(f"[ERROR] Dataset directory not found: {mobilenet_ssd_path}")
        return 1
        
    print(f"[INFO] Dataset path: {mobilenet_ssd_path}")
    
    # Convert each split
    splits = ['train', 'val', 'test']
    total_converted = 0
    
    for split in splits:
        images_dir = mobilenet_ssd_path / "images" / split
        if images_dir.exists():
            convert_dataset_split(mobilenet_ssd_path, split)
        else:
            print(f"[WARNING] Skipping {split} - directory not found: {images_dir}")
    
    # Validation
    print(f"\n[VALIDATION] Checking VOC format output...")
    voc_base = mobilenet_ssd_path / "voc_format"
    
    for split in splits:
        voc_split_dir = voc_base / split
        if voc_split_dir.exists():
            xml_count = len(list(voc_split_dir.glob("*.xml")))
            print(f"[OK] {split}: {xml_count} XML files")
        else:
            print(f"[MISSING] {split}: No VOC annotations")
    
    print(f"\n[SUCCESS] YOLO to VOC conversion completed!")
    print(f"[INFO] VOC annotations saved to: {voc_base}")
    print(f"[READY] MobileNet-SSD training scripts can now use VOC format")
    print("="*70)
    
    return 0

if __name__ == "__main__":
    exit(main())