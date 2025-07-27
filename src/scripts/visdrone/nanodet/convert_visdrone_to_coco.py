#!/usr/bin/env python3
"""
VisDrone to COCO Format Converter for NanoDet Training
Master's Thesis: Robust Object Detection for Surveillance Drones

This script converts VisDrone dataset from YOLO format to COCO format JSON
required for NanoDet training. Implements proper COCO structure with
images, annotations, and categories following COCO dataset specifications.

Key Features:
- Converts YOLO normalized coordinates to COCO absolute coordinates
- Creates proper COCO JSON structure for train/val/test splits
- Handles VisDrone 10-class mapping
- Validates conversion integrity
- Protocol Version 2.0 compliant

Author: Burak Kağan Yılmazer
Date: July 2025
Environment: nanodet_env
"""

import os
import json
import cv2
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# VisDrone class mapping (1-based to 0-based)
VISDRONE_CLASSES = [
    'pedestrian',      # 0 (was 1)
    'people',          # 1 (was 2)  
    'bicycle',         # 2 (was 3)
    'car',             # 3 (was 4)
    'van',             # 4 (was 5)
    'truck',           # 5 (was 6)
    'tricycle',        # 6 (was 7)
    'awning-tricycle', # 7 (was 8)
    'bus',             # 8 (was 9)
    'motor'            # 9 (was 10)
]

def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging configuration"""
    log_file = output_dir / f"visdrone_to_coco_conversion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def yolo_to_coco_bbox(yolo_bbox: List[float], img_width: int, img_height: int) -> List[float]:
    """
    Convert YOLO format bbox to COCO format
    
    Args:
        yolo_bbox: [x_center, y_center, width, height] (normalized 0-1)
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        [x_min, y_min, width, height] in absolute pixels (COCO format)
    """
    x_center, y_center, width, height = yolo_bbox
    
    # Convert normalized to absolute coordinates
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    width_abs = width * img_width
    height_abs = height * img_height
    
    # Convert center format to top-left format (COCO)
    x_min = x_center_abs - (width_abs / 2)
    y_min = y_center_abs - (height_abs / 2)
    
    # Ensure coordinates are within image bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    width_abs = min(width_abs, img_width - x_min)
    height_abs = min(height_abs, img_height - y_min)
    
    return [x_min, y_min, width_abs, height_abs]

def calculate_area(bbox: List[float]) -> float:
    """Calculate area of COCO bbox"""
    return bbox[2] * bbox[3]  # width * height

def process_split(split_name: str, dataset_root: Path, output_dir: Path, logger: logging.Logger) -> Dict:
    """
    Process a single dataset split (train/val/test)
    
    Args:
        split_name: 'train', 'val', or 'test'
        dataset_root: Path to VisDrone dataset root
        output_dir: Output directory for COCO format
        logger: Logger instance
        
    Returns:
        Dictionary containing COCO format data
    """
    logger.info(f"Processing {split_name} split...")
    
    # Paths
    images_dir = dataset_root / split_name / "images"
    labels_dir = dataset_root / split_name / "labels"
    
    if not images_dir.exists():
        logger.warning(f"Images directory not found: {images_dir}")
        return None
    
    if not labels_dir.exists():
        logger.warning(f"Labels directory not found: {labels_dir}")
        return None
    
    # Initialize COCO structure
    coco_data = {
        "info": {
            "year": 2025,
            "version": "1.0",
            "description": f"VisDrone {split_name} dataset converted to COCO format for NanoDet",
            "contributor": "Burak Kağan Yılmazer - Master's Thesis",
            "url": "https://github.com/VisDrone/VisDrone-Dataset",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [
            {
                "id": 1,
                "name": "VisDrone License",
                "url": "https://github.com/VisDrone/VisDrone-Dataset"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Add categories
    for i, class_name in enumerate(VISDRONE_CLASSES):
        coco_data["categories"].append({
            "id": i + 1,  # COCO categories start from 1
            "name": class_name,
            "supercategory": "object"
        })
    
    # Process images and annotations
    annotation_id = 1
    processed_images = 0
    processed_annotations = 0
    
    for img_file in images_dir.glob("*.jpg"):
        # Load image to get dimensions
        img_path = str(img_file)
        image = cv2.imread(img_path)
        
        if image is None:
            logger.warning(f"Could not load image: {img_file.name}")
            continue
        
        height, width = image.shape[:2]
        image_id = processed_images + 1
        
        # Add image info
        coco_data["images"].append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": img_file.name,
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": ""
        })
        
        # Process corresponding label file
        label_file = labels_dir / (img_file.stem + ".txt")
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        bbox_width = float(parts[3])
                        bbox_height = float(parts[4])
                        
                        # Skip invalid class IDs
                        if class_id < 0 or class_id >= len(VISDRONE_CLASSES):
                            continue
                        
                        # Convert YOLO to COCO bbox
                        yolo_bbox = [x_center, y_center, bbox_width, bbox_height]
                        coco_bbox = yolo_to_coco_bbox(yolo_bbox, width, height)
                        
                        # Skip invalid bboxes
                        if coco_bbox[2] <= 0 or coco_bbox[3] <= 0:
                            continue
                        
                        # Calculate area
                        area = calculate_area(coco_bbox)
                        
                        # Add annotation
                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": class_id + 1,  # COCO categories start from 1
                            "segmentation": [],  # Empty for bounding box only
                            "area": area,
                            "bbox": coco_bbox,
                            "iscrowd": 0
                        })
                        
                        annotation_id += 1
                        processed_annotations += 1
                        
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Invalid annotation in {label_file.name}: {line}")
                        continue
        
        processed_images += 1
        
        if processed_images % 100 == 0:
            logger.info(f"Processed {processed_images} images, {processed_annotations} annotations")
    
    logger.info(f"Split {split_name} completed:")
    logger.info(f"  • Images: {processed_images}")
    logger.info(f"  • Annotations: {processed_annotations}")
    logger.info(f"  • Categories: {len(VISDRONE_CLASSES)}")
    
    return coco_data

def validate_coco_format(coco_data: Dict, split_name: str, logger: logging.Logger) -> bool:
    """
    Validate COCO format data structure
    
    Args:
        coco_data: COCO format dictionary
        split_name: Split name for logging
        logger: Logger instance
        
    Returns:
        True if valid, False otherwise
    """
    logger.info(f"Validating COCO format for {split_name}...")
    
    # Check required fields
    required_fields = ["images", "annotations", "categories"]
    for field in required_fields:
        if field not in coco_data:
            logger.error(f"Missing required field: {field}")
            return False
    
    # Check categories
    if len(coco_data["categories"]) != len(VISDRONE_CLASSES):
        logger.error(f"Expected {len(VISDRONE_CLASSES)} categories, got {len(coco_data['categories'])}")
        return False
    
    # Check image-annotation consistency
    image_ids = {img["id"] for img in coco_data["images"]}
    annotation_image_ids = {ann["image_id"] for ann in coco_data["annotations"]}
    
    orphaned_annotations = annotation_image_ids - image_ids
    if orphaned_annotations:
        logger.warning(f"Found {len(orphaned_annotations)} annotations with no corresponding image")
    
    # Check category IDs
    category_ids = {cat["id"] for cat in coco_data["categories"]}
    annotation_cat_ids = {ann["category_id"] for ann in coco_data["annotations"]}
    
    invalid_categories = annotation_cat_ids - category_ids
    if invalid_categories:
        logger.error(f"Found annotations with invalid category IDs: {invalid_categories}")
        return False
    
    logger.info(f"COCO format validation passed for {split_name}")
    return True

def convert_visdrone_to_coco(dataset_root: str, output_dir: str) -> bool:
    """
    Main conversion function
    
    Args:
        dataset_root: Path to VisDrone dataset root
        output_dir: Output directory for COCO format
        
    Returns:
        True if conversion successful, False otherwise
    """
    # Setup paths
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    logger.info("="*80)
    logger.info("VisDrone to COCO Format Conversion")
    logger.info("="*80)
    logger.info(f"Input: {dataset_root}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Classes: {len(VISDRONE_CLASSES)}")
    logger.info("")
    
    # Create images subdirectories
    images_output_dir = output_dir / "images"
    for split in ["train", "val", "test"]:
        (images_output_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Process each split
    conversion_success = True
    
    for split_name in ["train", "val", "test"]:
        logger.info(f"Processing {split_name} split...")
        
        coco_data = process_split(split_name, dataset_root, output_dir, logger)
        
        if coco_data is None:
            logger.warning(f"Skipping {split_name} split - data not found")
            continue
        
        # Validate COCO format
        if not validate_coco_format(coco_data, split_name, logger):
            logger.error(f"COCO validation failed for {split_name}")
            conversion_success = False
            continue
        
        # Save COCO JSON file
        output_file = output_dir / f"{split_name}.json"
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        logger.info(f"Saved COCO annotations: {output_file}")
        
        # Copy images to output directory
        images_src_dir = dataset_root / split_name / "images"
        images_dst_dir = images_output_dir / split_name
        
        if images_src_dir.exists():
            import shutil
            
            logger.info(f"Copying images for {split_name}...")
            copied_count = 0
            
            for img_file in images_src_dir.glob("*.jpg"):
                dst_file = images_dst_dir / img_file.name
                if not dst_file.exists():
                    shutil.copy2(img_file, dst_file)
                    copied_count += 1
            
            logger.info(f"Copied {copied_count} images to {images_dst_dir}")
        
        logger.info("")
    
    # Generate summary
    logger.info("="*80)
    logger.info("CONVERSION SUMMARY")
    logger.info("="*80)
    
    for split_name in ["train", "val", "test"]:
        json_file = output_dir / f"{split_name}.json"
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            logger.info(f"{split_name.upper()} Split:")
            logger.info(f"  • Images: {len(data['images'])}")
            logger.info(f"  • Annotations: {len(data['annotations'])}")
            logger.info(f"  • File: {json_file}")
        else:
            logger.info(f"{split_name.upper()} Split: Not found")
    
    logger.info(f"Categories: {len(VISDRONE_CLASSES)}")
    logger.info(f"Classes: {', '.join(VISDRONE_CLASSES)}")
    
    if conversion_success:
        logger.info("")
        logger.info("✅ CONVERSION COMPLETED SUCCESSFULLY")
        logger.info("NanoDet training data is ready!")
    else:
        logger.error("")
        logger.error("❌ CONVERSION FAILED")
        logger.error("Check logs for detailed error information")
    
    return conversion_success

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="Convert VisDrone dataset to COCO format for NanoDet")
    parser.add_argument('--dataset-root', type=str, 
                       default='./data/my_dataset/visdrone',
                       help='Path to VisDrone dataset root')
    parser.add_argument('--output-dir', type=str,
                       default='./data/my_dataset/visdrone/nanodet_format',
                       help='Output directory for COCO format')
    
    args = parser.parse_args()
    
    print("="*80)
    print("VisDrone to COCO Format Converter for NanoDet")
    print("="*80)
    print(f"Dataset Root: {args.dataset_root}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Classes: {len(VISDRONE_CLASSES)}")
    print("="*80)
    print("")
    
    try:
        success = convert_visdrone_to_coco(args.dataset_root, args.output_dir)
        
        if success:
            print("\n" + "="*80)
            print("✅ [SUCCESS] VisDrone to COCO conversion completed!")
            print(f"COCO format data available at: {args.output_dir}")
            print("Ready for NanoDet training!")
            print("="*80)
        else:
            print("\n" + "="*80)
            print("❌ [ERROR] Conversion failed!")
            print("Check the logs for detailed error information.")
            print("="*80)
            exit(1)
        
    except Exception as e:
        print(f"\n❌ [ERROR] Conversion failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()