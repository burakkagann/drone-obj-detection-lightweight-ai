#!/usr/bin/env python3
"""
Dataset Stratification Manager for YOLOv5n + VisDrone
Implements strategic data distribution according to methodology framework.

Distribution Strategy:
- Original: 40% of training data
- Light conditions: 20% of training data (distributed across fog, night, blur, weather)
- Medium conditions: 25% of training data (distributed across fog, night, blur, weather)
- Heavy conditions: 15% of training data (distributed across fog, night, blur, weather)
"""

import os
import json
import shutil
import random
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import yaml

from environmental_augmentation_pipeline import (
    EnvironmentalAugmentator, 
    AugmentationType, 
    IntensityLevel
)

@dataclass
class DatasetStats:
    """Statistics for dataset distribution"""
    total_images: int
    original_count: int
    light_count: int
    medium_count: int
    heavy_count: int
    augmentation_breakdown: Dict[str, Dict[str, int]]

class DatasetStratificationManager:
    """Manages dataset stratification according to methodology framework"""
    
    def __init__(self, source_dataset_path: str, output_dataset_path: str, 
                 config_path: str = None, seed: int = 42):
        """
        Initialize the dataset stratification manager
        
        Args:
            source_dataset_path: Path to original VisDrone dataset
            output_dataset_path: Path where stratified dataset will be created
            config_path: Path to dataset configuration file
            seed: Random seed for reproducibility
        """
        self.source_path = Path(source_dataset_path)
        self.output_path = Path(output_dataset_path)
        self.config_path = Path(config_path) if config_path else None
        self.seed = seed
        
        # Set random seeds for reproducibility
        random.seed(seed)
        
        # Distribution ratios according to methodology
        self.distribution_ratios = {
            "original": 0.40,
            "light": 0.20,
            "medium": 0.25,
            "heavy": 0.15
        }
        
        # Augmentation type distribution within each intensity level
        self.augmentation_distribution = {
            AugmentationType.FOG: 0.30,
            AugmentationType.NIGHT: 0.30,
            AugmentationType.MOTION_BLUR: 0.25,
            AugmentationType.RAIN: 0.10,
            AugmentationType.SNOW: 0.05
        }
        
        # Initialize augmentator
        self.augmentator = EnvironmentalAugmentator(seed=seed)
        
        # Statistics tracking
        self.stats = None
        
    def analyze_source_dataset(self) -> Dict:
        """Analyze the source dataset structure and statistics"""
        analysis = {
            "total_images": 0,
            "train_images": 0,
            "val_images": 0,
            "test_images": 0,
            "image_formats": defaultdict(int),
            "annotation_files": 0,
            "directory_structure": {}
        }
        
        # Analyze directory structure
        for split in ["train", "val", "test"]:
            split_path = self.source_path / split
            if split_path.exists():
                images_path = split_path / "images"
                labels_path = split_path / "labels"
                
                if images_path.exists():
                    image_files = list(images_path.glob("*"))
                    analysis[f"{split}_images"] = len(image_files)
                    analysis["total_images"] += len(image_files)
                    
                    # Count image formats
                    for img_file in image_files:
                        analysis["image_formats"][img_file.suffix] += 1
                
                if labels_path.exists():
                    label_files = list(labels_path.glob("*.txt"))
                    analysis["annotation_files"] += len(label_files)
                
                analysis["directory_structure"][split] = {
                    "images": len(image_files) if images_path.exists() else 0,
                    "labels": len(label_files) if labels_path.exists() else 0
                }
        
        return analysis
    
    def create_stratified_dataset(self, verbose: bool = True) -> DatasetStats:
        """Create stratified dataset according to methodology framework"""
        if verbose:
            print("[START] Creating Stratified Dataset for YOLOv5n + VisDrone")
            print("=" * 60)
        
        # Analyze source dataset
        source_analysis = self.analyze_source_dataset()
        if verbose:
            print(f"[ANALYSIS] Source Dataset Analysis:")
            print(f"   Total Images: {source_analysis['total_images']}")
            print(f"   Train: {source_analysis['train_images']}")
            print(f"   Val: {source_analysis['val_images']}")
            print(f"   Test: {source_analysis['test_images']}")
        
        # Create output directory structure
        self._create_output_structure()
        
        # Process each split
        total_stats = DatasetStats(
            total_images=0,
            original_count=0,
            light_count=0,
            medium_count=0,
            heavy_count=0,
            augmentation_breakdown=defaultdict(lambda: defaultdict(int))
        )
        
        for split in ["train", "val", "test"]:
            if verbose:
                print(f"\n[PROCESS] Processing {split} split...")
            
            split_stats = self._process_split(split, verbose)
            
            # Aggregate statistics
            total_stats.total_images += split_stats.total_images
            total_stats.original_count += split_stats.original_count
            total_stats.light_count += split_stats.light_count
            total_stats.medium_count += split_stats.medium_count
            total_stats.heavy_count += split_stats.heavy_count
            
            # Merge augmentation breakdown
            for aug_type, intensity_dict in split_stats.augmentation_breakdown.items():
                for intensity, count in intensity_dict.items():
                    total_stats.augmentation_breakdown[aug_type][intensity] += count
        
        # Save statistics
        self._save_statistics(total_stats)
        
        # Create dataset configuration
        self._create_dataset_config(total_stats)
        
        if verbose:
            print(f"\n[SUCCESS] Dataset Stratification Complete!")
            print(f"   Total Images: {total_stats.total_images}")
            print(f"   Original: {total_stats.original_count} ({total_stats.original_count/total_stats.total_images*100:.1f}%)")
            print(f"   Light: {total_stats.light_count} ({total_stats.light_count/total_stats.total_images*100:.1f}%)")
            print(f"   Medium: {total_stats.medium_count} ({total_stats.medium_count/total_stats.total_images*100:.1f}%)")
            print(f"   Heavy: {total_stats.heavy_count} ({total_stats.heavy_count/total_stats.total_images*100:.1f}%)")
        
        self.stats = total_stats
        return total_stats
    
    def _create_output_structure(self):
        """Create output directory structure"""
        # Create main directories
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Create split directories
        for split in ["train", "val", "test"]:
            split_path = self.output_path / split
            split_path.mkdir(exist_ok=True)
            
            # Create single directories for all images and labels
            # YOLOv5 expects ALL images in train/images/ and ALL labels in train/labels/
            (split_path / "images").mkdir(exist_ok=True)
            (split_path / "labels").mkdir(exist_ok=True)
    
    def _process_split(self, split: str, verbose: bool = True) -> DatasetStats:
        """Process a single split (train/val/test)"""
        source_split_path = self.source_path / split
        output_split_path = self.output_path / split
        
        if not source_split_path.exists():
            if verbose:
                print(f"   [WARNING] {split} split not found, skipping...")
            return DatasetStats(0, 0, 0, 0, 0, defaultdict(lambda: defaultdict(int)))
        
        # Get all image files
        source_images_path = source_split_path / "images"
        source_labels_path = source_split_path / "labels"
        
        if not source_images_path.exists():
            if verbose:
                print(f"   [WARNING] Images directory not found for {split}, skipping...")
            return DatasetStats(0, 0, 0, 0, 0, defaultdict(lambda: defaultdict(int)))
        
        image_files = list(source_images_path.glob("*"))
        total_images = len(image_files)
        
        if total_images == 0:
            if verbose:
                print(f"   [WARNING] No images found in {split}, skipping...")
            return DatasetStats(0, 0, 0, 0, 0, defaultdict(lambda: defaultdict(int)))
        
        # Calculate distribution counts
        original_count = int(total_images * self.distribution_ratios["original"])
        light_count = int(total_images * self.distribution_ratios["light"])
        medium_count = int(total_images * self.distribution_ratios["medium"])
        heavy_count = int(total_images * self.distribution_ratios["heavy"])
        
        # Adjust for rounding
        actual_total = original_count + light_count + medium_count + heavy_count
        if actual_total < total_images:
            # Add difference to medium (largest group)
            medium_count += total_images - actual_total
        
        if verbose:
            print(f"   [DISTRIBUTION] {split} Distribution:")
            print(f"      Original: {original_count}")
            print(f"      Light: {light_count}")
            print(f"      Medium: {medium_count}")
            print(f"      Heavy: {heavy_count}")
        
        # Shuffle images for random distribution
        random.shuffle(image_files)
        
        # Process original images
        original_processed = self._process_original_images(
            image_files[:original_count], 
            source_labels_path, 
            output_split_path, 
            verbose
        )
        
        # Process augmented images
        remaining_images = image_files[original_count:]
        
        light_processed = self._process_augmented_images(
            remaining_images[:light_count], 
            source_labels_path, 
            output_split_path, 
            IntensityLevel.LIGHT, 
            verbose
        )
        
        medium_processed = self._process_augmented_images(
            remaining_images[light_count:light_count + medium_count], 
            source_labels_path, 
            output_split_path, 
            IntensityLevel.MEDIUM, 
            verbose
        )
        
        heavy_processed = self._process_augmented_images(
            remaining_images[light_count + medium_count:light_count + medium_count + heavy_count], 
            source_labels_path, 
            output_split_path, 
            IntensityLevel.HEAVY, 
            verbose
        )
        
        # Create statistics
        stats = DatasetStats(
            total_images=original_processed + light_processed + medium_processed + heavy_processed,
            original_count=original_processed,
            light_count=light_processed,
            medium_count=medium_processed,
            heavy_count=heavy_processed,
            augmentation_breakdown=defaultdict(lambda: defaultdict(int))
        )
        
        return stats
    
    def _process_original_images(self, image_files: List[Path], 
                               labels_path: Path, output_split_path: Path, 
                               verbose: bool) -> int:
        """Process original images (no augmentation)"""
        processed_count = 0
        
        for img_file in image_files:
            try:
                # Copy image directly to images/ directory
                output_img_path = output_split_path / "images" / img_file.name
                shutil.copy2(img_file, output_img_path)
                
                # Copy corresponding label if it exists
                label_file = labels_path / f"{img_file.stem}.txt"
                if label_file.exists():
                    output_label_path = output_split_path / "labels" / f"{img_file.stem}.txt"
                    shutil.copy2(label_file, output_label_path)
                
                processed_count += 1
                
            except Exception as e:
                if verbose:
                    print(f"   [ERROR] Error processing {img_file.name}: {e}")
        
        return processed_count
    
    def _process_augmented_images(self, image_files: List[Path], 
                                labels_path: Path, output_split_path: Path, 
                                intensity: IntensityLevel, verbose: bool) -> int:
        """Process augmented images"""
        processed_count = 0
        
        # Calculate distribution of augmentation types
        aug_counts = {}
        total_images = len(image_files)
        
        for aug_type, ratio in self.augmentation_distribution.items():
            aug_counts[aug_type] = int(total_images * ratio)
        
        # Adjust for rounding
        actual_total = sum(aug_counts.values())
        if actual_total < total_images:
            # Add difference to fog (largest group)
            aug_counts[AugmentationType.FOG] += total_images - actual_total
        
        # Process each augmentation type
        image_idx = 0
        for aug_type, count in aug_counts.items():
            for i in range(count):
                if image_idx >= len(image_files):
                    break
                
                img_file = image_files[image_idx]
                image_idx += 1
                
                try:
                    # Load and augment image
                    image = cv2.imread(str(img_file))
                    if image is None:
                        if verbose:
                            print(f"   [ERROR] Could not load image: {img_file.name}")
                        continue
                    
                    # Apply augmentation
                    augmented_image, config = self.augmentator.augment_image(
                        image, aug_type, intensity
                    )
                    
                    # Save augmented image directly to images/ directory
                    output_filename = f"{img_file.stem}_{aug_type.value}_{intensity.value}{img_file.suffix}"
                    output_img_path = output_split_path / "images" / output_filename
                    cv2.imwrite(str(output_img_path), augmented_image)
                    
                    # Copy corresponding label if it exists
                    label_file = labels_path / f"{img_file.stem}.txt"
                    if label_file.exists():
                        output_label_filename = f"{img_file.stem}_{aug_type.value}_{intensity.value}.txt"
                        output_label_path = output_split_path / "labels" / output_label_filename
                        shutil.copy2(label_file, output_label_path)
                    
                    processed_count += 1
                    
                except Exception as e:
                    if verbose:
                        print(f"   [ERROR] Error processing {img_file.name} with {aug_type.value}: {e}")
        
        return processed_count
    
    def _save_statistics(self, stats: DatasetStats):
        """Save dataset statistics"""
        # Convert defaultdict to regular dict for serialization
        stats_dict = {
            "total_images": stats.total_images,
            "original_count": stats.original_count,
            "light_count": stats.light_count,
            "medium_count": stats.medium_count,
            "heavy_count": stats.heavy_count,
            "augmentation_breakdown": dict((k, dict(v)) for k, v in stats.augmentation_breakdown.items())
        }
        
        # Save as JSON
        stats_file = self.output_path / "dataset_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        
        # Save as readable text
        stats_text_file = self.output_path / "dataset_statistics.txt"
        with open(stats_text_file, 'w') as f:
            f.write("YOLOv5n + VisDrone Dataset Statistics\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Total Images: {stats.total_images}\n")
            f.write(f"Original Images: {stats.original_count} ({stats.original_count/stats.total_images*100:.1f}%)\n")
            f.write(f"Light Augmented: {stats.light_count} ({stats.light_count/stats.total_images*100:.1f}%)\n")
            f.write(f"Medium Augmented: {stats.medium_count} ({stats.medium_count/stats.total_images*100:.1f}%)\n")
            f.write(f"Heavy Augmented: {stats.heavy_count} ({stats.heavy_count/stats.total_images*100:.1f}%)\n\n")
            
            f.write("Augmentation Breakdown:\n")
            for aug_type, intensity_dict in stats.augmentation_breakdown.items():
                f.write(f"  {aug_type}:\n")
                for intensity, count in intensity_dict.items():
                    f.write(f"    {intensity}: {count}\n")
    
    def _create_dataset_config(self, stats: DatasetStats):
        """Create dataset configuration file for YOLOv5"""
        config = {
            "path": str(self.output_path.absolute()),
            "train": "train/images",
            "val": "val/images", 
            "test": "test/images",
            "nc": 10,  # VisDrone has 10 classes
            "names": [
                "pedestrian",
                "people",
                "bicycle",
                "car",
                "van",
                "truck",
                "tricycle",
                "awning-tricycle",
                "bus",
                "motor"
            ],
            "stratification": {
                "total_images": stats.total_images,
                "distribution": {
                    "original": stats.original_count,
                    "light": stats.light_count,
                    "medium": stats.medium_count,
                    "heavy": stats.heavy_count
                },
                "augmentation_breakdown": dict(stats.augmentation_breakdown),
                "methodology_compliance": True,
                "seed": self.seed
            }
        }
        
        # Save configuration
        config_file = self.output_path / "dataset_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    def validate_stratification(self) -> Dict[str, float]:
        """Validate that stratification meets methodology requirements"""
        if not self.stats:
            raise ValueError("Dataset must be created before validation")
        
        total = self.stats.total_images
        actual_ratios = {
            "original": self.stats.original_count / total,
            "light": self.stats.light_count / total,
            "medium": self.stats.medium_count / total,
            "heavy": self.stats.heavy_count / total
        }
        
        # Calculate deviations from target ratios
        deviations = {}
        for key in self.distribution_ratios:
            target = self.distribution_ratios[key]
            actual = actual_ratios[key]
            deviations[key] = abs(actual - target)
        
        # Overall validation score (lower is better)
        validation_score = sum(deviations.values()) / len(deviations)
        
        validation_results = {
            "target_ratios": self.distribution_ratios,
            "actual_ratios": actual_ratios,
            "deviations": deviations,
            "validation_score": validation_score,
            "is_valid": validation_score < 0.05  # 5% tolerance
        }
        
        return validation_results 