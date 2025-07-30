#!/usr/bin/env python3
"""
Phase 1 Synthetic Test Set Evaluation for YOLOv5n
Master's Thesis: Robust Object Detection for Surveillance Drones

This script tests the Phase 1 baseline model on both clean and synthetic test sets
to establish baseline degradation metrics under environmental conditions.

Author: Burak Kağan Yılmazer
Date: July 30, 2025
Environment: yolov5n_visdrone_env
Protocol: Version 2.0 - Phase 1 Testing Requirements
"""

import os
import sys
import logging
import argparse
import warnings
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import cv2
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
script_path = Path(__file__).resolve()
project_root = script_path.parents[6]
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

try:
    import torch
    import yaml
    from PIL import Image
    
    # Import augmentation modules
    from augmentation_pipeline.augment_scripts.fog import add_fog
    from augmentation_pipeline.augment_scripts.night import simulate_night
    from augmentation_pipeline.augment_scripts.sensor_distortions import add_sensor_effects
    
except ImportError as e:
    print(f"[ERROR] Failed to import required packages: {e}")
    print("Please ensure you're using the yolov5n_visdrone_env environment")
    print("Activation: .\\venvs\\visdrone\\yolov5n_visdrone_env\\Scripts\\Activate.ps1")
    sys.exit(1)


class SyntheticTestGenerator:
    """
    Generates synthetic test sets with environmental augmentation
    for Phase 1 baseline degradation testing.
    """
    
    def __init__(self, test_dir: Path, output_dir: Path):
        """Initialize synthetic test generator"""
        self.test_dir = test_dir
        self.output_dir = output_dir
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for synthetic generation"""
        logger = logging.getLogger('SyntheticTest')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
        
    def generate_synthetic_test_set(self) -> Dict[str, Path]:
        """
        Generate synthetic test sets with different environmental conditions
        
        Returns:
            Dictionary mapping condition names to output directories
        """
        self.logger.info("Generating synthetic test sets...")
        
        # Define augmentation configurations
        augmentations = {
            'fog_light': {
                'type': 'fog',
                'config': {'intensity': 0.3, 'depth_blend': 0.2}
            },
            'fog_heavy': {
                'type': 'fog',
                'config': {'intensity': 0.7, 'depth_blend': 0.5}
            },
            'night_mild': {
                'type': 'night',
                'config': {'gamma': 2.0, 'brightness_reduction': 0.7, 'desaturate': True}
            },
            'night_severe': {
                'type': 'night',
                'config': {'gamma': 3.0, 'brightness_reduction': 0.4, 'desaturate': True}
            },
            'sensor_light': {
                'type': 'sensor',
                'config': {'blur_ksize': 3, 'noise_std': 3, 'shift_pixels': 1}
            },
            'sensor_heavy': {
                'type': 'sensor',
                'config': {'blur_ksize': 5, 'noise_std': 8, 'shift_pixels': 2}
            },
            'combined_mild': {
                'type': 'combined',
                'config': {
                    'fog': {'intensity': 0.2, 'depth_blend': 0.1},
                    'night': {'gamma': 1.8, 'brightness_reduction': 0.8, 'desaturate': False}
                }
            },
            'combined_severe': {
                'type': 'combined',
                'config': {
                    'fog': {'intensity': 0.5, 'depth_blend': 0.3},
                    'night': {'gamma': 2.5, 'brightness_reduction': 0.5, 'desaturate': True},
                    'sensor': {'blur_ksize': 3, 'noise_std': 5, 'shift_pixels': 1}
                }
            }
        }
        
        # Get test images
        test_images = list(self.test_dir.glob("*.jpg")) + list(self.test_dir.glob("*.png"))
        self.logger.info(f"Found {len(test_images)} test images")
        
        # Create output directories and process images
        synthetic_dirs = {}
        
        for aug_name, aug_config in augmentations.items():
            # Create output directory
            aug_output_dir = self.output_dir / aug_name / "images"
            aug_output_dir.mkdir(parents=True, exist_ok=True)
            synthetic_dirs[aug_name] = aug_output_dir.parent
            
            self.logger.info(f"Generating {aug_name} augmentation...")
            
            # Process each image
            for img_path in tqdm(test_images, desc=f"Processing {aug_name}"):
                try:
                    # Load image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                        
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Apply augmentation
                    aug_img = self._apply_augmentation(img_rgb, aug_config)
                    
                    # Convert back to BGR and save
                    aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                    output_path = aug_output_dir / img_path.name
                    cv2.imwrite(str(output_path), aug_img_bgr)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process {img_path.name}: {e}")
                    continue
                    
            # Copy labels (same for all augmentations)
            self._copy_labels(self.test_dir.parent / "labels", 
                            synthetic_dirs[aug_name] / "labels")
                            
        self.logger.info(f"Generated {len(synthetic_dirs)} synthetic test sets")
        return synthetic_dirs
        
    def _apply_augmentation(self, img: np.ndarray, aug_config: Dict) -> np.ndarray:
        """Apply specified augmentation to image"""
        aug_type = aug_config['type']
        config = aug_config['config']
        
        if aug_type == 'fog':
            return add_fog(img, config)
        elif aug_type == 'night':
            return simulate_night(img, config)
        elif aug_type == 'sensor':
            return add_sensor_effects(img, config)
        elif aug_type == 'combined':
            # Apply multiple augmentations sequentially
            aug_img = img.copy()
            if 'fog' in config:
                aug_img = add_fog(aug_img, config['fog'])
            if 'night' in config:
                aug_img = simulate_night(aug_img, config['night'])
            if 'sensor' in config:
                aug_img = add_sensor_effects(aug_img, config['sensor'])
            return aug_img
        else:
            return img
            
    def _copy_labels(self, src_labels: Path, dst_labels: Path):
        """Copy label files to synthetic directory"""
        if not src_labels.exists():
            self.logger.warning(f"Source labels directory not found: {src_labels}")
            return
            
        dst_labels.mkdir(parents=True, exist_ok=True)
        
        for label_file in src_labels.glob("*.txt"):
            shutil.copy2(label_file, dst_labels / label_file.name)


class Phase1SyntheticTester:
    """
    Tests Phase 1 baseline model on clean and synthetic test sets
    """
    
    def __init__(self, model_path: Path, data_yaml: Path = None):
        """Initialize tester with trained model"""
        self.model_path = model_path
        # Use dedicated dataset config instead of training config
        self.data_yaml = data_yaml or (project_root / "config" / "dataset" / "visdrone_dataset.yaml")
        self.logger = self._setup_logging()
        self.results = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('Phase1Tester')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
        
    def test_model(self, test_name: str, test_path: Path) -> Dict[str, Any]:
        """
        Test model on specified dataset
        
        Args:
            test_name: Name of the test (e.g., 'clean', 'fog_heavy')
            test_path: Path to test images
            
        Returns:
            Test results dictionary
        """
        self.logger.info(f"Testing on {test_name} dataset...")
        
        # Prepare temporary data yaml for this test
        temp_yaml = self._create_temp_yaml(test_path)
        
        # Run YOLOv5 validation
        yolov5_path = project_root / "src" / "models" / "YOLOv5"
        
        # Default confidence and IoU thresholds (YOLOv5 validation optimized)
        conf_thres = 0.001
        iou_thres = 0.6
        
        val_cmd = [
            sys.executable,
            str(yolov5_path / "val.py"),
            "--weights", str(self.model_path),
            "--data", str(temp_yaml),
            "--img", "640",
            "--batch-size", "8",
            "--conf-thres", str(conf_thres),
            "--iou-thres", str(iou_thres),
            "--device", "",
            "--save-txt",
            "--save-conf",
            "--project", str(self.model_path.parent.parent / "test_results"),
            "--name", f"phase1_{test_name}",
            "--exist-ok"
        ]
        
        # Execute validation (run from project root to avoid git path issues)
        import subprocess
        result = subprocess.run(val_cmd, capture_output=True, text=True, cwd=str(project_root))
        
        # Debug: Log YOLOv5 output to understand what's happening
        self.logger.info(f"YOLOv5 return code: {result.returncode}")
        if result.stdout:
            stdout_lines = result.stdout.strip().split('\n')
            self.logger.info(f"YOLOv5 stdout (last 15 lines):")
            for line in stdout_lines[-15:]:
                self.logger.info(f"  {line}")
        if result.stderr:
            self.logger.info(f"YOLOv5 stderr: {result.stderr}")
        
        if result.returncode != 0:
            self.logger.error(f"Validation failed for {test_name}")
            self.logger.error(result.stderr)
            return {}
            
        # Parse results from output
        results = self._parse_validation_output(result.stdout)
        results['test_name'] = test_name
        results['test_path'] = str(test_path)
        
        # Clean up temp yaml
        temp_yaml.unlink()
        
        return results
        
    def _create_temp_yaml(self, test_path: Path) -> Path:
        """Create temporary data yaml for specific test set"""
        with open(self.data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
            
        # Use the clean dataset config (no training hyperparameters)
        dataset_root = project_root / "data" / "my_dataset" / "visdrone"
        clean_config = {
            'path': str(dataset_root),
            'train': data_config['train'],
            'val': data_config['val'], 
            'nc': data_config['nc'],
            'names': data_config['names']
        }
        
        # Robust path calculation using Path.relative_to() instead of string manipulation
        try:
            if test_path.name == "images":
                # Clean test case: test_path is already pointing to images folder
                # e.g., .../visdrone/test/images
                images_path = test_path
            else:
                # Synthetic test case: test_path is directory containing images subfolder
                # e.g., .../visdrone/test_synthetic/fog_light -> .../visdrone/test_synthetic/fog_light/images
                images_path = test_path / "images"
            
            # Use Path.relative_to() method (robust, handles normalization)
            relative_path = images_path.relative_to(dataset_root)
            
            # Convert to POSIX format (forward slashes) for YOLOv5 compatibility
            clean_config['test'] = relative_path.as_posix()
            
            self.logger.info(f"YAML Config - Dataset root: {dataset_root}")
            self.logger.info(f"YAML Config - Test images: {images_path}")
            self.logger.info(f"YAML Config - Relative path: {relative_path.as_posix()}")
            
        except ValueError as e:
            # Fallback with explicit error handling
            self.logger.error(f"Path calculation failed: {e}")
            self.logger.error(f"test_path: {test_path}")
            self.logger.error(f"dataset_root: {dataset_root}")
            
            # Manual fallback based on known structure
            if test_path.name == "images":
                clean_config['test'] = "test/images"
                self.logger.info("Using fallback path: test/images")
            else:
                synthetic_dir = test_path.name
                clean_config['test'] = f"test_synthetic/{synthetic_dir}/images"
                self.logger.info(f"Using fallback path: test_synthetic/{synthetic_dir}/images")
        
        # Save temporary yaml with CLEAN config (no training hyperparameters)
        temp_yaml = test_path / "temp_data.yaml"
        with open(temp_yaml, 'w') as f:
            yaml.dump(clean_config, f)
        
        # Debug: Log the generated YAML content
        self.logger.info(f"Generated YAML at: {temp_yaml}")
        self.logger.info(f"Clean YAML contents: {clean_config}")
            
        return temp_yaml
        
    def _parse_validation_output(self, output: str) -> Dict[str, Any]:
        """Parse validation results from YOLOv5 output"""
        results = {
            'mAP50': 0.0,
            'mAP50-95': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'classes': {}
        }
        
        # Parse metrics from output with multiple patterns
        lines = output.split('\n')
        
        # Pattern 1: Look for results table with "all" class
        for line in lines:
            if 'all' in line and any(keyword in line for keyword in ['mAP', 'AP']):
                self.logger.info(f"Found results line: {line}")
                parts = line.split()
                
                # Try different parsing approaches
                try:
                    if 'all' in parts:
                        idx = parts.index('all')
                        # YOLOv5 format: Class Images Instances P R mAP50 mAP50-95
                        if len(parts) > idx + 6:
                            results['precision'] = float(parts[idx + 3])
                            results['recall'] = float(parts[idx + 4])  
                            results['mAP50'] = float(parts[idx + 5])
                            results['mAP50-95'] = float(parts[idx + 6])
                            self.logger.info(f"Parsed: P={results['precision']}, R={results['recall']}, mAP50={results['mAP50']}")
                            break
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Failed to parse line '{line}': {e}")
                    continue
        
        # Pattern 2: Look for summary metrics if pattern 1 failed
        if results['mAP50'] == 0.0:
            for line in lines:
                if 'mAP_0.5' in line or 'mAP@0.5' in line:
                    self.logger.info(f"Found mAP line: {line}")
                    # Try to extract number after mAP
                    import re
                    match = re.search(r'mAP[@_]0\.5[:\s=]+([0-9.]+)', line)
                    if match:
                        results['mAP50'] = float(match.group(1))
                        self.logger.info(f"Extracted mAP50: {results['mAP50']}")
                        break
        
        # Pattern 3: Look for any decimal numbers that might be metrics
        if results['mAP50'] == 0.0:
            self.logger.warning("Could not find standard mAP format, looking for any metrics...")
            for line in lines:
                if any(keyword in line.lower() for keyword in ['results', 'metrics', 'summary']):
                    self.logger.info(f"Potential results line: {line}")
        
        return results
        
    def run_all_tests(self, clean_test_path: Path, synthetic_dirs: Dict[str, Path]) -> Dict[str, Any]:
        """Run tests on all datasets and compile results"""
        all_results = {}
        
        # Test on clean dataset  
        self.logger.info("Testing on clean dataset...")
        all_results['clean'] = self.test_model('clean', clean_test_path)
        
        # Test on each synthetic dataset
        for aug_name, aug_path in synthetic_dirs.items():
            self.logger.info(f"Testing on {aug_name} dataset...")
            all_results[aug_name] = self.test_model(aug_name, aug_path)
            
        return all_results


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Phase 1 Synthetic Test Evaluation for YOLOv5n"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='runs/train/yolov5n_phase1_baseline_20250730_034928/weights/best.pt',
        help='Path to trained model weights'
    )
    parser.add_argument(
        '--generate-only',
        action='store_true',
        help='Only generate synthetic test sets without testing'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Phase 1 Synthetic Test Set Evaluation")
    print("Protocol: Version 2.0 - Baseline Degradation Testing")
    print("="*80)
    
    # Paths
    visdrone_path = project_root / "data" / "my_dataset" / "visdrone"
    test_images_path = visdrone_path / "test" / "images"
    synthetic_output = visdrone_path / "test_synthetic"
    data_yaml = project_root / "config" / "dataset" / "visdrone_dataset.yaml"  # Use dedicated dataset config
    model_path = project_root / args.model
    
    # Verify paths
    if not test_images_path.exists():
        print(f"[ERROR] Test images not found: {test_images_path}")
        return 1
        
    if not model_path.exists() and not args.generate_only:
        print(f"[ERROR] Model weights not found: {model_path}")
        return 1
        
    # Generate synthetic test sets
    generator = SyntheticTestGenerator(test_images_path, synthetic_output)
    synthetic_dirs = generator.generate_synthetic_test_set()
    
    if args.generate_only:
        print("\n[SUCCESS] Synthetic test sets generated!")
        print(f"Output directory: {synthetic_output}")
        return 0
        
    # Test model on all datasets
    tester = Phase1SyntheticTester(model_path, data_yaml)
    results = tester.run_all_tests(test_images_path, synthetic_dirs)
    
    # Save results
    results_file = model_path.parent.parent / "phase1_synthetic_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\n[SUCCESS] Testing completed!")
    print(f"Results saved to: {results_file}")
    
    # Print summary
    print("\nResults Summary:")
    print("-" * 50)
    for test_name, test_results in results.items():
        if 'mAP50' in test_results:
            print(f"{test_name:20} mAP@0.5: {test_results['mAP50']:.3f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())