#!/usr/bin/env python3
"""
YOLOv5n TRUE BASELINE Training Script (Raw Performance)
Thesis: "Robust Object Detection for Surveillance Drones in Low-Visibility Environments"

Purpose: Establish TRUE baseline using YOLOv5n default hyperparameters
Dataset: Original VisDrone (7,019 images) with NO augmentation
Expected Performance: 15-18% mAP@0.5 (unoptimized raw performance)

This script demonstrates RAW YOLOv5n capability before any optimization,
providing the foundation for measuring optimization improvements.

METHODOLOGY COMPLIANCE:
1. Uses YOLOv5n true defaults (416px, batch=8)
2. Disables ALL augmentations for pure dataset performance
3. Provides comprehensive evaluation metrics
4. Enables direct comparison with YOLOv8n baseline

Usage:
    python train_true_baseline_yolov5n.py [--epochs EPOCHS] [--quick-test]
    
    --epochs: Number of epochs to train (default: 100 for full training, 20 for quick test)
    --quick-test: Run 20-epoch validation test first
"""

import argparse
import subprocess
import sys
import os
import time
import json
import warnings
from pathlib import Path
from datetime import datetime
import logging

# Suppress warnings for clean output
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*", category=UserWarning)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('true_baseline_training.log'),
        logging.StreamHandler()
    ]
)

def setup_paths():
    """Set up project paths"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent.parent.parent  # Go up 5 levels
    yolov5_path = project_root / "src" / "models" / "YOLOv5"
    
    return {
        'project_root': project_root,
        'yolov5_path': yolov5_path,
        'script_dir': script_dir,
        'config_dir': project_root / "config" / "visdrone" / "yolov5n_v1"
    }

def validate_environment(paths):
    """Validate true baseline training environment"""
    logging.info("[VALIDATION] Validating TRUE Baseline training environment...")
    
    # Check if YOLOv5 directory exists
    if not paths['yolov5_path'].exists():
        logging.error(f"[ERROR] YOLOv5 directory not found: {paths['yolov5_path']}")
        return False
    
    # Check if baseline dataset config exists
    dataset_config = paths['config_dir'] / "baseline_dataset_config.yaml"
    if not dataset_config.exists():
        logging.error(f"[ERROR] Baseline dataset config not found: {dataset_config}")
        return False
    
    # Check if true baseline hyperparameter file exists
    hyp_file = paths['config_dir'] / "hyp_visdrone_true_baseline.yaml"
    if not hyp_file.exists():
        logging.error(f"[ERROR] True baseline hyperparameter file not found: {hyp_file}")
        logging.info("[INFO] Creating true baseline hyperparameter file...")
        create_true_baseline_hyperparameters(paths)
    
    # Check if baseline dataset actually exists
    dataset_path = paths['project_root'] / "data" / "my_dataset" / "visdrone"
    train_images = dataset_path / "train" / "images"
    train_labels = dataset_path / "train" / "labels"
    val_images = dataset_path / "val" / "images"
    val_labels = dataset_path / "val" / "labels"
    
    for path, desc in [(train_images, "baseline training images"), (train_labels, "baseline training labels"), 
                       (val_images, "baseline validation images"), (val_labels, "baseline validation labels")]:
        if not path.exists():
            logging.error(f"[ERROR] Baseline dataset path missing: {path} ({desc})")
            return False
        
        # Count files
        file_count = len(list(path.glob("*")))
        logging.info(f"[SUCCESS] Found {file_count} {desc} files")
    
    # Validate expected file counts for baseline dataset
    train_count = len(list(train_images.glob("*.jpg")))
    val_count = len(list(val_images.glob("*.jpg")))
    
    if train_count != 6471:
        logging.warning(f"[WARNING] Expected 6,471 training images, found {train_count}")
    if val_count != 548:
        logging.warning(f"[WARNING] Expected 548 validation images, found {val_count}")
    
    logging.info(f"[DATASET] TRUE Baseline Dataset: {train_count} train, {val_count} val images")
    
    # Validate YOLOv5 pre-trained weights availability
    weights_path = paths['yolov5_path'] / "yolov5n.pt"
    if not weights_path.exists():
        logging.warning(f"[WARNING] Pre-trained weights not found at {weights_path}")
        logging.info("[INFO] YOLOv5 will download yolov5n.pt automatically")
    else:
        logging.info(f"[SUCCESS] Pre-trained weights found: {weights_path}")
    
    logging.info("[SUCCESS] TRUE Baseline environment validation passed")
    return True

def create_true_baseline_hyperparameters(paths):
    """Create true baseline hyperparameters file with YOLOv5 defaults"""
    
    hyp_content = """# YOLOv5n TRUE BASELINE Hyperparameters
# Purpose: Raw YOLOv5n performance without optimization
# Based on: YOLOv5 default hyperparameters (hyp.scratch-low.yaml)
# Expected Performance: 15-18% mAP@0.5 (unoptimized)
# 
# This configuration represents UNOPTIMIZED YOLOv5n performance
# to establish the true baseline for measuring optimization improvements.

# ===== LEARNING RATE CONFIGURATION (YOLOv5 DEFAULTS) =====
lr0: 0.01               # Default initial learning rate (NOT optimized)
lrf: 0.01               # Default final learning rate factor
momentum: 0.937         # Default SGD momentum
weight_decay: 0.0005    # Default optimizer weight decay
warmup_epochs: 3.0      # Default warmup epochs (NOT extended)
warmup_momentum: 0.8    # Default warmup initial momentum
warmup_bias_lr: 0.1     # Default warmup initial bias learning rate

# ===== LOSS FUNCTION WEIGHTS (YOLOv5 DEFAULTS) =====
box: 0.05               # Default box loss gain (NOT optimized for small objects)
cls: 0.5                # Default class loss gain (NOT optimized for small objects)
cls_pw: 1.0             # Class BCELoss positive_weight
obj: 1.0                # Default object loss gain (NOT optimized)
obj_pw: 1.0             # Object BCELoss positive_weight
iou_t: 0.2              # Default IoU training threshold (NOT lowered for small objects)
anchor_t: 4.0           # Default anchor-multiple threshold
fl_gamma: 0.0           # Focal loss gamma (disabled - YOLOv5 default)

# ===== AUGMENTATION SETTINGS (YOLOv5 DEFAULTS) =====
# Color augmentation (YOLOv5 defaults - NOT optimized for drone imagery)
hsv_h: 0.015            # Default HSV-Hue augmentation
hsv_s: 0.7              # Default HSV-Saturation augmentation
hsv_v: 0.4              # Default HSV-Value augmentation

# Geometric augmentation (YOLOv5 defaults - NOT optimized for aerial perspective)
degrees: 0.0            # Default image rotation (NO rotation)
translate: 0.1          # Default image translation
scale: 0.5              # Default image scale
shear: 0.0              # Default image shear
perspective: 0.0        # Default image perspective

# Flip augmentation (YOLOv5 defaults)
flipud: 0.0             # Image flip up-down (disabled)
fliplr: 0.5             # Image flip left-right (default)

# Advanced augmentation (YOLOv5 defaults - MINIMAL)
mosaic: 1.0             # Default mosaic augmentation (FULL - standard YOLOv5)
mixup: 0.0              # Default mixup augmentation (DISABLED - YOLOv5 default)
copy_paste: 0.0         # Default copy-paste augmentation (DISABLED)

# ===== TRAINING CONFIGURATION (YOLOv5 DEFAULTS) =====
# These represent unoptimized YOLOv5 training settings
batch_size: 8           # Default YOLOv5 batch size (SMALLER than optimized)
img_size: 416           # Default YOLOv5 image size (LOWER resolution)

# ===== TRUE BASELINE NOTES =====
# This configuration represents RAW YOLOv5n capability:
# 1. No optimization for small objects (higher box/cls loss, higher iou_t)
# 2. No optimization for drone imagery (no rotation, default HSV)
# 3. Minimal augmentation (no mixup, no copy-paste)
# 4. Standard resolution (416px vs optimized 640px)
# 5. Smaller batch size (8 vs optimized 16)
# 6. Higher learning rate (0.01 vs optimized 0.005)

# Expected performance vs optimized:
# - True Baseline: 15-18% mAP@0.5 (this configuration)
# - Optimized Baseline: 23-25% mAP@0.5 (Trial-2 configuration)
# - Improvement potential: +8-10% mAP@0.5 through optimization

# Scientific value:
# - Shows raw YOLOv5n capability without any domain-specific optimization
# - Establishes true baseline for measuring optimization impact
# - Demonstrates full improvement potential from defaults to optimized
# - Provides complete context for thesis analysis

# Created: January 22, 2025
# Purpose: True baseline for scientific comparison and optimization impact measurement
# Based on: YOLOv5 default hyperparameters (hyp.scratch-low.yaml equivalent)
"""
    
    hyp_file = paths['config_dir'] / "hyp_visdrone_true_baseline.yaml"
    with open(hyp_file, 'w') as f:
        f.write(hyp_content)
    
    logging.info(f"[CREATED] True baseline hyperparameters created: {hyp_file}")

def check_gpu_availability():
    """Check GPU availability with detailed reporting"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            # Parse GPU info from nvidia-smi output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'RTX' in line or 'GTX' in line:
                    gpu_info = line.strip()
                    logging.info(f"[GPU] GPU detected: {gpu_info}")
                    break
            logging.info("[SUCCESS] GPU available for true baseline training")
            return "0"
        else:
            logging.warning("[WARNING] GPU not detected, will use CPU (much slower)")
            return "cpu"
    except FileNotFoundError:
        logging.warning("[WARNING] nvidia-smi not found, will use CPU")
        return "cpu"

def create_training_config(paths, epochs, quick_test=False):
    """Create comprehensive TRUE baseline training configuration"""
    config = {
        'experiment': {
            'phase': 'True Baseline - Raw Performance',
            'purpose': 'Establish unoptimized YOLOv5n performance baseline',
            'dataset_type': 'original',
            'augmentation_strategy': 'none',  # CRITICAL: No augmentation for true baseline
            'optimization_level': 'none'
        },
        'performance_context': {
            'true_baseline_expected': '15-18% mAP@0.5',
            'optimized_baseline_expected': '23-25% mAP@0.5', 
            'optimization_potential': '+8-10% mAP@0.5',
            'significance': 'Shows raw model capability and optimization impact'
        },
        'training_params': {
            'epochs': epochs,
            'batch_size': 8,                # YOLOv5n TRUE default (do not change)
            'img_size': 416,                # YOLOv5n TRUE default (do not change)
            'lr0': 0.01,                    # YOLOv5n default learning rate
            'device': check_gpu_availability(),
            'quick_test': quick_test,
            'weights': 'yolov5n.pt',        # Pre-trained weights
            # AUGMENTATION CONTROLS - All disabled for true baseline
            'augment': False,               # Master switch for augmentation
            'hsv_h': 0.0,                   # Disable HSV-Hue augmentation
            'hsv_s': 0.0,                   # Disable HSV-Saturation augmentation
            'hsv_v': 0.0,                   # Disable HSV-Value augmentation
            'degrees': 0.0,                 # Disable rotation
            'translate': 0.0,               # Disable translation
            'scale': 0.0,                   # Disable scaling
            'shear': 0.0,                   # Disable shear
            'perspective': 0.0,             # Disable perspective
            'flipud': 0.0,                  # Disable vertical flip
            'fliplr': 0.0,                  # Disable horizontal flip
            'mosaic': 0.0,                  # CRITICAL: Disable mosaic
            'mixup': 0.0,                   # CRITICAL: Disable mixup
            'copy_paste': 0.0,              # Disable copy-paste
            # TRAINING CONTROLS
            'multi_scale': False,           # DEFAULT: disabled
            'cos_lr': False,                # DEFAULT: disabled
            'cache': False                  # DEFAULT: disabled
        },
        'hyperparameter_comparison': {
            'vs_optimized_baseline': {
                'batch_size': '8 vs 16',
                'img_size': '416 vs 640',
                'lr0': '0.01 vs 0.005',
                'mosaic': '0.0 vs 0.8',
                'mixup': '0.0 vs 0.4',
                'multi_scale': 'False vs True',
                'cos_lr': 'False vs True',
                'box_loss': '0.05 vs 0.03',
                'cls_loss': '0.5 vs 0.3',
                'obj_loss': '1.0 vs 1.2',
                'iou_threshold': '0.2 vs 0.15'
            }
        },
        'scientific_value': [
            'Establishes raw YOLOv5n capability',
            'Uses TRUE YOLOv5n defaults (416px, batch=8)',
            'Disables ALL augmentations for pure performance',
            'Enables direct comparison with YOLOv8n baseline',
            'Provides foundation for measuring improvements'
        ]
    }
    
    # Save config for reproducibility
    config_file = paths['script_dir'] / f"true_baseline_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logging.info(f"[CONFIG] True baseline training configuration saved: {config_file}")
    return config

def run_true_baseline_training(paths, config):
    """Run YOLOv5n TRUE baseline training with default hyperparameters"""
    logging.info("[START] Starting YOLOv5n TRUE BASELINE Training (Raw Performance)")
    logging.info(f"[EXPECTED] Expected Performance: {config['performance_context']['true_baseline_expected']}")
    logging.info(f"[PURPOSE] Purpose: {config['experiment']['purpose']}")
    
    logging.info("[BASELINE] TRUE BASELINE CONFIGURATION:")
    logging.info("  - Using YOLOv5n TRUE defaults (416px, batch=8)")
    logging.info("  - NO synthetic augmentation (fog, night, blur)")
    logging.info("  - NO standard augmentation (mosaic, mixup, HSV, geometric)")
    logging.info("  - Pure original dataset performance")
    logging.info("  - Enables direct comparison with YOLOv8n baseline")
    
    # Clear GPU cache before training to prevent memory issues
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("[GPU] GPU cache cleared")
    except ImportError:
        logging.info("[GPU] PyTorch not available for cache clearing")
    
    # Change to YOLOv5 directory
    original_dir = os.getcwd()
    os.chdir(paths['yolov5_path'])
    
    try:
        # Prepare training arguments with YOLOv5 DEFAULTS
        train_args = [
            sys.executable, "train.py",
            
            # Dataset and model configuration
            "--data", str(paths['config_dir'] / "baseline_dataset_config.yaml"),
            "--weights", config['training_params']['weights'],    # Pre-trained weights
            "--hyp", str(paths['config_dir'] / "hyp_visdrone_true_baseline.yaml"),  # TRUE baseline hyperparameters
            
            # Training parameters (YOLOv5 DEFAULTS - NOT optimized)
            "--epochs", str(config['training_params']['epochs']),
            "--batch-size", str(config['training_params']['batch_size']),        # 8 (default)
            "--imgsz", str(config['training_params']['img_size']),               # 416 (default)
            "--device", config['training_params']['device'],
            
            # NO ADVANCED TRAINING OPTIONS (keep YOLOv5 defaults)
            # --multi-scale: DISABLED (default)
            # --cos-lr: DISABLED (default)
            # --cache: DISABLED (default)
            
            # Output configuration
            "--name", f"yolov5n_true_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "--project", str(paths['project_root'] / "runs" / "train"),
            "--exist-ok",
            
            # Basic settings
            "--workers", "2",               # Reduced workers for stability
            "--save-period", "10",          # Save checkpoint every 10 epochs
        ]
        
        # Add quick test logging
        if config['training_params']['quick_test']:
            logging.info(f"[TEST] Running true baseline validation ({config['training_params']['epochs']} epochs)")
        else:
            logging.info(f"[TRAINING] Running full true baseline training ({config['training_params']['epochs']} epochs)")
        
        # Log complete training command for reproducibility
        logging.info(f"[COMMAND] True baseline training command: {' '.join(train_args)}")
        
        # Log key differences from optimized baseline
        logging.info("[DEFAULTS] Using YOLOv5 default settings:")
        logging.info("  • Batch Size: 8 (vs optimized: 16)")
        logging.info("  • Image Size: 416px (vs optimized: 640px)")
        logging.info("  • Learning Rate: 0.01 (vs optimized: 0.005)")
        logging.info("  • Multi-scale: DISABLED (vs optimized: ENABLED)")
        logging.info("  • Cosine LR: DISABLED (vs optimized: ENABLED)")
        logging.info("  • Mixup: DISABLED (vs optimized: 0.4)")
        logging.info("  • Object Loss: 1.0 (vs optimized: 1.2)")
        logging.info("")
        
        # Start training
        start_time = time.time()
        logging.info("[START] True baseline training started...")
        
        # Run training process with real-time output
        process = subprocess.Popen(
            train_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            universal_newlines=True,
            bufsize=1
        )
        
        # Log output in real-time with progress indicators
        epoch_count = 0
        for line in process.stdout:
            line = line.strip()
            if line:
                # Track epoch progress
                if "Epoch" in line and "/" in line:
                    epoch_count += 1
                    if epoch_count % 5 == 0:  # Log every 5 epochs
                        logging.info(f"[PROGRESS] True Baseline Progress: Epoch {epoch_count}")
                
                # Log all output for debugging
                logging.info(line)
        
        # Wait for completion
        process.wait()
        end_time = time.time()
        
        # Check training success
        if process.returncode == 0:
            training_time = end_time - start_time
            logging.info(f"[SUCCESS] True baseline training completed successfully in {training_time:.2f} seconds")
            logging.info(f"[DURATION] True baseline training duration: {training_time/3600:.2f} hours")
            return True
        else:
            logging.error(f"[ERROR] True baseline training failed with return code: {process.returncode}")
            return False
            
    except Exception as e:
        logging.error(f"[ERROR] True baseline training error: {str(e)}")
        return False
    finally:
        # Return to original directory
        os.chdir(original_dir)

def analyze_true_baseline_results(paths, config):
    """Comprehensive analysis of true baseline results"""
    logging.info("[ANALYSIS] Running comprehensive baseline analysis...")
    
    try:
        # Import evaluation module
        sys.path.append(str(paths['project_root'] / 'src' / 'evaluation'))
        from thesis_metrics import YOLOEvaluationMetrics
        
        # Find best model weights
        best_weights = paths['project_root'] / 'runs' / 'train' / config['training_params']['name'] / 'weights' / 'best.pt'
        if not best_weights.exists():
            logging.error("[ERROR] Best model weights not found")
            return
        
        # Initialize evaluator
        evaluator = YOLOEvaluationMetrics(
            model_path=str(best_weights),
            data_yaml=str(paths['config_dir'] / "baseline_dataset_config.yaml"),
            img_size=config['training_params']['img_size']
        )
        
        # Run comprehensive evaluation
        metrics = evaluator.evaluate_all()
        
        # Log detailed metrics
        logging.info("[METRICS] Comprehensive Evaluation Results:")
        logging.info(f"  mAP@0.5: {metrics['map50']:.3f}")
        logging.info(f"  mAP@0.5:0.95: {metrics['map']:.3f}")
        logging.info(f"  Precision: {metrics['precision']:.3f}")
        logging.info(f"  Recall: {metrics['recall']:.3f}")
        logging.info(f"  F1-Score: {metrics['f1']:.3f}")
        
        # Class-wise performance
        logging.info("[CLASSES] Per-class Performance:")
        for cls_name, cls_metrics in metrics['per_class'].items():
            logging.info(f"  {cls_name}:")
            logging.info(f"    AP: {cls_metrics['ap']:.3f}")
            logging.info(f"    Precision: {cls_metrics['precision']:.3f}")
            logging.info(f"    Recall: {cls_metrics['recall']:.3f}")
        
        # Save detailed metrics
        metrics_file = paths['project_root'] / 'results' / 'thesis_metrics' / f"yolov5n_true_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logging.info(f"[SUCCESS] Detailed metrics saved to: {metrics_file}")
        
    except Exception as e:
        logging.error(f"[ERROR] Failed to analyze results: {str(e)}")
        raise

def main():
    """Main true baseline training function"""
    parser = argparse.ArgumentParser(description='YOLOv5n TRUE Baseline Training (Raw Performance)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--quick-test', action='store_true', help='Run 20-epoch validation test')
    args = parser.parse_args()
    
    # Override epochs for quick test
    if args.quick_test:
        args.epochs = 20
    
    logging.info("[START] YOLOv5n TRUE BASELINE Training Starting")
    logging.info("="*70)
    logging.info("[EXPERIMENT] TRUE BASELINE TRAINING:")
    logging.info("   Purpose: Establish raw YOLOv5n performance without optimization")
    logging.info("   Dataset: Original VisDrone (7,019 images)")
    logging.info("   Hyperparameters: YOLOv5 defaults (unoptimized)")
    logging.info("   Expected: 15-18% mAP@0.5 (raw performance)")
    logging.info("   Scientific Value: Shows optimization impact potential")
    logging.info("="*70)
    
    try:
        # Setup paths
        paths = setup_paths()
        logging.info(f"[PATHS] Project root: {paths['project_root']}")
        
        # Validate environment
        if not validate_environment(paths):
            logging.error("[ERROR] True baseline environment validation failed")
            sys.exit(1)
        
        # Create training configuration
        config = create_training_config(paths, args.epochs, args.quick_test)
        
        # Run true baseline training
        success = run_true_baseline_training(paths, config)
        
        if success:
            # Analyze results
            analyze_true_baseline_results(paths, config)
            logging.info("[SUCCESS] True baseline training completed successfully!")
            logging.info("[CONTEXT] Raw YOLOv5n performance established")
            logging.info("[READY] Ready for optimization impact comparison")
            sys.exit(0)
        else:
            logging.error("[ERROR] True baseline training failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logging.warning("[INTERRUPTED] True baseline training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"[ERROR] Unexpected error in true baseline training: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()