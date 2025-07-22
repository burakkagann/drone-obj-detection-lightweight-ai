#!/usr/bin/env python3
"""
YOLOv5n TRUE BASELINE Training Script (Raw Performance)
Thesis: "Robust Object Detection for Surveillance Drones in Low-Visibility Environments"

Purpose: Establish TRUE baseline using YOLOv5n default hyperparameters
Dataset: Original VisDrone (7,019 images) with YOLOv5 DEFAULT augmentation settings
Expected Performance: 15-18% mAP@0.5 (unoptimized raw performance)

This script demonstrates RAW YOLOv5n capability before any optimization,
providing the foundation for measuring optimization improvements.

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
            'augmentation_strategy': 'yolov5_defaults_minimal',
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
            'batch_size': 8,                # YOLOv5 default (smaller than optimized)
            'img_size': 416,                # YOLOv5 default (lower than optimized)
            'lr0': 0.01,                    # YOLOv5 default (higher than optimized)
            'device': check_gpu_availability(),
            'quick_test': quick_test,
            'weights': 'yolov5n.pt',        # Pre-trained weights
            'multi_scale': False,           # DEFAULT: disabled (vs optimized: enabled)
            'cos_lr': False,                # DEFAULT: disabled (vs optimized: enabled)
            'cache': False                  # DEFAULT: disabled (vs optimized: enabled)
        },
        'hyperparameter_comparison': {
            'vs_optimized_baseline': {
                'batch_size': '8 vs 16',
                'img_size': '416 vs 640',
                'lr0': '0.01 vs 0.005',
                'mosaic': '1.0 vs 0.8',
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
            'Measures optimization impact quantitatively',
            'Provides complete baseline context',
            'Demonstrates improvement potential',
            'Enables fair comparison with literature'
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
    """Comprehensive analysis of TRUE baseline results"""
    logging.info("[ANALYSIS] Analyzing TRUE BASELINE results...")
    
    # Find the latest true baseline training results
    runs_dir = paths['project_root'] / "runs" / "train"
    if not runs_dir.exists():
        logging.error("[ERROR] No training results found")
        return
    
    # Find the latest true baseline run
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and "yolov5n_true_baseline" in d.name]
    if not run_dirs:
        logging.error("[ERROR] No true baseline results found")
        return
    
    latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
    logging.info(f"[RESULTS] Latest true baseline run directory: {latest_run}")
    
    # Analyze results.csv if available
    results_csv = latest_run / "results.csv"
    if results_csv.exists():
        logging.info("[ANALYSIS] Analyzing true baseline results.csv...")
        
        try:
            import pandas as pd
            df = pd.read_csv(results_csv)
            
            # Get final epoch results
            final_row = df.iloc[-1]
            final_map_50 = final_row['metrics/mAP_0.5'] * 100  # Convert to percentage
            final_map_50_95 = final_row['metrics/mAP_0.5:0.95'] * 100
            final_precision = final_row['metrics/precision'] * 100
            final_recall = final_row['metrics/recall'] * 100
            
            # Compare with expectations
            expected_range = config['performance_context']['true_baseline_expected']
            optimization_potential = config['performance_context']['optimization_potential']
            
            logging.info("[RESULTS] TRUE BASELINE RESULTS ANALYSIS:")
            logging.info("="*60)
            logging.info(f"[METRICS] Final mAP@0.5: {final_map_50:.3f}%")
            logging.info(f"[METRICS] Final mAP@0.5:0.95: {final_map_50_95:.3f}%")
            logging.info(f"[METRICS] Final Precision: {final_precision:.3f}%")
            logging.info(f"[METRICS] Final Recall: {final_recall:.3f}%")
            logging.info("")
            logging.info("[BASELINE] TRUE BASELINE PERFORMANCE CONTEXT:")
            logging.info(f"   Expected Range: {expected_range}")
            logging.info(f"   Achieved Performance: {final_map_50:.3f}% mAP@0.5")
            logging.info(f"   Optimization Potential: {optimization_potential}")
            logging.info("")
            
            # Evaluate baseline success
            if 15.0 <= final_map_50 <= 18.0:
                logging.info("[SUCCESS] EXCELLENT! Performance within expected true baseline range!")
                logging.info("[ANALYSIS] This establishes a clear baseline for optimization impact measurement")
            elif final_map_50 < 15.0:
                logging.warning("[WARNING] Performance below expected true baseline range")
                logging.info("[ANALYSIS] May indicate training issues or dataset challenges")
            elif final_map_50 > 18.0:
                logging.info("[HIGHER] Performance above expected true baseline range")
                logging.info("[ANALYSIS] YOLOv5n defaults perform better than expected on VisDrone")
            
            # Calculate optimization potential
            if final_map_50 > 0:
                # Assume optimized baseline will achieve ~23-25%
                estimated_optimized = 24.0  # Conservative estimate
                optimization_gap = estimated_optimized - final_map_50
                logging.info(f"[POTENTIAL] Estimated optimization improvement: +{optimization_gap:.1f}% mAP@0.5")
                logging.info(f"[POTENTIAL] Relative improvement: +{(optimization_gap/final_map_50)*100:.1f}%")
            
            # Training efficiency analysis
            total_epochs = len(df)
            logging.info(f"[DURATION] True baseline training completed in {total_epochs} epochs")
            
            # Check for early stopping
            if total_epochs < config['training_params']['epochs']:
                logging.info("[STOPPED] Training stopped early (patience triggered)")
            
            # Save true baseline performance for comparison
            true_baseline_results = {
                'experiment': 'True Baseline - Raw Performance',
                'map_50': final_map_50,
                'map_50_95': final_map_50_95,
                'precision': final_precision,
                'recall': final_recall,
                'total_epochs': total_epochs,
                'training_date': datetime.now().isoformat(),
                'dataset_type': 'original_visdrone',
                'augmentation': 'yolov5_defaults_minimal',
                'hyperparameters': 'yolov5_defaults',
                'batch_size': config['training_params']['batch_size'],
                'img_size': config['training_params']['img_size'],
                'learning_rate': config['training_params']['lr0'],
                'scientific_significance': 'Establishes raw model capability baseline'
            }
            
            # Save for optimization comparison
            baseline_file = paths['script_dir'] / f"true_baseline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(baseline_file, 'w') as f:
                json.dump(true_baseline_results, f, indent=2)
            
            logging.info(f"[SAVED] True baseline results saved for optimization comparison: {baseline_file}")
            
            # Next steps recommendations
            logging.info("")
            logging.info("[NEXT STEPS] Recommendations:")
            logging.info("  1. Run optimized baseline (Trial-2 hyperparameters on original data)")
            logging.info("  2. Compare true baseline vs optimized baseline")
            logging.info("  3. Run environmental augmentation experiments")
            logging.info("  4. Measure total improvement: true baseline → environmental")
            logging.info("")
            
        except Exception as e:
            logging.error(f"[ERROR] Error analyzing true baseline results: {str(e)}")
            logging.info("[INFO] Please check results manually in the run directory")
    
    else:
        logging.warning("[WARNING] results.csv not found - checking for other result files")
        
        # List available files for manual analysis
        result_files = list(latest_run.glob("*"))
        logging.info(f"[FILES] Available result files: {[f.name for f in result_files]}")

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