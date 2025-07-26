#!/usr/bin/env python3
"""
YOLOv5n Baseline Training Script (Phase 1 - Control Group)
Thesis: "Robust Object Detection for Surveillance Drones in Low-Visibility Environments"

Purpose: Establish baseline performance benchmarks using original VisDrone dataset
Dataset: Original VisDrone (7,019 images) with YOLOv5 real-time augmentation only
Expected Performance: 22-25% mAP@0.5 (based on Trial-2 proven results)

Usage:
    python train_baseline_yolov5n.py [--epochs EPOCHS] [--quick-test]
    
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
        logging.FileHandler('baseline_training.log'),
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
    """Validate baseline training environment"""
    logging.info("🔍 Validating Baseline (Phase 1) training environment...")
    
    # Check if YOLOv5 directory exists
    if not paths['yolov5_path'].exists():
        logging.error(f"❌ YOLOv5 directory not found: {paths['yolov5_path']}")
        return False
    
    # Check if baseline dataset config exists
    dataset_config = paths['config_dir'] / "baseline_dataset_config.yaml"
    if not dataset_config.exists():
        logging.error(f"❌ Baseline dataset config not found: {dataset_config}")
        return False
    
    # Check if hyperparameter file exists (use Trial-2 proven configuration)
    hyp_file = paths['config_dir'] / "hyp_visdrone_trial-2_optimized.yaml"
    if not hyp_file.exists():
        logging.error(f"❌ Hyperparameter file not found: {hyp_file}")
        return False
    
    # Check if baseline dataset actually exists
    dataset_path = paths['project_root'] / "data" / "my_dataset" / "visdrone"
    train_images = dataset_path / "train" / "images"
    train_labels = dataset_path / "train" / "labels"
    val_images = dataset_path / "val" / "images"
    val_labels = dataset_path / "val" / "labels"
    
    for path, desc in [(train_images, "baseline training images"), (train_labels, "baseline training labels"), 
                       (val_images, "baseline validation images"), (val_labels, "baseline validation labels")]:
        if not path.exists():
            logging.error(f"❌ Baseline dataset path missing: {path} ({desc})")
            return False
        
        # Count files
        file_count = len(list(path.glob("*")))
        logging.info(f"✅ Found {file_count} {desc} files")
    
    # Validate expected file counts for baseline dataset
    train_count = len(list(train_images.glob("*.jpg")))
    val_count = len(list(val_images.glob("*.jpg")))
    
    if train_count != 6471:
        logging.warning(f"⚠️  Expected 6,471 training images, found {train_count}")
    if val_count != 548:
        logging.warning(f"⚠️  Expected 548 validation images, found {val_count}")
    
    logging.info(f"📊 Baseline Dataset: {train_count} train, {val_count} val images")
    
    # Validate YOLOv5 pre-trained weights availability
    weights_path = paths['yolov5_path'] / "yolov5n.pt"
    if not weights_path.exists():
        logging.warning(f"⚠️  Pre-trained weights not found at {weights_path}")
        logging.info("🔄 YOLOv5 will download yolov5n.pt automatically")
    else:
        logging.info(f"✅ Pre-trained weights found: {weights_path}")
    
    logging.info("✅ Baseline environment validation passed")
    return True

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
                    logging.info(f"🎮 GPU detected: {gpu_info}")
                    break
            logging.info("✅ GPU available for baseline training")
            return "0"
        else:
            logging.warning("⚠️  GPU not detected, will use CPU (much slower)")
            return "cpu"
    except FileNotFoundError:
        logging.warning("⚠️  nvidia-smi not found, will use CPU")
        return "cpu"

def create_training_config(paths, epochs, quick_test=False):
    """Create comprehensive baseline training configuration"""
    config = {
        'experiment': {
            'phase': 'Phase 1 - Baseline',
            'purpose': 'Control group - establish baseline performance',
            'dataset_type': 'original',
            'augmentation_strategy': 'real_time_only'
        },
        'baseline_targets': {
            'yolov5n_map50_min': 20.0,      # Minimum acceptable performance
            'yolov5n_map50_target': 23.0,   # Target based on Trial-2
            'yolov5n_map50_excellent': 25.0 # Excellent performance
        },
        'training_params': {
            'epochs': epochs,
            'batch_size': 16,               # Trial-2 proven batch size
            'img_size': 640,                # High resolution for small objects
            'lr0': 0.005,                   # Trial-2 proven learning rate
            'device': check_gpu_availability(),
            'quick_test': quick_test,
            'weights': 'yolov5n.pt',        # Pre-trained weights (CRITICAL)
            'multi_scale': True,            # Multi-scale training (CRITICAL)
            'cos_lr': True,                 # Cosine LR scheduling (CRITICAL)
            'cache': 'ram'                  # RAM caching (CRITICAL)
        },
        'augmentation_config': {
            'real_time_only': True,
            'environmental_preprocessing': False,
            'yolov5_standard': {
                'mosaic': 0.8,              # Object diversity
                'mixup': 0.4,               # Decision boundary learning
                'hsv_h': 0.02,              # Hue variation
                'hsv_s': 0.5,               # Saturation variation
                'hsv_v': 0.3,               # Value variation
                'degrees': 5.0,             # Rotation for aerial perspective
                'translate': 0.2,           # Translation variation
                'scale': 0.8,               # Scale variation
                'fliplr': 0.5,              # Horizontal flip
                'copy_paste': 0.3           # Copy-paste augmentation
            }
        },
        'evaluation_metrics': [
            'mAP@0.5',                      # Primary metric
            'mAP@0.5:0.95',                 # COCO-style mAP
            'Precision',                    # Detection precision
            'Recall',                       # Detection recall
            'FPS',                          # Inference speed
            'Model_Size',                   # Parameter count
            'FLOPs'                         # Computational efficiency
        ]
    }
    
    # Save config for reproducibility
    config_file = paths['script_dir'] / f"baseline_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logging.info(f"📋 Baseline training configuration saved: {config_file}")
    return config

def run_baseline_training(paths, config):
    """Run YOLOv5n baseline training (Phase 1)"""
    logging.info("🚀 Starting YOLOv5n Baseline Training (Phase 1)")
    logging.info(f"🎯 Target: {config['baseline_targets']['yolov5n_map50_target']}% mAP@0.5")
    logging.info(f"📊 Dataset: Original VisDrone (real-time augmentation only)")
    
    # Change to YOLOv5 directory
    original_dir = os.getcwd()
    os.chdir(paths['yolov5_path'])
    
    try:
        # Prepare training arguments with all proven parameters
        train_args = [
            sys.executable, "train.py",
            
            # Dataset and model configuration
            "--data", str(paths['config_dir'] / "baseline_dataset_config.yaml"),
            "--weights", config['training_params']['weights'],    # Pre-trained weights
            "--hyp", str(paths['config_dir'] / "hyp_visdrone_trial-2_optimized.yaml"),  # Proven hyperparameters
            
            # Training parameters (proven from Trial-2)
            "--epochs", str(config['training_params']['epochs']),
            "--batch-size", str(config['training_params']['batch_size']),
            "--imgsz", str(config['training_params']['img_size']),
            "--device", config['training_params']['device'],
            
            # Critical training enhancements
            "--multi-scale",                                     # Multi-scale training (CRITICAL)
            "--cos-lr",                                         # Cosine learning rate scheduling (CRITICAL)
            "--cache", config['training_params']['cache'],      # RAM caching (CRITICAL)
            
            # Output configuration
            "--name", f"yolov5n_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "--project", str(paths['project_root'] / "runs" / "train"),
            "--exist-ok",
            
            # Metrics and logging
            "--save-period", "5",           # Save checkpoint every 5 epochs
            "--workers", "4",               # Number of data loader workers
        ]
        
        # Add quick test logging
        if config['training_params']['quick_test']:
            logging.info(f"🔬 Running baseline validation ({config['training_params']['epochs']} epochs)")
        else:
            logging.info(f"🏃 Running full baseline training ({config['training_params']['epochs']} epochs)")
        
        # Log complete training command for reproducibility
        logging.info(f"🔧 Baseline training command: {' '.join(train_args)}")
        
        # Start training with real-time output
        start_time = time.time()
        logging.info("⏰ Baseline training started...")
        
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
                        logging.info(f"📈 Baseline Progress: Epoch {epoch_count}")
                
                # Log all output for debugging
                logging.info(line)
        
        # Wait for completion
        process.wait()
        end_time = time.time()
        
        # Check training success
        if process.returncode == 0:
            training_time = end_time - start_time
            logging.info(f"✅ Baseline training completed successfully in {training_time:.2f} seconds")
            logging.info(f"⏱️  Baseline training duration: {training_time/3600:.2f} hours")
            return True
        else:
            logging.error(f"❌ Baseline training failed with return code: {process.returncode}")
            return False
            
    except Exception as e:
        logging.error(f"💥 Baseline training error: {str(e)}")
        return False
    finally:
        # Return to original directory
        os.chdir(original_dir)

def analyze_baseline_results(paths, config):
    """Comprehensive analysis of baseline results"""
    logging.info("📊 Analyzing Baseline (Phase 1) results...")
    
    # Find the latest baseline training results
    runs_dir = paths['project_root'] / "runs" / "train"
    if not runs_dir.exists():
        logging.error("❌ No training results found")
        return
    
    # Find the latest baseline run
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and "yolov5n_baseline" in d.name]
    if not run_dirs:
        logging.error("❌ No baseline results found")
        return
    
    latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
    logging.info(f"📁 Latest baseline run directory: {latest_run}")
    
    # Analyze results.csv if available
    results_csv = latest_run / "results.csv"
    if results_csv.exists():
        logging.info("📈 Analyzing baseline results.csv...")
        
        try:
            import pandas as pd
            df = pd.read_csv(results_csv)
            
            # Get final epoch results
            final_row = df.iloc[-1]
            final_map_50 = final_row['metrics/mAP_0.5'] * 100  # Convert to percentage
            final_map_50_95 = final_row['metrics/mAP_0.5:0.95'] * 100
            final_precision = final_row['metrics/precision'] * 100
            final_recall = final_row['metrics/recall'] * 100
            
            # Compare with baseline targets
            targets = config['baseline_targets']
            
            logging.info("🎯 BASELINE (PHASE 1) RESULTS ANALYSIS:")
            logging.info("="*60)
            logging.info(f"📊 Final mAP@0.5: {final_map_50:.3f}%")
            logging.info(f"📊 Final mAP@0.5:0.95: {final_map_50_95:.3f}%")
            logging.info(f"📊 Final Precision: {final_precision:.3f}%")
            logging.info(f"📊 Final Recall: {final_recall:.3f}%")
            logging.info("")
            logging.info("📈 BASELINE PERFORMANCE EVALUATION:")
            logging.info(f"   Minimum Target: {targets['yolov5n_map50_min']:.1f}% mAP@0.5")
            logging.info(f"   Expected Target: {targets['yolov5n_map50_target']:.1f}% mAP@0.5")
            logging.info(f"   Excellence Target: {targets['yolov5n_map50_excellent']:.1f}% mAP@0.5")
            logging.info(f"   Achieved Performance: {final_map_50:.3f}% mAP@0.5")
            logging.info("")
            
            # Evaluate baseline success
            if final_map_50 >= targets['yolov5n_map50_excellent']:
                logging.info("🏆 EXCELLENT! Outstanding baseline performance!")
                logging.info("✅ Recommend: Proceed immediately to Phase 2 (Environmental)")
            elif final_map_50 >= targets['yolov5n_map50_target']:
                logging.info("🎯 SUCCESS! Target baseline performance achieved!")
                logging.info("✅ Recommend: Proceed to Phase 2 (Environmental)")
            elif final_map_50 >= targets['yolov5n_map50_min']:
                logging.info("✅ ACCEPTABLE! Minimum baseline performance achieved!")
                logging.info("💡 Recommend: Proceed to Phase 2 with baseline documented")
            else:
                logging.warning("⚠️  CAUTION: Baseline performance below minimum target")
                logging.info("🔍 Recommend: Debug training configuration before Phase 2")
            
            # Training efficiency analysis
            total_epochs = len(df)
            logging.info(f"⏱️  Baseline training completed in {total_epochs} epochs")
            
            # Check for early stopping
            if total_epochs < config['training_params']['epochs']:
                logging.info("🛑 Training stopped early (patience triggered)")
            
            # Save baseline performance for Phase 2 comparison
            baseline_results = {
                'phase': 'Phase 1 - Baseline',
                'map_50': final_map_50,
                'map_50_95': final_map_50_95,
                'precision': final_precision,
                'recall': final_recall,
                'total_epochs': total_epochs,
                'training_date': datetime.now().isoformat(),
                'dataset_type': 'original_visdrone',
                'augmentation': 'real_time_only'
            }
            
            # Save for Phase 2 comparison
            baseline_file = paths['script_dir'] / f"baseline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(baseline_file, 'w') as f:
                json.dump(baseline_results, f, indent=2)
            
            logging.info(f"💾 Baseline results saved for Phase 2 comparison: {baseline_file}")
            
        except Exception as e:
            logging.error(f"❌ Error analyzing baseline results: {str(e)}")
            logging.info("📊 Please check results manually in the run directory")
    
    else:
        logging.warning("⚠️  results.csv not found - checking for other result files")
        
        # List available files for manual analysis
        result_files = list(latest_run.glob("*"))
        logging.info(f"📁 Available result files: {[f.name for f in result_files]}")

def main():
    """Main baseline training function"""
    parser = argparse.ArgumentParser(description='YOLOv5n Baseline Training (Phase 1)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--quick-test', action='store_true', help='Run 20-epoch validation test')
    args = parser.parse_args()
    
    # Override epochs for quick test
    if args.quick_test:
        args.epochs = 20
    
    logging.info("🚀 YOLOv5n Baseline Training (Phase 1) Starting")
    logging.info("="*70)
    logging.info("📋 EXPERIMENTAL PROTOCOL:")
    logging.info("   Phase: 1 - Baseline (Control Group)")
    logging.info("   Dataset: Original VisDrone (7,019 images)")
    logging.info("   Augmentation: YOLOv5 real-time only")
    logging.info("   Purpose: Establish baseline performance benchmarks")
    logging.info("="*70)
    
    try:
        # Setup paths
        paths = setup_paths()
        logging.info(f"📁 Project root: {paths['project_root']}")
        
        # Validate environment
        if not validate_environment(paths):
            logging.error("❌ Baseline environment validation failed")
            sys.exit(1)
        
        # Create training configuration
        config = create_training_config(paths, args.epochs, args.quick_test)
        
        # Run baseline training
        success = run_baseline_training(paths, config)
        
        if success:
            # Analyze results
            analyze_baseline_results(paths, config)
            logging.info("🎉 Baseline training (Phase 1) completed successfully!")
            logging.info("📊 Results documented for Phase 2 comparison")
            logging.info("🔄 Ready to proceed to Phase 2 (Environmental)")
            sys.exit(0)
        else:
            logging.error("💥 Baseline training (Phase 1) failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logging.warning("⚠️  Baseline training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"💥 Unexpected error in baseline training: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()