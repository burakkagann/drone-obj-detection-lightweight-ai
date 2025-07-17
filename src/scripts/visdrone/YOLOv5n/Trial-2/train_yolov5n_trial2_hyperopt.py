#!/usr/bin/env python3
"""
YOLOv5n Trial-2 Hyperparameter Optimization Training Script
Optimized for VisDrone dataset with research-backed settings

Key Optimizations Applied:
1. Enabled mosaic and mixup augmentation (critical for performance)
2. Increased image resolution from 416 to 640 pixels
3. Reduced learning rate for small object detection
4. Increased batch size for stable gradients
5. Optimized loss function weights for small objects

Expected Improvements:
- Baseline: 17.80% mAP@0.5
- Target: 22-25% mAP@0.5 (+3-5% improvement)
- Success threshold: >18.8% mAP@0.5 (+1% minimum)

Usage:
    python train_yolov5n_trial2_hyperopt.py [--epochs EPOCHS] [--quick-test]
    
    --epochs: Number of epochs to train (default: 100 for full training, 20 for quick test)
    --quick-test: Run 20-epoch validation test first
"""

import argparse
import subprocess
import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trial2_training.log'),
        logging.StreamHandler()
    ]
)

def setup_paths():
    """Set up project paths"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent.parent
    yolov5_path = project_root / "src" / "models" / "YOLOv5"
    
    return {
        'project_root': project_root,
        'yolov5_path': yolov5_path,
        'script_dir': script_dir,
        'config_dir': project_root / "config" / "visdrone" / "yolov5n_v1"
    }

def validate_environment(paths):
    """Validate training environment"""
    logging.info("Validating training environment...")
    
    # Check if YOLOv5 directory exists
    if not paths['yolov5_path'].exists():
        logging.error(f"YOLOv5 directory not found: {paths['yolov5_path']}")
        return False
    
    # Check if hyperparameter file exists
    hyp_file = paths['config_dir'] / "hyp_visdrone_trial-2_optimized.yaml"
    if not hyp_file.exists():
        logging.error(f"Hyperparameter file not found: {hyp_file}")
        return False
    
    # Check if dataset config exists
    dataset_config = paths['config_dir'] / "yolov5n_visdrone_config.yaml"
    if not dataset_config.exists():
        logging.error(f"Dataset config not found: {dataset_config}")
        return False
    
    logging.info("✅ Environment validation passed")
    return True

def check_gpu_availability():
    """Check GPU availability"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            logging.info("✅ GPU detected and available")
            return "0"
        else:
            logging.warning("⚠️ GPU not detected, using CPU")
            return "cpu"
    except FileNotFoundError:
        logging.warning("⚠️ nvidia-smi not found, using CPU")
        return "cpu"

def create_training_config(paths, epochs, quick_test=False):
    """Create training configuration"""
    config = {
        'trial_name': 'trial-2-hyperopt',
        'baseline_performance': {
            'map_50': 17.80,
            'map_50_95': 8.03,
            'precision': 29.77,
            'recall': 17.44,
            'fps': 28.68
        },
        'target_performance': {
            'map_50_min': 18.8,  # +1% minimum
            'map_50_target': 21.0,  # +3% target
            'map_50_excellent': 23.0  # +5% excellent
        },
        'optimizations_applied': [
            'Enabled mosaic augmentation (0.0 → 0.8)',
            'Enabled mixup augmentation (0.0 → 0.4)',
            'Increased image resolution (416 → 640)',
            'Reduced learning rate (0.01 → 0.005)',
            'Increased batch size (8 → 16)',
            'Extended warmup epochs (3.0 → 5.0)',
            'Optimized loss function weights'
        ],
        'training_params': {
            'epochs': epochs,
            'batch_size': 16,
            'img_size': 640,
            'lr0': 0.005,
            'device': check_gpu_availability(),
            'quick_test': quick_test
        }
    }
    
    # Save config for reference
    config_file = paths['script_dir'] / f"trial2_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logging.info(f"📝 Training configuration saved to: {config_file}")
    return config

def run_training(paths, config):
    """Run YOLOv5n training with optimized hyperparameters"""
    logging.info("🚀 Starting YOLOv5n Trial-2 Hyperparameter Optimization Training")
    
    # Change to YOLOv5 directory
    original_dir = os.getcwd()
    os.chdir(paths['yolov5_path'])
    
    try:
        # Prepare training arguments
        train_args = [
            sys.executable, "train.py",
            
            # Dataset and model configuration
            "--data", str(paths['config_dir'] / "yolov5n_visdrone_config.yaml"),
            "--weights", "yolov5n.pt",
            "--hyp", str(paths['config_dir'] / "hyp_visdrone_trial-2_optimized.yaml"),
            
            # Training parameters (optimized for Trial-2)
            "--epochs", str(config['training_params']['epochs']),
            "--batch-size", str(config['training_params']['batch_size']),
            "--imgsz", str(config['training_params']['img_size']),
            "--device", config['training_params']['device'],
            
            # Advanced training options
            "--multi-scale",  # Enable multi-scale training
            "--cos-lr",      # Cosine learning rate scheduling
            
            # Output configuration
            "--name", f"yolov5n_visdrone_trial2_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "--project", str(paths['project_root'] / "runs" / "train"),
            "--exist-ok",
            
            # Metrics and logging
            "--save-period", "5",  # Save checkpoint every 5 epochs
            "--cache", "ram",     # Cache images in RAM for faster training
            "--workers", "4",     # Number of data loader workers
        ]
        
        # Add quick test flag if specified
        if config['training_params']['quick_test']:
            logging.info("🔍 Running quick validation test (20 epochs)")
        else:
            logging.info(f"📚 Running full training ({config['training_params']['epochs']} epochs)")
        
        # Log training command
        logging.info(f"Training command: {' '.join(train_args)}")
        
        # Start training
        start_time = time.time()
        logging.info("⏱️ Training started...")
        
        # Run training process
        process = subprocess.Popen(
            train_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            universal_newlines=True
        )
        
        # Log output in real-time
        for line in process.stdout:
            logging.info(line.strip())
        
        # Wait for completion
        process.wait()
        end_time = time.time()
        
        # Check if training was successful
        if process.returncode == 0:
            training_time = end_time - start_time
            logging.info(f"✅ Training completed successfully in {training_time:.2f} seconds")
            return True
        else:
            logging.error(f"❌ Training failed with return code: {process.returncode}")
            return False
            
    except Exception as e:
        logging.error(f"❌ Training error: {str(e)}")
        return False
    finally:
        # Return to original directory
        os.chdir(original_dir)

def analyze_results(paths, config):
    """Analyze training results"""
    logging.info("📊 Analyzing training results...")
    
    # Look for the latest training results
    runs_dir = paths['project_root'] / "runs" / "train"
    if not runs_dir.exists():
        logging.error("No training results found")
        return
    
    # Find the latest run directory
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and "yolov5n_visdrone_trial2" in d.name]
    if not run_dirs:
        logging.error("No Trial-2 results found")
        return
    
    latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
    logging.info(f"📁 Latest run directory: {latest_run}")
    
    # Check for results.txt
    results_file = latest_run / "results.txt"
    if results_file.exists():
        logging.info("📈 Training results found")
        
        # Read and analyze results
        try:
            with open(results_file, 'r') as f:
                lines = f.readlines()
                
            # Get final epoch results
            if lines:
                final_line = lines[-1].strip()
                values = final_line.split()
                
                if len(values) >= 7:
                    # Extract metrics (assuming standard YOLOv5 format)
                    final_map_50 = float(values[5]) * 100  # Convert to percentage
                    final_map_50_95 = float(values[6]) * 100
                    
                    # Compare with baseline and targets
                    baseline_map_50 = config['baseline_performance']['map_50']
                    improvement = final_map_50 - baseline_map_50
                    
                    logging.info(f"🎯 RESULTS ANALYSIS:")
                    logging.info(f"   Baseline mAP@0.5: {baseline_map_50:.2f}%")
                    logging.info(f"   Trial-2 mAP@0.5: {final_map_50:.2f}%")
                    logging.info(f"   Improvement: {improvement:+.2f}%")
                    
                    # Evaluate success
                    targets = config['target_performance']
                    if final_map_50 >= targets['map_50_excellent']:
                        logging.info("🎉 EXCELLENT PERFORMANCE ACHIEVED!")
                    elif final_map_50 >= targets['map_50_target']:
                        logging.info("✅ TARGET PERFORMANCE ACHIEVED!")
                    elif final_map_50 >= targets['map_50_min']:
                        logging.info("✅ MINIMUM IMPROVEMENT ACHIEVED!")
                    else:
                        logging.warning("⚠️ Performance below minimum threshold")
                    
                    # Recommendations
                    if improvement > 3:
                        logging.info("💡 RECOMMENDATION: Proceed to full 100-epoch training")
                    elif improvement > 1:
                        logging.info("💡 RECOMMENDATION: Consider Phase 2 optimizations")
                    else:
                        logging.info("💡 RECOMMENDATION: Debug and try alternative approaches")
                        
        except Exception as e:
            logging.error(f"Error analyzing results: {str(e)}")
    
    else:
        logging.warning("⚠️ results.txt not found in training directory")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='YOLOv5n Trial-2 Hyperparameter Optimization')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--quick-test', action='store_true', help='Run 20-epoch validation test')
    args = parser.parse_args()
    
    # Override epochs for quick test
    if args.quick_test:
        args.epochs = 20
    
    # Setup paths
    paths = setup_paths()
    
    # Validate environment
    if not validate_environment(paths):
        logging.error("❌ Environment validation failed")
        sys.exit(1)
    
    # Create training configuration
    config = create_training_config(paths, args.epochs, args.quick_test)
    
    # Run training
    success = run_training(paths, config)
    
    if success:
        # Analyze results
        analyze_results(paths, config)
        logging.info("🎉 Trial-2 training completed successfully!")
        sys.exit(0)
    else:
        logging.error("❌ Trial-2 training failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 