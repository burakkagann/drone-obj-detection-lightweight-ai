#!/usr/bin/env python3
"""
YOLOv5n Trial-4 Conservative Training Script
Based on proven Trial-2 architecture with conservative hyperparameter modifications

Key Features:
1. Robust Python wrapper architecture (proven from Trial-2)
2. Pre-trained weight initialization (yolov5n.pt)
3. Multi-scale training and cosine LR scheduling
4. Comprehensive validation and error handling
5. Warning suppression for clean output
6. Conservative modifications: obj loss (1.2→1.25), batch size (16→16)

Expected Performance:
- Baseline: Trial-2 achieved 23.557% mAP@0.5
- Target: 24-25% mAP@0.5 (+0.5-1.5% conservative improvement)
- Minimum: >23.8% mAP@0.5 (+0.2% improvement)

Usage:
    python train_yolov5n_trial4.py [--epochs EPOCHS] [--quick-test]
    
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

# Suppress torch.cuda.amp.autocast deprecation warning
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*", category=UserWarning)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trial4_training.log'),
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
    """Validate training environment with enhanced checks"""
    logging.info("[VALIDATION] Validating Trial-4 training environment...")
    
    # Check if YOLOv5 directory exists
    if not paths['yolov5_path'].exists():
        logging.error(f"[ERROR] YOLOv5 directory not found: {paths['yolov5_path']}")
        return False
    
    # Check if hyperparameter file exists (Trial-4 specific)
    hyp_file = paths['config_dir'] / "hyp_visdrone_trial4.yaml"
    if not hyp_file.exists():
        logging.error(f"[ERROR] Trial-4 hyperparameter file not found: {hyp_file}")
        return False
    
    # Check if dataset config exists
    dataset_config = paths['config_dir'] / "yolov5n_visdrone_config.yaml"
    if not dataset_config.exists():
        logging.error(f"[ERROR] Dataset config not found: {dataset_config}")
        return False
    
    # Check if dataset actually exists
    dataset_path = paths['project_root'] / "data" / "my_dataset" / "visdrone"
    train_images = dataset_path / "train" / "images"
    train_labels = dataset_path / "train" / "labels"
    val_images = dataset_path / "val" / "images"
    val_labels = dataset_path / "val" / "labels"
    
    for path, desc in [(train_images, "training images"), (train_labels, "training labels"), 
                       (val_images, "validation images"), (val_labels, "validation labels")]:
        if not path.exists():
            logging.error(f"[ERROR] Dataset path missing: {path} ({desc})")
            return False
        
        # Count files
        file_count = len(list(path.glob("*")))
        logging.info(f"[SUCCESS] Found {file_count} {desc} files")
    
    # Validate YOLOv5 pre-trained weights availability
    weights_path = paths['yolov5_path'] / "yolov5n.pt"
    if not weights_path.exists():
        logging.warning(f"[WARNING] Pre-trained weights not found at {weights_path}")
        logging.info("[INFO] YOLOv5 will download yolov5n.pt automatically")
    else:
        logging.info(f"[SUCCESS] Pre-trained weights found: {weights_path}")
    
    logging.info("[SUCCESS] Environment validation passed")
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
                    logging.info(f"[GPU] GPU detected: {gpu_info}")
                    break
            logging.info("[SUCCESS] GPU available for training")
            return "0"
        else:
            logging.warning("[WARNING] GPU not detected, will use CPU (much slower)")
            return "cpu"
    except FileNotFoundError:
        logging.warning("[WARNING] nvidia-smi not found, will use CPU")
        return "cpu"

def create_training_config(paths, epochs, quick_test=False):
    """Create comprehensive training configuration for Trial-4"""
    config = {
        'trial_name': 'trial-4-conservative',
        'trial_version': '4.0',
        'baseline_performance': {
            'trial_2_map_50': 23.557,  # Trial-2 proven baseline
            'trial_2_map_50_95': 10.97,
            'trial_3_map_50': 0.002,   # Trial-3 catastrophic failure
        },
        'target_performance': {
            'map_50_min': 23.8,        # +0.2% minimum improvement
            'map_50_target': 24.0,     # +0.5% target improvement
            'map_50_excellent': 24.5   # +1.0% excellent improvement
        },
        'modifications_from_trial2': [
            'Object loss: 1.2 → 1.25 (+4.2% increase)',
            'Batch size: 16 → 16 (kept same - Trial-2 proven)',
            'All other parameters: IDENTICAL to Trial-2',
            'Pre-trained weights: ENABLED (critical for performance)',
            'Multi-scale training: ENABLED (critical for performance)',
            'Cosine LR scheduling: ENABLED (critical for performance)'
        ],
        'training_params': {
            'epochs': epochs,
            'batch_size': 16,  # Restored to Trial-2 proven setting
            'img_size': 640,
            'lr0': 0.005,      # Keep Trial-2 learning rate
            'device': check_gpu_availability(),
            'quick_test': quick_test,
            'weights': 'yolov5n.pt',  # Critical: pre-trained weights
            'multi_scale': True,       # Critical: multi-scale training
            'cos_lr': True,           # Critical: cosine LR scheduling
            'cache': 'ram'            # Critical: RAM caching
        },
        'risk_mitigation': [
            'Conservative modifications only (2 parameters changed)',
            'Based on proven Trial-2 architecture',
            'Comprehensive validation before training',
            'Real-time monitoring and logging',
            'Automatic results analysis and comparison'
        ]
    }
    
    # Save config for reproducibility
    config_file = paths['script_dir'] / f"trial4_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logging.info(f"[CONFIG] Training configuration saved: {config_file}")
    return config

def handle_memory_error_and_retry(paths, config, error_msg):
    """Handle CUDA memory errors with automatic batch size reduction"""
    current_batch = config['training_params']['batch_size']
    
    if "CUDA out of memory" in str(error_msg) or "cuDNN" in str(error_msg):
        if current_batch >= 14:
            new_batch = 14
            logging.warning(f"[MEMORY] CUDA memory error detected. Reducing batch size: {current_batch} → {new_batch}")
        elif current_batch >= 12:
            new_batch = 12
            logging.warning(f"[MEMORY] CUDA memory error detected. Reducing batch size: {current_batch} → {new_batch}")
        else:
            logging.error(f"[ERROR] CUDA memory error with batch size {current_batch}. Cannot reduce further.")
            return None
        
        # Update config with new batch size
        config['training_params']['batch_size'] = new_batch
        logging.info(f"[RETRY] Retrying training with batch size {new_batch}")
        
        return new_batch
    
    return None

def run_training(paths, config):
    """Run YOLOv5n Trial-4 training with robust error handling and memory management"""
    logging.info("[START] Starting YOLOv5n Trial-4 Conservative Training")
    logging.info(f"[TARGET] Target: {config['target_performance']['map_50_target']}% mAP@0.5")
    logging.info(f"[BASELINE] Baseline: {config['baseline_performance']['trial_2_map_50']}% mAP@0.5 (Trial-2)")
    
    # Clear GPU cache before training to prevent memory issues
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logging.info("[GPU] GPU cache cleared to prevent memory issues")
    except ImportError:
        logging.info("[GPU] PyTorch not available for cache clearing")
    
    # Change to YOLOv5 directory
    original_dir = os.getcwd()
    os.chdir(paths['yolov5_path'])
    
    try:
        # Prepare training arguments with all Trial-2 proven parameters
        train_args = [
            sys.executable, "train.py",
            
            # Dataset and model configuration
            "--data", str(paths['config_dir'] / "yolov5n_visdrone_config.yaml"),
            "--weights", config['training_params']['weights'],  # Pre-trained weights
            "--hyp", str(paths['config_dir'] / "hyp_visdrone_trial4.yaml"),
            
            # Training parameters (Trial-4 conservative modifications)
            "--epochs", str(config['training_params']['epochs']),
            "--batch-size", str(config['training_params']['batch_size']),
            "--imgsz", str(config['training_params']['img_size']),
            "--device", config['training_params']['device'],
            
            # Critical training enhancements (from Trial-2)
            "--multi-scale",    # Multi-scale training (CRITICAL)
            "--cos-lr",         # Cosine learning rate scheduling (CRITICAL)
            "--cache", config['training_params']['cache'],  # RAM caching (CRITICAL)
            
            # Output configuration
            "--name", f"yolov5n_trial4_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "--project", str(paths['project_root'] / "runs" / "train"),
            "--exist-ok",
            
            # Memory optimization
            "--workers", "2",       # Reduced workers to save memory
            "--save-period", "10",  # Reduced checkpoint frequency to save memory
            
            # Training optimization
            "--patience", "20",     # Early stopping patience
        ]
        
        # Add quick test logging
        if config['training_params']['quick_test']:
            logging.info(f"[TEST] Running validation test ({config['training_params']['epochs']} epochs)")
        else:
            logging.info(f"[TRAINING] Running full training ({config['training_params']['epochs']} epochs)")
        
        # Log complete training command for reproducibility
        logging.info(f"[COMMAND] Training command: {' '.join(train_args)}")
        
        # Start training with memory error handling
        max_retries = 2
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                start_time = time.time()
                logging.info(f"[START] Training started (attempt {retry_count + 1}/{max_retries + 1})...")
                if retry_count > 0:
                    logging.info(f"[RETRY] Using batch size: {config['training_params']['batch_size']}")
                
                # Update batch size in training arguments if retrying
                if retry_count > 0:
                    # Find and update batch-size argument
                    for i, arg in enumerate(train_args):
                        if arg == "--batch-size":
                            train_args[i + 1] = str(config['training_params']['batch_size'])
                            break
                
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
                training_output = []
                
                for line in process.stdout:
                    line = line.strip()
                    if line:
                        training_output.append(line)
                        
                        # Track epoch progress
                        if "Epoch" in line and "/" in line:
                            epoch_count += 1
                            if epoch_count % 5 == 0:  # Log every 5 epochs
                                logging.info(f"[PROGRESS] Progress: Epoch {epoch_count}")
                        
                        # Check for memory errors early
                        if "CUDA out of memory" in line or "cuDNN error" in line:
                            logging.warning(f"[MEMORY] Memory error detected: {line}")
                        
                        # Log all output for debugging
                        logging.info(line)
                
                # Wait for completion
                process.wait()
                end_time = time.time()
                
                # Check training success
                if process.returncode == 0:
                    training_time = end_time - start_time
                    logging.info(f"[SUCCESS] Training completed successfully in {training_time:.2f} seconds")
                    logging.info(f"[DURATION] Training duration: {training_time/3600:.2f} hours")
                    return True
                else:
                    # Check if it's a memory error
                    error_output = '\n'.join(training_output[-10:])  # Last 10 lines
                    new_batch = handle_memory_error_and_retry(paths, config, error_output)
                    
                    if new_batch is not None and retry_count < max_retries:
                        retry_count += 1
                        logging.info(f"[RETRY] Attempting retry {retry_count}/{max_retries} with batch size {new_batch}")
                        continue
                    else:
                        logging.error(f"[ERROR] Training failed with return code: {process.returncode}")
                        return False
                        
            except Exception as e:
                error_msg = str(e)
                new_batch = handle_memory_error_and_retry(paths, config, error_msg)
                
                if new_batch is not None and retry_count < max_retries:
                    retry_count += 1
                    logging.info(f"[RETRY] Attempting retry {retry_count}/{max_retries} with batch size {new_batch}")
                    continue
                else:
                    logging.error(f"[ERROR] Training exception: {error_msg}")
                    return False
        
        logging.error("[ERROR] All training attempts failed")
        return False
            
    except Exception as e:
        logging.error(f"[ERROR] Training error: {str(e)}")
        return False
    finally:
        # Return to original directory
        os.chdir(original_dir)

def analyze_results(paths, config):
    """Comprehensive analysis of Trial-4 results with Trial-2 comparison"""
    logging.info("[ANALYSIS] Analyzing Trial-4 results...")
    
    # Find the latest training results
    runs_dir = paths['project_root'] / "runs" / "train"
    if not runs_dir.exists():
        logging.error("[ERROR] No training results found")
        return
    
    # Find the latest Trial-4 run
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and "yolov5n_trial4" in d.name]
    if not run_dirs:
        logging.error("[ERROR] No Trial-4 results found")
        return
    
    latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
    logging.info(f"[RESULTS] Latest run directory: {latest_run}")
    
    # Analyze results.csv if available
    results_csv = latest_run / "results.csv"
    if results_csv.exists():
        logging.info("[ANALYSIS] Analyzing results.csv...")
        
        try:
            import pandas as pd
            df = pd.read_csv(results_csv)
            
            # Get final epoch results
            final_row = df.iloc[-1]
            final_map_50 = final_row['metrics/mAP_0.5'] * 100  # Convert to percentage
            final_map_50_95 = final_row['metrics/mAP_0.5:0.95'] * 100
            final_precision = final_row['metrics/precision'] * 100
            final_recall = final_row['metrics/recall'] * 100
            
            # Compare with baselines
            trial_2_baseline = config['baseline_performance']['trial_2_map_50']
            improvement = final_map_50 - trial_2_baseline
            
            logging.info("[RESULTS] TRIAL-4 RESULTS ANALYSIS:")
            logging.info("="*50)
            logging.info(f"[METRICS] Final mAP@0.5: {final_map_50:.3f}%")
            logging.info(f"[METRICS] Final mAP@0.5:0.95: {final_map_50_95:.3f}%")
            logging.info(f"[METRICS] Final Precision: {final_precision:.3f}%")
            logging.info(f"[METRICS] Final Recall: {final_recall:.3f}%")
            logging.info("")
            logging.info("[COMPARISON] COMPARISON WITH BASELINES:")
            logging.info(f"   Trial-2 Baseline: {trial_2_baseline:.3f}% mAP@0.5")
            logging.info(f"   Trial-4 Result: {final_map_50:.3f}% mAP@0.5")
            logging.info(f"   Improvement: {improvement:+.3f}%")
            logging.info("")
            
            # Evaluate success against targets
            targets = config['target_performance']
            if final_map_50 >= targets['map_50_excellent']:
                logging.info("[EXCELLENT] EXCELLENT! Outstanding performance achieved!")
                logging.info("[RECOMMEND] Recommend: Proceed with multi-model comparison")
            elif final_map_50 >= targets['map_50_target']:
                logging.info("[SUCCESS] SUCCESS! Target performance achieved!")
                logging.info("[RECOMMEND] Recommend: Consider Trial-5 with additional optimizations")
            elif final_map_50 >= targets['map_50_min']:
                logging.info("[SUCCESS] SUCCESS! Minimum improvement achieved!")
                logging.info("[RECOMMEND] Recommend: Analyze for further optimization opportunities")
            elif final_map_50 >= trial_2_baseline:
                logging.info("[POSITIVE] POSITIVE: Performance maintained or slightly improved")
                logging.info("[RECOMMEND] Recommend: Conservative approach validated")
            else:
                logging.warning("[WARNING] CAUTION: Performance below Trial-2 baseline")
                logging.info("[RECOMMEND] Recommend: Debug configuration differences")
            
            # Training efficiency analysis
            total_epochs = len(df)
            logging.info(f"[DURATION] Training completed in {total_epochs} epochs")
            
            # Check for early stopping
            if total_epochs < config['training_params']['epochs']:
                logging.info("[STOPPED] Training stopped early (patience triggered)")
            
        except Exception as e:
            logging.error(f"[ERROR] Error analyzing results: {str(e)}")
            logging.info("[INFO] Please check results manually in the run directory")
    
    else:
        logging.warning("[WARNING] results.csv not found - checking for other result files")
        
        # List available files for manual analysis
        result_files = list(latest_run.glob("*"))
        logging.info(f"[FILES] Available result files: {[f.name for f in result_files]}")

def main():
    """Main training function with comprehensive error handling"""
    parser = argparse.ArgumentParser(description='YOLOv5n Trial-4 Conservative Training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--quick-test', action='store_true', help='Run 20-epoch validation test')
    args = parser.parse_args()
    
    # Override epochs for quick test
    if args.quick_test:
        args.epochs = 20
    
    logging.info("[START] YOLOv5n Trial-4 Conservative Training Starting")
    logging.info("="*60)
    
    try:
        # Setup paths
        paths = setup_paths()
        logging.info(f"[PATHS] Project root: {paths['project_root']}")
        
        # Validate environment
        if not validate_environment(paths):
            logging.error("[ERROR] Environment validation failed")
            sys.exit(1)
        
        # Create training configuration
        config = create_training_config(paths, args.epochs, args.quick_test)
        
        # Run training
        success = run_training(paths, config)
        
        if success:
            # Analyze results
            analyze_results(paths, config)
            logging.info("[SUCCESS] Trial-4 training completed successfully!")
            logging.info("[INFO] Check the analysis above for performance comparison")
            sys.exit(0)
        else:
            logging.error("[ERROR] Trial-4 training failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logging.warning("[INTERRUPTED] Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"[ERROR] Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()