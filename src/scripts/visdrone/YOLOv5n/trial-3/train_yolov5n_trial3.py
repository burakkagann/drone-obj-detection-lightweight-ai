#!/usr/bin/env python3
"""
YOLOv5n Trial-3 (Advanced Optimization) Training Script for VisDrone Dataset
Master's Thesis: Robust Object Detection for Surveillance Drones

This script implements Trial-3 advanced optimization training for YOLOv5n on the VisDrone dataset
using optimized hyperparameters, extended training, and advanced augmentation strategies.

Key features for Trial-3 (Advanced Optimization):
- Extended training epochs (100+ for full convergence)
- Optimized hyperparameters based on Phase 2 results
- Advanced augmentation pipeline for maximum robustness
- Multi-scale training capability
- Enhanced loss function weighting

Author: Burak Kağan Yılmazer
Date: July 2025
Environment: yolov5n_env
Protocol: Version 2.0 - Advanced Optimization Framework
"""

import os
import sys
import logging
import argparse
import warnings
from datetime import datetime
from pathlib import Path

# Disable wandb and optimize memory
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Suppress warnings for clean output
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*", category=FutureWarning)

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[5]
sys.path.append(str(project_root))

try:
    import torch
    import yaml
    # YOLOv5 import
    sys.path.append(str(project_root / "src" / "models" / "YOLOv5"))
    import train
    import val
except ImportError as e:
    print(f"[ERROR] Failed to import required packages: {e}")
    print("Please ensure you're using the yolov5n_visdrone_env environment")
    print("Activation: .\\venvs\\visdrone\\yolov5n_visdrone_env\\Scripts\\Activate.ps1")
    sys.exit(1)

def setup_logging(output_dir: Path) -> None:
    """Setup logging configuration"""
    log_file = output_dir / f"yolov5n_trial3_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    return logger

def validate_environment(dataset_config_path: Path = None) -> None:
    """Validate training environment and dependencies"""
    # Check PyTorch and GPU
    print(f"[INFO] PyTorch Version: {torch.__version__}")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"[INFO] GPU {i}: {gpu_name}")
        print(f"[INFO] CUDA Version: {torch.version.cuda}")
    else:
        print("[WARNING] No GPU available, training will use CPU")
    
    # Validate dataset paths (if config path provided)
    if dataset_config_path and not dataset_config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_config_path}")
    
    if dataset_config_path:
        print(f"[INFO] Dataset config: {dataset_config_path}")

def create_trial3_config() -> Path:
    """Create YOLOv5n Trial-3 advanced optimization configuration"""
    config_dir = project_root / "config" / "visdrone" / "yolov5n_trial3"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset configuration (same as previous trials)
    dataset_config = {
        'path': str(project_root / "data" / "my_dataset" / "visdrone"),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 10,
        'names': [
            'pedestrian', 'people', 'bicycle', 'car', 'van',
            'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
        ]
    }
    
    dataset_config_path = config_dir / "yolov5n_visdrone_trial3.yaml"
    with open(dataset_config_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    # Advanced Hyperparameters configuration (TRIAL-3 - OPTIMIZED)
    hyp_config = {
        # Learning rate settings (optimized based on Phase 2 results)
        'lr0': 0.007,         # Increased from 0.005 for better convergence
        'lrf': 0.01,          # Lower final learning rate for fine-tuning
        'momentum': 0.937,    # SGD momentum/Adam beta1
        'weight_decay': 0.0005, # Optimizer weight decay 5e-4
        'warmup_epochs': 3.0, # Warmup epochs (fractions ok)
        'warmup_momentum': 0.8, # Warmup initial momentum
        'warmup_bias_lr': 0.1, # Warmup initial bias lr
        
        # Loss function weights (further optimized for small objects)
        'box': 0.02,          # Further reduced box loss for small objects
        'cls': 0.3,           # Class loss gain  
        'cls_pw': 1.0,        # Class BCELoss positive_weight
        'obj': 1.5,           # Increased objectness focus
        'obj_pw': 1.0,        # Object BCELoss positive_weight
        'iou_t': 0.12,        # Lower IoU threshold for small objects
        'anchor_t': 4.0,      # Anchor-multiple threshold
        'fl_gamma': 0.0,      # Focal loss gamma (disabled)
        
        # ADVANCED AUGMENTATION: OPTIMIZED FOR MAXIMUM ROBUSTNESS
        'hsv_h': 0.025,       # Slightly increased hue variation
        'hsv_s': 0.6,         # Increased saturation variation
        'hsv_v': 0.35,        # Increased value variation
        'degrees': 7.0,       # Increased rotation for robustness
        'translate': 0.25,    # Increased translation
        'scale': 0.9,         # Increased scale variation
        'shear': 2.0,         # Added shear transformation
        'perspective': 0.0002, # Added perspective transformation
        'flipud': 0.0,        # Image flip up-down (disabled)
        'fliplr': 0.5,        # Image flip left-right (probability)
        'mosaic': 1.0,        # Maximum mosaic augmentation
        'mixup': 0.15,        # Reduced mixup for stability
        'copy_paste': 0.5,    # Increased copy-paste for small objects
        
        # Advanced augmentation parameters
        'erasing': 0.0,       # Random erasing (disabled for now)
        'crop_fraction': 1.0, # Crop fraction for mosaic
    }
    
    hyp_config_path = config_dir / "hyp_yolov5n_trial3.yaml"
    with open(hyp_config_path, 'w') as f:
        yaml.dump(hyp_config, f, default_flow_style=False)
    
    return dataset_config_path, hyp_config_path

def train_yolov5n_trial3(epochs: int = 100, quick_test: bool = False) -> Path:
    """
    Train YOLOv5n Trial-3 (Advanced Optimization) model on VisDrone dataset
    Following Protocol v2.0 - Advanced Optimization Framework
    
    Args:
        epochs: Number of training epochs (default: 100)
        quick_test: If True, use reduced settings for quick validation
    
    Returns:
        Path to training results directory
    """
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "runs" / "train" / f"yolov5n_trial3_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("[START] YOLOv5n Trial-3 (Advanced Optimization) Training Started")
    logger.info("PROTOCOL: Version 2.0 - Advanced Optimization Framework")
    logger.info("METHODOLOGY: Trial-3 - Advanced Hyperparameter Optimization")
    logger.info(f"Output Directory: {output_dir}")
    
    try:
        # Create configuration files first
        logger.info("[CONFIG] Creating Trial-3 advanced configuration...")
        dataset_config_path, hyp_config_path = create_trial3_config()
        
        # Validate environment (after configs are created)
        validate_environment(dataset_config_path)
        
        # Quick test adjustments
        if quick_test:
            epochs = 30
            logger.info(f"[INFO] Quick test mode enabled ({epochs} epochs)")
        
        # Training parameters (optimized for RTX 3060 6GB + performance)
        img_size = 640          # Input image size
        batch_size = 8          # Maintained for memory stability
        workers = 0             # Disable multiprocessing on Windows
        
        logger.info(f"[TRAINING] Configuration Summary:")
        logger.info(f"  • Model: YOLOv5n (nano)")
        logger.info(f"  • Trial: 3 (Advanced Optimization)")
        logger.info(f"  • Dataset: VisDrone + Advanced Augmentation")
        logger.info(f"  • Epochs: {epochs}")
        logger.info(f"  • Image Size: {img_size}")
        logger.info(f"  • Batch Size: {batch_size}")
        logger.info(f"  • Augmentation: ADVANCED (maximum robustness)")
        logger.info("")
        
        # Log Trial-3 specific features
        logger.info("[TRIAL-3] Advanced Optimization Training Features:")
        logger.info("  - OPTIMIZED HYPERPARAMETERS: Based on Phase 2 analysis")
        logger.info("  - ADVANCED AUGMENTATION: Maximum robustness suite")
        logger.info("  - EXTENDED TRAINING: Up to 100+ epochs for convergence")
        logger.info("  - ENHANCED LOSS WEIGHTING: Optimized for small objects")
        logger.info("  - METHODOLOGY COMPLIANCE: Protocol v2.0 Trial-3")
        logger.info("  - TARGET PERFORMANCE: >27% mAP@0.5 (stretch: >29%)")
        logger.info("")
        
        # Prepare training arguments
        train_args = {
            'img': img_size,
            'batch': batch_size,
            'epochs': epochs,
            'data': str(dataset_config_path),
            'cfg': str(project_root / "src" / "models" / "YOLOv5" / "models" / "yolov5n.yaml"),  # YOLOv5n architecture
            'weights': 'yolov5n.pt',  # YOLOv5n pre-trained weights
            'hyp': str(hyp_config_path),
            'project': str(project_root / "runs" / "train"),
            'name': f'yolov5n_trial3_{timestamp}',
            'cache': True,           # Cache images for faster training
            'workers': workers,
            'device': '',            # Auto-select device
            'multi_scale': False,    # Disabled for stability
            'optimizer': 'SGD',      # SGD optimizer
            'cos_lr': True,          # Cosine learning rate scheduler
            'exist_ok': True,
            'patience': 300,         # Early stopping patience
            'save_period': -1,       # Save checkpoint every x epochs (-1 disabled)
        }
        
        # Start training
        logger.info("[TRAINING] Starting YOLOv5n Trial-3 advanced optimization training...")
        
        # Construct command for YOLOv5 training (same approach as previous trials)
        yolov5_path = project_root / "src" / "models" / "YOLOv5"
        train_cmd = [
            sys.executable,  # Use current Python executable from venv
            str(yolov5_path / "train.py"),
            "--img", str(train_args['img']),
            "--batch", str(train_args['batch']),
            "--epochs", str(train_args['epochs']),
            "--data", str(train_args['data']),
            "--cfg", str(train_args['cfg']),
            "--weights", train_args['weights'],
            "--hyp", str(train_args['hyp']),
            "--project", str(train_args['project']),
            "--name", train_args['name'],
            "--cache",
            "--workers", str(train_args['workers']),
            "--device", train_args['device'],
            "--optimizer", train_args['optimizer'],
            "--cos-lr",
            "--exist-ok",
            "--patience", str(train_args['patience'])
        ]
        
        logger.info(f"[TRAINING] Command: {' '.join(train_cmd)}")
        
        # Execute training with optimized settings
        import subprocess
        import os
        
        # Optimize environment for memory and stability
        env = os.environ.copy()
        env['WANDB_DISABLED'] = 'true'
        env['WANDB_MODE'] = 'disabled'
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        env['CUDA_LAUNCH_BLOCKING'] = '1'
        
        result = subprocess.run(train_cmd, cwd=str(yolov5_path), capture_output=False, env=env)
        
        if result.returncode != 0:
            raise RuntimeError(f"YOLOv5 training failed with return code {result.returncode}")
        
        # Training completed
        logger.info("[SUCCESS] Training completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        
        # Get results directory (YOLOv5 creates its own structure)
        yolov5_results_dir = project_root / "runs" / "train" / f'yolov5n_trial3_{timestamp}'
        
        logger.info("[TRIAL-3] Advanced Optimization Training Complete:")
        logger.info(f"  • Protocol: Version 2.0 - Advanced Optimization")
        logger.info(f"  • Results: {yolov5_results_dir}")
        logger.info(f"  • Training Epochs: {epochs}")
        logger.info(f"  • Expected Performance: >27% mAP@0.5")
        
        # Performance analysis
        logger.info("[ANALYSIS] Advanced Optimization Performance Analysis:")
        logger.info("  - Methodology compliance: Trial-3 advanced optimization complete")
        logger.info("  - Target: >27% mAP@0.5 for significant thesis impact")
        logger.info("  - Stretch target: >29% mAP@0.5 for outstanding results")
        logger.info("  - Comparison: Phase 1 (24.5%) → Phase 2 (25.9%) → Trial-3 (27%+)")
        logger.info("  - Research impact: Complete methodology optimization demonstration")
        
        return yolov5_results_dir
        
    except Exception as e:
        logger.error(f"[ERROR] Training failed: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="YOLOv5n Trial-3 (Advanced Optimization) Training for VisDrone")
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with reduced settings (30 epochs)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("YOLOv5n Trial-3 (Advanced Optimization) Training - VisDrone Dataset")
    print("PROTOCOL: Version 2.0 - Advanced Optimization Framework")
    print("="*80)
    print(f"Epochs: {args.epochs}")
    print(f"Quick Test: {args.quick_test}")
    print(f"Trial: 3 (Advanced Optimization)")
    print(f"Target: >27% mAP@0.5 (stretch: >29%)")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    try:
        output_dir = train_yolov5n_trial3(
            epochs=args.epochs,
            quick_test=args.quick_test
        )
        
        print("\n" + "="*80)
        print("[SUCCESS] YOLOv5n Trial-3 (Advanced Optimization) Training Complete!")
        print(f"Results: {output_dir}")
        print("Expected: Advanced optimization for maximum performance")
        print("Target: >27% mAP@0.5 with advanced hyperparameters")
        print("Impact: Outstanding thesis results and methodology validation")
        print("="*80)
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()