#!/usr/bin/env python3
"""
YOLOv5n Baseline Training Script for VisDrone Dataset
Master's Thesis: Robust Object Detection for Surveillance Drones

This script implements Phase 1 (True Baseline) training for YOLOv5n on the VisDrone dataset
using NO augmentation to establish absolute model performance reference point.

Key features for Phase 1 (True Baseline):
- Original VisDrone dataset only (no synthetic augmentation)
- NO real-time augmentation (all disabled)
- Minimal preprocessing (resize, normalize only)
- Pure model capability measurement

Author: Burak Kağan Yılmazer
Date: July 2025
Environment: yolov5n_env
Protocol: Version 2.0 - True Baseline Framework
"""

import os
import sys
import logging
import argparse
import warnings
from datetime import datetime
from pathlib import Path

# Disable wandb completely
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'

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
    print("Please ensure you're using the yolov5n_env environment")
    print("Activation: .\\venvs\\yolov5n_env\\Scripts\\Activate.ps1")
    sys.exit(1)

def setup_logging(output_dir: Path) -> None:
    """Setup logging configuration"""
    log_file = output_dir / f"yolov5n_baseline_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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

def create_baseline_config() -> Path:
    """Create YOLOv5n baseline configuration following Protocol v2.0"""
    config_dir = project_root / "config" / "visdrone" / "yolov5n_baseline"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset configuration
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
    
    dataset_config_path = config_dir / "yolov5n_visdrone_baseline.yaml"
    with open(dataset_config_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    # Hyperparameters configuration (TRUE BASELINE - NO AUGMENTATION)
    hyp_config = {
        # Learning rate settings
        'lr0': 0.01,          # Initial learning rate (SGD=1E-2, Adam=1E-3)
        'lrf': 0.01,          # Final learning rate (lr0 * lrf)
        'momentum': 0.937,    # SGD momentum/Adam beta1
        'weight_decay': 0.0005, # Optimizer weight decay 5e-4
        'warmup_epochs': 3.0, # Warmup epochs (fractions ok)
        'warmup_momentum': 0.8, # Warmup initial momentum
        'warmup_bias_lr': 0.1, # Warmup initial bias lr
        
        # Loss function weights
        'box': 0.05,          # Box loss gain
        'cls': 0.3,           # Class loss gain  
        'cls_pw': 1.0,        # Class BCELoss positive_weight
        'obj': 0.7,           # Object loss gain (scale with pixels)
        'obj_pw': 1.0,        # Object BCELoss positive_weight
        'iou_t': 0.20,        # IoU training threshold
        'anchor_t': 4.0,      # Anchor-multiple threshold
        'fl_gamma': 0.0,      # Focal loss gamma (disabled for baseline)
        
        # AUGMENTATION: ALL DISABLED FOR TRUE BASELINE
        'hsv_h': 0.0,         # Image HSV-Hue augmentation (fraction)
        'hsv_s': 0.0,         # Image HSV-Saturation augmentation (fraction)
        'hsv_v': 0.0,         # Image HSV-Value augmentation (fraction)
        'degrees': 0.0,       # Image rotation (+/- deg)
        'translate': 0.0,     # Image translation (+/- fraction)
        'scale': 0.0,         # Image scale (+/- gain)
        'shear': 0.0,         # Image shear (+/- deg)
        'perspective': 0.0,   # Image perspective (+/- fraction), range 0-0.001
        'flipud': 0.0,        # Image flip up-down (probability)
        'fliplr': 0.0,        # Image flip left-right (probability)
        'mosaic': 0.0,        # Image mosaic (probability)
        'mixup': 0.0,         # Image mixup (probability)
        'copy_paste': 0.0,    # Segment copy-paste (probability)
    }
    
    hyp_config_path = config_dir / "hyp_yolov5n_baseline.yaml"
    with open(hyp_config_path, 'w') as f:
        yaml.dump(hyp_config, f, default_flow_style=False)
    
    return dataset_config_path, hyp_config_path

def train_yolov5n_baseline(epochs: int = 100, quick_test: bool = False) -> Path:
    """
    Train YOLOv5n Baseline (Phase 1) model on VisDrone dataset
    Following Protocol v2.0 - True Baseline Framework
    
    Args:
        epochs: Number of training epochs (default: 100)
        quick_test: If True, use reduced settings for quick validation
    
    Returns:
        Path to training results directory
    """
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "runs" / "train" / f"yolov5n_baseline_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("[START] YOLOv5n Baseline (Phase 1) Training Started")
    logger.info("PROTOCOL: Version 2.0 - True Baseline Framework")
    logger.info("METHODOLOGY: Phase 1 - True Baseline (No Augmentation)")
    logger.info(f"Output Directory: {output_dir}")
    
    try:
        # Create configuration files first
        logger.info("[CONFIG] Creating baseline configuration...")
        dataset_config_path, hyp_config_path = create_baseline_config()
        
        # Validate environment (after configs are created)
        validate_environment(dataset_config_path)
        
        # Quick test adjustments
        if quick_test:
            epochs = 20
            logger.info(f"[INFO] Quick test mode enabled ({epochs} epochs)")
        
        # Training parameters (optimized for RTX 3060 6GB)
        img_size = 640          # Input image size
        batch_size = 8          # Reduced batch size for memory stability
        workers = 0             # Disable multiprocessing on Windows to avoid shared memory errors
        
        logger.info(f"[TRAINING] Configuration Summary:")
        logger.info(f"  • Model: YOLOv5n (nano)")
        logger.info(f"  • Phase: 1 (True Baseline)")
        logger.info(f"  • Dataset: VisDrone (original only)")
        logger.info(f"  • Epochs: {epochs}")
        logger.info(f"  • Image Size: {img_size}")
        logger.info(f"  • Batch Size: {batch_size}")
        logger.info(f"  • Augmentation: DISABLED (true baseline)")
        logger.info("")
        
        # Log Phase 1 specific features
        logger.info("[PHASE-1] True Baseline Training Features:")
        logger.info("  - ORIGINAL DATASET ONLY: No synthetic augmentation")
        logger.info("  - NO REAL-TIME AUGMENTATION: All augmentation disabled")
        logger.info("  - MINIMAL PREPROCESSING: Resize and normalize only")
        logger.info("  - METHODOLOGY COMPLIANCE: Protocol v2.0 Phase 1")
        logger.info("  - TARGET PERFORMANCE: >18% mAP@0.5")
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
            'name': f'yolov5n_baseline_{timestamp}',
            'cache': True,           # Cache images for faster training
            'workers': workers,
            'device': '',            # Auto-select device
            'multi_scale': False,    # Disabled for true baseline
            'optimizer': 'SGD',      # SGD optimizer
            'cos_lr': True,          # Cosine learning rate scheduler
            'exist_ok': True,
            'patience': 300,         # Early stopping patience
            'save_period': -1,       # Save checkpoint every x epochs (-1 disabled)
        }
        
        # Start training
        logger.info("[TRAINING] Starting YOLOv5n baseline training...")
        
        # Construct command for YOLOv5 training
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
        
        # Add multi-scale flag if enabled
        if not train_args['multi_scale']:
            # Multi-scale is enabled by default, we need to disable it for baseline
            # YOLOv5 doesn't have a direct --no-multi-scale flag, so we'll work with default
            pass
        
        logger.info(f"[TRAINING] Command: {' '.join(train_cmd)}")
        
        # Execute training with optimized settings
        import subprocess
        import os
        
        # Optimize environment for memory and stability
        env = os.environ.copy()
        env['WANDB_DISABLED'] = 'true'
        env['WANDB_MODE'] = 'disabled'
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # Better memory management
        env['CUDA_LAUNCH_BLOCKING'] = '1'  # Better error reporting
        
        result = subprocess.run(train_cmd, cwd=str(yolov5_path), capture_output=False, env=env)
        
        if result.returncode != 0:
            raise RuntimeError(f"YOLOv5 training failed with return code {result.returncode}")
        
        # Training completed
        logger.info("[SUCCESS] Training completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        
        # Get results directory (YOLOv5 creates its own structure)
        yolov5_results_dir = project_root / "runs" / "train" / f'yolov5n_baseline_{timestamp}'
        
        logger.info("[BASELINE] Phase 1 Baseline Training Complete:")
        logger.info(f"  • Protocol: Version 2.0 - True Baseline Framework")
        logger.info(f"  • Results: {yolov5_results_dir}")
        logger.info(f"  • Training Epochs: {epochs}")
        logger.info(f"  • Expected Performance: >18% mAP@0.5")
        
        # Performance analysis
        logger.info("[ANALYSIS] Baseline Performance Analysis:")
        logger.info("  - Methodology compliance: Phase 1 true baseline established")
        logger.info("  - Target: >18% mAP@0.5 for thesis requirements")
        logger.info("  - Comparison: Will be compared against Phase 2 environmental training")
        logger.info("  - Next step: Phase 2 training with environmental augmentation")
        
        return yolov5_results_dir
        
    except Exception as e:
        logger.error(f"[ERROR] Training failed: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="YOLOv5n Baseline (Phase 1) Training for VisDrone")
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with reduced settings (20 epochs)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("YOLOv5n Baseline (Phase 1) Training - VisDrone Dataset")
    print("PROTOCOL: Version 2.0 - True Baseline Framework")
    print("="*80)
    print(f"Epochs: {args.epochs}")
    print(f"Quick Test: {args.quick_test}")
    print(f"Phase: 1 (True Baseline - No Augmentation)")
    print(f"Target: >18% mAP@0.5")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    try:
        output_dir = train_yolov5n_baseline(
            epochs=args.epochs,
            quick_test=args.quick_test
        )
        
        print("\n" + "="*80)
        print("[SUCCESS] YOLOv5n Baseline (Phase 1) Training Complete!")
        print(f"Results: {output_dir}")
        print("Expected: True baseline established for Phase 2 comparison")
        print("Target: >18% mAP@0.5 with no augmentation")
        print("Next: Phase 2 environmental robustness training")
        print("="*80)
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()