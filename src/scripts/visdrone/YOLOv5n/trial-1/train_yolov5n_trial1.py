#!/usr/bin/env python3
"""
YOLOv5n Trial-1 Training Script for VisDrone Dataset
Master's Thesis: Robust Object Detection for Surveillance Drones

This script implements Phase 2 (Environmental Robustness) training for YOLOv5n on the VisDrone dataset
using environmental augmentation AND optimized hyperparameters for robustness testing.

Key features for Phase 2 (Environmental Robustness):
- Environmental augmented dataset (original + synthetic conditions)
- Standard real-time augmentation enabled
- Optimized hyperparameters for robustness
- Focus on low-visibility robustness vs. Phase 1 baseline

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
    log_file = output_dir / f"yolov5n_trial1_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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

def validate_environment() -> None:
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
    
    # Validate environmental dataset paths
    env_dataset_path = project_root / "data" / "environmental_augmented_dataset" / "visdrone"
    if not env_dataset_path.exists():
        print(f"[WARNING] Environmental dataset not found: {env_dataset_path}")
        print("[INFO] Falling back to original dataset with real-time augmentation")
        return False
    
    print(f"[INFO] Environmental dataset: {env_dataset_path}")
    return True

def create_trial1_config(use_env_dataset: bool = True) -> Path:
    """Create YOLOv5n Trial-1 configuration following Protocol v2.0 Phase 2"""
    config_dir = project_root / "config" / "visdrone" / "yolov5n_trial1"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset configuration
    if use_env_dataset:
        # Use environmental augmented dataset
        dataset_config = {
            'path': str(project_root / "data" / "environmental_augmented_dataset" / "visdrone"),
            'train': 'train/images',
            'val': 'val/images', 
            'test': 'test/images',
            'nc': 10,
            'names': [
                'pedestrian', 'people', 'bicycle', 'car', 'van',
                'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
            ]
        }
    else:
        # Fallback to original dataset
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
    
    dataset_config_path = config_dir / "yolov5n_visdrone_trial1.yaml"
    with open(dataset_config_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    # Hyperparameters configuration (PHASE 2 - ENVIRONMENTAL ROBUSTNESS)
    hyp_config = {
        # Learning rate settings (optimized for robustness)
        'lr0': 0.005,         # Reduced initial learning rate for stability
        'lrf': 0.02,          # Lower final learning rate
        'momentum': 0.937,    # SGD momentum/Adam beta1
        'weight_decay': 0.0005, # Optimizer weight decay 5e-4
        'warmup_epochs': 3.0, # Warmup epochs (fractions ok)
        'warmup_momentum': 0.8, # Warmup initial momentum
        'warmup_bias_lr': 0.1, # Warmup initial bias lr
        
        # Loss function weights (optimized for small objects)
        'box': 0.03,          # Box loss gain (reduced for small objects)
        'cls': 0.3,           # Class loss gain  
        'cls_pw': 1.0,        # Class BCELoss positive_weight
        'obj': 1.2,           # Object loss gain (increased for objectness)
        'obj_pw': 1.0,        # Object BCELoss positive_weight
        'iou_t': 0.15,        # IoU training threshold (reduced for small objects)
        'anchor_t': 4.0,      # Anchor-multiple threshold
        'fl_gamma': 0.0,      # Focal loss gamma (disabled)
        
        # AUGMENTATION: ENABLED FOR ROBUSTNESS (Protocol v2.0 Phase 2)
        'hsv_h': 0.02,        # Image HSV-Hue augmentation (fraction)
        'hsv_s': 0.5,         # Image HSV-Saturation augmentation (fraction)
        'hsv_v': 0.3,         # Image HSV-Value augmentation (fraction)
        'degrees': 5.0,       # Image rotation (+/- deg)
        'translate': 0.2,     # Image translation (+/- fraction)
        'scale': 0.8,         # Image scale (+/- gain)
        'shear': 0.0,         # Image shear (+/- deg)
        'perspective': 0.0,   # Image perspective (+/- fraction), range 0-0.001
        'flipud': 0.0,        # Image flip up-down (probability)
        'fliplr': 0.5,        # Image flip left-right (probability)
        'mosaic': 0.8,        # Image mosaic (probability)
        'mixup': 0.4,         # Image mixup (probability)
        'copy_paste': 0.3,    # Segment copy-paste (probability)
    }
    
    hyp_config_path = config_dir / "hyp_yolov5n_trial1.yaml"
    with open(hyp_config_path, 'w') as f:
        yaml.dump(hyp_config, f, default_flow_style=False)
    
    return dataset_config_path, hyp_config_path

def train_yolov5n_trial1(epochs: int = 100, quick_test: bool = False) -> Path:
    """
    Train YOLOv5n Trial-1 (Phase 2) model on VisDrone dataset
    Following Protocol v2.0 - Environmental Robustness Framework
    
    Args:
        epochs: Number of training epochs (default: 100)
        quick_test: If True, use reduced settings for quick validation
    
    Returns:
        Path to training results directory
    """
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "runs" / "train" / f"yolov5n_trial1_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("[START] YOLOv5n Trial-1 (Phase 2) Training Started")
    logger.info("PROTOCOL: Version 2.0 - True Baseline Framework")
    logger.info("METHODOLOGY: Phase 2 - Environmental Robustness")
    logger.info(f"Output Directory: {output_dir}")
    
    try:
        # Validate environment and check for environmental dataset
        use_env_dataset = validate_environment()
        
        # Create configuration files
        logger.info("[CONFIG] Creating Trial-1 configuration...")
        dataset_config_path, hyp_config_path = create_trial1_config(use_env_dataset)
        
        # Quick test adjustments
        if quick_test:
            epochs = 20
            logger.info(f"[INFO] Quick test mode enabled ({epochs} epochs)")
        
        # Training parameters
        img_size = 640          # Input image size
        batch_size = 16         # Batch size
        workers = 4             # Data loader workers
        
        logger.info(f"[TRAINING] Configuration Summary:")
        logger.info(f"  • Model: YOLOv5n (nano)")
        logger.info(f"  • Phase: 2 (Environmental Robustness)")
        logger.info(f"  • Dataset: {'Environmental Augmented' if use_env_dataset else 'Original + Real-time Aug'}")
        logger.info(f"  • Epochs: {epochs}")
        logger.info(f"  • Image Size: {img_size}")
        logger.info(f"  • Batch Size: {batch_size}")
        logger.info(f"  • Augmentation: ENABLED (robustness training)")
        logger.info("")
        
        # Log Phase 2 specific features
        logger.info("[PHASE-2] Environmental Robustness Training Features:")
        if use_env_dataset:
            logger.info("  - ENVIRONMENTAL DATASET: Original + synthetic conditions")
        else:
            logger.info("  - ORIGINAL DATASET: With enhanced real-time augmentation")
        logger.info("  - REAL-TIME AUGMENTATION: Mosaic, mixup, HSV, geometric enabled")
        logger.info("  - OPTIMIZED HYPERPARAMETERS: Reduced LR, balanced loss weights")
        logger.info("  - METHODOLOGY COMPLIANCE: Protocol v2.0 Phase 2")
        logger.info("  - TARGET PERFORMANCE: >25% mAP@0.5 (+7pp from baseline)")
        logger.info("")
        
        # Prepare training arguments
        train_args = {
            'img': img_size,
            'batch': batch_size,
            'epochs': epochs,
            'data': str(dataset_config_path),
            'weights': 'yolov5n.pt',  # Pre-trained weights
            'hyp': str(hyp_config_path),
            'project': str(project_root / "runs" / "train"),
            'name': f'yolov5n_trial1_{timestamp}',
            'cache': True,           # Cache images for faster training
            'workers': workers,
            'device': '',            # Auto-select device
            'multi_scale': True,     # Multi-scale training enabled
            'optimizer': 'SGD',      # SGD optimizer
            'cos_lr': True,          # Cosine learning rate scheduler
            'exist_ok': True,
            'patience': 300,         # Early stopping patience
            'save_period': -1,       # Save checkpoint every x epochs (-1 disabled)
        }
        
        # Start training
        logger.info("[TRAINING] Starting YOLOv5n Trial-1 training...")
        
        # Import and run YOLOv5 training
        sys.argv = ['train.py']  # Reset sys.argv for YOLOv5
        for key, value in train_args.items():
            sys.argv.extend([f'--{key}', str(value)])
        
        # Run training
        train.main()
        
        # Training completed
        logger.info("[SUCCESS] Training completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        
        # Get results directory (YOLOv5 creates its own structure)
        yolov5_results_dir = project_root / "runs" / "train" / f'yolov5n_trial1_{timestamp}'
        
        logger.info("[TRIAL-1] Phase 2 Environmental Robustness Training Complete:")
        logger.info(f"  • Protocol: Version 2.0 - Environmental Robustness")
        logger.info(f"  • Results: {yolov5_results_dir}")
        logger.info(f"  • Training Epochs: {epochs}")
        logger.info(f"  • Expected Performance: >25% mAP@0.5")
        
        # Performance analysis
        logger.info("[ANALYSIS] Environmental Robustness Analysis:")
        logger.info("  - Methodology compliance: Phase 2 environmental training complete")
        logger.info("  - Target: >25% mAP@0.5 for thesis requirements")
        logger.info("  - Comparison: Phase 1 baseline vs Phase 2 robustness")
        logger.info("  - Expected improvement: +7pp absolute mAP improvement")
        logger.info("  - Research impact: Complete methodology demonstration")
        
        return yolov5_results_dir
        
    except Exception as e:
        logger.error(f"[ERROR] Training failed: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="YOLOv5n Trial-1 (Phase 2) Training for VisDrone")
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with reduced settings (20 epochs)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("YOLOv5n Trial-1 (Phase 2) Training - VisDrone Dataset")
    print("PROTOCOL: Version 2.0 - Environmental Robustness Framework")
    print("="*80)
    print(f"Epochs: {args.epochs}")
    print(f"Quick Test: {args.quick_test}")
    print(f"Phase: 2 (Environmental Robustness)")
    print(f"Target: >25% mAP@0.5 (+7pp from baseline)")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    try:
        output_dir = train_yolov5n_trial1(
            epochs=args.epochs,
            quick_test=args.quick_test
        )
        
        print("\n" + "="*80)
        print("[SUCCESS] YOLOv5n Trial-1 (Phase 2) Training Complete!")
        print(f"Results: {output_dir}")
        print("Expected: Environmental robustness vs baseline comparison")
        print("Target: >25% mAP@0.5 with environmental augmentation")
        print("Impact: Complete methodology demonstration")
        print("="*80)
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()