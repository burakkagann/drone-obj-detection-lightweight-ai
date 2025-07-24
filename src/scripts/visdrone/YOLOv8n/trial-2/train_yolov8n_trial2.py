#!/usr/bin/env python3
"""
YOLOv8n Trial-2 Training Script for VisDrone Dataset
Master's Thesis: Robust Object Detection for Surveillance Drones

This script implements Trial-2 training for YOLOv8n on the VisDrone dataset
focusing on Enhanced Small Object Detection strategy to improve pedestrian class performance.

Key Trial-2 Strategy - Enhanced Small Object Detection:
- Higher resolution (832px) for small object visibility
- Optimized augmentation for small objects (enhanced mosaic, copy-paste)
- Refined hyperparameters for small object detection
- Focus on reducing class imbalance (pedestrian vs people performance gap)

Expected Performance Target:
- mAP@0.5: 30-32% (+1-3% improvement over Trial-1)
- Pedestrian class: 2.5-4% mAP@0.5 (+1-2.5% improvement)
- Maintain: >55 FPS inference speed, <6 MB model size

Author: Burak Kağan Yılmazer
Date: January 2025
Environment: yolov8n-visdrone_venv
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
    from ultralytics import YOLO
    import torch
    import yaml
except ImportError as e:
    print(f"[ERROR] Failed to import required packages: {e}")
    print("Please ensure you're using the yolov8n-visdrone_venv environment")
    print("Activation: .\\venvs\\yolov8n-visdrone_venv\\Scripts\\Activate.ps1")
    sys.exit(1)

def setup_logging(output_dir: Path) -> None:
    """Setup logging configuration"""
    log_file = output_dir / f"yolov8n_trial2_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def validate_environment() -> None:
    """Validate training environment and dependencies"""
    # Check CUDA availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // 1024**3
        print(f"[INFO] GPU: {gpu_name} ({gpu_memory}GB)")
        print(f"[INFO] CUDA Version: {torch.version.cuda}")
    else:
        print("[WARNING] CUDA not available, training will use CPU")
    
    # Check PyTorch version
    print(f"[INFO] PyTorch Version: {torch.__version__}")
    
    # Validate dataset paths
    dataset_config = project_root / "config" / "visdrone" / "yolov8n_v1" / "yolov8n_visdrone_config.yaml"
    if not dataset_config.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {dataset_config}")
    
    print(f"[INFO] Dataset Config: {dataset_config}")
    print(f"[INFO] Project Root: {project_root}")

def create_yolov8n_trial2_config(output_dir: Path) -> Path:
    """
    Create YOLOv8n Trial-2 hyperparameter configuration
    Strategy: Enhanced Small Object Detection
    """
    
    # YOLOv8n Trial-2 hyperparameters (Enhanced Small Object Detection)
    yolov8n_trial2_config = {
        # Learning rate configuration (optimized for small objects)
        'lr0': 0.003,           # Further reduced for fine-grained small object features
        'lrf': 0.01,            # Lower final ratio for better convergence
        'momentum': 0.95,       # Increased momentum for stability
        'weight_decay': 0.0003, # Reduced weight decay to prevent small feature suppression
        'warmup_epochs': 6.0,   # Extended warmup for small object learning
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Loss function weights (optimized for small object detection)
        'box': 8.0,             # Increased box loss weight for precise localization
        'cls': 0.4,             # Slightly reduced to balance with box loss
        'dfl': 1.3,             # Reduced DFL for small object focus
        
        # Small Object Detection Augmentation (CRITICAL OPTIMIZATIONS)
        'hsv_h': 0.015,         # Reduced hue to preserve small object features
        'hsv_s': 0.4,           # Balanced saturation for small objects
        'hsv_v': 0.25,          # Controlled value changes for visibility
        
        # Geometric augmentation (small object optimized)
        'degrees': 3.0,         # Reduced rotation to preserve small object integrity
        'translate': 0.15,      # Reduced translation for small object stability
        'scale': 0.9,           # Reduced scale variation to maintain small object size
        'shear': 0.0,           # Disabled shear (can distort small objects)
        'perspective': 0.00005, # Minimal perspective for small object preservation
        
        # Flip augmentation
        'flipud': 0.0,          # No vertical flip for drone imagery
        'fliplr': 0.5,          # Standard horizontal flip
        
        # Advanced augmentation (ENHANCED FOR SMALL OBJECTS)
        'mosaic': 0.9,          # INCREASED: More multi-scale training for small objects
        'mixup': 0.3,           # REDUCED: Less feature blending to preserve small features
        'copy_paste': 0.4,      # INCREASED: Critical for small object augmentation
        
        # YOLOv8 specific optimizations
        'close_mosaic': 15,     # Extended mosaic training for small object learning
        'auto_augment': 'randaugment',  # Advanced auto-augmentation
        'erasing': 0.2,         # Reduced random erasing to preserve small objects
    }
    
    # Save configuration
    config_file = output_dir / "yolov8n_trial2_hyperparameters.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(yolov8n_trial2_config, f, default_flow_style=False, sort_keys=False)
    
    return config_file

def train_yolov8n_trial2(epochs: int = 50, quick_test: bool = False) -> Path:
    """
    Train YOLOv8n Trial-2 model on VisDrone dataset
    Strategy: Enhanced Small Object Detection
    
    Args:
        epochs: Number of training epochs (default: 50 for Trial-2)
        quick_test: If True, use minimal settings for quick validation
    
    Returns:
        Path to training results directory
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "runs" / "train" / f"yolov8n_trial2_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("[START] YOLOv8n Trial-2 Training Started")
    logger.info("Strategy: Enhanced Small Object Detection")
    logger.info(f"Output Directory: {output_dir}")
    
    try:
        # Validate environment
        validate_environment()
        
        # Create Trial-2 hyperparameter configuration
        config_file = create_yolov8n_trial2_config(output_dir)
        logger.info(f"[CONFIG] Trial-2 hyperparameters saved: {config_file}")
        
        # Load YOLOv8n model with pretrained weights
        model_path = Path(__file__).parent / "yolov8n.pt"
        if not model_path.exists():
            logger.info("Downloading YOLOv8n pretrained weights...")
        
        model = YOLO("yolov8n.pt")  # Will auto-download if not exists
        logger.info(f"[SUCCESS] YOLOv8n model loaded: {model.model}")
        
        # Dataset configuration - using YOLOv8n-specific VisDrone config with absolute paths
        dataset_config = project_root / "config" / "visdrone" / "yolov8n_v1" / "yolov8n_visdrone_config.yaml"
        
        # Verify dataset exists at correct location
        dataset_path = project_root / "data" / "my_dataset" / "visdrone"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        # Verify train/val/test directories exist
        for subset in ['train', 'val', 'test']:
            subset_path = dataset_path / subset / "images"
            if not subset_path.exists():
                raise FileNotFoundError(f"Dataset subset not found: {subset_path}")
        
        logger.info(f"[DATASET] VisDrone dataset verified at: {dataset_path}")
        
        # Training parameters (Trial-2 Enhanced Small Object Detection - Windows Stable)
        train_params = {
            'data': str(dataset_config),
            'epochs': epochs,
            'imgsz': 832,   # CRITICAL: Higher resolution for small objects (vs 640 in Trial-1)
            'batch': 6,     # FURTHER REDUCED: Conservative batch size for stability
            'project': str(output_dir.parent),
            'name': output_dir.name,
            'exist_ok': True,
            'save_period': 8,  # Save every 8 epochs for longer training
            'device': 0 if torch.cuda.is_available() else 'cpu',
            'workers': 0,   # DISABLED: No multiprocessing to avoid worker crashes on Windows
            'verbose': True,
            'seed': 42,  # Reproducibility
            'cfg': str(config_file),  # Use Trial-2 hyperparameters
            'cache': False, # DISABLED: No RAM caching due to 832px memory requirements
            'amp': True,    # Automatic Mixed Precision
            'cos_lr': True, # Cosine LR scheduling for better convergence
            'patience': 25, # Early stopping patience
            'multi_scale': True, # CRITICAL: Multi-scale training for small objects
        }
        
        # Quick test adjustments
        if quick_test:
            train_params.update({
                'epochs': 15,
                'batch': 4,     # Even smaller batch for quick test
                'workers': 0,   # Already disabled in main config for stability
                'cache': False, # Already disabled in main config
                'imgsz': 640,   # Reduced resolution for quick test
                'multi_scale': False
            })
            logger.info("[INFO] Quick test mode enabled (15 epochs, reduced settings)")
        
        logger.info(f"[CONFIG] Training Parameters:")
        for key, value in train_params.items():
            logger.info(f"  {key}: {value}")
        
        # Log Trial-2 specific strategy
        logger.info("[TRIAL-2] Enhanced Small Object Detection Strategy:")
        logger.info("  - HIGHER RESOLUTION: 832px (vs 640px) for small object visibility")
        logger.info("  - OPTIMIZED AUGMENTATION: Enhanced mosaic (0.9), copy-paste (0.4)")
        logger.info("  - REFINED HYPERPARAMETERS: Lower LR (0.003), increased box loss (8.0)")
        logger.info("  - MULTI-SCALE TRAINING: Enabled for scale invariance")
        logger.info("  - TARGET: 30-32% mAP@0.5, improved pedestrian class performance")
        logger.info("")
        logger.info("[IMPROVEMENTS] Expected improvements over Trial-1:")
        logger.info("  - Overall mAP@0.5: 30-32% (+1-3% improvement)")
        logger.info("  - Pedestrian class: 2.5-4% mAP@0.5 (+1-2.5% improvement)")
        logger.info("  - Small object recall: +2-4% improvement")
        logger.info("  - Class imbalance: Reduced performance gap")
        
        # Start training
        logger.info("[TRAINING] Starting YOLOv8n Trial-2 training...")
        results = model.train(**train_params)
        
        # Training completed
        logger.info("[SUCCESS] Training completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        
        # Log final metrics if available
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            logger.info("[METRICS] Final Training Metrics:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value}")
        
        # METHODOLOGY COMPLIANCE: Comprehensive evaluation with Trial-1 comparison
        logger.info("[EVALUATION] Running comprehensive evaluation with Trial-1 comparison...")
        try:
            # Import evaluation module
            sys.path.append(str(Path(__file__).parent.parent))
            from evaluation_metrics import YOLOv8nEvaluationMetrics, load_baseline_metrics
            
            # Find best model weights
            best_weights = output_dir / "weights" / "best.pt"
            if not best_weights.exists():
                # Fallback to last weights
                best_weights = output_dir / "weights" / "last.pt"
            
            if best_weights.exists():
                # Create evaluation output directory
                eval_output_dir = output_dir / "evaluation"
                
                # Try to load Trial-1 metrics for comparison
                trial1_metrics = None
                trial1_pattern = project_root / "runs" / "train" / "yolov8n_trial1_*" / "evaluation" / "evaluation_results_*.json"
                
                # Search for most recent Trial-1 results
                import glob
                trial1_files = sorted(glob.glob(str(trial1_pattern)), reverse=True)
                if trial1_files:
                    trial1_metrics = load_baseline_metrics(trial1_files[0])
                    logger.info(f"[TRIAL-1] Loaded Trial-1 metrics: {trial1_files[0]}")
                else:
                    logger.warning("[WARNING] No Trial-1 metrics found for comparison")
                
                # Run comprehensive evaluation
                evaluator = YOLOv8nEvaluationMetrics(
                    model_path=str(best_weights),
                    dataset_config=dataset_config,
                    output_dir=eval_output_dir
                )
                
                eval_results = evaluator.run_comprehensive_evaluation(trial1_metrics)
                
                logger.info("[EVALUATION] Comprehensive evaluation completed!")
                logger.info(f"[EVALUATION] Results saved to: {eval_output_dir}")
                
                # Log key Trial-2 metrics for thesis
                acc_metrics = eval_results.get('detection_accuracy', {})
                speed_metrics = eval_results.get('inference_speed', {})
                size_metrics = eval_results.get('model_size', {})
                rob_metrics = eval_results.get('robustness', {})
                
                logger.info("[TRIAL-2] Key Trial-2 Metrics for Thesis:")
                logger.info(f"  • mAP@0.5: {acc_metrics.get('mAP_50', 'N/A'):.4f}")
                logger.info(f"  • mAP@0.5:0.95: {acc_metrics.get('mAP_50_95', 'N/A'):.4f}")
                logger.info(f"  • Precision: {acc_metrics.get('precision', 'N/A'):.4f}")
                logger.info(f"  • Recall: {acc_metrics.get('recall', 'N/A'):.4f}")
                logger.info(f"  • FPS: {speed_metrics.get('fps', 'N/A')}")
                logger.info(f"  • Model Size: {size_metrics.get('model_file_size_mb', 'N/A')} MB")
                
                if trial1_metrics:
                    logger.info("[COMPARISON] Trial-1 vs Trial-2 Comparison:")
                    logger.info(f"  • Performance Change: {rob_metrics.get('performance_change_percent', 'N/A'):.2f}%")
                    logger.info(f"  • Improvement Category: {rob_metrics.get('robustness_category', 'N/A')}")
                    logger.info(f"  • Trial-1 mAP@0.5: {rob_metrics.get('baseline_mAP_50', 'N/A'):.4f}")
                    logger.info(f"  • Trial-2 mAP@0.5: {rob_metrics.get('current_mAP_50', 'N/A'):.4f}")
                
                # Small object detection analysis
                logger.info("[ANALYSIS] Small Object Detection Impact Analysis:")
                logger.info("  - Strategy: Enhanced small object detection with higher resolution")
                logger.info("  - Key factors: 832px resolution, optimized augmentation, multi-scale training")
                logger.info("  - Research value: Quantifies small object detection improvements for thesis")
                
            else:
                logger.warning("[WARNING] Model weights not found, skipping evaluation")
                
        except Exception as e:
            logger.error(f"[ERROR] Comprehensive evaluation failed: {e}")
            logger.info("[INFO] Training completed but evaluation failed - check model weights")
        
        return output_dir
        
    except Exception as e:
        logger.error(f"[ERROR] Training failed: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="YOLOv8n Trial-2 Training for VisDrone")
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with reduced settings')
    
    args = parser.parse_args()
    
    print("="*80)
    print("YOLOv8n Trial-2 Training - VisDrone Dataset")
    print("Strategy: Enhanced Small Object Detection")
    print("="*80)
    print(f"Epochs: {args.epochs}")
    print(f"Quick Test: {args.quick_test}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    try:
        output_dir = train_yolov8n_trial2(
            epochs=args.epochs,
            quick_test=args.quick_test
        )
        
        print("\n" + "="*80)
        print("[SUCCESS] YOLOv8n Trial-2 Training Complete!")
        print(f"Results: {output_dir}")
        print("Expected: Improved small object detection performance")
        print("="*80)
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()