#!/usr/bin/env python3
"""
YOLOv8n Baseline Training Script for VisDrone Dataset
Master's Thesis: Robust Object Detection for Surveillance Drones

This script implements baseline training for YOLOv8n on the VisDrone dataset
using default hyperparameters to establish performance benchmarks.

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
except ImportError as e:
    print(f"[ERROR] Failed to import required packages: {e}")
    print("Please ensure you're using the yolov8n-visdrone_venv environment")
    print("Activation: .\\venvs\\yolov8n-visdrone_venv\\Scripts\\Activate.ps1")
    sys.exit(1)

def setup_logging(output_dir: Path) -> None:
    """Setup logging configuration"""
    log_file = output_dir / f"yolov8n_baseline_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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

def train_yolov8n_baseline(epochs: int = 20, quick_test: bool = False) -> Path:
    """
    Train YOLOv8n baseline model on VisDrone dataset
    
    Args:
        epochs: Number of training epochs (default: 20 for baseline)
        quick_test: If True, use minimal settings for quick validation
    
    Returns:
        Path to training results directory
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "runs" / "train" / f"yolov8n_baseline_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("[START] YOLOv8n Baseline Training Started")
    logger.info(f"Output Directory: {output_dir}")
    
    try:
        # Validate environment
        validate_environment()
        
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
        
        # Training parameters (TRUE baseline - minimal augmentation)
        train_params = {
            'data': str(dataset_config),
            'epochs': epochs,
            'imgsz': 640,  # Standard resolution
            'batch': 16,   # Standard batch size for RTX 3060 (5GB)
            'project': str(output_dir.parent),
            'name': output_dir.name,
            'exist_ok': True,
            'save_period': 5,  # Save every 5 epochs
            'device': 0 if torch.cuda.is_available() else 'cpu',
            'workers': 4,
            'verbose': True,
            'seed': 42,  # Reproducibility
            # TRUE BASELINE: Disable augmentation for pure dataset performance
            'hsv_h': 0.0,      # Disable HSV-Hue augmentation
            'hsv_s': 0.0,      # Disable HSV-Saturation augmentation  
            'hsv_v': 0.0,      # Disable HSV-Value augmentation
            'degrees': 0.0,    # Disable rotation
            'translate': 0.0,  # Disable translation
            'scale': 0.0,      # Disable scaling
            'shear': 0.0,      # Disable shear
            'perspective': 0.0, # Disable perspective
            'flipud': 0.0,     # Disable vertical flip
            'fliplr': 0.0,     # Disable horizontal flip
            'mosaic': 0.0,     # CRITICAL: Disable mosaic (default enabled)
            'mixup': 0.0,      # CRITICAL: Disable mixup (default enabled)
            'copy_paste': 0.0, # Disable copy-paste
        }
        
        # Quick test adjustments
        if quick_test:
            train_params.update({
                'epochs': 5,
                'batch': 8,
                'workers': 2
            })
            logger.info("[INFO] Quick test mode enabled (5 epochs, reduced batch)")
        
        logger.info(f"[CONFIG] Training Parameters:")
        for key, value in train_params.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("[BASELINE] TRUE BASELINE CONFIGURATION:")
        logger.info("  - NO synthetic augmentation (fog, night, blur)")
        logger.info("  - NO standard augmentation (mosaic, mixup, HSV, geometric)")
        logger.info("  - Pure original dataset performance")
        logger.info("  - Benchmark for comparing augmented models")
        
        # Start training
        logger.info("[TRAINING] Starting YOLOv8n baseline training...")
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
        
        # METHODOLOGY COMPLIANCE: Comprehensive evaluation
        logger.info("[EVALUATION] Running comprehensive evaluation (Methodology Section 4.1)...")
        try:
            # Import evaluation module
            sys.path.append(str(Path(__file__).parent.parent))
            from evaluation_metrics import YOLOv8nEvaluationMetrics
            
            # Find best model weights
            best_weights = output_dir / "weights" / "best.pt"
            if not best_weights.exists():
                # Fallback to last weights
                best_weights = output_dir / "weights" / "last.pt"
            
            if best_weights.exists():
                # Create evaluation output directory
                eval_output_dir = output_dir / "evaluation"
                
                # Run comprehensive evaluation
                evaluator = YOLOv8nEvaluationMetrics(
                    model_path=str(best_weights),
                    dataset_config=dataset_config,
                    output_dir=eval_output_dir
                )
                
                eval_results = evaluator.run_comprehensive_evaluation()
                
                logger.info("[EVALUATION] Comprehensive evaluation completed!")
                logger.info(f"[EVALUATION] Results saved to: {eval_output_dir}")
                
                # Log key baseline metrics for thesis
                acc_metrics = eval_results.get('detection_accuracy', {})
                speed_metrics = eval_results.get('inference_speed', {})
                size_metrics = eval_results.get('model_size', {})
                
                logger.info("[BASELINE] Key Baseline Metrics for Thesis:")
                logger.info(f"  • mAP@0.5: {acc_metrics.get('mAP_50', 'N/A'):.4f}")
                logger.info(f"  • mAP@0.5:0.95: {acc_metrics.get('mAP_50_95', 'N/A'):.4f}")
                logger.info(f"  • Precision: {acc_metrics.get('precision', 'N/A'):.4f}")
                logger.info(f"  • Recall: {acc_metrics.get('recall', 'N/A'):.4f}")
                logger.info(f"  • FPS: {speed_metrics.get('fps', 'N/A')}")
                logger.info(f"  • Model Size: {size_metrics.get('model_file_size_mb', 'N/A')} MB")
                logger.info(f"  • Memory Usage: {size_metrics.get('memory_usage_mb', 'N/A')} MB")
                
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
    parser = argparse.ArgumentParser(description="YOLOv8n Baseline Training for VisDrone")
    parser.add_argument('--epochs', type=int, default=20, 
                       help='Number of training epochs (default: 20)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with reduced settings')
    
    args = parser.parse_args()
    
    print("="*80)
    print("YOLOv8n Baseline Training - VisDrone Dataset")
    print("="*80)
    print(f"Epochs: {args.epochs}")
    print(f"Quick Test: {args.quick_test}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    try:
        output_dir = train_yolov8n_baseline(
            epochs=args.epochs,
            quick_test=args.quick_test
        )
        
        print("\n" + "="*80)
        print("[SUCCESS] YOLOv8n Baseline Training Complete!")
        print(f"Results: {output_dir}")
        print("="*80)
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()