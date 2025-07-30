#!/usr/bin/env python3
"""
YOLOv5n Phase 1 (True Baseline) Training Script for VisDrone Dataset
Master's Thesis: Robust Object Detection for Surveillance Drones

This script implements Phase 1 (True Baseline) training following Protocol v2.0
with NO augmentation to establish absolute model performance reference point.

Key Phase 1 Requirements:
- Original VisDrone dataset only (no synthetic augmentation)
- ALL real-time augmentation disabled (hsv, rotation, translation, etc.)
- Minimal preprocessing: resize to 640x640 and normalize only
- Pure model capability measurement for thesis baseline

Author: Burak Kağan Yılmazer
Date: July 2025
Environment: yolov5n_visdrone_env
Protocol: Version 2.0 - True Baseline Framework
"""

import os
import sys
import logging
import argparse
import warnings
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Disable wandb completely
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'

# Suppress warnings for clean output
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path for imports
script_path = Path(__file__).resolve()
project_root = script_path.parents[6]  # Fixed: need 6 parents, not 5

# Debug path calculation
print(f"[DEBUG] Script location: {script_path}")
print(f"[DEBUG] Project root: {project_root}")
print(f"[DEBUG] Expected YOLOv5 path: {project_root / 'src' / 'models' / 'YOLOv5'}")

sys.path.append(str(project_root))

try:
    import torch
    import yaml
    import numpy as np
    from torch.utils.data import DataLoader
    
except ImportError as e:
    print(f"[ERROR] Failed to import required packages: {e}")
    print("Please ensure you're using the correct virtual environment")
    print("Activation: .\\venvs\\visdrone\\yolov5n_visdrone_env\\Scripts\\Activate.ps1")
    sys.exit(1)


class Phase1BaselineTrainer:
    """
    YOLOv5n Phase 1 (True Baseline) Trainer
    
    Implements Protocol v2.0 requirements for establishing true baseline
    performance with no augmentation for thesis methodology compliance.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize Phase 1 trainer with configuration"""
        self.project_root = project_root
        self.config_path = config_path or (project_root / "config" / "phase1_baseline" / "yolov5n_visdrone.yaml")
        self.yolov5_path = project_root / "src" / "models" / "YOLOv5"
        
        # Validate YOLOv5 path exists
        if not self.yolov5_path.exists():
            raise FileNotFoundError(f"YOLOv5 directory not found: {self.yolov5_path}")
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize training metrics
        self.training_metrics = {
            'start_time': None,
            'end_time': None,
            'total_duration': None,
            'best_map': 0.0,
            'final_map': 0.0,
            'epochs_completed': 0,
            'early_stopped': False
        }
        
    def _load_config(self) -> Dict[str, Any]:
        """Load training configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Validate Phase 1 requirements
        self._validate_phase1_config(config)
        return config
        
    def _validate_phase1_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration meets Phase 1 requirements"""
        required_disabled = [
            'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 
            'scale', 'shear', 'perspective', 'flipud', 'fliplr',
            'mosaic', 'mixup', 'copy_paste'
        ]
        
        for param in required_disabled:
            if param in config and config[param] != 0.0:
                raise ValueError(f"Phase 1 violation: {param} must be 0.0 (found: {config[param]})")
                
        # Verify phase identifier
        if config.get('phase') != 1:
            raise ValueError(f"Configuration must specify phase: 1 (found: {config.get('phase')})")
            
        print("[VALIDATION] Phase 1 configuration validated - all augmentation disabled ✓")
        
    def setup_logging(self, output_dir: Path) -> logging.Logger:
        """Setup comprehensive logging for Phase 1 training"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = output_dir / f"yolov5n_phase1_baseline_{timestamp}.log"
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Setup file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Setup console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Create logger
        logger = logging.getLogger('Phase1Baseline')
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
        
    def validate_environment(self) -> Dict[str, Any]:
        """Validate training environment and return system info"""
        env_info = {
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'device_info': {},
            'dataset_path': str(self.project_root / self.config['path'])
        }
        
        # GPU information
        if torch.cuda.is_available():
            env_info['device_info'] = {
                'gpu_count': torch.cuda.device_count(),
                'cuda_version': torch.version.cuda,
                'devices': []
            }
            
            for i in range(torch.cuda.device_count()):
                gpu_info = {
                    'index': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_total': torch.cuda.get_device_properties(i).total_memory / (1024**3),
                    'memory_allocated': torch.cuda.memory_allocated(i) / (1024**3),
                    'memory_cached': torch.cuda.memory_reserved(i) / (1024**3)
                }
                env_info['device_info']['devices'].append(gpu_info)
        
        # Validate dataset paths
        dataset_path = self.project_root / self.config['path']
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
            
        # Check train/val directories
        for split in ['train', 'val']:
            split_path = dataset_path / self.config[split]
            if not split_path.exists():
                raise FileNotFoundError(f"{split} directory not found: {split_path}")
                
        return env_info
        
    def create_training_directories(self) -> Path:
        """Create output directories for training results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.project_root / "runs" / "train" / f"yolov5n_phase1_baseline_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (output_dir / "logs").mkdir(exist_ok=True)
        (output_dir / "configs").mkdir(exist_ok=True)
        (output_dir / "metrics").mkdir(exist_ok=True)
        
        return output_dir
        
    def save_training_config(self, output_dir: Path) -> None:
        """Save training configuration and environment info"""
        # Save config copy
        config_copy = output_dir / "configs" / "training_config.yaml"
        with open(config_copy, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
            
        # Save environment info
        env_info = self.validate_environment()
        env_file = output_dir / "configs" / "environment_info.json"
        with open(env_file, 'w') as f:
            json.dump(env_info, f, indent=2, default=str)
            
    def prepare_dataset_config(self, output_dir: Path) -> Path:
        """Prepare YOLOv5-compatible dataset configuration"""
        # Use the dedicated dataset config file for consistency
        dedicated_dataset_config = self.project_root / "config" / "dataset" / "visdrone_dataset.yaml"
        
        if dedicated_dataset_config.exists():
            # Copy the dedicated dataset config
            dataset_config_path = output_dir / "configs" / "dataset.yaml"
            import shutil
            shutil.copy2(dedicated_dataset_config, dataset_config_path)
            return dataset_config_path
        else:
            # Fallback: create from training config
            dataset_config = {
                'path': str(self.project_root / self.config['path']),
                'train': self.config['train'],
                'val': self.config['val'],
                'test': self.config.get('test', self.config['val']),
                'nc': self.config['nc'],
                'names': {i: name for i, name in enumerate(self.config['names'].values())}
            }
            
            # Save dataset config
            dataset_config_path = output_dir / "configs" / "dataset.yaml"
            with open(dataset_config_path, 'w') as f:
                yaml.dump(dataset_config, f, default_flow_style=False)
                
            return dataset_config_path
        
    def prepare_hyperparameters(self, output_dir: Path) -> Path:
        """Prepare YOLOv5 hyperparameter configuration"""
        hyp_config = {
            # Learning rate settings
            'lr0': self.config.get('lr0', 0.01),
            'lrf': self.config.get('lrf', 0.01),
            'momentum': self.config.get('momentum', 0.937),
            'weight_decay': self.config.get('weight_decay', 0.0005),
            'warmup_epochs': self.config.get('warmup_epochs', 3.0),
            'warmup_momentum': self.config.get('warmup_momentum', 0.8),
            'warmup_bias_lr': self.config.get('warmup_bias_lr', 0.1),
            
            # Loss function weights
            'box': self.config.get('box', 0.05),
            'cls': self.config.get('cls', 0.3),
            'cls_pw': self.config.get('cls_pw', 1.0),
            'obj': self.config.get('obj', 0.7),
            'obj_pw': self.config.get('obj_pw', 1.0),
            'iou_t': self.config.get('iou_t', 0.20),
            'anchor_t': self.config.get('anchor_t', 4.0),
            'fl_gamma': self.config.get('fl_gamma', 0.0),
            
            # PHASE 1: ALL AUGMENTATION DISABLED
            'hsv_h': 0.0,
            'hsv_s': 0.0,
            'hsv_v': 0.0,
            'degrees': 0.0,
            'translate': 0.0,
            'scale': 0.0,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.0,
            'mosaic': 0.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
        }
        
        # Save hyperparameters
        hyp_config_path = output_dir / "configs" / "hyperparameters.yaml"
        with open(hyp_config_path, 'w') as f:
            yaml.dump(hyp_config, f, default_flow_style=False)
            
        return hyp_config_path
        
    def train(self, epochs: int = None, quick_test: bool = False) -> Dict[str, Any]:
        """
        Execute Phase 1 (True Baseline) training
        
        Args:
            epochs: Number of training epochs (default from config)
            quick_test: If True, run with reduced settings for validation
            
        Returns:
            Dictionary containing training results and metrics
        """
        # Setup training environment
        output_dir = self.create_training_directories()
        logger = self.setup_logging(output_dir)
        
        # Training start
        self.training_metrics['start_time'] = datetime.now()
        
        logger.info("="*80)
        logger.info("[START] YOLOv5n Phase 1 (True Baseline) Training")
        logger.info("PROTOCOL: Version 2.0 - True Baseline Framework")
        logger.info("METHODOLOGY: Phase 1 - No Augmentation Baseline")
        logger.info("="*80)
        
        try:
            # Environment validation
            logger.info("[VALIDATION] Validating training environment...")
            env_info = self.validate_environment()
            
            # Log environment info
            logger.info(f"PyTorch Version: {env_info['pytorch_version']}")
            logger.info(f"CUDA Available: {env_info['cuda_available']}")
            if env_info['cuda_available']:
                for gpu in env_info['device_info']['devices']:
                    logger.info(f"GPU {gpu['index']}: {gpu['name']} ({gpu['memory_total']:.1f}GB)")
            
            # Save configurations
            self.save_training_config(output_dir)
            dataset_config_path = self.prepare_dataset_config(output_dir)
            hyp_config_path = self.prepare_hyperparameters(output_dir)
            
            # Training parameters
            epochs = epochs or self.config.get('epochs', 100)
            if quick_test:
                epochs = min(epochs, 20)
                logger.info(f"[INFO] Quick test mode enabled - reduced to {epochs} epochs")
                
            batch_size = self.config.get('batch_size', 8)
            img_size = self.config.get('imgsz', 640)
            
            logger.info(f"[CONFIG] Training Configuration:")
            logger.info(f"  • Model: YOLOv5n (nano)")
            logger.info(f"  • Phase: 1 (True Baseline)")
            logger.info(f"  • Dataset: VisDrone (original only)")
            logger.info(f"  • Epochs: {epochs}")
            logger.info(f"  • Batch Size: {batch_size}")
            logger.info(f"  • Image Size: {img_size}x{img_size}")
            logger.info(f"  • Augmentation: ALL DISABLED")
            logger.info(f"  • Target mAP@0.5: >{self.config.get('target_map', 0.18)}")
            logger.info("")
            
            # Phase 1 methodology compliance logging
            logger.info("[PHASE-1] True Baseline Training Features:")
            logger.info("  - ORIGINAL DATASET ONLY: No synthetic augmentation")
            logger.info("  - NO REAL-TIME AUGMENTATION: All disabled (hsv, rotation, etc.)")
            logger.info("  - MINIMAL PREPROCESSING: Resize 640x640 and normalize only")
            logger.info("  - METHODOLOGY COMPLIANCE: Protocol v2.0 Phase 1")
            logger.info("  - PURPOSE: Establish pure model performance baseline")
            logger.info("")
            
            # Prepare YOLOv5 training arguments
            train_args = [
                '--img', str(img_size),
                '--batch', str(batch_size),
                '--epochs', str(epochs),
                '--data', str(dataset_config_path),
                '--cfg', str(self.yolov5_path / "models" / "yolov5n.yaml"),
                '--weights', self.config.get('model', 'yolov5n.pt'),
                '--hyp', str(hyp_config_path),
                '--project', str(output_dir.parent),
                '--name', output_dir.name,
                '--cache',
                '--workers', str(self.config.get('workers', 0)),
                '--device', self.config.get('device', ''),
                '--optimizer', self.config.get('optimizer', 'SGD'),
                '--exist-ok',
                '--patience', str(self.config.get('patience', 300)),
            ]
            
            # Add cosine LR if enabled
            if self.config.get('cos_lr', True):
                train_args.append('--cos-lr')
                
            # Execute training
            logger.info("[TRAINING] Starting YOLOv5n Phase 1 baseline training...")
            logger.info(f"Command: python train.py {' '.join(train_args)}")
            
            # Prepare subprocess command
            train_cmd = [
                sys.executable,  # Use current Python executable from venv
                str(self.yolov5_path / "train.py")
            ] + train_args
            
            # Set environment variables for stable training
            env = os.environ.copy()
            env['WANDB_DISABLED'] = 'true'
            env['WANDB_MODE'] = 'disabled'
            env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            env['CUDA_LAUNCH_BLOCKING'] = '1'  # Better error reporting
            
            # Execute training with subprocess for better reliability
            import subprocess
            
            logger.info(f"Full command: {' '.join(train_cmd)}")
            
            try:
                result = subprocess.run(
                    train_cmd, 
                    cwd=str(self.yolov5_path),
                    env=env,
                    capture_output=False,  # Show output in real-time
                    text=True
                )
                
                if result.returncode != 0:
                    raise RuntimeError(f"YOLOv5 training failed with return code {result.returncode}")
                    
            except Exception as training_error:
                logger.error(f"YOLOv5 training failed: {training_error}")
                raise
                
            # Training completed successfully
            self.training_metrics['end_time'] = datetime.now()
            self.training_metrics['total_duration'] = (
                self.training_metrics['end_time'] - self.training_metrics['start_time']
            ).total_seconds()
            self.training_metrics['epochs_completed'] = epochs
            
            logger.info("[SUCCESS] Phase 1 training completed successfully!")
            logger.info(f"Total Duration: {self.training_metrics['total_duration']:.1f} seconds")
            logger.info(f"Output Directory: {output_dir}")
            
            # Save training metrics
            metrics_file = output_dir / "metrics" / "training_summary.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.training_metrics, f, indent=2, default=str)
                
            # Final analysis
            logger.info("")
            logger.info("[ANALYSIS] Phase 1 Baseline Training Analysis:")
            logger.info("  • Methodology: Protocol v2.0 compliance achieved")
            logger.info("  • Baseline: True baseline established (no augmentation)")
            logger.info("  • Purpose: Reference point for Phase 2 comparison") 
            logger.info("  • Next Step: Phase 2 synthetic augmentation training")
            logger.info("  • Expected: Foundation for thesis comparative analysis")
            
            return {
                'success': True,
                'output_dir': output_dir,
                'training_metrics': self.training_metrics,
                'config_used': self.config,
                'environment_info': env_info
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Phase 1 training failed: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            self.training_metrics['end_time'] = datetime.now()
            self.training_metrics['total_duration'] = (
                self.training_metrics['end_time'] - self.training_metrics['start_time']
            ).total_seconds() if self.training_metrics['start_time'] else 0
            
            return {
                'success': False,
                'error': str(e),
                'output_dir': output_dir if 'output_dir' in locals() else None,
                'training_metrics': self.training_metrics
            }


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(
        description="YOLOv5n Phase 1 (True Baseline) Training for VisDrone Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard Phase 1 training (100 epochs)
    python train_phase1_baseline.py

    # Quick test (20 epochs)
    python train_phase1_baseline.py --quick-test

    # Custom epoch count
    python train_phase1_baseline.py --epochs 150

Protocol: Version 2.0 - True Baseline Framework
Phase: 1 (No Augmentation Baseline)
Target: >18% mAP@0.5 for thesis requirements
        """
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=None,
        help='Number of training epochs (default: from config file)'
    )
    parser.add_argument(
        '--quick-test', 
        action='store_true',
        help='Run quick test with reduced settings (20 epochs)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (default: config/phase1_baseline/yolov5n_visdrone.yaml)'
    )
    
    args = parser.parse_args()
    
    # Display header
    print("="*80)
    print("YOLOv5n Phase 1 (True Baseline) Training - VisDrone Dataset")
    print("PROTOCOL: Version 2.0 - True Baseline Framework")
    print("="*80)
    print(f"Configuration: {args.config or 'Default'}")
    print(f"Epochs: {args.epochs or 'From config'}")
    print(f"Quick Test: {args.quick_test}")
    print(f"Phase: 1 (True Baseline - NO Augmentation)")
    print(f"Target: >18% mAP@0.5")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    try:
        # Initialize trainer
        config_path = Path(args.config) if args.config else None
        trainer = Phase1BaselineTrainer(config_path=config_path)
        
        # Execute training
        results = trainer.train(
            epochs=args.epochs,
            quick_test=args.quick_test
        )
        
        # Display results
        print("\n" + "="*80)
        if results['success']:
            print("[SUCCESS] YOLOv5n Phase 1 (True Baseline) Training Complete!")
            print(f"Results Directory: {results['output_dir']}")
            print(f"Training Duration: {results['training_metrics']['total_duration']:.1f}s")
            print("Expected: True baseline established for thesis methodology")
            print("Target: >18% mAP@0.5 with NO augmentation")
            print("Next Step: Phase 2 synthetic augmentation training")
        else:
            print("[ERROR] Training failed!")
            print(f"Error: {results['error']}")
            return 1
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Training script failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())