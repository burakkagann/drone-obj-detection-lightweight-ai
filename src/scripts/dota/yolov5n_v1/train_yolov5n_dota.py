#!/usr/bin/env python3
"""
Train YOLOv5n model on DOTA v1.0 dataset
This script handles the training of YOLOv5n model on the DOTA dataset
"""

import os
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO

def train_yolov5n_dota():
    """Train YOLOv5n model on DOTA dataset"""
    print("Starting YOLOv5n training on DOTA dataset...")
    
    # Set up paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent.parent
    data_yaml = project_root / 'config' / 'dota' / 'yolov5n_v1' / 'dota_v1_yolov5n.yaml'
    
    # Verify config file exists
    if not data_yaml.exists():
        raise FileNotFoundError(f"Config file not found at {data_yaml}")
    
    # Training hyperparameters
    hyperparameters = {
        'epochs': 100,  # Total training epochs
        'batch': 16,    # Batch size (-1 for AutoBatch)
        'imgsz': 640,   # Image size for training
        'patience': 50,  # Early stopping patience
        'device': 0 if torch.cuda.is_available() else 'cpu',  # Device to train on (GPU or CPU)
        'workers': 8,   # Number of worker threads for data loading
        'project': str(project_root / 'models' / 'dota' / 'yolov5n_v1'),  # Save results to project/name
        'name': 'train',  # Save results to project/name
        'exist_ok': True,  # Existing project/name ok, do not increment
        'pretrained': True,  # Use pretrained YOLOv5n weights
        'optimizer': 'auto',  # Optimizer (auto, SGD, Adam, etc.)
        'verbose': True,  # Print verbose output
        'seed': 42,  # Random seed for reproducibility
        'deterministic': True,  # Enable deterministic mode
        'single_cls': False,  # Train as single-class dataset
        'rect': False,  # Rectangular training
        'cos_lr': True,  # Cosine LR scheduler
        'close_mosaic': 10,  # Disable mosaic augmentation for final epochs
        'resume': False,  # Resume training from last checkpoint
        'amp': True,  # Automatic Mixed Precision
        'fraction': 1.0,  # Dataset fraction to train on
        'profile': False,  # Profile ONNX and TensorRT speeds
    }
    
    # Create output directory
    os.makedirs(hyperparameters['project'], exist_ok=True)
    
    # Print config file path and contents
    print(f"\nUsing config file: {data_yaml}")
    with open(data_yaml, 'r') as f:
        print("\nConfig contents:")
        print(f.read())
    
    # Initialize model
    print("\nInitializing YOLOv5n model...")
    model = YOLO('yolov5n.pt')  # Load pretrained YOLOv5n model
    
    # Train the model
    try:
        print("\nStarting training...")
        results = model.train(
            data=str(data_yaml),
            **hyperparameters
        )
        print("Training completed successfully!")
        
        # Save training results
        results_file = Path(hyperparameters['project']) / 'training_results.yaml'
        with open(results_file, 'w') as f:
            yaml.dump(results, f)
            
    except Exception as e:
        print(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    train_yolov5n_dota() 