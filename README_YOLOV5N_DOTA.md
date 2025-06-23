# YOLOv5n Training on DOTA Dataset

This document describes the process of training a YOLOv5n model on the DOTA (Dataset for Object deTection in Aerial images) v1.0 dataset.

## Project Structure

```
drone-obj-detection-lightweight-ai/
├── config/
│   └── dota/
│       └── yolov5n_v1/
│           └── dota_v1_yolov5n.yaml    # Dataset configuration
├── data/
│   └── my_dataset/
│       └── dota/
│           └── dota-v1.0/
│               ├── images/             # Original images
│               │   ├── train/
│               │   └── val/
│               ├── labels/             # Original DOTA format labels
│               │   ├── train_original/
│               │   └── val_original/
│               └── yolov5n/            # Converted YOLO format dataset
│                   ├── images/
│                   │   ├── train/
│                   │   └── val/
│                   └── labels/
│                       ├── train/
│                       └── val/
├── models/
│   └── dota/
│       └── yolov5n_v1/
│           └── train/                  # Training outputs
├── src/
│   └── scripts/
│       └── dota/
│           └── yolov5n_v1/
│               ├── convert_dota_to_yolo.py  # Dataset conversion script
│               └── train_yolov5n_dota.py    # Training script
└── venvs/
    └── dota/
        └── venvs/
            └── yolov5n_dota_env/      # Python virtual environment
```

## Setup Process

1. Create Python Virtual Environment:
```bash
python -m venv venvs/dota/venvs/yolov5n_dota_env
```

2. Activate Virtual Environment:
```bash
# Windows
.\\venvs\\dota\\venvs\\yolov5n_dota_env\\Scripts\\Activate.ps1

# Linux/Mac
source venvs/dota/venvs/yolov5n_dota_env/bin/activate
```

3. Install Dependencies:
```bash
pip install ultralytics torch numpy==1.24.3 opencv-python tqdm
```

## Dataset Preparation

1. Dataset Structure:
   - The DOTA v1.0 dataset contains 1409 training images and 456 validation images
   - Original format: text files with polygon coordinates (x1,y1,x2,y2,x3,y3,x4,y4,class_id,difficult)
   - Target format: YOLO format (class_id, x_center, y_center, width, height)

2. Dataset Configuration:
   - Configuration file: `config/dota/yolov5n_v1/dota_v1_yolov5n.yaml`
   - Contains dataset paths and 15 DOTA classes:
     - plane, ship, storage-tank, baseball-diamond, tennis-court
     - basketball-court, ground-track-field, harbor, bridge, large-vehicle
     - small-vehicle, helicopter, roundabout, soccer-ball-field, swimming-pool

3. Convert Dataset:
```bash
python src/scripts/dota/yolov5n_v1/convert_dota_to_yolo.py
```
This script:
- Converts polygon coordinates to bounding boxes
- Normalizes coordinates to [0,1] range
- Creates YOLO format labels
- Organizes images and labels in YOLOv5 expected structure

## Training Process

1. Training Configuration:
   - Model: YOLOv5n (nano) - lightweight version
   - Image size: 640x640
   - Batch size: 16
   - Epochs: 100
   - Early stopping patience: 50
   - Optimizer: Auto
   - Learning rate scheduler: Cosine
   - Data augmentation: Mosaic, RandomAffine

2. Start Training:
```bash
python src/scripts/dota/yolov5n_v1/train_yolov5n_dota.py
```

The training script:
- Uses pretrained YOLOv5n weights
- Saves checkpoints and training metrics
- Implements early stopping
- Supports both CPU and GPU training
- Saves results in `models/dota/yolov5n_v1/train/`

## Training Outputs

The training process generates:
- Model checkpoints (best.pt, last.pt)
- Training metrics (results.csv)
- Confusion matrix
- Training plots (loss curves, PR curves)
- Validation results

## Monitoring Training

Training progress can be monitored through:
1. Terminal output showing batch progress and metrics
2. TensorBoard logs (if enabled)
3. Results files in the training output directory

## Notes

1. Hardware Requirements:
   - Minimum 8GB RAM
   - GPU recommended but not required
   - SSD storage recommended for faster data loading

2. Performance Optimization:
   - Adjust batch size based on available memory
   - Modify number of workers based on CPU cores
   - Enable GPU training if available

3. Troubleshooting:
   - If encountering CUDA out of memory: reduce batch size
   - If training is slow: increase number of workers
   - If validation metrics are poor: adjust training hyperparameters

## Future Improvements

1. Implement multi-GPU training support
2. Add support for custom augmentation strategies
3. Integrate with experiment tracking platforms
4. Add model export functionality (ONNX, TensorRT) 