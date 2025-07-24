# YOLOv8 Model Directory

This directory contains YOLOv8 model configurations, weights, and results for the drone object detection project.

## Directory Structure

```
YOLOv8/
├── configs/          # YOLOv8 configuration files
├── weights/          # Pretrained and trained model weights
├── results/          # Training results and metrics
└── README.md         # This file
```

## Model Information

- **Model**: YOLOv8n (nano variant)
- **Framework**: Ultralytics YOLOv8
- **Dataset**: VisDrone (10 classes)
- **Purpose**: Lightweight drone object detection for edge devices

## Usage

YOLOv8 models are managed through the ultralytics package:

```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO('yolov8n.pt')

# Train on VisDrone dataset
model.train(data='path/to/visdrone/config.yaml', epochs=20)

# Validate model
model.val()

# Export for deployment
model.export(format='onnx')
```

## Training Scripts

Training scripts are located in:
- `src/scripts/visdrone/YOLOv8n/`

## Configuration Files

- Dataset configuration: `config/visdrone/yolov8n_v1/`
- Hyperparameter configs: `configs/` (this directory)

## Model Weights

- Pretrained weights: `weights/yolov8n.pt`
- Trained weights: `weights/yolov8n_visdrone_*.pt`

## Results

Training results and metrics are saved in `results/` directory.