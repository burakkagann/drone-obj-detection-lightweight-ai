# NanoDet Training for VisDrone Dataset

This document explains how to set up and train a NanoDet model on the VisDrone dataset for efficient drone-based object detection.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Monitoring and Evaluation](#monitoring-and-evaluation)
- [Directory Structure](#directory-structure)
- [Configuration Details](#configuration-details)
- [Troubleshooting](#troubleshooting)

## Overview

This implementation uses NanoDet-Plus with a ShuffleNetV2 backbone for efficient drone-based object detection. The model is designed to be lightweight and fast while maintaining good accuracy on drone-captured images.

Key Features:
- Lightweight architecture suitable for edge devices
- Optimized for drone-view object detection
- Support for 10 VisDrone object classes
- Integrated data augmentation pipeline
- PyTorch Lightning integration for training

## Prerequisites

- Python 3.7 or higher
- CUDA-capable GPU (recommended)
- VisDrone dataset
- Windows/Linux operating system

## Installation

1. Create and activate the virtual environment:
```bash
# Windows
python -m venv venvs/nanodet_env
.\venvs\nanodet_env\Scripts\activate.ps1

# Linux
python -m venv venvs/nanodet_env
source venvs/nanodet_env/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements/visdrone/requirements_nanodet.txt
```

3. Run the setup script:
```bash
python src/scripts/visdrone/nanodet/setup_nanodet.py
```

## Dataset Preparation

1. Download the VisDrone dataset and place it in the following structure:
```
data/my_dataset/visdrone/
├── VisDrone2019-DET-train/
│   ├── annotations/
│   └── images/
├── VisDrone2019-DET-val/
│   ├── annotations/
│   └── images/
└── VisDrone2019-DET-test/
    └── images/
```

2. Convert the dataset to NanoDet format:
```bash
python src/scripts/visdrone/nanodet/convert_visdrone.py
```

This will create the following structure:
```
data/my_dataset/visdrone/nanodet_format/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

## Model Architecture

The implementation uses NanoDet-Plus with the following components:

- **Backbone**: ShuffleNetV2 (0.5x)
  - Efficient feature extraction
  - Pretrained on ImageNet

- **Neck**: GhostPAN
  - Input channels: [48, 96, 192]
  - Output channels: 96
  - Kernel size: 5
  - Depthwise convolutions

- **Head**: NanoDetPlusHead
  - 10 classes (VisDrone)
  - Quality Focal Loss
  - Distribution Focal Loss
  - GIoU Loss

## Training Process

1. Review and modify the configuration file if needed:
```yaml
# config/nanodet/train_config_nanodet_visdrone.yaml
```

2. Start training:
```bash
python src/scripts/visdrone/nanodet/train_nanodet.py --config config/nanodet/train_config_nanodet_visdrone.yaml
```

Key Training Parameters:
- Input size: 416x416
- Batch size: 8 per GPU
- Initial learning rate: 0.001
- Optimizer: AdamW
- Total epochs: 300
- Learning rate schedule: Cosine annealing
- Validation interval: Every 10 epochs

## Monitoring and Evaluation

1. Monitor training with TensorBoard:
```bash
tensorboard --logdir runs/nanodet
```

2. Training outputs:
- Model checkpoints: `runs/nanodet/checkpoints/`
- Training logs: `runs/nanodet/logs/`
- Evaluation results: Saved every 10 epochs

## Directory Structure

```
├── config/
│   └── nanodet/
│       └── train_config_nanodet_visdrone.yaml
├── src/
│   ├── models/
│   │   └── nanodet/
│   └── scripts/
│       └── visdrone/
│           └── nanodet/
│               ├── setup_nanodet.py
│               ├── train_nanodet.py
│               └── convert_visdrone.py
└── data/
    └── my_dataset/
        └── visdrone/
            └── nanodet_format/
```

## Configuration Details

Key configuration sections in `train_config_nanodet_visdrone.yaml`:

1. Model Architecture:
```yaml
model:
  arch:
    name: NanoDetPlus
    backbone:
      name: ShuffleNetV2
      model_size: 0.5x
```

2. Data Augmentation:
```yaml
data:
  train:
    pipeline:
      perspective: 0.0
      scale: [0.6, 1.4]
      stretch: [[1, 1], [1, 1]]
      rotation: 0
      shear: 0
      translate: 0.2
      flip: 0.5
```

3. Training Schedule:
```yaml
schedule:
  optimizer:
    name: AdamW
    lr: 0.001
  total_epochs: 300
  lr_schedule:
    name: CosineAnnealingLR
```

## Troubleshooting

Common issues and solutions:

1. **CUDA Out of Memory**:
   - Reduce batch size in config
   - Use smaller input size
   - Use model with smaller backbone

2. **Import Errors**:
   - Ensure setup script was run
   - Check virtual environment activation
   - Verify all dependencies are installed

3. **Dataset Errors**:
   - Verify dataset structure
   - Check annotation format
   - Ensure all images are readable

For additional help, check the error messages in the training logs or create an issue in the repository. 