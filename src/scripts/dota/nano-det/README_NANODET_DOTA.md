# NanoDet Training Implementation for DOTA Dataset

This document outlines the implementation of NanoDet-Plus model training on the DOTA (Dataset for Object deTection in Aerial images) dataset. The implementation focuses on efficient aerial object detection using a lightweight architecture.

## Table of Contents
- [Overview](#overview)
- [Implementation Steps](#implementation-steps)
- [Project Structure](#project-structure)
- [Configuration Details](#configuration-details)
- [Training Process](#training-process)
- [Notes and Observations](#notes-and-observations)

## Overview

This implementation uses NanoDet-Plus with ShuffleNetV2 backbone for aerial image object detection. Key features:
- Lightweight architecture optimized for aerial view detection
- Support for 15 DOTA object classes
- Integration with PyTorch Lightning for training
- Custom data pipeline for DOTA dataset

## Implementation Steps

### 1. Environment Setup
- Created virtual environment: `venvs/dota/venvs/nanodet_dota_env`
- Dependencies managed through requirements file
- Environment activation: `.\venvs\dota\venvs\nanodet_dota_env\Scripts\activate.ps1`

### 2. Dataset Preparation
- Dataset structure:
```
data/my_dataset/dota/dota-v1.0/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train_original/
│   └── val_original/
└── nano-det/
    ├── images/
    │   ├── train/
    │   └── val/
    └── annotations/
        ├── instances_train.json
        └── instances_val.json
```

### 3. Model Architecture
Implemented NanoDet-Plus with:
- Backbone: ShuffleNetV2 (1.0x)
- Neck: GhostPAN
  - Input channels: [116, 232, 464]
  - Output channels: 96
  - Kernel size: 5
- Head: NanoDetPlusHead
  - 15 classes (DOTA-specific)
  - Quality Focal Loss
  - Distribution Focal Loss
  - GIoU Loss

## Project Structure

```
src/scripts/dota/nano-det/
├── README_NANODET_DOTA.md
├── train_nanodet.py
├── verify_annotations.py
└── convert_dota_to_nanodet.py

config/dota/nano-det/
└── nanodet_dota.yml
```

## Configuration Details

Key settings in `nanodet_dota.yml`:

```yaml
model:
  arch:
    name: NanoDetPlus
    backbone:
      name: ShuffleNetV2
      model_size: 1.0x
    fpn:
      name: GhostPAN
      out_channels: 96

data:
  train:
    input_size: [512,512]
    pipeline:
      scale: [0.6, 1.4]
      flip: 0.5

schedule:
  optimizer:
    name: AdamW
    lr: 0.001
  total_epochs: 300
  val_intervals: 10
```

## Training Process

1. **Data Loading**
   - Successfully implemented dataset loading
   - Proper index creation for both train and val sets

2. **Model Initialization**
   - Successful model architecture setup
   - Pretrained ShuffleNetV2 weights loaded
   - Total parameters: 1.6M

3. **Training Configuration**
   - Batch size: 4 per GPU
   - Workers per GPU: 2
   - Learning rate: 0.001 with AdamW
   - Validation interval: Every 10 epochs

## Notes and Observations

### Important Implementation Details
1. Dataset Conversion
   - DOTA annotations converted to COCO format
   - Proper handling of rotated bounding boxes
   - Image paths verified and validated

2. Model Setup
   - Model architecture properly initialized
   - All components (backbone, neck, head) verified
   - Loss functions implemented and tested

### Current Status
- Environment setup complete
- Dataset loading successful
- Model initialization verified
- Training can proceed on CPU (slower but functional)
- GPU support can be added later for faster training

### Future Improvements
1. GPU Support
   - Add CUDA support for faster training
   - Optimize batch size for GPU memory

2. Performance Monitoring
   - Add TensorBoard logging
   - Implement validation metrics tracking
   - Add checkpoint saving

### Troubleshooting Notes
- If encountering CUDA issues, training can proceed on CPU
- Virtual environment may need recreation if pip issues occur
- Verify dataset paths if encountering data loading issues

### Usage Instructions
1. Activate environment:
   ```bash
   .\venvs\dota\venvs\nanodet_dota_env\Scripts\activate.ps1
   ```

2. Start training:
   ```bash
   python src/scripts/dota/nano-det/train_nanodet.py --config config/dota/nano-det/nanodet_dota.yml
   ```

3. Monitor training:
   - Check `runs/nanodet_dota` for outputs
   - Model checkpoints saved in `runs/nanodet_dota/checkpoints`
   - Logs available in `runs/nanodet_dota/logs` 