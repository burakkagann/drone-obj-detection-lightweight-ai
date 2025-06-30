# MobileNet-SSD Setup for DOTA Dataset

This document details the setup and validation process for training a MobileNet-SSD model on the DOTA (Dataset for Object deTection in Aerial images) dataset.

## Project Structure

```
src/
├── models/
│   └── mobilenet_ssd.py       # MobileNet-SSD model implementation
├── utils/
│   └── datasets.py            # Dataset loader for VOC format
└── scripts/
    └── dota/
        ├── test_mobilenet_ssd_dota_training.py  # Training validation script
        └── README_mobilenet_ssd_dota_setup.md   # This file
```

## Dataset Format
The DOTA dataset is organized in VOC format:
```
data/my_dataset/dota/dota-v1.0/mobilenet-ssd/
├── train/
│   ├── JPEGImages/     # Contains image files (.jpg or .png)
│   └── Annotations/    # Contains XML annotation files
└── val/
    ├── JPEGImages/
    └── Annotations/
```

## Model Architecture

### MobileNet-SSD Components:
1. **Backbone**: MobileNetV2 (pretrained on ImageNet)
   - Uses intermediate feature maps from layer 13 (96 channels)
   - Final feature map (1280 channels)

2. **Additional Feature Layers**:
   - Conv layers that progressively reduce spatial dimensions
   - Channel progression: 1280 → 512 → 256 → 256 → 256

3. **Detection Heads**:
   - Regression (loc) heads: Predict bounding box coordinates
   - Classification (conf) heads: Predict class probabilities
   - Each head processes features from different scales

### Channel Dimensions:
```python
Feature Maps:
- Backbone layer 13: 96 channels
- Backbone output: 1280 channels
- Additional layers: 512, 256, 256, 256 channels

Detection Heads (per feature map):
- Loc head: out_channels = num_anchors * 4
- Conf head: out_channels = num_anchors * (num_classes + 1)
```

## Setup Process and Debugging

### 1. Dataset Loading
- Initially failed due to directory structure mismatch
- Updated `VOCDataset` class to handle VOC format
- Added support for both .jpg and .png images
- Successfully loaded 1,409 training images

### 2. DataLoader Setup
- Added custom `collate_fn` to handle variable number of objects per image
- Set `batch_size=2` for testing
- Disabled multiprocessing (`num_workers=0`) for debugging

### 3. Model Architecture Debugging
- Fixed channel dimension mismatches:
  1. Initial attempt used incorrect backbone channels
  2. Updated to match MobileNetV2's actual architecture
  3. Adjusted additional layers to maintain proper channel progression

### 4. Training Components
- Optimizer: Adam with learning rate 0.001
- Loss Functions:
  - Localization: Smooth L1 Loss
  - Classification: Cross Entropy Loss

## Validation Results

The setup was validated through a test script that checks:

1. ✅ Dataset Loading
   - Successfully loads 1,409 training images
   - Proper parsing of XML annotations

2. ✅ DataLoader
   - Correct batch formation
   - Proper handling of variable-sized object lists

3. ✅ Model Creation
   - Successful initialization with pretrained weights
   - Correct architecture setup

4. ✅ Forward Pass
   - Proper tensor dimensions throughout the network
   - Successful feature extraction and detection

5. ✅ Training Loop
   - Successful gradient calculation
   - Weight updates confirmed
   - Loss values show expected behavior

## Notes for Implementation

1. **Data Preparation**:
   - Ensure images are in JPEGImages directory
   - Annotations must be in VOC XML format
   - Class names must match the model's expected classes

2. **Model Configuration**:
   - Input size: 300x300 pixels
   - Number of classes: 15 (plus background)
   - 6 anchor boxes per feature map location

3. **Training Tips**:
   - Start with small batch size (2-4) for validation
   - Monitor both localization and classification losses
   - Consider class weighting if dataset is imbalanced

4. **Future Improvements**:
   - Implement proper anchor box generation
   - Add validation loop
   - Add metrics calculation (mAP, precision, recall)
   - Add TensorBoard logging
   - Implement model checkpointing

## Common Issues and Solutions

1. **Channel Dimension Mismatch**:
   - Symptom: RuntimeError about channel dimensions
   - Solution: Ensure feature map channels match between layers
   - Check backbone output channels (1280 for MobileNetV2)

2. **Batch Collation**:
   - Symptom: Error about tensor sizes in batch
   - Solution: Use custom collate_fn for variable objects
   - Ensure all tensors in batch are properly padded

3. **Memory Issues**:
   - Start with small batch size
   - Gradually increase based on available memory
   - Monitor GPU memory usage if available

## References

1. MobileNetV2 Architecture: [torchvision.models.mobilenet_v2](https://pytorch.org/vision/stable/models/mobilenetv2.html)
2. SSD Paper: [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
3. DOTA Dataset: [DOTA: A Large-scale Dataset for Object Detection in Aerial Images](https://captain-whu.github.io/DOTA/) 