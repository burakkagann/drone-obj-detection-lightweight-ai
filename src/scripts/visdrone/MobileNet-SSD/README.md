# MobileNet-SSD Training Framework for VisDrone Dataset

**Master's Thesis**: "Robust Object Detection for Surveillance Drones in Low-Visibility Environments Using Lightweight AI Models"

## Overview

This framework implements MobileNet-SSD training for the VisDrone dataset following the thesis experimental protocol:
- **Phase 2 (Baseline)**: Original dataset training with minimal augmentation
- **Phase 3 (Trial-1)**: Synthetic environmental augmentation training
- **Comparative Analysis**: Baseline vs augmented performance evaluation

## Architecture

**MobileNet-SSD**: Single Shot Multibox Detector with MobileNet backbone
- **Backbone**: MobileNetV2 (lightweight, efficient)
- **Detector**: SSD (fast, single-stage)
- **Target Performance**: >18% mAP@0.5 (protocol requirement)
- **Model Size**: <10MB (edge deployment ready)

## Framework Structure

```
MobileNet-SSD/
├── baseline/                    # Phase 2: TRUE baseline training
│   ├── train_mobilenet_ssd_baseline.py
│   └── run_mobilenet_ssd_baseline.ps1
├── trial-1/                     # Phase 3: Synthetic augmentation
│   ├── train_mobilenet_ssd_trial1.py
│   └── run_mobilenet_ssd_trial1.ps1
├── evaluation_metrics.py        # Comprehensive evaluation framework
└── README.md                    # This file
```

## Implementation Approach

### **PyTorch-Based Implementation**
- **Library**: PyTorch + torchvision (no Caffe dependencies)
- **Pre-trained Weights**: torchvision.models.detection.ssd300_vgg16
- **Custom Adaptation**: MobileNet backbone replacement
- **Data Pipeline**: PyTorch DataLoader with VisDrone format support

### **Training Configuration**
```yaml
Phase 2 (Baseline):
- Dataset: Original VisDrone only
- Augmentation: Minimal (resize, normalize only)
- Epochs: 50-100
- Target: >18% mAP@0.5

Phase 3 (Trial-1):
- Dataset: Original + synthetic environmental data
- Augmentation: Fog, night, blur, rain + standard augmentation
- Epochs: 50-100
- Target: >20% mAP@0.5 (improvement over baseline)
```

## Environment Setup

### **Virtual Environment**
```powershell
# Activate MobileNet-SSD environment
.\venvs\mobilenet_ssd_env\Scripts\Activate.ps1
```

### **Dependencies**
```
torch>=1.12.0
torchvision>=0.13.0
opencv-python>=4.6.0
numpy>=1.21.0
matplotlib>=3.5.0
tqdm>=4.64.0
tensorboard>=2.9.0
```

## Dataset Configuration

### **VisDrone Classes (10 classes)**
```
0: pedestrian
1: people  
2: bicycle
3: car
4: van
5: truck
6: tricycle
7: awning-tricycle
8: bus
9: motor
```

### **Data Paths**
```
data/my_dataset/visdrone/
├── train/images/     # Training images
├── train/labels/     # Training annotations (YOLO format)
├── val/images/       # Validation images
├── val/labels/       # Validation annotations
└── test/images/      # Test images
```

## Training Commands

### **Phase 2 (Baseline)**
```powershell
# Quick test (20 epochs)
.\src\scripts\visdrone\MobileNet-SSD\baseline\run_mobilenet_ssd_baseline.ps1 -QuickTest

# Full training (100 epochs)
.\src\scripts\visdrone\MobileNet-SSD\baseline\run_mobilenet_ssd_baseline.ps1 -Epochs 100
```

### **Phase 3 (Trial-1)**
```powershell
# Quick test (20 epochs)
.\src\scripts\visdrone\MobileNet-SSD\trial-1\run_mobilenet_ssd_trial1.ps1 -QuickTest

# Full training (100 epochs)
.\src\scripts\visdrone\MobileNet-SSD\trial-1\run_mobilenet_ssd_trial1.ps1 -Epochs 100
```

## Expected Performance

### **Protocol Targets**
| Phase | Dataset | Target mAP@0.5 | Model Size | FPS |
|-------|---------|----------------|------------|-----|
| **Phase 2** | Original | >18% | <10MB | >15 |
| **Phase 3** | Augmented | >20% | <10MB | >15 |

### **Comparison Models**
- **YOLOv8n**: 32.82% mAP@0.5 (reference)
- **YOLOv5n**: ~20% mAP@0.5 (target)
- **MobileNet-SSD**: >18% mAP@0.5 (minimum)

## Methodology Compliance

### **Section 4.1 - Evaluation Metrics**
✅ **Detection Accuracy**: mAP@0.5, mAP@0.5:0.95, Precision, Recall  
✅ **Inference Speed**: FPS, inference time (ms)  
✅ **Model Size**: File size (MB), memory usage  
✅ **Robustness**: Performance degradation analysis

### **Section 4.2 - Comparative Analysis**
✅ **Baseline vs Augmented**: Phase 2 vs Phase 3 comparison  
✅ **Cross-Model Comparison**: vs YOLOv5n, YOLOv8n  
✅ **Edge Performance**: Real-time capability assessment

## Implementation Status

- [ ] **Environment Setup**: Validate mobilenet_ssd_env dependencies
- [ ] **Baseline Implementation**: Phase 2 training framework
- [ ] **Trial-1 Implementation**: Phase 3 augmentation framework
- [ ] **Evaluation Framework**: Comprehensive metrics collection
- [ ] **Baseline Training**: Execute Phase 2 training
- [ ] **Trial-1 Training**: Execute Phase 3 training
- [ ] **Comparison Analysis**: Generate comparative reports

## Next Steps

1. **Immediate**: Validate PyTorch MobileNet-SSD implementation
2. **Phase 2**: Execute baseline training (original dataset)
3. **Phase 3**: Execute augmented training (synthetic data)
4. **Analysis**: Generate thesis-compliant comparative analysis
5. **Integration**: Add to multi-model comparison framework

## Technical Notes

### **Key Differences from YOLO**
- **Architecture**: Single-stage detector with anchor boxes
- **Backbone**: MobileNet (vs YOLO's custom backbone)
- **Training**: Different loss functions and optimization
- **Inference**: Different post-processing pipeline

### **Implementation Challenges**
- **Custom MobileNet-SSD**: May need custom implementation
- **Data Loading**: Convert YOLO format to SSD requirements
- **Anchor Configuration**: Optimize for VisDrone object scales
- **Performance Tuning**: Balance accuracy vs speed

---

**Author**: Burak Kağan Yılmazer  
**Date**: January 2025  
**Environment**: mobilenet_ssd_env  
**Status**: Framework preparation phase