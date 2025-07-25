# NanoDet Training Framework for VisDrone Dataset

**Master's Thesis**: "Robust Object Detection for Surveillance Drones in Low-Visibility Environments Using Lightweight AI Models"

## Overview

This framework implements NanoDet training for the VisDrone dataset following the thesis experimental protocol:
- **Phase 2 (Baseline)**: Original dataset training with minimal augmentation
- **Phase 3 (Trial-1)**: Synthetic environmental augmentation training
- **Comparative Analysis**: Baseline vs augmented performance evaluation

## Architecture

**NanoDet**: Ultra-lightweight anchor-free object detector
- **Backbone**: ShuffleNetV2 (0.5x) - minimal parameter count
- **Neck**: GhostPAN - efficient feature pyramid
- **Head**: NanoDetPlusHead - anchor-free detection
- **Target Performance**: >15% mAP@0.5 (protocol requirement)
- **Model Size**: <3MB (ultra-lightweight edge deployment)

## Framework Structure

```
nanodet/
├── baseline/                    # Phase 2: TRUE baseline training
│   ├── train_nanodet_baseline.py
│   └── run_nanodet_baseline.ps1
├── trial-1/                     # Phase 3: Synthetic augmentation
│   ├── train_nanodet_trial1.py
│   └── run_nanodet_trial1.ps1
├── evaluation_metrics.py        # Comprehensive evaluation framework
└── README.md                    # This file
```

## Implementation Approach

### **PyTorch-Based Implementation**
- **Library**: PyTorch + NanoDet official implementation
- **Pre-trained Weights**: COCO pre-trained NanoDet model
- **Custom Adaptation**: VisDrone 10-class configuration
- **Data Pipeline**: COCO format with VisDrone adaptation

### **Training Configuration**
```yaml
Phase 2 (Baseline):
- Dataset: Original VisDrone only
- Augmentation: Minimal (resize, normalize only)
- Epochs: 100-150
- Target: >15% mAP@0.5

Phase 3 (Trial-1):
- Dataset: Original + synthetic environmental data
- Augmentation: Fog, night, blur, rain + standard augmentation
- Epochs: 100-150
- Target: >17% mAP@0.5 (improvement over baseline)
```

## Environment Setup

### **Virtual Environment**
```powershell
# Activate NanoDet environment
.\venvs\nanodet_env\Scripts\Activate.ps1
```

### **Dependencies**
```
torch>=1.12.0
torchvision>=0.13.0
pytorch-lightning>=1.8.0
opencv-python>=4.6.0
numpy>=1.21.0
matplotlib>=3.5.0
tqdm>=4.64.0
tensorboard>=2.9.0
pycocotools>=2.0.6
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

### **Data Format**
NanoDet uses COCO format annotations:
```
data/my_dataset/visdrone/nanodet_format/
├── annotations/
│   ├── train.json        # COCO format training annotations
│   ├── val.json          # COCO format validation annotations
│   └── test.json         # COCO format test annotations
└── images/
    ├── train/           # Training images
    ├── val/             # Validation images
    └── test/            # Test images
```

## Training Commands

### **Phase 2 (Baseline)**
```powershell
# Quick test (20 epochs)
.\src\scripts\visdrone\nanodet\baseline\run_nanodet_baseline.ps1 -QuickTest

# Full training (150 epochs)
.\src\scripts\visdrone\nanodet\baseline\run_nanodet_baseline.ps1 -Epochs 150
```

### **Phase 3 (Trial-1)**
```powershell
# Quick test (20 epochs)
.\src\scripts\visdrone\nanodet\trial-1\run_nanodet_trial1.ps1 -QuickTest

# Full training (150 epochs)
.\src\scripts\visdrone\nanodet\trial-1\run_nanodet_trial1.ps1 -Epochs 150
```

## Expected Performance

### **Protocol Targets**
| Phase | Dataset | Target mAP@0.5 | Model Size | FPS |
|-------|---------|----------------|------------|-----|
| **Phase 2** | Original | >15% | <3MB | >30 |
| **Phase 3** | Augmented | >17% | <3MB | >30 |

### **Comparison Models**
- **YOLOv8n**: 32.82% mAP@0.5 (reference)
- **YOLOv5n**: ~20% mAP@0.5 (target)
- **MobileNet-SSD**: >18% mAP@0.5 (comparison)
- **NanoDet**: >15% mAP@0.5 (ultra-lightweight)

## Methodology Compliance

### **Section 4.1 - Evaluation Metrics**
✅ **Detection Accuracy**: mAP@0.5, mAP@0.5:0.95, Precision, Recall  
✅ **Inference Speed**: FPS, inference time (ms)  
✅ **Model Size**: File size (MB), memory usage  
✅ **Robustness**: Performance degradation analysis

### **Section 4.2 - Comparative Analysis**
✅ **Baseline vs Augmented**: Phase 2 vs Phase 3 comparison  
✅ **Cross-Model Comparison**: vs YOLOv5n, YOLOv8n, MobileNet-SSD  
✅ **Edge Performance**: Ultra-lightweight deployment assessment

## Implementation Status

- [ ] **Environment Setup**: Validate nanodet_env dependencies
- [ ] **Data Preparation**: Convert VisDrone to COCO format
- [ ] **Baseline Implementation**: Phase 2 training framework
- [ ] **Trial-1 Implementation**: Phase 3 augmentation framework
- [ ] **Evaluation Framework**: Comprehensive metrics collection
- [ ] **Baseline Training**: Execute Phase 2 training
- [ ] **Trial-1 Training**: Execute Phase 3 training
- [ ] **Comparison Analysis**: Generate comparative reports

## Next Steps

1. **Immediate**: Setup PyTorch NanoDet environment and dependencies
2. **Data Preparation**: Convert VisDrone to COCO format for NanoDet
3. **Phase 2**: Execute baseline training (original dataset)
4. **Phase 3**: Execute augmented training (synthetic data)
5. **Analysis**: Generate thesis-compliant comparative analysis
6. **Integration**: Add to multi-model comparison framework

## Technical Notes

### **Key Differences from YOLO/MobileNet-SSD**
- **Architecture**: Anchor-free detection (vs anchor-based)
- **Backbone**: ShuffleNetV2 (ultra-lightweight)
- **Format**: COCO JSON annotations (vs YOLO txt or VOC XML)
- **Training**: PyTorch Lightning framework
- **Inference**: Different post-processing pipeline

### **Implementation Challenges**
- **Data Format**: Need COCO format conversion from existing data
- **Model Size**: Ensure <3MB for ultra-lightweight classification
- **Performance**: Balance extreme efficiency with detection accuracy
- **Framework**: Integration with PyTorch Lightning training loop

### **Ultra-Lightweight Optimization**
- **Model Pruning**: Consider post-training optimization
- **Quantization**: INT8 optimization for edge deployment
- **Knowledge Distillation**: Optional teacher-student training
- **Architecture Search**: Optimize backbone for VisDrone specifics

---

**Author**: Burak Kağan Yılmazer  
**Date**: January 2025  
**Environment**: nanodet_env  
**Status**: Framework preparation phase  
**Priority**: Ultra-lightweight edge deployment ready