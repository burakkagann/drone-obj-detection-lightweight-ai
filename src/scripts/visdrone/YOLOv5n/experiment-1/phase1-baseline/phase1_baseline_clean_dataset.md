# YOLOv5n Phase 1 Baseline Results - Clean Dataset

**Master's Thesis**: Robust Object Detection for Surveillance Drones in Low-Visibility Environments  
**Model**: YOLOv5n (nano)  
**Dataset**: VisDrone (Clean/Original)  
**Protocol**: Version 2.0 - True Baseline Framework  
**Date**: July 30, 2025  

## Training Configuration

### Phase 1 Requirements Compliance ✅
- **No Augmentation**: All data augmentation disabled (hsv, rotation, mosaic, etc.)
- **Original Dataset**: VisDrone training set without synthetic modifications
- **Minimal Preprocessing**: Resize to 640×640 and normalization only
- **Pure Model Capability**: Baseline performance measurement achieved

### Training Parameters
- **Epochs**: 100
- **Batch Size**: 8
- **Image Size**: 640×640
- **Optimizer**: SGD with cosine LR
- **Learning Rate**: 0.01 (initial)
- **Device**: NVIDIA GeForce RTX 3060 Laptop GPU (6GB)
- **Training Duration**: 12.535 hours (45,372.6 seconds)

## Model Architecture
- **Model**: YOLOv5n
- **Layers**: 157
- **Parameters**: 1,772,695
- **Gradients**: 0
- **GFLOPs**: 4.2
- **Model Size**: 3.8MB (optimized)

## Training Results Summary

### Overall Performance
| Metric | Value |
|--------|-------|
| **mAP@0.5** | **0.249 (24.9%)** ✅ |
| **mAP@0.5:0.95** | 0.120 (12.0%) |
| **Precision** | 0.410 |
| **Recall** | 0.217 |
| **Target** | >18% mAP@0.5 |
| **Status** | **ACHIEVED** ✅ |

### Class-wise Performance
| Class | Images | Instances | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|--------|-----------|-----------|--------|---------|---------------|
| All | 548 | 40,168 | 0.410 | 0.217 | 0.249 | 0.120 |
| Pedestrian | 548 | 1,410 | 0.199 | 0.00496 | 0.0177 | 0.00569 |
| People | 548 | 38,758 | 0.620 | 0.430 | 0.480 | 0.234 |

### Key Observations
1. **Overall mAP@0.5 of 24.9%** exceeds the thesis requirement of >18%
2. **People class** performs significantly better (48.0% mAP@0.5) than pedestrian class (1.77%)
3. **Class imbalance**: 38,758 people instances vs 1,410 pedestrian instances
4. **Precision-Recall trade-off**: Higher precision (0.41) but lower recall (0.217)

## Model Outputs
- **Best Weights**: `runs/train/yolov5n_phase1_baseline_20250730_034928/weights/best.pt`
- **Last Weights**: `runs/train/yolov5n_phase1_baseline_20250730_034928/weights/last.pt`
- **Training Logs**: `runs/train/yolov5n_phase1_baseline_20250730_034928/`

## Validation Dataset Statistics
- **Total Images**: 548
- **Total Instances**: 40,168
- **Average Instances per Image**: 73.3

## Training Efficiency
- **Training Time**: 12.535 hours
- **Epochs per Hour**: 7.98
- **Average Time per Epoch**: 7.5 minutes
- **GPU Utilization**: Efficient with RTX 3060 6GB

## Phase 1 Baseline Establishment ✅

### Methodology Compliance
- ✅ **Protocol v2.0** requirements met
- ✅ **True baseline** established with no augmentation
- ✅ **Pure model performance** measured on clean dataset
- ✅ **Reference point** created for Phase 2 comparison

### Baseline Performance Summary
- **Primary Metric (mAP@0.5)**: 24.9%
- **Secondary Metric (mAP@0.5:0.95)**: 12.0%
- **Inference Speed**: TBD (requires separate measurement)
- **Model Size**: 3.8MB (deployment-ready)

## Next Steps
1. **Test on synthetic dataset** with environmental augmentation
2. **Measure degradation** under fog, night, blur conditions
3. **Establish baseline degradation** metrics
4. **Compare with Phase 2** augmented training results

## Conclusion
Phase 1 baseline training successfully completed with 24.9% mAP@0.5, exceeding the 18% target. This establishes a solid foundation for evaluating the effectiveness of synthetic augmentation in Phase 2. The model shows strong performance on the "people" class but struggles with "pedestrian" detection, likely due to class imbalance and smaller object sizes.

---
*Generated: July 30, 2025*  
*Training ID: yolov5n_phase1_baseline_20250730_034928*