# YOLOv5n-Visdrone Trial-3 Optimization

**Date**: January 21, 2025  
**Model**: YOLOv5n  
**Dataset**: VisDrone  
**Baseline Performance**: 23.557% mAP@0.5 (Trial-2 BEST result)  
**Optimization Goal**: Push beyond 25% mAP@0.5 thesis excellence threshold  

## Background

After achieving 23.557% mAP@0.5 with Trial-2 (surpassing the initial 22.6% noted in CLAUDE.md), we performed detailed analysis of the training curves and implemented targeted optimizations to reach the 25% mAP@0.5 target required for thesis excellence.

## Training Curve Analysis from Trial-2

Based on `runs/train/yolov5n_trial2_BEST_23.557mAP/results.csv` analysis:

### Key Observations:
1. **Learning Rate Convergence**: Final LR reached very low values (~0.0002) suggesting potential for finer optimization
2. **Loss Patterns**: 
   - Box loss: 0.068-0.070 range (stable)
   - Object loss: 0.23-0.25 range (room for improvement)
   - Class loss: 0.0027-0.0030 range (well optimized)
3. **Performance Trajectory**: Steady improvement through epoch 18 where 23.557% was achieved
4. **Validation Stability**: Low overfitting, suggesting capacity for more aggressive optimization

## Trial-3 Optimization Strategy

### Learning Rate Optimization
```yaml
# Original Trial-2
lr0: 0.005
lrf: 0.02
momentum: 0.937
weight_decay: 0.0005

# Trial-3
lr0: 0.004              # Reduced by 20% for finer convergence
lrf: 0.015               # Reduced final LR for better fine-tuning
momentum: 0.940          # Increased momentum for stability
weight_decay: 0.0008     # Enhanced regularization (+60%)
```

**Rationale**: Trial-2 showed good convergence but reached very low LR too quickly. Trial-3 provides more controlled learning rate decay for better fine-tuning.

### Loss Weight Rebalancing
```yaml
# Original Trial-2
box: 0.03
cls: 0.3
obj: 1.2
iou_t: 0.15

# Trial-3
box: 0.025               # Slight reduction (-17%)
cls: 0.25                # Reduced class emphasis (-17%)
obj: 1.4                 # Increased objectness (+17%)
iou_t: 0.12              # Lowered IoU threshold (-20%)
```

**Rationale**: Training curves showed object loss had room for improvement. Increased objectness weight should improve detection confidence while reduced IoU threshold helps with small object recall.

### Augmentation Balance Refinement
```yaml
# Original Trial-2
mosaic: 0.8
mixup: 0.4
copy_paste: 0.3
degrees: 5.0
translate: 0.2
scale: 0.8

# Trial-3
mosaic: 0.9              # Increased context learning (+12.5%)
mixup: 0.3               # Reduced to preserve small objects (-25%)
copy_paste: 0.4          # Increased small object augmentation (+33%)
degrees: 3.0             # Reduced rotation to preserve small objects (-40%)
translate: 0.15          # Reduced translation (-25%)
scale: 0.7               # Reduced scale variation (-12.5%)
```

**Rationale**: Balance between augmentation strength and small object preservation. Higher mosaic for better context, but reduced geometric transformations to maintain small object integrity.

### Training Configuration Enhancement
```yaml
# Trial-3 additions
batch_size: 20           # Increased from 16 (+25%)
warmup_epochs: 6.0       # Extended warmup (+20%)
fl_gamma: 0.5            # Added mild focal loss for hard examples
```

**Rationale**: Larger batch size for more stable gradients, extended warmup for better convergence, focal loss to handle difficult small objects.

## Expected Performance Improvements

### Quantitative Predictions
- **Learning Rate Optimization**: +0.5-1.0% mAP@0.5
- **Loss Weight Rebalancing**: +0.8-1.2% mAP@0.5  
- **Augmentation Refinement**: +0.7-1.3% mAP@0.5
- **Training Configuration**: +0.5-0.8% mAP@0.5

**Total Expected Range**: 24.5-27.1% mAP@0.5

### Performance Thresholds
- **Minimum Success**: >24.0% mAP@0.5 (+0.5% improvement)
- **Target Excellence**: >25.0% mAP@0.5 (+1.5% improvement)
- **Outstanding Result**: >26.0% mAP@0.5 (+2.5% improvement)

## Implementation Files

### Configuration Files
- **Hyperparameters**: `config/visdrone/yolov5n_v1/hyp_visdrone_trial3.yaml`
- **Training Script**: `src/scripts/visdrone/YOLOv5n/Trial-3/run_trial3_training.ps1`

### Validation Protocol
1. **Quick Test**: 20-epoch validation run
2. **Baseline Comparison**: Against 23.557% mAP@0.5
3. **Success Criteria**: >24.0% mAP@0.5 for full 100-epoch training
4. **Performance Monitoring**: TensorBoard + comprehensive metrics

## Research Justification

### Literature Support
- **Small Object Detection**: Reduced IoU thresholds improve recall (YOLOv5 evolution studies)
- **Objectness Weighting**: Higher object loss weights benefit dense detection scenarios (ST-YOLO 2025)
- **Augmentation Balance**: Mosaic enhancement with geometric moderation optimal for aerial imagery (VisDrone-YOLOv8)
- **Learning Rate Scheduling**: Finer LR control critical for convergence beyond 20% mAP@0.5 (Hyperparameter evolution guide)

### Thesis Contribution
This optimization represents a systematic approach to pushing lightweight model performance beyond baseline thresholds through:
1. **Data-Driven Analysis**: Using training curves to identify optimization opportunities
2. **Balanced Enhancement**: Improving performance without compromising small object detection
3. **Methodical Validation**: Clear success criteria and comparison protocols

## Next Steps

1. **Execute Quick Test**: Run 20-epoch validation with enhanced v2 configuration
2. **Performance Analysis**: Compare results against 23.557% baseline
3. **Decision Point**: Proceed to 100-epoch training if >24.0% achieved
4. **Documentation**: Record final results and insights for thesis analysis

---

**Created**: January 21, 2025  
**Author**: Burak Kağan Yılmazer  
**Status**: Ready for Implementation  
**Files**: Configuration and training script prepared