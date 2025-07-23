# YOLOv5n Trial-5 Strategy: "Recall Enhancement"

**Date**: July 23, 2025  
**Objective**: Address persistent low recall issue while maintaining high precision  
**Target**: Achieve >24.5% mAP@0.5 with improved recall (20% → 25-30%)

## Strategic Foundation

### Performance Analysis Summary
| Trial | mAP@0.5 | Precision | Recall | Status | Key Insights |
|-------|---------|-----------|--------|--------|--------------|
| Trial-2 | 23.557% | 81.08% | 19.71% | ✅ BASELINE | Proven configuration |
| Trial-3 | 0.002% | - | - | ❌ FAILURE | Focal loss disaster |
| Trial-4 | 23.70% | 81.15% | 19.97% | ✅ CRASH | Slight improvement trend |

### Core Problem Identification
**Consistent Pattern**: High precision (~80%) but low recall (~20%) across all successful trials.

**Root Cause Analysis**:
- Model is highly conservative in detections
- High confidence threshold for positive predictions
- Potentially too restrictive anchor assignments
- IoU threshold may be too strict for small objects

## Trial-5 Strategic Approach

### Primary Objective: **Recall Enhancement**
- **Current Recall**: ~20%
- **Target Recall**: 25-30%
- **Precision Target**: Maintain 75-80%
- **Overall mAP@0.5 Target**: 24.5-26%

### Scientific Hypothesis
By reducing detection thresholds and strengthening objectness confidence, we can achieve more positive detections while maintaining precision through the model's inherent conservative nature.

## Hyperparameter Optimization Strategy

### Core Modifications (vs Trial-4 baseline)

#### 1. Detection Threshold Optimization
```yaml
# RECALL ENHANCEMENT - Primary changes
iou_t: 0.10              # REDUCED from 0.15 (more lenient matching)
anchor_t: 3.5            # REDUCED from 4.0 (more anchor assignments)
```

**Scientific Rationale**:
- Lower IoU threshold enables more ground truth boxes to be matched with predictions
- Reduced anchor threshold allows more anchors to be assigned to targets
- Both changes should increase positive sample assignment during training

#### 2. Loss Function Rebalancing
```yaml
# OBJECTNESS EMPHASIS
obj: 1.5                 # INCREASED from 1.25 (stronger objectness signal)
box: 0.025               # REDUCED from 0.03 (less restrictive box regression)
cls: 0.3                 # UNCHANGED (proven effective)
```

**Scientific Rationale**:
- Higher object loss weight emphasizes detection confidence
- Lower box loss weight reduces penalty for slightly imperfect localizations
- Class loss maintained at proven effective level

#### 3. Training Stability Enhancements
```yaml
# EXTENDED TRAINING
epochs: 25               # INCREASED from 20 (more convergence time)
warmup_epochs: 6.0       # INCREASED from 5.0 (better stability)
```

**Scientific Rationale**:
- Extended training allows model more time to balance precision-recall trade-off
- Longer warmup provides more stable gradient updates early in training

### Preserved Proven Settings

#### Critical Lessons from Trial-3 Failure
```yaml
# ABSOLUTELY UNCHANGED - Critical for stability
fl_gamma: 0.0            # DISABLED - focal loss caused Trial-3 disaster
lr0: 0.005               # Proven learning rate
momentum: 0.937          # Optimal SGD momentum
weight_decay: 0.0005     # Proven regularization
```

#### Essential Augmentation Settings
```yaml
# PROVEN PERFORMANCE DRIVERS - Unchanged
mosaic: 0.8              # Essential for mAP improvement
mixup: 0.4               # Critical for robustness
copy_paste: 0.3          # Important for small object detection
batch_size: 16           # Proven batch size for stability
img_size: 640            # Optimal resolution for VisDrone
```

## Expected Performance Impact

### Quantitative Predictions

#### Recall Improvement Analysis
- **IoU threshold reduction (0.15→0.10)**: Expected +2-4% recall
- **Anchor threshold reduction (4.0→3.5)**: Expected +1-3% recall  
- **Increased objectness loss (1.25→1.5)**: Expected +1-2% recall
- **Combined effect**: +4-9% recall improvement

#### Precision Impact Assessment
- **Risk**: Slight precision decrease due to more liberal detection
- **Mitigation**: Model's inherent conservative nature should limit decrease
- **Expected**: 75-80% precision (vs current 81%)

#### Overall mAP@0.5 Projection
- **Conservative**: 24.2% (+0.5% from Trial-4)
- **Realistic**: 24.8% (+1.1% from Trial-4)
- **Optimistic**: 25.5% (+1.8% from Trial-4)

### Success Criteria

#### Minimum Success Threshold
- **mAP@0.5**: >23.8% (beat Trial-4 baseline)
- **Recall**: >21% (meaningful improvement)
- **Precision**: >75% (acceptable trade-off)

#### Target Success Metrics
- **mAP@0.5**: >24.5% (significant improvement)
- **Recall**: >25% (substantial enhancement)
- **Precision**: >77% (minimal degradation)

#### Excellent Performance Goals
- **mAP@0.5**: >25.5% (thesis target achievement)
- **Recall**: >28% (major breakthrough)
- **Precision**: >78% (maintained high precision)

## Risk Assessment and Mitigation

### Low Risk Factors ✅
- **Conservative base**: Building on proven Trial-2/4 foundation
- **Small changes**: Incremental modifications only
- **Scientific basis**: Each change has clear theoretical justification
- **Fallback available**: Can revert to Trial-4 configuration if needed

### Moderate Risk Factors ⚠️
- **Precision trade-off**: May see slight precision decrease
- **Convergence time**: Extended training may require more resources
- **Hyperparameter interaction**: Multiple changes may have unexpected interactions

### Mitigation Strategies
1. **Monitor early epochs**: Check for convergence issues in first 5 epochs
2. **Precision tracking**: Stop training if precision drops below 70%
3. **Comparative analysis**: Direct comparison with Trial-4 metrics throughout training
4. **Checkpoint strategy**: Save models every 5 epochs for rollback capability

## Implementation Timeline

### Phase 1: Configuration Setup (30 minutes)
- Create Trial-5 hyperparameter YAML
- Set up training scripts and environment
- Verify configuration against Trial-4 baseline

### Phase 2: Training Execution (6-8 hours)
- 25-epoch training run with monitoring
- Real-time performance tracking
- Early stopping if critical issues detected

### Phase 3: Analysis and Validation (2 hours)
- Comprehensive performance comparison with previous trials
- Precision-recall curve analysis
- Model weight and checkpoint validation

## Success Validation Protocol

### Real-time Monitoring
- **Epoch 5**: Check for stable loss convergence
- **Epoch 10**: Evaluate precision-recall balance
- **Epoch 15**: Assess mAP@0.5 trajectory
- **Epoch 20**: Compare with Trial-4 performance
- **Epoch 25**: Final evaluation and model selection

### Post-training Analysis
1. **Quantitative comparison**: Direct metrics comparison with all previous trials
2. **Qualitative assessment**: Visual inspection of detection quality
3. **Statistical significance**: Determine if improvements are meaningful
4. **Model characteristics**: Analyze learned features and behaviors

## Next Steps After Trial-5

### If Successful (>24.5% mAP@0.5)
- **Multi-model comparison**: Proceed to YOLOv8n, MobileNet-SSD, NanoDet training
- **Edge device testing**: Begin deployment optimization
- **Thesis analysis**: Start comprehensive results documentation

### If Moderate Success (23.8-24.5% mAP@0.5)
- **Fine-tuning**: Minor hyperparameter adjustments
- **Trial-6**: Consider alternative recall enhancement strategies
- **Augmentation optimization**: Focus on data augmentation improvements

### If Insufficient (<23.8% mAP@0.5)
- **Rollback analysis**: Identify which changes caused performance degradation
- **Alternative strategies**: Consider architecture modifications or different datasets
- **Research pivot**: Evaluate alternative lightweight models

## Conclusion

Trial-5 represents a scientifically grounded approach to addressing the fundamental precision-recall imbalance observed across all successful training runs. By focusing specifically on recall enhancement while preserving proven configuration elements, this strategy offers the highest probability of achieving meaningful performance improvements within the remaining thesis timeline.

The conservative nature of the modifications, combined with the strong foundation of Trial-2/4 results, minimizes risk while maximizing potential for breakthrough performance that could significantly enhance the thesis contribution.

---

**Documentation**: This strategy will be implemented with comprehensive logging and analysis to support thesis methodology and results sections.

**Timeline Impact**: Trial-5 execution fits within the 40-day thesis completion timeline and provides clear path forward regardless of outcome.

**Research Value**: Win-win scenario - either achieves performance breakthrough or provides valuable insights into model optimization limitations.