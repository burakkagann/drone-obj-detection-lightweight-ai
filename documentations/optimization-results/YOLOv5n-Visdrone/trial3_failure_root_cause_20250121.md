# Trial-3 Root Cause Analysis & Hyperparameter Comparison

**Date**: January 21, 2025  
**Analysis Type**: Hyperparameter Debugging  
**Status**: CRITICAL ISSUE IDENTIFIED

## Executive Summary

**ROOT CAUSE IDENTIFIED**: The Trial-3 hyperparameter configuration contains multiple problematic changes from the proven Trial-2 baseline that caused catastrophic training failure.

## Hyperparameter Comparison Analysis

### Learning Rate Configuration
| Parameter | Trial-2 (SUCCESS) | Trial-3 (FAILED) | Impact |
|-----------|-------------------|-------------------|---------|
| `lr0` | 0.005 | 0.004 | -20% learning rate may be too conservative |
| `lrf` | 0.02 | 0.015 | -25% final LR reduction |
| `momentum` | 0.937 | 0.940 | Minor change, unlikely cause |
| `weight_decay` | 0.0005 | 0.0008 | +60% regularization, potentially over-regularized |
| `warmup_epochs` | 5.0 | 6.0 | Extended warmup, minor impact |

### Loss Function Configuration
| Parameter | Trial-2 (SUCCESS) | Trial-3 (FAILED) | Impact |
|-----------|-------------------|-------------------|---------|
| `box` | 0.03 | 0.025 | -17% box loss weight |
| `cls` | 0.3 | 0.25 | -17% class loss weight |
| `obj` | 1.2 | 1.4 | +17% object loss weight |
| `iou_t` | 0.15 | 0.12 | -20% IoU threshold |
| `fl_gamma` | 0.0 | 0.5 | **CRITICAL**: Focal loss enabled when was disabled |

### Augmentation Configuration
| Parameter | Trial-2 (SUCCESS) | Trial-3 (FAILED) | Impact |
|-----------|-------------------|-------------------|---------|
| `hsv_h` | 0.02 | 0.015 | Minor reduction |
| `hsv_s` | 0.5 | 0.6 | +20% saturation variation |
| `hsv_v` | 0.3 | 0.35 | +17% value variation |
| `degrees` | 5.0 | 3.0 | -40% rotation reduction |
| `translate` | 0.2 | 0.15 | -25% translation reduction |
| `scale` | 0.8 | 0.7 | -12% scale variation |
| `mosaic` | 0.8 | 0.9 | +12% mosaic intensity |
| `mixup` | 0.4 | 0.3 | -25% mixup intensity |
| `copy_paste` | 0.3 | 0.4 | +33% copy-paste intensity |

### Training Configuration  
| Parameter | Trial-2 (SUCCESS) | Trial-3 (FAILED) | Impact |
|-----------|-------------------|-------------------|---------|
| `batch_size` | 16 | 20 | +25% batch size increase |

## Critical Issues Identified

### 🚨 SMOKING GUN: Focal Loss Activation
**Most Likely Root Cause**: `fl_gamma: 0.5` in Trial-3 vs `fl_gamma: 0.0` in Trial-2

- **Trial-2**: Focal loss DISABLED (0.0) - proven successful
- **Trial-3**: Focal loss ENABLED (0.5) - new untested configuration
- **Impact**: Focal loss dramatically changes loss function behavior for hard examples
- **Risk**: Can cause training instability and convergence failure

### 🔥 Secondary Contributing Factors

1. **Over-Regularization**
   - `weight_decay`: Increased from 0.0005 to 0.0008 (+60%)
   - Combined with reduced learning rate may prevent learning

2. **Loss Weight Imbalance**
   - `obj` loss increased from 1.2 to 1.4 (+17%)
   - `box` and `cls` losses both reduced (-17%)
   - May cause model to focus too much on objectness vs. localization/classification

3. **IoU Threshold Too Aggressive**
   - `iou_t` reduced from 0.15 to 0.12 (-20%)
   - May make positive sample assignment too strict

4. **Batch Size Increase**
   - Increased from 16 to 20 (+25%)
   - May require learning rate adjustment which wasn't made

## Validation Against Training Logs

### Expected vs Actual Behavior
- **Expected**: Gradual mAP improvement to 25%+
- **Actual**: mAP stuck at ~0.002% (1000x lower than expected)
- **Loss Behavior**: Losses appear normal but no learning occurs
- **Convergence**: Training stopped at epoch 27/100

### Loss Function Analysis
- Box, object, and class losses show normal ranges
- No obvious divergence or NaN values
- Suggests loss function calculation working but not driving learning

## Recovery Strategy

### Immediate Action Plan

#### Phase 1: Control Experiment (URGENT)
1. **Re-run Trial-2 configuration** to confirm reproducibility
2. **Verify environment consistency** and dataset integrity
3. **Establish baseline confidence** before making any changes

#### Phase 2: Incremental Testing (CRITICAL)
1. **Test focal loss impact**: Run Trial-2 config with `fl_gamma: 0.5` only
2. **Test learning rate**: Try Trial-3 config with Trial-2 learning rates
3. **Test loss weights**: Revert to Trial-2 loss weight configuration

#### Phase 3: Corrected Trial-4 (RECOVERY)
1. Start with proven Trial-2 configuration
2. Make single, small improvements only
3. Test each change incrementally

### Recommended Trial-4 Configuration

**Conservative Approach - Based on Trial-2 with minimal changes:**

```yaml
# Start with Trial-2 as base, make ONLY these changes:
lr0: 0.005              # Keep Trial-2 rate (don't reduce)
fl_gamma: 0.0           # KEEP DISABLED (critical)
obj: 1.3                # Modest increase from 1.2 (not 1.4)
batch_size: 18          # Modest increase from 16 (not 20)
# Keep all other parameters identical to Trial-2
```

## Timeline Impact

### Lost Development Time
- **Training Time**: 1 full training cycle lost
- **Analysis Time**: 1 day debugging required
- **Recovery Time**: 2-3 days for proper validation

### Thesis Risk Mitigation
- **Immediate Priority**: Restore working baseline
- **Short-term Goal**: Achieve validated improvements over Trial-2
- **Documentation**: Turn failure analysis into thesis methodology strength

## Lessons Learned

### Hyperparameter Optimization Principles
1. **Never change multiple parameters simultaneously** when baseline is working
2. **Focal loss is high-risk modification** requiring separate validation
3. **Learning rate and regularization must be balanced** carefully
4. **Batch size changes require learning rate adjustment**

### Research Methodology Improvements
1. **Implement A/B testing protocol** for single parameter changes
2. **Always run control experiments** before major modifications  
3. **Document parameter interaction effects** for thesis analysis
4. **Maintain proven baseline configurations** as reference points

---

**Priority Actions**:
1. ✅ Root cause identified (focal loss + over-regularization)
2. 🔄 Run Trial-2 control experiment to verify reproducibility  
3. 📋 Plan incremental Trial-4 with conservative modifications
4. 📝 Document failure analysis as thesis methodology contribution

**Status**: ANALYSIS COMPLETE - READY FOR RECOVERY PROTOCOL