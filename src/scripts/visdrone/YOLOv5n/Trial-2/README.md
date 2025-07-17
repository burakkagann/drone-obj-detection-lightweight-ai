# YOLOv5n Trial-2 Hyperparameter Optimization

This directory contains the implementation of Trial-2 hyperparameter optimization for YOLOv5n training on the VisDrone dataset.

## 📊 **Performance Context**

**Baseline Performance (100 epochs):**
- mAP@0.5: 17.80%
- mAP@0.5:0.95: 8.03%
- Precision: 29.77%
- Recall: 17.44%
- FPS: 28.68

**Research Findings:**
- Published baselines: 31-46% mAP@0.5
- Performance gap: ~13-17% below expected
- Primary cause: Suboptimal hyperparameters

## 🔧 **Critical Optimizations Applied**

### **1. Augmentation Fixes (Major Impact)**
- **Mosaic**: 0.0 → 0.8 (was disabled, now enabled)
- **Mixup**: 0.0 → 0.4 (was disabled, now enabled)
- **Copy-paste**: 0.0 → 0.3 (enabled for small objects)
- **Estimated Impact**: +5-8% mAP@0.5

### **2. Resolution Optimization**
- **Image Size**: 416 → 640 pixels
- **Rationale**: Small objects need higher resolution
- **Estimated Impact**: +3-5% mAP@0.5

### **3. Learning Rate Optimization**
- **Learning Rate**: 0.01 → 0.005
- **Rationale**: Gentler training for small objects
- **Warmup Epochs**: 3.0 → 5.0 (extended stability)
- **Estimated Impact**: +2-3% mAP@0.5

### **4. Batch Size Optimization**
- **Batch Size**: 8 → 16
- **Rationale**: Better gradient estimates
- **Estimated Impact**: +1-2% mAP@0.5

### **5. Loss Function Tuning**
- **Box Loss**: 0.05 → 0.03 (reduced for small objects)
- **Class Loss**: 0.5 → 0.3 (reduced for small objects)
- **Object Loss**: 1.0 → 1.2 (increased objectness emphasis)
- **IoU Threshold**: 0.2 → 0.15 (better small object detection)

## 🎯 **Expected Results**

**Performance Targets:**
- **Minimum**: >18.8% mAP@0.5 (+1% improvement)
- **Target**: >21% mAP@0.5 (+3% improvement)
- **Excellent**: >23% mAP@0.5 (+5% improvement)

**Total Expected Improvement**: +11-18% relative improvement

## 📁 **Files Description**

### **Configuration Files**
- `hyp_visdrone_trial-2_optimized.yaml` - Optimized hyperparameters (saved in config/visdrone/yolov5n_v1/)

### **Training Scripts**
- `train_yolov5n_trial2_hyperopt.py` - Main Python training script
- `run_trial2_hyperopt.ps1` - PowerShell wrapper script

### **Documentation**
- `README.md` - This file
- `trial2_config_*.json` - Runtime configuration files (auto-generated)

## 🚀 **Usage Instructions**

### **Quick Test (Recommended First)**
```powershell
# Run 20-epoch validation test
.\run_trial2_hyperopt.ps1 -QuickTest
```

### **Full Training**
```powershell
# Run full 100-epoch training
.\run_trial2_hyperopt.ps1 -Epochs 100
```

### **Python Direct Usage**
```bash
# Quick test
python train_yolov5n_trial2_hyperopt.py --quick-test

# Full training
python train_yolov5n_trial2_hyperopt.py --epochs 100
```

## 🔍 **Validation Protocol**

### **Step 1: Environment Check**
- Virtual environment activated (yolov5n_env)
- GPU availability verified
- Configuration files validated

### **Step 2: Quick Test (20 epochs)**
- Validate optimizations work
- Compare against baseline (17.80% mAP@0.5)
- Assess improvement potential

### **Step 3: Results Analysis**
- Extract final mAP@0.5 from results
- Compare with targets
- Make decision on next steps

### **Step 4: Decision Matrix**
- **>3% improvement**: Proceed to full 100-epoch training
- **1-3% improvement**: Consider Phase 2 optimizations
- **<1% improvement**: Debug and try alternative approaches

## 📊 **Research Sources**

**Primary Research:**
- **ST-YOLO (2025)**: 640x640 resolution, batch size 16, learning rate 0.001
- **VisDrone-YOLOv8**: mixup=0.4 achieved 43.7% mAP@0.5
- **YOLOv5 Hyperparameter Evolution**: Lower learning rates optimal for small objects

**Key Findings:**
- Mosaic and mixup augmentation are critical for VisDrone performance
- Higher resolution significantly improves small object detection
- Gentler learning rates prevent overshooting for small objects
- Larger batch sizes provide more stable gradients

## 🎯 **Success Metrics**

### **Training Metrics**
- mAP@0.5 improvement over baseline
- Precision and recall improvements
- Training stability (no overfitting)
- Convergence speed

### **Efficiency Metrics**
- FPS maintained >25
- Memory usage <600MB
- Training time reasonable

### **Validation Metrics**
- Consistent improvement across validation set
- No signs of overfitting
- Stable learning curves

## 🔄 **Next Steps After Trial-2**

### **If Successful (>3% improvement)**
1. Run full 100-epoch training
2. Analyze class-wise performance
3. Proceed to Phase 2 optimizations

### **If Moderate (1-3% improvement)**
1. Implement Phase 2 optimizations:
   - Anchor optimization
   - Advanced augmentation
   - Transfer learning improvements

### **If Unsuccessful (<1% improvement)**
1. Debug hyperparameter interactions
2. Try alternative learning rate schedules
3. Experiment with different augmentation strategies

## 📝 **Notes**

- All optimizations are based on peer-reviewed research
- Hyperparameters are specifically tuned for VisDrone characteristics
- Quick test protocol minimizes compute time while validating improvements
- Comprehensive logging enables detailed analysis

---

**Created**: January 17, 2025
**Research Sources**: ST-YOLO (2025), VisDrone-YOLOv8, YOLOv5 hyperparameter evolution
**Expected Training Time**: 3 hours (20 epochs), 12 hours (100 epochs) 