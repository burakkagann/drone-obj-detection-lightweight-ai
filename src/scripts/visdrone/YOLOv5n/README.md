# YOLOv5n Training Framework for VisDrone Dataset

**Master's Thesis**: "Robust Object Detection for Surveillance Drones in Low-Visibility Environments Using Lightweight AI Models"  
**Student**: Burak Kağan Yılmazer  
**Protocol**: Version 2.0 - True Baseline Framework  
**Model**: YOLOv5n (nano) - Lightweight edge-optimized architecture

## Overview

This directory contains the complete YOLOv5n training framework implementing the **True Baseline Protocol v2.0** for drone object detection research. The framework follows a two-phase comparative study to demonstrate the complete impact of environmental robustness methodology.

## Methodology Framework

### **Phase 1: True Baseline (Control Group)**
- **Purpose**: Establish absolute model performance reference point
- **Dataset**: Original VisDrone dataset only (no augmentation)
- **Augmentation**: DISABLED (all real-time augmentation set to 0.0)
- **Target**: >18% mAP@0.5
- **Rationale**: Pure model capability measurement for maximum effect size

### **Phase 2: Environmental Robustness (Treatment Group)**
- **Purpose**: Demonstrate complete methodology impact vs. true baseline
- **Dataset**: Environmental augmented dataset (original + synthetic conditions)
- **Augmentation**: ENABLED (standard real-time + environmental pre-processing)
- **Target**: >25% mAP@0.5 (+7pp improvement)
- **Rationale**: Show total research contribution and robustness improvement

## Directory Structure

```
YOLOv5n/
├── baseline/                    # Phase 1: True Baseline Training
│   ├── train_yolov5n_baseline.py    # Python training script
│   └── run_yolov5n_baseline.ps1     # PowerShell wrapper script
├── trial-1/                     # Phase 2: Environmental Robustness
│   ├── train_yolov5n_trial1.py      # Python training script
│   └── run_yolov5n_trial1.ps1       # PowerShell wrapper script
├── evaluation_metrics.py        # Comprehensive evaluation framework
└── README.md                    # This documentation
```

## Training Execution

### **Phase 1: True Baseline Training**

**Prerequisites**:
- YOLOv5n environment activated: `.\venvs\yolov5n_env\Scripts\Activate.ps1`
- Original VisDrone dataset available

**Quick Start**:
```powershell
# Navigate to repository root
cd "C:\Users\burak\OneDrive\Desktop\Git Repos\drone-obj-detection-lightweight-ai"

# Activate environment
.\venvs\yolov5n_env\Scripts\Activate.ps1

# Run Phase 1 baseline training
.\src\scripts\visdrone\YOLOv5n\baseline\run_yolov5n_baseline.ps1
```

**Training Features**:
- ✅ **NO Augmentation**: All real-time augmentation disabled
- ✅ **Original Dataset**: VisDrone images and labels only
- ✅ **True Baseline**: Pure model performance measurement
- ✅ **Target**: >18% mAP@0.5 (Protocol v2.0 requirement)

### **Phase 2: Environmental Robustness Training**

**Prerequisites**:
- Phase 1 baseline completed and results available
- Environmental augmented dataset (if available) or original dataset for fallback

**Quick Start**:
```powershell
# Same environment and repository setup as Phase 1

# Run Phase 2 environmental robustness training
.\src\scripts\visdrone\YOLOv5n\trial-1\run_yolov5n_trial1.ps1
```

**Training Features**:
- ✅ **Environmental Dataset**: Original + synthetic conditions (fog, night, blur, rain)
- ✅ **Real-time Augmentation**: Mosaic, mixup, HSV, geometric enabled
- ✅ **Optimized Hyperparameters**: Reduced LR, balanced loss weights
- ✅ **Target**: >25% mAP@0.5 (+7pp improvement from baseline)

## Configuration Parameters

### **Phase 1: True Baseline Configuration**
```yaml
# Hyperparameters (NO AUGMENTATION)
lr0: 0.01                        # Default learning rate
mosaic: 0.0                      # Disabled
mixup: 0.0                       # Disabled
hsv_h: 0.0                       # Disabled
hsv_s: 0.0                       # Disabled
hsv_v: 0.0                       # Disabled
degrees: 0.0                     # Disabled
translate: 0.0                   # Disabled
scale: 0.0                       # Disabled
fliplr: 0.0                      # Disabled
copy_paste: 0.0                  # Disabled
```

### **Phase 2: Environmental Robustness Configuration**
```yaml
# Hyperparameters (FULL AUGMENTATION)
lr0: 0.005                       # Reduced for stability
mosaic: 0.8                      # Enabled
mixup: 0.4                       # Enabled
hsv_h: 0.02                      # Color variation
hsv_s: 0.5                       # Saturation variation
hsv_v: 0.3                       # Value variation
degrees: 5.0                     # Rotation
translate: 0.2                   # Translation
scale: 0.8                       # Scaling
fliplr: 0.5                      # Horizontal flip
copy_paste: 0.3                  # Copy-paste augmentation
```

## Evaluation Framework

### **Comprehensive Metrics Collection**
The `evaluation_metrics.py` module implements thesis-compliant evaluation:

**Detection Accuracy**:
- mAP@0.5, mAP@0.5:0.95
- Precision, Recall, F1-Score
- Per-class performance analysis

**Inference Speed**:
- FPS (Frames Per Second)
- Average inference time (ms)
- Throughput analysis

**Model Efficiency**:
- Model file size (MB)
- Memory usage during inference
- Parameter count

**Hardware Configuration**:
- GPU specifications
- Memory availability
- CUDA version compatibility

### **Usage Example**:
```python
from evaluation_metrics import YOLOv5nEvaluationMetrics

# Initialize evaluator
evaluator = YOLOv5nEvaluationMetrics(
    model_path="runs/train/yolov5n_baseline_*/weights/best.pt",
    dataset_config="config/visdrone/yolov5n_baseline/yolov5n_visdrone_baseline.yaml",
    output_dir="evaluation_results"
)

# Run complete evaluation
results = evaluator.run_complete_evaluation()
```

## Expected Performance Targets

### **Phase 1 (True Baseline)**
- **Target mAP@0.5**: >18% (Protocol v2.0 requirement)
- **Model Size**: <7MB (edge deployment ready)
- **Inference Speed**: >20 FPS (real-time capability)
- **Memory Usage**: <2GB GPU memory during training

### **Phase 2 (Environmental Robustness)**
- **Target mAP@0.5**: >25% (+7pp improvement from baseline)
- **Robustness**: <15% performance degradation under adverse conditions
- **Speed Maintenance**: >20 FPS (maintained real-time performance)
- **Cross-condition Consistency**: σ < 3% mAP variance

## Research Impact Demonstration

### **Thesis Contribution Quantification**
```
Phase 1 Baseline:    ~18% mAP@0.5 (pure model capability)
                           ↓
Phase 2 Robustness:  ~25% mAP@0.5 (complete methodology)
                           ↓
Research Impact:     +7 percentage points absolute improvement
Effect Size:         38% relative improvement
```

### **Comparative Analysis Framework**
1. **Baseline Establishment**: True performance reference (Phase 1)
2. **Methodology Validation**: Complete approach effectiveness (Phase 2)
3. **Statistical Significance**: p < 0.05 for all improvements
4. **Practical Impact**: Edge device deployment readiness
5. **Academic Contribution**: Reproducible framework for future research

## Troubleshooting

### **Common Issues**

**Environment Activation Fails**:
```powershell
# Fix execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Recreate environment if needed
python -m venv venvs\yolov5n_env
.\venvs\yolov5n_env\Scripts\Activate.ps1
pip install -r requirements.txt
```

**CUDA Memory Issues**:
- Reduce batch size in training scripts (default: 16 → 8)
- Enable gradient checkpointing
- Clear CUDA cache between training runs

**Dataset Path Issues**:
- Ensure VisDrone dataset is in: `data\my_dataset\visdrone\`
- Verify train/val/test splits exist
- Check label file format compatibility

### **Performance Validation**

**Phase 1 Validation Checklist**:
- [ ] mAP@0.5 > 18% achieved
- [ ] NO augmentation confirmed in logs
- [ ] Training converged without overfitting
- [ ] Results saved to `runs/train/yolov5n_baseline_*`

**Phase 2 Validation Checklist**:
- [ ] mAP@0.5 > 25% achieved
- [ ] +7pp improvement over Phase 1 confirmed
- [ ] Augmentation enabled and functioning
- [ ] Environmental robustness demonstrated

## Integration with Thesis Framework

This YOLOv5n framework integrates seamlessly with the overall thesis research:

**Cross-Model Consistency**: Same structure as YOLOv8n, MobileNet-SSD, NanoDet
**Protocol Compliance**: Full adherence to Version 2.0 True Baseline Framework
**Reproducibility**: Complete documentation and configuration management
**Thesis Integration**: Direct results export for comparative analysis

## References and Documentation

- **Protocol Document**: `documentations/methodology/thesis_experimental_protocol.md`
- **Methodology**: `Z_Methodology.txt`
- **YOLOv5 Documentation**: Official Ultralytics YOLOv5 repository
- **VisDrone Dataset**: Official VisDrone challenge documentation

---

**Status**: Ready for execution following Protocol v2.0  
**Next Steps**: Execute Phase 1 baseline training  
**Timeline**: 2-3 days for complete two-phase training and evaluation  
**Expected Outcome**: +7pp mAP improvement demonstration for thesis impact