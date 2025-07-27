# NanoDet Model Training Framework - VisDrone Dataset

**Protocol Version 2.0 Compliant Framework**

Master's Thesis: Robust Object Detection for Surveillance Drones in Low-Visibility Environments Using Lightweight AI Models

**Author**: Burak Kağan Yılmazer  
**Environment**: `nanodet_env`  
**Dataset**: VisDrone (10 classes)  
**Target**: Ultra-lightweight model (<3MB) with real-time performance

---

## 🎯 Framework Overview

This NanoDet training framework implements **Protocol Version 2.0 - True Baseline Framework** for systematic evaluation of synthetic environmental augmentation effectiveness on ultra-lightweight object detection models.

### Protocol Version 2.0 Compliance

✅ **Phase 1**: True Baseline (NO augmentation)  
✅ **Phase 2**: Environmental Robustness (Synthetic augmentation)  
✅ **Baseline Comparison**: Quantified improvement analysis  
✅ **Ultra-lightweight**: <3MB model target  
✅ **COCO Format**: Proper data format for NanoDet  

---

## 📁 Directory Structure

```
src/scripts/visdrone/nanodet/
├── baseline/                           # Phase 1: True Baseline
│   ├── train_nanodet_baseline.py      # Phase 1 training script
│   └── run_nanodet_baseline.ps1       # Phase 1 PowerShell wrapper
├── trial-1/                           # Phase 2: Environmental Robustness  
│   ├── train_nanodet_trial1.py        # Phase 2 training script
│   └── run_nanodet_trial1.ps1         # Phase 2 PowerShell wrapper
├── convert_visdrone_to_coco.py        # YOLO → COCO conversion
├── evaluation_metrics.py              # Comprehensive evaluation framework
└── README.md                          # This documentation
```

---

## 🔧 Setup and Prerequisites

### 1. Virtual Environment Activation

**CRITICAL**: Always activate the NanoDet environment before training:

```powershell
# Navigate to repository root
cd "C:\Users\burak\OneDrive\Desktop\Git Repos\drone-obj-detection-lightweight-ai"

# Activate NanoDet environment
.\venvs\nanodet_env\Scripts\Activate.ps1

# Verify activation (should show environment name in prompt)
```

### 2. Data Format Conversion

NanoDet requires COCO format data. Convert VisDrone YOLO format first:

```bash
# Run YOLO to COCO conversion
python src/scripts/visdrone/nanodet/convert_visdrone_to_coco.py \
    --dataset-root ./data/my_dataset/visdrone \
    --output-dir ./data/my_dataset/visdrone/nanodet_format
```

**Expected Output Structure**:
```
data/my_dataset/visdrone/nanodet_format/
├── train.json              # COCO annotations for training
├── val.json                # COCO annotations for validation  
├── test.json               # COCO annotations for testing
└── images/                 # Copied image files
    ├── train/
    ├── val/
    └── test/
```

### 3. Required Packages

Ensure all dependencies are installed in `nanodet_env`:

- `torch` (PyTorch)
- `torchvision`
- `opencv-python`
- `pycocotools`
- `albumentations` (for Phase 2 augmentation)
- `matplotlib`
- `numpy`

---

## 🚀 Training Execution

### Phase 1: True Baseline Training

**Objective**: Establish true baseline performance without any augmentation

```powershell
# Standard training (100 epochs)
.\src\scripts\visdrone\nanodet\baseline\run_nanodet_baseline.ps1

# Custom epochs
.\src\scripts\visdrone\nanodet\baseline\run_nanodet_baseline.ps1 -Epochs 150

# Quick test (20 epochs)
.\src\scripts\visdrone\nanodet\baseline\run_nanodet_baseline.ps1 -QuickTest
```

**Phase 1 Features**:
- ❌ **NO AUGMENTATION** (Protocol v2.0 requirement)
- ✅ Original VisDrone dataset only
- ✅ Minimal preprocessing (resize + normalize)
- ✅ Ultra-lightweight architecture
- ✅ Target: >12% mAP@0.5

### Phase 2: Environmental Robustness Training

**Objective**: Test synthetic environmental augmentation effectiveness

```powershell
# Standard training (100 epochs)
.\src\scripts\visdrone\nanodet\trial-1\run_nanodet_trial1.ps1

# With baseline comparison
.\src\scripts\visdrone\nanodet\trial-1\run_nanodet_trial1.ps1 -BaselineDir "runs\train\nanodet_phase1_baseline_20250727_143022"

# Quick test with comparison
.\src\scripts\visdrone\nanodet\trial-1\run_nanodet_trial1.ps1 -QuickTest -BaselineDir "runs\train\nanodet_phase1_baseline_20250727_143022"
```

**Phase 2 Features**:
- ✅ **Synthetic Environmental Augmentation**:
  - Fog simulation
  - Night/low-light conditions
  - Motion blur and noise
  - Advanced geometric transforms
- ✅ Enhanced standard augmentation
- ✅ Robustness features (dropout, regularization)
- ✅ Baseline comparison analysis
- ✅ Target: >18% mAP@0.5

---

## 📊 Methodology Compliance

### Protocol Version 2.0 Requirements

| Requirement | Phase 1 | Phase 2 | Status |
|-------------|---------|---------|--------|
| True Baseline (No Aug) | ✅ | ❌ | ✅ |
| Environmental Aug | ❌ | ✅ | ✅ |
| COCO Data Format | ✅ | ✅ | ✅ |
| Ultra-lightweight (<3MB) | ✅ | ✅ | ✅ |
| Baseline Comparison | N/A | ✅ | ✅ |
| Comprehensive Metrics | ✅ | ✅ | ✅ |

### Performance Targets

| Metric | Phase 1 Target | Phase 2 Target | Methodology Requirement |
|--------|----------------|----------------|------------------------|
| mAP@0.5 | >12% | >18% | Baseline vs Improved |
| Model Size | <3MB | <3MB | Ultra-lightweight |
| FPS | >10 | >10 | Real-time capable |
| Improvement | Baseline | +5% vs Phase 1 | Synthetic aug effectiveness |

---

## 🔬 Technical Implementation

### Model Architecture

**Ultra-Lightweight NanoDet Design**:
- **Backbone**: ShuffleNetV2-inspired (32→64→128 channels)
- **Head**: Simple detection head (64 channels + final layer)
- **Features**: BatchNorm, ReLU, AdaptiveAvgPool
- **Robustness**: Dropout layers (Phase 2), gradient clipping
- **Target Size**: <3MB for edge deployment

### Data Augmentation Strategy

**Phase 1 (True Baseline)**:
```python
# MINIMAL preprocessing only
transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

**Phase 2 (Environmental Robustness)**:
```python
# ENHANCED augmentation with environmental simulation
A.Compose([
    # Geometric augmentations
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
    
    # Environmental conditions
    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.MotionBlur(blur_limit=7, p=0.2),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    
    # Final processing
    A.Resize(416, 416),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

---

## 📈 Results and Analysis

### Training Outputs

Each training session generates:

```
runs/train/nanodet_phase{1|2}_{baseline|trial1}_YYYYMMDD_HHMMSS/
├── best_model.pth                 # Best performing model
├── final_model.pth                # Final epoch model
├── training_history.json          # Loss curves and metrics
├── checkpoint_epoch_*.pth         # Periodic checkpoints
└── nanodet_phase*_*.log           # Detailed training logs
```

### Evaluation Framework

Execute comprehensive evaluation using:

```bash
python src/scripts/visdrone/nanodet/evaluation_metrics.py \
    --model-path runs/train/nanodet_phase2_trial1_*/best_model.pth \
    --dataset-path data/my_dataset/visdrone/nanodet_format \
    --output-dir evaluation_results/nanodet
```

**Evaluation Metrics**:
- Detection accuracy (mAP@0.5, mAP@0.5:0.95)
- Inference speed (FPS, latency)
- Model efficiency (size, parameters)
- Hardware usage (CPU, GPU, memory)
- Per-class performance analysis

---

## 🎓 Thesis Integration

### Research Contributions

1. **Ultra-Lightweight Architecture**: <3MB model suitable for edge devices
2. **Environmental Robustness**: Synthetic augmentation effectiveness validation
3. **True Baseline Comparison**: Rigorous methodology compliance
4. **Real-time Performance**: >10 FPS capability demonstration

### Expected Findings

- **Phase 1 Performance**: ~12-15% mAP@0.5 (true baseline)
- **Phase 2 Improvement**: ~15-20% mAP@0.5 (5%+ improvement)
- **Model Efficiency**: <3MB with >10 FPS performance
- **Robustness Validation**: Quantified improvement under adverse conditions

### Methodology Section Integration

This framework directly supports thesis methodology sections:
- **Section 3.3**: Two-phase training approach
- **Section 4.1**: Comprehensive evaluation metrics
- **Section 4.2**: Comparative analysis framework
- **Section 5**: Results and discussion material

---

## 🛠️ Troubleshooting

### Common Issues

1. **Virtual Environment Issues**:
   ```bash
   # Ensure proper activation
   .\venvs\nanodet_env\Scripts\Activate.ps1
   
   # Verify packages
   pip list | findstr torch
   ```

2. **COCO Format Missing**:
   ```bash
   # Run conversion first
   python src/scripts/visdrone/nanodet/convert_visdrone_to_coco.py
   ```

3. **GPU Memory Issues**:
   - Reduce batch size in training scripts
   - Use gradient checkpointing
   - Enable mixed precision training

4. **Albumentations Import Error**:
   ```bash
   pip install albumentations
   ```

### Performance Optimization

- **GPU Utilization**: Monitor with `nvidia-smi`
- **Memory Usage**: Adjust batch size based on available memory
- **Training Speed**: Use mixed precision training for faster convergence

---

## 📝 Summary

This NanoDet framework provides:

✅ **Protocol v2.0 Compliance**: Two-phase methodology  
✅ **Ultra-lightweight Design**: <3MB target achievement  
✅ **Environmental Robustness**: Synthetic augmentation validation  
✅ **Comprehensive Evaluation**: All required metrics  
✅ **Thesis Integration**: Direct methodology support  

**Next Steps**: Execute Phase 1 → Phase 2 → Evaluation → Multi-model comparison

---

*Framework developed for Master's Thesis research - July 2025*