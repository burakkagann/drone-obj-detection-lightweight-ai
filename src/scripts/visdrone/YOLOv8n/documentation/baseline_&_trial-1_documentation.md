# YOLOv8n Baseline and Trial-1 Training for VisDrone Dataset

This directory contains YOLOv8n baseline and synthetic augmentation training scripts for the VisDrone dataset as part of the Master's Thesis: "Robust Object Detection for Surveillance Drones in Low-Visibility Environments Using Lightweight AI Models."

## Directory Contents

### Training Scripts

#### Python Training Scripts
- **`train_yolov8n_baseline.py`** - Baseline training with default YOLOv8n hyperparameters
- **`train_yolov8n_trial1.py`** - Trial-1 optimization with hyperparameters adapted from YOLOv5n Trial-2

#### PowerShell Execution Scripts  
- **`run_yolov8n_baseline.ps1`** - Professional wrapper for baseline training
- **`run_yolov8n_trial1.ps1`** - Professional wrapper for Trial-1 training

#### Model Weights
- **`yolov8n.pt`** - YOLOv8n pretrained weights (auto-downloaded)

## Research Methodology Compliance

This implementation follows the thesis methodology phases:
- **Phase 2**: Baseline training on original datasets (no augmentation)
- **Phase 3**: Synthetic augmentation training for robustness testing

## Training Approach

### Baseline Training (20 epochs) - Phase 2
**Purpose**: Establish TRUE baseline performance on original VisDrone dataset without any augmentation.

**Key Settings**:
- **NO synthetic augmentation** (fog, night, blur, rain)
- **NO standard augmentation** (mosaic, mixup, HSV, geometric)
- **Original VisDrone images and labels ONLY**
- Pure dataset performance benchmark
- 640px image resolution
- Batch size: 16 (optimized for RTX 3060)

**Usage**:
```powershell
# Standard baseline training
.\run_yolov8n_baseline.ps1

# Quick validation test
.\run_yolov8n_baseline.ps1 -QuickTest

# Custom epoch count
.\run_yolov8n_baseline.ps1 -Epochs 30
```

### Trial-1 Training (50 epochs) - Phase 3
**Purpose**: Test synthetic augmentation impact and robustness using optimized hyperparameters.

**Key Features**:
- **Synthetic environmental augmentation** (fog, night, blur, rain simulation)
- **Enhanced standard augmentation** (mosaic 0.8, mixup 0.4, HSV, geometric)
- **Optimized hyperparameters** adapted from YOLOv5n Trial-2 success
- **Robustness focus** for low-visibility conditions
- **Methodology compliance** with Phase 3 requirements

**Optimizations** (adapted from YOLOv5n Trial-2):
- **Learning Rate**: 0.005 (reduced for small object detection)
- **Warmup**: 5 epochs (extended for training stability)
- **Resolution**: 640px (higher resolution for small objects)
- **Loss Weights**: Adapted for YOLOv8 architecture (box: 7.5, cls: 0.5, dfl: 1.5)

**Usage**:
```powershell
# Standard Trial-1 training
.\run_yolov8n_trial1.ps1

# Quick validation test
.\run_yolov8n_trial1.ps1 -QuickTest

# Extended training
.\run_yolov8n_trial1.ps1 -Epochs 100
```

## Environment Requirements

### Virtual Environment
- **Environment**: `yolov8n-visdrone_venv`
- **Location**: `.\venvs\yolov8n-visdrone_venv`
- **Activation**: `.\venvs\yolov8n-visdrone_venv\Scripts\Activate.ps1`

### Hardware Requirements
- **GPU**: NVIDIA RTX 3060 Laptop (5GB VRAM)
- **CUDA**: Version 11.8
- **PyTorch**: 2.7.1+cu118 (CUDA-enabled)

### Key Dependencies
- `ultralytics==8.3.169` (YOLOv8 framework)
- `torch==2.7.1+cu118` (CUDA-enabled PyTorch)
- `torchvision==0.22.1+cu118`
- `opencv-python==4.12.0.88`
- `numpy`, `matplotlib`, `PyYAML`, `tqdm`, `pandas`

## Dataset Configuration

**Dataset**: VisDrone (10 classes)
**Classes**: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor
**Configuration**: `config/visdrone/yolov5n_v1/yolov5n_visdrone_config.yaml`

## Expected Performance and Research Impact

### Baseline Performance (Phase 2)
- **Target**: Establish TRUE benchmark performance
- **Expected**: Raw dataset performance without augmentation benefits
- **Purpose**: Pure performance baseline for measuring augmentation impact
- **Methodology**: Original VisDrone dataset only

### Trial-1 Performance (Phase 3)
- **Target**: Demonstrate synthetic augmentation benefits
- **Expected Improvements over Baseline**:
  - Synthetic environmental augmentation: Improved robustness in low-visibility
  - Enhanced standard augmentation: +5-8% mAP@0.5
  - Optimized hyperparameters: +2-3% mAP@0.5
- **Research Value**: Quantifies augmentation impact for thesis
- **Methodology**: Original dataset + synthetic conditions

## Results and Logging

### Output Locations
- **Training Results**: `runs/train/yolov8n_baseline_*` or `runs/train/yolov8n_trial1_*`
- **Training Logs**: Timestamped log files in each results directory
- **Model Weights**: `best.pt` and `last.pt` in results/weights/ directory
- **Metrics**: `results.csv` with training metrics
- **Visualizations**: Loss curves, confusion matrices, sample predictions

### Key Metrics to Monitor
- **mAP@0.5**: Primary evaluation metric
- **mAP@0.5:0.95**: Comprehensive evaluation metric
- **Precision/Recall**: Per-class and overall performance
- **Training Loss**: Box, class, and DFL loss components
- **Training Speed**: Epochs/hour, images/second

## Training Configuration Comparison

### Baseline vs Trial-1 Settings

| Parameter | Baseline (Phase 2) | Trial-1 (Phase 3) | Purpose |
|-----------|-------------------|-------------------|---------|
| **Synthetic Augmentation** | ❌ Disabled | ✅ Enabled (fog, night, blur) | Test environmental robustness |
| **Standard Augmentation** | ❌ All disabled | ✅ Enhanced (mosaic, mixup, HSV) | Pure baseline vs optimized |
| **Learning Rate** | Default (0.01) | Optimized (0.005) | Baseline vs optimization |
| **Epochs** | 20 | 50 | Quick baseline vs full training |
| **Purpose** | Pure benchmark | Augmentation impact | Research comparison |

### Synthetic Augmentation Features (Trial-1 Only)
- **Environmental**: Fog simulation, night conditions, rain effects
- **Sensor Effects**: Motion blur, chromatic aberration, noise
- **Standard Enhanced**: `mosaic: 0.8, mixup: 0.4, copy_paste: 0.3`
- **Geometric**: `degrees: 5.0, translate: 0.2, scale: 0.8`

## Troubleshooting

### Common Issues
1. **CUDA not available**: Ensure CUDA-enabled PyTorch installation
2. **Out of memory**: Reduce batch size or image resolution
3. **Dataset not found**: Check dataset configuration paths
4. **Import errors**: Verify virtual environment activation

### Performance Optimization
1. **Enable AMP**: Automatic Mixed Precision for faster training
2. **Image caching**: Cache images in memory for faster data loading
3. **Worker optimization**: Adjust number of workers based on CPU cores
4. **Batch size tuning**: Find optimal batch size for available GPU memory

## Research Integration

### Thesis Methodology Compliance
- **Phase 2**: Baseline training on original datasets ✅
- **Phase 3**: Synthetic augmentation implementation ✅
- **Objective**: Quantify synthetic augmentation benefits for thesis
- **Timeline**: 40 days remaining for thesis completion
- **Research Question**: Environmental robustness through synthetic data

### Documentation Requirements
- Baseline vs Trial-1 performance comparison
- Synthetic augmentation impact quantification
- Low-visibility robustness metrics
- Edge device deployment feasibility
- Methodology validation for thesis

## Usage Examples

### Phase 2: Baseline Training (Original Dataset Only)
```powershell
# 1. Activate environment
cd "C:\Users\burak\OneDrive\Desktop\Git Repos\drone-obj-detection-lightweight-ai"
.\venvs\activate_yolov8n_visdrone.ps1

# 2. Run TRUE baseline (no augmentation)
.\src\scripts\visdrone\YOLOv8n\run_yolov8n_baseline.ps1
```

### Phase 3: Synthetic Augmentation Training
```powershell
# 3. Run Trial-1 (synthetic augmentation + optimization)
.\src\scripts\visdrone\YOLOv8n\run_yolov8n_trial1.ps1
```

### Research Workflow
```powershell
# Complete methodology workflow
.\run_yolov8n_baseline.ps1           # Phase 2: Pure baseline
.\run_yolov8n_trial1.ps1             # Phase 3: Augmented training

# Quick validation tests
.\run_yolov8n_baseline.ps1 -QuickTest  # 5-epoch baseline test
.\run_yolov8n_trial1.ps1 -QuickTest    # 10-epoch augmentation test

# Help and documentation
.\run_yolov8n_baseline.ps1 -Help       # Baseline training help
.\run_yolov8n_trial1.ps1 -Help         # Trial-1 training help
```

---

**Author**: Burak Kağan Yılmazer  
**Date**: January 2025  
**Thesis**: Robust Object Detection for Surveillance Drones in Low-Visibility Environments Using Lightweight AI Models  
**Methodology**: Phase 2 (Baseline) + Phase 3 (Synthetic Augmentation) Training Framework