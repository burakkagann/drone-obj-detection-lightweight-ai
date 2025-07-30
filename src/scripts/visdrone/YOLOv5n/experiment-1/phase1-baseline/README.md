# YOLOv5n Experiment-1: Phase 1 (True Baseline) Training Framework

**Master's Thesis**: Robust Object Detection for Surveillance Drones in Low-Visibility Environments  
**Protocol**: Version 2.0 - True Baseline Framework  
**Author**: Burak Kağan Yılmazer  
**Date**: July 2025  

## Overview

This directory contains the complete implementation of **Phase 1 (True Baseline)** training for YOLOv5n on the VisDrone dataset, following Protocol v2.0 requirements for establishing absolute model performance baselines.

## Phase 1 Requirements Compliance

### ✅ True Baseline Training Features

- **ORIGINAL DATASET ONLY**: No synthetic augmentation (fog, night, blur, rain)
- **NO REAL-TIME AUGMENTATION**: All augmentation disabled (hsv, rotation, translation, etc.)
- **MINIMAL PREPROCESSING**: Resize to 640x640 and normalize only
- **PURE MODEL CAPABILITY**: Baseline performance measurement without enhancement
- **METHODOLOGY COMPLIANCE**: Protocol v2.0 Phase 1 requirements

### 🎯 Training Objectives

1. **Establish True Baseline**: Measure pure YOLOv5n model capability
2. **Reference Point Creation**: Foundation for Phase 2 comparison
3. **Thesis Methodology**: Support comparative analysis framework
4. **Performance Target**: >18% mAP@0.5 for thesis requirements

## Directory Structure

```
experiment-1/
├── phase1-baseline/
│   ├── train_phase1_baseline.py    # Main training script
│   └── run_phase1_baseline.ps1     # PowerShell wrapper
├── evaluation_metrics.py           # Comprehensive evaluation
├── README.md                       # This documentation
└── config/                         # Configuration files
    └── yolov5n_phase1_baseline.yaml
```

## Quick Start

### Prerequisites

1. **Environment**: Activate yolov5n_visdrone_env virtual environment
   ```powershell
   .\venvs\visdrone\yolov5n_visdrone_env\Scripts\Activate.ps1
   ```

2. **Dataset**: Ensure VisDrone dataset is prepared in `data/my_dataset/visdrone/`

3. **Dependencies**: PyTorch, YOLOv5, OpenCV, etc. (installed in venv)

### Training Execution

#### Option 1: PowerShell Wrapper (Recommended)
```powershell
# Standard Phase 1 training (100 epochs)
.\src\scripts\visdrone\YOLOv5n\experiment-1\phase1-baseline\run_phase1_baseline.ps1

# Quick test (20 epochs)
.\src\scripts\visdrone\YOLOv5n\experiment-1\phase1-baseline\run_phase1_baseline.ps1 -QuickTest

# Custom configuration
.\src\scripts\visdrone\YOLOv5n\experiment-1\phase1-baseline\run_phase1_baseline.ps1 -Epochs 150 -Verbose
```

#### Option 2: Direct Python Execution
```python
# Navigate to project root first
cd "C:\Users\burak\OneDrive\Desktop\Git Repos\drone-obj-detection-lightweight-ai"

# Activate environment
.\venvs\visdrone\yolov5n_visdrone_env\Scripts\Activate.ps1

# Run training
python src\scripts\visdrone\YOLOv5n\experiment-1\phase1-baseline\train_phase1_baseline.py

# With options
python src\scripts\visdrone\YOLOv5n\experiment-1\phase1-baseline\train_phase1_baseline.py --epochs 100 --quick-test
```

## Configuration Details

### Dataset Configuration (`config/phase1_baseline/yolov5n_visdrone.yaml`)

**Key Phase 1 Settings:**
- **Dataset**: VisDrone (10 classes)
- **Input Size**: 640x640 pixels
- **Batch Size**: 8 (optimized for RTX 3060 6GB)
- **Epochs**: 100 (default)
- **ALL AUGMENTATION DISABLED**: Essential for true baseline

**Critical Baseline Parameters:**
```yaml
# ALL AUGMENTATION DISABLED FOR PHASE 1
hsv_h: 0.0         # HSV-Hue (DISABLED)
hsv_s: 0.0         # HSV-Saturation (DISABLED)
hsv_v: 0.0         # HSV-Value (DISABLED)
degrees: 0.0       # Rotation (DISABLED)
translate: 0.0     # Translation (DISABLED)
scale: 0.0         # Scale (DISABLED)
mosaic: 0.0        # Mosaic (DISABLED)
mixup: 0.0         # Mixup (DISABLED)
# ... all other augmentation parameters set to 0.0
```

### Model Configuration
- **Architecture**: YOLOv5n (nano - lightweight)
- **Pre-trained Weights**: yolov5n.pt (COCO pre-trained)
- **Optimizer**: SGD with cosine learning rate scheduling
- **Loss Function**: Standard YOLOv5 loss (box + class + objectness)

## Training Process

### 1. Environment Validation
- PyTorch and CUDA compatibility check
- Dataset path validation
- Virtual environment verification
- GPU memory assessment

### 2. Configuration Setup
- Dynamic configuration file generation
- Hyperparameter validation for Phase 1 compliance
- Dataset format verification

### 3. Training Execution
- YOLOv5 training with disabled augmentation
- Comprehensive logging and monitoring
- Checkpoint saving (best and last models)
- Memory optimization for stable training

### 4. Results Collection
- Training metrics (loss curves, mAP progression)
- Model checkpoints and weights
- Comprehensive logging files
- Configuration backups

## Expected Results

### Performance Targets
- **Primary Target**: >18% mAP@0.5
- **Training Time**: 2-4 hours (RTX 3060, 100 epochs)
- **Model Size**: ~15 MB (YOLOv5n weights)
- **Inference Speed**: >25 FPS (640x640 input)

### Output Structure
```
runs/train/yolov5n_phase1_baseline_YYYYMMDD_HHMMSS/
├── weights/
│   ├── best.pt           # Best model checkpoint
│   └── last.pt           # Final model checkpoint
├── results.csv           # Training metrics
├── hyp.yaml             # Hyperparameters used
├── opt.yaml             # Training options
├── train_batch*.jpg     # Training batch visualizations
├── val_batch*.jpg       # Validation batch visualizations
└── results.png          # Training curves
```

## Evaluation and Analysis

### Comprehensive Evaluation
```python
# Run evaluation metrics
python src\scripts\visdrone\YOLOv5n\experiment-1\evaluation_metrics.py \
    --model runs/train/yolov5n_phase1_baseline_*/weights/best.pt \
    --data config/phase1_baseline/yolov5n_visdrone.yaml \
    --output evaluation_results
```

### Key Evaluation Metrics
1. **Detection Accuracy**: mAP@0.5, mAP@0.5:0.95, Precision, Recall
2. **Inference Performance**: FPS, latency breakdown
3. **Model Efficiency**: Size, parameters, memory usage
4. **Hardware Compatibility**: GPU/CPU performance analysis

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch_size from 8 to 4 or 2
   - Enable gradient checkpointing
   - Close other GPU-intensive applications

2. **Dataset Path Errors**
   - Verify VisDrone dataset structure
   - Check absolute paths in configuration
   - Ensure train/val/test directories exist

3. **Environment Issues**
   - Verify yolov5n_visdrone_env activation
   - Check YOLOv5 installation and imports
   - Validate PyTorch CUDA compatibility

4. **Training Failures**
   - Check log files for detailed error messages
   - Verify configuration file syntax
   - Ensure sufficient disk space for outputs

### Performance Optimization

1. **Memory Optimization**
   - Use batch_size=4 for 6GB GPU
   - Set workers=0 on Windows
   - Enable mixed precision training (amp=true)

2. **Speed Optimization**
   - Cache images for faster data loading
   - Use SSD storage for dataset
   - Optimize number of data loading workers

## Integration with Thesis Framework

### Phase 1 → Phase 2 Transition
1. **Baseline Established**: Phase 1 provides reference performance
2. **Configuration Template**: Serves as base for Phase 2 augmentation
3. **Comparative Analysis**: Results enable quantified improvement measurement
4. **Methodology Validation**: Demonstrates protocol compliance

### Multi-Model Framework
This Phase 1 implementation serves as the template for:
- **YOLOv8n Phase 1**: Similar baseline training
- **MobileNet-SSD Phase 1**: Lightweight alternative baseline
- **NanoDet Phase 1**: Ultra-lightweight baseline
- **Cross-Model Comparison**: Unified evaluation framework

## Next Steps

### Immediate Actions
1. **Execute Phase 1 Training**: Run baseline training to completion
2. **Results Analysis**: Evaluate baseline performance metrics
3. **Documentation**: Record results for thesis methodology

### Phase 2 Preparation
1. **Synthetic Augmentation**: Implement environmental condition simulation
2. **Enhanced Training**: Phase 2 with fog, night, blur augmentation
3. **Comparative Analysis**: Quantify augmentation impact vs baseline

### Multi-Model Expansion
1. **YOLOv8n Implementation**: Apply same framework structure
2. **MobileNet-SSD Training**: Lightweight alternative comparison
3. **NanoDet Evaluation**: Ultra-lightweight performance analysis

## Technical Specifications

### Hardware Requirements
- **GPU**: NVIDIA RTX 3060 (6GB) or better
- **RAM**: 16GB+ recommended
- **Storage**: 50GB+ free space for datasets and results
- **CUDA**: Compatible version with PyTorch

### Software Dependencies
- **Python**: 3.8+
- **PyTorch**: 1.12+ with CUDA support
- **YOLOv5**: Latest ultralytics implementation
- **OpenCV**: 4.5+
- **Additional**: NumPy, PIL, YAML, etc.

### Dataset Requirements
- **VisDrone Dataset**: Properly formatted in YOLO format
- **Directory Structure**: train/val/test splits
- **Labels**: Normalized YOLO format annotations
- **Classes**: 10 VisDrone object categories

## Methodology Compliance

### Protocol v2.0 Requirements
✅ **Phase 1 Compliance**: True baseline with no augmentation  
✅ **Original Dataset**: VisDrone without synthetic enhancement  
✅ **Minimal Processing**: Resize and normalize only  
✅ **Pure Performance**: Baseline model capability measurement  
✅ **Documentation**: Comprehensive logging and analysis  
✅ **Reproducibility**: Detailed configuration and scripts  

### Thesis Integration
- **Baseline Reference**: Foundation for all comparative analysis
- **Methodology Validation**: Demonstrates systematic approach
- **Performance Benchmark**: Quantified baseline for improvement measurement
- **Framework Template**: Reusable structure for other models

---

**Status**: Ready for Phase 1 execution  
**Next**: Execute baseline training and proceed to Phase 2 synthetic augmentation  
**Expected Duration**: 2-4 hours training + 1 hour evaluation  
**Success Criteria**: >18% mAP@0.5 baseline performance achieved