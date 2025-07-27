# YOLOv5n Trial-3 (Advanced Optimization) - VisDrone Dataset

**Master's Thesis Project**: Robust Object Detection for Surveillance Drones in Low-Visibility Environments Using Lightweight AI Models  
**Student**: Burak Kağan Yılmazer  
**Protocol**: Version 2.0 - Advanced Optimization Framework  
**Date**: July 2025  

---

## 🎯 Trial-3 Overview

**Trial-3** represents the **advanced optimization phase** of the YOLOv5n training methodology, building upon the successful results from Phase 1 baseline (24.5% mAP@0.5) and Phase 2 environmental robustness (25.9% mAP@0.5). This trial implements cutting-edge hyperparameter optimization and maximum augmentation strategies to achieve **outstanding thesis performance**.

### Performance Targets
- **Primary Target**: >27% mAP@0.5 (+1.1pp from Phase 2)
- **Stretch Goal**: >29% mAP@0.5 (exceptional thesis results)
- **Total Improvement**: >2.5pp absolute improvement from baseline
- **Research Impact**: Complete methodology optimization demonstration

---

## 🔬 Methodology Framework

### Protocol Version 2.0 Compliance
Trial-3 follows the established **Protocol Version 2.0** framework with advanced optimization:

| Phase | Purpose | Augmentation | Target mAP@0.5 | Status |
|-------|---------|--------------|----------------|---------|
| **Phase 1** | True Baseline | None (disabled) | >18% | ✅ **24.5%** |
| **Phase 2** | Environmental Robustness | Standard suite | >25% | ✅ **25.9%** |
| **Trial-3** | Advanced Optimization | Maximum suite | >27% | 🚀 **In Progress** |

### Research Progression
```
Phase 1 Baseline:     24.5% mAP@0.5
Phase 2 Robustness:   25.9% mAP@0.5  (+1.4pp)
Trial-3 Target:       27.0% mAP@0.5  (+1.1pp minimum)
                     ─────────────────────────────────
Total Improvement:    >2.5pp absolute (+10% relative)
```

---

## ⚙️ Advanced Optimization Features

### Hyperparameter Optimization (vs Phase 2)

| Parameter | Phase 2 Value | Trial-3 Value | Improvement | Rationale |
|-----------|---------------|---------------|-------------|-----------|
| **Learning Rate** | 0.005 | 0.007 | +40% | Enhanced convergence |
| **Mosaic Aug** | 0.8 | 1.0 | +25% | Maximum multi-image training |
| **Copy-paste Aug** | 0.3 | 0.5 | +67% | Enhanced small object focus |
| **Box Loss Weight** | 0.03 | 0.02 | -33% | Small object optimization |
| **Obj Loss Weight** | 1.2 | 1.5 | +25% | Enhanced objectness |

### Advanced Augmentation Suite

**Geometric Transformations:**
- **Rotation**: 7.0° (enhanced viewpoint robustness)
- **Translation**: 0.25 (increased position invariance)
- **Scale**: 0.9 (improved scale robustness)
- **Shear**: 2.0 (added geometric distortion)
- **Perspective**: 0.0002 (subtle perspective transformation)

**Color Augmentations:**
- **HSV Hue**: 0.025 (optimized color variation)
- **HSV Saturation**: 0.6 (enhanced lighting robustness)
- **HSV Value**: 0.35 (improved brightness adaptation)

**Advanced Techniques:**
- **Mosaic**: 1.0 (maximum multi-image composition)
- **Mixup**: 0.15 (conservative image blending)
- **Copy-paste**: 0.5 (enhanced small object augmentation)

---

## 🏗️ Implementation Architecture

### Training Configuration
```yaml
Model: YOLOv5n (nano)
Architecture: 157 layers, 1.77M parameters, 4.2 GFLOPs
Input Size: 640x640 pixels
Batch Size: 8 (RTX 3060 optimized)
Workers: 0 (Windows stability)
Training Epochs: 100 (default, configurable)
Optimizer: SGD with cosine learning rate scheduling
```

### Memory Optimization
- **GPU**: NVIDIA RTX 3060 6GB optimized
- **Batch Size**: 8 (prevents CUDA OOM)
- **Workers**: 0 (eliminates Windows multiprocessing issues)
- **Environment Variables**: Optimized for stability and memory management

### Dataset Configuration
- **Dataset**: VisDrone (10 classes)
- **Training Set**: 6,471 images (293,751 instances)
- **Validation Set**: 548 images (12,740 instances) 
- **Test Set**: 1,610 images (43,235 instances)
- **Augmentation**: Maximum robustness suite applied

---

## 📁 File Structure

```
trial-3/
├── train_yolov5n_trial3.py          # Main training script
├── run_yolov5n_trial3.ps1           # PowerShell execution wrapper
├── README.md                        # This documentation
└── [Generated during training:]
    ├── yolov5n_trial3_comprehensive_analysis.md
    └── training_logs/
```

### Configuration Files (Auto-generated)
```
config/visdrone/yolov5n_trial3/
├── yolov5n_visdrone_trial3.yaml     # Dataset configuration
└── hyp_yolov5n_trial3.yaml          # Advanced hyperparameters
```

---

## 🚀 Execution Instructions

### Prerequisites
1. **Environment Activation** (MANDATORY):
   ```powershell
   .\venvs\visdrone\yolov5n_visdrone_env\Scripts\Activate.ps1
   ```

2. **Repository Location**:
   ```
   C:\Users\burak\OneDrive\Desktop\Git Repos\drone-obj-detection-lightweight-ai
   ```

### Training Commands

#### Standard Training (100 epochs)
```powershell
.\src\scripts\visdrone\YOLOv5n\trial-3\run_yolov5n_trial3.ps1
```

#### Extended Training (150 epochs)
```powershell
.\src\scripts\visdrone\YOLOv5n\trial-3\run_yolov5n_trial3.ps1 -Epochs 150
```

#### Quick Test (30 epochs)
```powershell
.\src\scripts\visdrone\YOLOv5n\trial-3\run_yolov5n_trial3.ps1 -QuickTest
```

#### Help Information
```powershell
.\src\scripts\visdrone\YOLOv5n\trial-3\run_yolov5n_trial3.ps1 -Help
```

### Direct Python Execution
```bash
# After environment activation
python src/scripts/visdrone/YOLOv5n/trial-3/train_yolov5n_trial3.py --epochs 100
python src/scripts/visdrone/YOLOv5n/trial-3/train_yolov5n_trial3.py --quick-test
```

---

## 📊 Expected Performance Analysis

### Training Duration
- **Quick Test (30 epochs)**: ~1-1.5 hours
- **Standard Training (100 epochs)**: ~4-5 hours  
- **Extended Training (150 epochs)**: ~6-7 hours

### Performance Predictions
Based on Phase 2 analysis (25.9% mAP@0.5), Trial-3 optimizations should achieve:

| Metric | Phase 2 Result | Trial-3 Target | Expected Improvement |
|--------|----------------|----------------|---------------------|
| **mAP@0.5** | 25.9% | >27% | +1.1pp minimum |
| **mAP@0.5:0.95** | 11.8% | >12.5% | +0.7pp |
| **Precision** | 39.1% | >42% | +3pp |
| **Recall** | 22.4% | >23.5% | +1.1pp |
| **Model Size** | 3.8MB | 3.8MB | Maintained |

### Research Significance
- **Outstanding Results**: 27-29% mAP@0.5 demonstrates exceptional methodology
- **Complete Framework**: Three-phase optimization fully validated
- **Edge Deployment**: Maintained efficiency throughout optimization
- **Thesis Impact**: Maximum performance with comprehensive documentation

---

## 🔬 Research Contribution

### Academic Significance
1. **Complete Methodology**: Demonstrates full optimization framework
2. **Quantified Improvements**: >2.5pp absolute improvement documented
3. **Edge Device Focus**: Maintains 3.8MB model throughout optimization
4. **Systematic Approach**: Validates progressive enhancement methodology

### Practical Impact
1. **Drone Surveillance**: Enhanced performance for real-world deployment
2. **Edge Computing**: Optimal balance of accuracy and efficiency
3. **Environmental Robustness**: Maximum augmentation for adverse conditions
4. **Scalable Framework**: Methodology applicable to other lightweight models

### Thesis Integration
- **Chapter 4**: Complete results from all three phases
- **Chapter 5**: Comprehensive comparative analysis  
- **Chapter 6**: Outstanding conclusions with 27-29% mAP@0.5
- **Appendix**: Full methodology validation and reproducible setup

---

## 📈 Comparative Context

### vs. Literature Benchmarks
- **VisDrone SOTA**: ~30-35% mAP@0.5
- **Trial-3 Target**: 27-29% mAP@0.5 
- **Position**: Strong performance for nano model on challenging dataset
- **Efficiency**: Superior model size (3.8MB) vs SOTA (>20MB)

### vs. Previous Phases
```
Research Progression:
Phase 1 (Baseline):      24.5% mAP@0.5  [TRUE BASELINE]
Phase 2 (Robustness):    25.9% mAP@0.5  [+1.4pp IMPROVEMENT]
Trial-3 (Optimization):  27%+ mAP@0.5   [+1.1pp+ IMPROVEMENT]
                         ────────────────────────────────────
Total Research Impact:    >2.5pp absolute (+10% relative)
```

---

## 🎓 Success Criteria

### Minimum Requirements
- [x] **Training Completion**: 100 epochs without errors
- [ ] **Performance Target**: >27% mAP@0.5
- [ ] **Improvement Validation**: +1.1pp over Phase 2
- [ ] **Model Efficiency**: Maintained 3.8MB size
- [ ] **Documentation**: Comprehensive analysis report

### Outstanding Achievement
- [ ] **Stretch Performance**: >29% mAP@0.5
- [ ] **Exceptional Improvement**: +3pp+ over Phase 2
- [ ] **Research Excellence**: Top-tier thesis results
- [ ] **Complete Framework**: Full methodology validation

---

## 🔧 Troubleshooting

### Common Issues
1. **Environment**: Ensure `yolov5n_visdrone_env` is activated
2. **CUDA Memory**: Use batch_size=8 for RTX 3060
3. **Windows Compatibility**: workers=0 prevents multiprocessing issues
4. **Path Issues**: Execute from repository root directory

### Performance Monitoring
- **Training Logs**: Detailed logging in output directory
- **TensorBoard**: Disabled to prevent compatibility issues
- **Results CSV**: Comprehensive metrics in runs/train/yolov5n_trial3_*/
- **Model Weights**: Best weights saved automatically

---

## 📝 Documentation Standards

### Automatic Documentation
Trial-3 generates comprehensive documentation including:
- **Training Analysis**: Detailed performance breakdown
- **Comparative Study**: Phase 1 vs Phase 2 vs Trial-3
- **Methodology Validation**: Complete framework demonstration
- **Thesis Integration**: Ready-to-use academic results

### File Naming Convention
- **Training Run**: `yolov5n_trial3_YYYYMMDD_HHMMSS`
- **Analysis**: `yolov5n_trial3_comprehensive_analysis.md`
- **Logs**: `yolov5n_trial3_training_YYYYMMDD_HHMMSS.log`

---

*This README provides complete guidance for executing YOLOv5n Trial-3 advanced optimization training, designed to achieve outstanding thesis results while maintaining comprehensive methodology validation and academic rigor.*