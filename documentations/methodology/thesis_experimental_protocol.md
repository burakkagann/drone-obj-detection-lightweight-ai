# Thesis Experimental Protocol: Environmental Robustness for Drone Object Detection

**Date**: January 21, 2025  
**Thesis**: "Robust Object Detection for Surveillance Drones in Low-Visibility Environments Using Lightweight AI Models"  
**Student**: Burak Kağan Yılmazer  
**Protocol Version**: 1.0

## Executive Summary

This document establishes the comprehensive experimental protocol for demonstrating environmental robustness in lightweight object detection models for drone surveillance applications.

## Research Objectives

### Primary Research Question
**How can lightweight AI models maintain robust object detection performance in low-visibility drone surveillance environments?**

### Specific Hypotheses
1. **H1**: Environmental augmentation improves model robustness in adverse conditions
2. **H2**: Combined real-time + pre-processed augmentation outperforms real-time only
3. **H3**: Performance gains are maintained across different environmental severities
4. **H4**: Robustness improvements are consistent across object classes

## Experimental Design

### **Two-Phase Comparative Study**

#### **Phase 1: Baseline Performance (Control Group)**
- **Dataset**: Original VisDrone (7,019 images)
- **Augmentation**: YOLOv5 real-time only (standard practice)
- **Purpose**: Establish baseline performance benchmarks
- **Models**: YOLOv5n, YOLOv8n, MobileNet-SSD, NanoDet

#### **Phase 2: Environmental Robustness (Treatment Group)**  
- **Dataset**: Environmental augmented (8,629 images: original + synthetic environmental)
- **Augmentation**: YOLOv5 real-time + environmental pre-processing
- **Purpose**: Demonstrate robustness improvement
- **Models**: Same architectures as Phase 1

### **Controlled Variables**
- ✅ **Model architectures** (identical across phases)
- ✅ **Training hyperparameters** (standardized configurations)
- ✅ **Hardware setup** (consistent GPU/CPU usage)
- ✅ **Evaluation metrics** (mAP@0.5, mAP@0.5:0.95, FPS, model size)
- ✅ **Test set composition** (fair comparison protocols)

### **Independent Variables**
- 🎯 **Augmentation strategy** (real-time vs real-time + environmental)
- 🎯 **Environmental conditions** (original, fog, night, motion blur, rain, snow)
- 🎯 **Condition severity** (light, medium, heavy)

### **Dependent Variables**
- 📊 **Detection accuracy** (mAP@0.5, precision, recall)
- 📊 **Inference speed** (FPS, latency)
- 📊 **Model efficiency** (parameters, FLOPS, memory usage)
- 📊 **Robustness metrics** (performance degradation under adverse conditions)

## Dataset Specification

### **Original VisDrone Dataset**
```
Structure:
├── Training: 6,471 images
├── Validation: 548 images  
├── Test: 1,610 images
├── Classes: 10 (pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor)
└── Characteristics: High-resolution drone imagery, small object challenges
```

### **Environmental Augmented Dataset**
```
Enhanced Structure:
├── Training: 8,629 images (original + 2,158 environmental variants)
├── Validation: 548 images (original validation set maintained for fair comparison)
├── Test: Multi-condition test sets
├── Environmental Conditions:
│   ├── Fog (light, medium, heavy)
│   ├── Night (light, medium, heavy) 
│   ├── Motion blur (light, medium, heavy)
│   ├── Rain (light, medium, heavy)
│   └── Snow (light, medium, heavy)
└── Augmentation Distribution: 40% original, 35% light, 20% medium, 5% heavy
```

### **Environmental Augmentation Pipeline**

#### **Fog Simulation**
```python
# Physics-based atmospheric scattering model
I_fog(x) = I_clear(x) * t(x) + A * (1 - t(x))
# Where t(x) = exp(-β * d(x)) # transmission map
# β = scattering coefficient, d(x) = depth map
```

#### **Night Simulation**  
```python
# Gamma correction with noise injection
I_night = γ * (I_original / 255)^gamma * 255
# gamma ∈ [0.3, 0.7] # darkness levels
# Add Gaussian noise: σ ∈ [5, 15] # sensor noise
```

#### **Motion Blur Simulation**
```python
# Linear motion kernel convolution
K = motion_kernel(length, angle)
I_blur = cv2.filter2D(I_original, -1, K)
# length ∈ [5, 25] pixels # blur intensity
```

#### **Rain/Snow Simulation**
```python
# Precipitation overlay with atmospheric effects
I_rain = I_original + rain_overlay(intensity, density, direction)
# intensity ∈ [0.1, 0.5] # visibility reduction
```

## Training Configuration Standards

### **YOLOv5n Standard Configuration**
```yaml
# Proven optimal settings for VisDrone
model: yolov5n.pt                # Pre-trained weights (CRITICAL)
img_size: 640                    # High resolution for small objects
batch_size: 16                   # Hardware-optimized
epochs: 100                      # Full training (20 for validation)

# Critical training enhancements
multi_scale: True                # Multi-scale training
cos_lr: True                     # Cosine learning rate scheduling
cache: ram                       # Fast data loading
workers: 4                       # Optimal data loader threads

# Learning rate configuration
lr0: 0.005                       # Initial learning rate
lrf: 0.02                        # Final LR factor
momentum: 0.937                  # SGD momentum
weight_decay: 0.0005             # Regularization

# Loss function weights (small object optimized)
box: 0.03                        # Box regression loss
cls: 0.3                         # Classification loss  
obj: 1.2                         # Objectness loss
iou_t: 0.15                      # IoU threshold
fl_gamma: 0.0                    # Focal loss (disabled)

# Real-time augmentation (MAINTAINED in both phases)
mosaic: 0.8                      # Object context diversity
mixup: 0.4                       # Decision boundary learning
hsv_h: 0.02                      # Hue variation
hsv_s: 0.5                       # Saturation variation  
hsv_v: 0.3                       # Value variation
degrees: 5.0                     # Rotation for aerial perspective
translate: 0.2                   # Translation variation
scale: 0.8                       # Scale variation
fliplr: 0.5                      # Horizontal flip
copy_paste: 0.3                  # Copy-paste augmentation
```

### **Cross-Model Consistency**
- **YOLOv8n**: Equivalent configuration adaptation
- **MobileNet-SSD**: Comparable training regime
- **NanoDet**: Architecture-appropriate settings

## Evaluation Protocol

### **Standard Metrics (Both Phases)**
```yaml
Primary Metrics:
- mAP@0.5: COCO-style mean Average Precision at IoU 0.5
- mAP@0.5:0.95: COCO-style mAP across IoU thresholds
- Precision: True positives / (True positives + False positives)
- Recall: True positives / (True positives + False negatives)

Efficiency Metrics:
- FPS: Frames per second (inference speed)
- Model Size: Parameter count and storage requirements
- FLOPs: Floating point operations per inference
- Memory Usage: Peak GPU memory during inference

Robustness Metrics:
- Condition-specific mAP: Performance on each environmental condition
- Degradation Factor: (Baseline mAP - Condition mAP) / Baseline mAP
- Cross-condition Consistency: Performance variance across conditions
```

### **Multi-Condition Test Sets**
```yaml
Test Conditions:
1. Original Conditions (control):
   - Standard VisDrone test set (1,610 images)
   - Clear visibility, standard lighting

2. Environmental Conditions (treatment):
   - Fog Test Set: 400 images (light: 200, medium: 150, heavy: 50)
   - Night Test Set: 400 images (light: 200, medium: 150, heavy: 50)  
   - Motion Blur Test Set: 400 images (light: 200, medium: 150, heavy: 50)
   - Rain Test Set: 300 images (light: 150, medium: 100, heavy: 50)
   - Snow Test Set: 300 images (light: 150, medium: 100, heavy: 50)

3. Combined Conditions (stress test):
   - Multi-condition scenarios (fog + night, rain + motion blur)
   - Progressive degradation testing
   - Real-world condition simulation
```

## Statistical Analysis Framework

### **Hypothesis Testing**
```yaml
Primary Comparisons:
- Baseline vs Environmental (paired t-test)
- Cross-model robustness (ANOVA)
- Condition-specific improvements (Tukey HSD)

Significance Levels:
- α = 0.05 (standard significance)
- α = 0.01 (high confidence)
- Effect size reporting (Cohen's d)

Sample Size Justification:
- Training: >6,000 images per condition
- Validation: >500 images (consistent across experiments)
- Test: >300 images per environmental condition
```

### **Performance Benchmarking**
```yaml
Baseline Targets (Phase 1):
- YOLOv5n: >22% mAP@0.5 (based on Trial-2: 23.557%)
- YOLOv8n: >24% mAP@0.5 (expected improvement)
- MobileNet-SSD: >18% mAP@0.5 (efficiency-focused)
- NanoDet: >15% mAP@0.5 (ultra-lightweight)

Robustness Targets (Phase 2):
- <20% performance degradation under adverse conditions
- >5% absolute improvement in low-visibility scenarios
- Maintained or improved inference speed
- Cross-condition consistency (σ < 3% mAP variance)
```

## Implementation Timeline

### **Week 1-2: Dataset Preparation**
- [x] Analyze existing dataset versions
- [ ] Create clean baseline dataset (Phase 1)
- [ ] Validate environmental augmented dataset (Phase 2)  
- [ ] Establish train/val/test splits
- [ ] Quality assurance and statistics validation

### **Week 3-4: Baseline Experiments (Phase 1)**
- [ ] YOLOv5n baseline training and evaluation
- [ ] YOLOv8n baseline training and evaluation
- [ ] MobileNet-SSD baseline training and evaluation
- [ ] NanoDet baseline training and evaluation
- [ ] Comprehensive results documentation

### **Week 5-6: Environmental Experiments (Phase 2)**  
- [ ] YOLOv5n environmental training and evaluation
- [ ] YOLOv8n environmental training and evaluation
- [ ] MobileNet-SSD environmental training and evaluation
- [ ] NanoDet environmental training and evaluation
- [ ] Multi-condition testing protocol

### **Week 7: Analysis and Documentation**
- [ ] Statistical analysis and hypothesis testing
- [ ] Cross-model comparison and ranking
- [ ] Robustness metrics calculation
- [ ] Thesis results compilation

## Quality Assurance

### **Reproducibility Requirements**
```yaml
Documentation Standards:
- Complete hyperparameter logging
- Seed-based random state control
- Environment specification (Python, CUDA, library versions)
- Hardware configuration documentation
- Training time and resource usage tracking

Validation Protocols:
- Cross-validation on baseline results
- Ablation studies on augmentation components
- Sensitivity analysis on hyperparameters
- Statistical significance verification
```

### **Bias Mitigation**
```yaml
Data Handling:
- Stratified sampling for environmental conditions
- Balanced class representation across datasets
- Consistent preprocessing pipelines
- Independent test set maintenance

Model Training:
- Identical initialization strategies
- Consistent optimization procedures
- Fair computational resource allocation
- Standardized early stopping criteria
```

## Expected Outcomes

### **Quantitative Targets**
```yaml
Phase 1 (Baseline):
- Establish reliable performance benchmarks
- Document standard YOLOv5n capabilities: ~23% mAP@0.5
- Cross-model performance ranking
- Efficiency vs accuracy trade-off analysis

Phase 2 (Environmental):
- Demonstrate robustness improvements: +3-7% mAP@0.5 in adverse conditions
- Maintain or improve baseline performance: ≥23% mAP@0.5 on original conditions
- Show cross-condition consistency: <15% variance across environmental tests
- Validate practical deployment readiness: >10 FPS inference speed
```

### **Qualitative Contributions**
```yaml
Research Impact:
- Novel environmental augmentation methodology
- Comprehensive lightweight model comparison
- Practical drone surveillance deployment guidelines
- Reproducible experimental framework

Academic Significance:
- Peer-reviewable methodology and results
- Open-source implementation for research community
- Benchmark establishment for future research
- Real-world problem solving demonstration
```

## Risk Mitigation

### **Technical Risks**
```yaml
Training Failures:
- Backup training configurations
- Multiple hardware options (RTX 3060 + RTX 2060)
- Cloud computing fallback (Google Colab, AWS)
- Checkpoint-based recovery systems

Data Issues:
- Dataset validation and integrity checks
- Backup copies of all processed datasets
- Version control for data preprocessing
- Quality metrics for synthetic augmentations
```

### **Timeline Risks**
```yaml
Schedule Management:
- Parallel training on multiple machines
- Prioritized experiment ordering (YOLOv5n first)
- Incremental result documentation
- Flexible scope adjustment protocols
```

## Success Criteria

### **Minimum Viable Research (40-day constraint)**
```yaml
Essential Deliverables:
- ✅ YOLOv5n baseline and environmental comparison
- ✅ Statistical significance demonstration  
- ✅ Robustness improvement quantification
- ✅ Thesis-quality documentation and analysis

Stretch Goals (if time permits):
- 🎯 Multi-model comprehensive comparison
- 🎯 Edge device deployment validation
- 🎯 Real-world condition testing
- 🎯 Publication-ready results
```

### **Academic Excellence Indicators**
```yaml
Research Quality:
- >3% absolute mAP improvement in adverse conditions
- Statistical significance (p < 0.05) across key comparisons
- Comprehensive ablation study results
- Reproducible methodology and code release

Practical Impact:
- Deployment-ready model configurations
- Real-world applicability demonstration
- Clear implementation guidelines
- Performance vs efficiency optimization guidance
```

---

**Protocol Status**: APPROVED FOR IMPLEMENTATION  
**Next Phase**: Clean dataset preparation and baseline experiment execution  
**Timeline**: 40 days remaining → Prioritize YOLOv5n comparison for guaranteed results

---

*This protocol ensures rigorous scientific methodology while maximizing practical impact within thesis timeline constraints.*