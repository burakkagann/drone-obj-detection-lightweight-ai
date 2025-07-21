# CLAUDE.md - Project Context and Instructions

## Project Overview
**Master's Thesis Project**: "Robust Object Detection for Surveillance Drones in Low-Visibility Environments Using Lightweight AI Models"

**Student**: Burak Kağan Yılmazer  
**Program**: Computer Science - Big Data & Artificial Intelligence  
**Supervisors**: Prof. Dr. Alexander Iliev, Prof. Dr. Reiner Creutzburg  
**Timeline**: 40 days remaining (as of current date)  
**Due Date**: September 15, 2025

## Research Objectives
The primary goal is to develop lightweight object detection models capable of:
1. **Real-time performance** on edge devices (NVIDIA Jetson Nano, Raspberry Pi)
2. **Robust detection** in low-visibility environments (fog, nighttime, sensor distortions)
3. **Optimal trade-offs** between accuracy, speed, and model size for drone deployment

## Repository Structure and Key Components

### Core Models Under Investigation
1. **YOLOv5n** - Primary focus, proven baseline (22.6% mAP@0.5 on VisDrone)
2. **YOLOv8n** - Secondary comparison model
3. **MobileNet-SSD** - Lightweight alternative architecture
4. **NanoDet** - Ultra-lightweight edge-optimized model

### Datasets
- **VisDrone** - Primary dataset (10 classes, drone surveillance imagery)
- **DOTA** - Secondary dataset (15 classes, aerial imagery with OBB support)
- **CIFAR** - Under consideration (discuss with supervisor)

### Current Status
- **Phase**: Model optimization and multi-model comparison
- **Best Performance**: Trial-2 YOLOv5n achieving 23.557% mAP@0.5
- **Current Priority**: Trial-3 optimization targeting 25%+ mAP@0.5 (ready for execution)
- **Next Steps**: Execute Trial-3, then implement multi-model comparison framework

## Directory Structure

### Source Code (`src/`)
- `models/` - Model implementations and architectures
- `scripts/` - Training scripts organized by dataset
- `augmentation_pipeline/` - Environmental condition simulation
- `data_preparation/` - Dataset processing utilities
- `evaluation/` - Comprehensive metrics framework
- `utils/` - Shared utilities and helper functions

### Configuration (`config/`)
- Model-specific YAML configurations
- Dataset format definitions
- Hyperparameter optimization settings
- Augmentation pipeline parameters

### Data (`data/`)
- `raw/` - Original datasets
- `augmented/` - Environmentally augmented datasets
- `my_dataset/` - Processed and formatted datasets

### Results and Logs
- `runs/` - Training runs and TensorBoard logs
- `logs/` - Detailed training logs
- `checkpoints/` - Model weights and validation results
- `results/` - Performance analysis and thesis metrics

## Current Training Approach

### YOLOv5n Training Trials
- **Trial-1**: 100 epochs baseline (completed)
- **Trial-2**: Hyperparameter optimization (23.557% mAP@0.5 - PROVEN BASELINE) ✅
- **Trial-3**: Enhanced optimization - CRITICAL FAILURE (0.002% mAP@0.5) ❌

**CRITICAL FAILURE ANALYSIS**: Trial-3 resulted in 99.99% performance degradation due to:
- **Primary Cause**: Focal loss activation (`fl_gamma: 0.5`) when Trial-2 had it disabled (`fl_gamma: 0.0`)
- **Secondary Factors**: Over-regularization, loss weight imbalance, reduced learning rate
- **Status**: Root cause identified, recovery protocol established
- **Next Steps**: Return to Trial-2 baseline, implement incremental improvements only

**Research Impact**: Failure analysis demonstrates critical importance of controlled hyperparameter modification and provides valuable thesis methodology insights.

### Synthetic Augmentation Pipeline
Located in `src/augmentation_pipeline/`:
- **Fog simulation** (`fog.py`) - Atmospheric fog with depth blending
- **Night conditions** (`night.py`) - Low-light with gamma correction
- **Sensor distortions** (`sensor_distortions.py`) - Blur, noise, chromatic aberration
- **Quality validation** - SSIM/PSNR metrics for augmentation realism

## Performance Targets and Thresholds

### Recommended Performance Thresholds
- **Minimum Acceptable mAP@0.5**: 25%
- **Target mAP@0.5**: 30-35%
- **Real-time Performance**: >10 FPS
- **Model Size**: <10MB for edge deployment
- **Power Consumption**: <5W for drone applications

### Evaluation Priority (for thesis impact)
1. **Model Size vs. Accuracy Trade-off** (highest priority)
2. **Real-time Performance** (essential for surveillance)
3. **Robustness under Adverse Conditions** (unique contribution)
4. **Power Consumption** (important but secondary)

## Edge Device Testing Strategy

### Simulation-First Approach (Recommended for timeline)
- **NVIDIA Docker containers** with resource constraints
- **Google Colab** with limited resources
- **AWS EC2 instances** for Raspberry Pi simulation

### Hardware Testing (if budget permits)
- **Jetson Nano**: ~$99 (optimal for research validation)
- **Raspberry Pi 4**: ~$75 (broader applicability)
- **Timeline**: Order by Day 7 for Week 3 testing

## Environment Management

### Virtual Environment Requirements (CRITICAL)
**⚠️ MANDATORY**: Virtual environment MUST be activated before running ANY training scripts or model operations.

### All Available Virtual Environments

#### Primary Model Environments
1. **YOLOv5n Environment (PRIMARY)**
   - **Location**: `.\venvs\yolov5n_env\`
   - **Activation**: `.\venvs\yolov5n_env\Scripts\Activate.ps1`
   - **Used For**: All YOLOv5n training, evaluation, and related tasks

2. **MobileNet-SSD Environment**
   - **Location**: `.\venvs\mobilenet_ssd_env\`
   - **Activation**: `.\venvs\mobilenet_ssd_env\Scripts\Activate.ps1`
   - **Used For**: MobileNet-SSD training and evaluation

3. **NanoDet Environment**
   - **Location**: `.\venvs\nanodet_env\`
   - **Activation**: `.\venvs\nanodet_env\Scripts\Activate.ps1`
   - **Used For**: NanoDet training and evaluation

4. **TensorFlow Environment**
   - **Location**: `.\venvs\tensorflow_env\`
   - **Activation**: `.\venvs\tensorflow_env\Scripts\Activate.ps1`
   - **Used For**: TensorFlow-based models and operations

5. **Augmentation Environment**
   - **Location**: `.\venvs\augment_venv\`
   - **Activation**: `.\venvs\augment_venv\Scripts\Activate.ps1`
   - **Used For**: Synthetic data augmentation pipeline

#### Dataset-Specific Environments

6. **VisDrone YOLOv5n Environment**
   - **Location**: `.\venvs\visdrone\yolov5n_visdrone_env\`
   - **Activation**: `.\venvs\visdrone\yolov5n_visdrone_env\Scripts\Activate.ps1`
   - **Used For**: VisDrone-specific YOLOv5n training

7. **DOTA Model Environments**
   - **YOLOv5n DOTA**: `.\venvs\dota\venvs\yolov5n_dota_env\Scripts\Activate.ps1`
   - **YOLOv8n DOTA**: `.\venvs\dota\venvs\yolov8n_dota_env\Scripts\Activate.ps1`
   - **MobileNet-SSD DOTA**: `.\venvs\dota\venvs\mobilenet_ssd_dota_env\Scripts\Activate.ps1`
   - **NanoDet DOTA**: `.\venvs\dota\venvs\nanodet_dota_env\Scripts\Activate.ps1`

### Virtual Environment Activation Protocol (MANDATORY)

**ALWAYS follow this sequence before ANY training:**

```powershell
# Step 1: Navigate to repository root
cd "C:\Users\burak\OneDrive\Desktop\Git Repos\drone-obj-detection-lightweight-ai"

# Step 2: Activate appropriate virtual environment
# For YOLOv5n training (PRIMARY - most common):
.\venvs\yolov5n_env\Scripts\Activate.ps1

# For other models, use corresponding activation script:
# .\venvs\mobilenet_ssd_env\Scripts\Activate.ps1
# .\venvs\nanodet_env\Scripts\Activate.ps1
# .\venvs\tensorflow_env\Scripts\Activate.ps1
# .\venvs\augment_venv\Scripts\Activate.ps1
# .\venvs\visdrone\yolov5n_visdrone_env\Scripts\Activate.ps1
# .\venvs\dota\venvs\yolov5n_dota_env\Scripts\Activate.ps1

# Step 3: Verify activation (should show environment name in prompt)
# Example: (yolov5n_env) PS C:\Users\burak\...

# Step 4: Then proceed with training scripts
```

### Environment Selection Guidelines

**Use this environment for each task:**
- **YOLOv5n VisDrone Training**: `yolov5n_env` (PRIMARY)
- **MobileNet-SSD Training**: `mobilenet_ssd_env`
- **NanoDet Training**: `nanodet_env`
- **Augmentation Pipeline**: `augment_venv`
- **DOTA Dataset Training**: Use corresponding DOTA environment
- **TensorFlow Operations**: `tensorflow_env`

### Dependencies and Troubleshooting
- **Execution Policy**: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- **Path Issues**: Always use absolute paths or navigate to repository root first
- **Verification**: Check for environment name prefix in terminal prompt after activation
- **Deactivation**: Use `deactivate` command when switching environments

## 40-Day Action Plan

### Week 1-2 (Days 1-14): Model Optimization & Core Results
1. Clean repository - delete Trial-3/4 folders
2. Optimize Trial-2 YOLOv5n baseline
3. Complete synthetic augmentation validation
4. Establish baseline performance metrics

### Week 3 (Days 15-21): Multi-Model Comparison
1. Train MobileNet-SSD on VisDrone
2. Train NanoDet on VisDrone
3. Comparative analysis framework
4. Edge device simulation setup

### Week 4-5 (Days 22-35): Edge Device Strategy & Analysis
1. Edge device performance testing
2. Results compilation and analysis
3. Begin thesis writing - methodology and results

### Week 6 (Days 36-40): Final Analysis & Submission
1. Final results visualization
2. Thesis completion and submission
3. Documentation finalization

## Training Commands and Scripts

### YOLOv5n Training
```bash
# Activate environment
# For YOLOv5n Trial-2 optimization
python src/models/YOLOv5/train.py --data config/visdrone/yolov5n_v1/yolov5n_visdrone_config.yaml --cfg src/models/YOLOv5/models/yolov5n.yaml --hyp config/visdrone/yolov5n_v1/hyp_visdrone_trial-2_optimized.yaml
```

### PowerShell Training Scripts - STANDARDIZED APPROACH

**MANDATORY**: Use Trial-2 wrapper script pattern for all future trials (most bulletproof):

#### Recommended Pattern (Trial-2 Style):
```powershell
# Create separate Python wrapper script for each trial
$pythonArgs = @(
    "train_yolov5n_trialX.py",
    "--epochs", $Epochs.ToString()
)

if ($QuickTest) {
    $pythonArgs += "--quick-test"
}

$process = Start-Process -FilePath "python" -ArgumentList $pythonArgs -NoNewWindow -PassThru -Wait
```

**Advantages:**
- ✅ No PowerShell path resolution issues
- ✅ Python handles all file paths internally  
- ✅ Proven reliability (Trial-2: 23.557% mAP@0.5)
- ✅ Simpler debugging and maintenance

#### Avoid Direct train.py Calls:
❌ Complex relative paths (Trial-3 style)
❌ Space-sensitive absolute paths  
❌ PowerShell argument escaping issues

#### Warning Suppression for Clean Output:
```python
# Add to training scripts to suppress torch.cuda.amp.autocast warnings
import warnings
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*", category=FutureWarning)
```

## Key Files and Configurations

### Critical Configuration Files
- `config/visdrone/yolov5n_v1/yolov5n_visdrone_config.yaml` - Dataset configuration
- `config/visdrone/yolov5n_v1/hyp_visdrone_trial-2_optimized.yaml` - Original Trial-2 hyperparameters
- `config/visdrone/yolov5n_v1/hyp_visdrone_trial3.yaml` - **CURRENT ACTIVE** Trial-3 hyperparameters
- `config/augmentation_config.yaml` - Augmentation pipeline settings

### Model Weights and Results
- `runs/train/yolov5n_trial2_BEST_23.557mAP/weights/best.pt` - **CURRENT BEST** Trial-2 model (23.557% mAP@0.5)
- `runs/train/yolov5n_trial2_BEST_23.557mAP/results.csv` - Current best training metrics

### Training Scripts (Trial-Based Organization)
- `src/scripts/visdrone/YOLOv5n/Trial-3/run_trial3_training.ps1` - **ACTIVE** Trial-3 training script

### Trial Organization Protocol (MANDATORY)
**General Practice**: Each trial gets its own folder under `src/scripts/visdrone/YOLOv5n/Trial-X/`

#### Trial Naming Convention:
- **Trial-1**: Baseline experiments
- **Trial-2**: First optimization iteration (23.557% mAP@0.5 achieved)
- **Trial-3**: Current optimization targeting 25%+ mAP@0.5
- **Trial-4**: Next iteration (if needed)
- **Trial-N**: Sequential numbering for all future trials

#### Folder Structure Requirements:
```
src/scripts/visdrone/YOLOv5n/
├── Trial-1/          [Baseline scripts]
├── Trial-2/          [First optimization scripts] 
├── Trial-3/          [Current active scripts]
└── Trial-X/          [Future trials...]
```

#### Configuration Naming:
- `config/visdrone/yolov5n_v1/hyp_visdrone_trial1.yaml`
- `config/visdrone/yolov5n_v1/hyp_visdrone_trial-2_optimized.yaml` 
- `config/visdrone/yolov5n_v1/hyp_visdrone_trial3.yaml`
- `config/visdrone/yolov5n_v1/hyp_visdrone_trial4.yaml` (future)

### Documentation
- `Z_Methodology.txt` - Complete thesis methodology and literature review
- `documentations/progression-notes/` - Detailed development history
- Various README files throughout the repository

## Common Issues and Solutions

### Training Issues
- **CUDA compatibility**: Ensure proper PyTorch CUDA version alignment
- **Dataset path issues**: Use absolute paths in configuration files
- **Memory issues**: Implement gradient checkpointing for large models

### Environment Issues
- **PowerShell execution policy**: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- **Virtual environment conflicts**: Use separate environments for each model
- **Cross-platform compatibility**: Consider Docker containers for reproducibility

## Research Validation Approach

### Synthetic Augmentation Validation
- Current simulation-based approach is sufficient for thesis scope
- SSIM/PSNR quality metrics implemented
- Cross-validation: train on augmented, test on both augmented and original
- Literature benchmarking against cited papers

### Performance Validation
- Comprehensive metrics framework in `src/evaluation/thesis_metrics.py`
- Automated CSV export and visualization
- TensorBoard integration for training monitoring
- Baseline vs. augmented model comparisons

## Thesis Contribution Areas

### Primary Contributions
1. **Comprehensive benchmarking framework** for lightweight models on edge devices
2. **Synthetic augmentation pipeline** for adverse environmental conditions
3. **Performance trade-off analysis** between accuracy, speed, and robustness
4. **Practical deployment guidelines** for drone-based surveillance

### Academic Significance
- Addresses critical gap in edge-based drone surveillance
- Novel combination of lightweight models + environmental robustness
- Practical relevance for defense, search-and-rescue, and monitoring applications

## Notes for Claude Assistant

### Working Approach
- **Focus on immediate priorities**: 40-day deadline requires strategic focus
- **Proven baseline optimization**: Trial-2 (22.6% mAP) is the foundation to build upon
- **Practical solutions**: Simulation over hardware when time-constrained
- **Documentation-heavy**: Extensive documentation supports thesis writing

### Communication Style
- **Direct and actionable**: Provide specific commands and file paths
- **Timeline-aware**: Always consider 40-day constraint in recommendations
- **Research-focused**: Maintain academic rigor while being practical

### Repository Cleanup Priorities
1. ~~Delete Trial-3 and Trial-4 folders to reduce confusion~~ ✅ COMPLETED
2. Consolidate virtual environments where possible
3. Organize results and logs for thesis analysis
4. Maintain comprehensive documentation for reproducibility

### Documentation Management Protocol
**CRITICAL**: After every progression, all relevant information must be saved in `documentations/` directory structure:

#### Documentation Structure Requirements
- **Location**: `c:\Users\burak\OneDrive\Desktop\Git Repos\drone-obj-detection-lightweight-ai\documentations\`
- **Format**: All documentation must be in `.md` format
- **Naming Convention**: File names must align with their contents (e.g., `trial2_optimization_results.md`, `edge_device_simulation_setup.md`)
- **Auto-Creation**: If no relevant folder exists, create appropriate subdirectory structure

#### Mandatory Documentation Categories
Each category must be organized by **Model-Dataset** combinations (e.g., `YOLOv5n-Visdrone/`, `YOLOv8n-Visdrone/`, `MobileNet-SSD-Visdrone/`):

1. **`optimization-results/`** - All model optimization findings and results
   - Structure: `optimization-results/[Model-Dataset]/[trial_name]_[date].md`
   - Example: `optimization-results/YOLOv5n-Visdrone/trial2_enhanced_v2_optimization_20250121.md`

2. **`training-logs/`** - Detailed training session summaries and analysis
   - Structure: `training-logs/[Model-Dataset]/[session_name]_[date].md`
   - Example: `training-logs/YOLOv5n-Visdrone/trial2_enhanced_v2_training_20250121.md`

3. **`edge-device-testing/`** - Edge device simulation and testing documentation
   - Structure: `edge-device-testing/[Model-Dataset]/[device_type]_results_[date].md`
   - Example: `edge-device-testing/YOLOv5n-Visdrone/jetson_nano_simulation_20250121.md`

4. **`comparative-analysis/`** - Multi-model comparison results and insights
   - Structure: `comparative-analysis/[comparison_type]_[date].md`
   - Example: `comparative-analysis/yolov5n_vs_yolov8n_visdrone_20250121.md`

5. **`augmentation-validation/`** - Synthetic augmentation effectiveness analysis
   - Structure: `augmentation-validation/[Model-Dataset]/[augmentation_type]_validation_[date].md`
   - Example: `augmentation-validation/YOLOv5n-Visdrone/fog_augmentation_validation_20250121.md`

6. **`performance-benchmarks/`** - Comprehensive performance metrics and benchmarking
   - Structure: `performance-benchmarks/[Model-Dataset]/[benchmark_type]_[date].md`
   - Example: `performance-benchmarks/YOLOv5n-Visdrone/mAP_precision_recall_analysis_20250121.md`

7. **`thesis-analysis/`** - Analysis specifically for thesis writing and conclusions
   - Structure: `thesis-analysis/[analysis_topic]_[date].md`
   - Example: `thesis-analysis/lightweight_model_tradeoffs_20250121.md`

8. **`troubleshooting/`** - Issues encountered and solutions implemented
   - Structure: `troubleshooting/[Model-Dataset]/[issue_type]_[date].md`
   - Example: `troubleshooting/YOLOv5n-Visdrone/cuda_memory_issues_20250121.md`

9. **`methodology-updates/`** - Any changes or refinements to research methodology
   - Structure: `methodology-updates/[update_topic]_[date].md`
   - Example: `methodology-updates/hyperparameter_optimization_strategy_20250121.md`

#### Documentation Workflow
- **During Training**: Save training summaries with timestamp and configuration details
- **After Optimization**: Document hyperparameter changes and performance impacts
- **Post-Analysis**: Create comparative analysis reports with visualizations
- **Problem Resolution**: Document all issues and solutions for reproducibility
- **Milestone Completion**: Create comprehensive summary reports for each major milestone

#### File Naming Examples
- `trial2_hyperparameter_optimization_YYYYMMDD.md`
- `yolov5n_vs_mobilenet_ssd_comparison_YYYYMMDD.md`
- `edge_device_jetson_simulation_results_YYYYMMDD.md`
- `synthetic_augmentation_validation_metrics_YYYYMMDD.md`

---

*Last Updated: [Current Date]*  
*This file serves as the primary context for Claude assistant when working on this drone object detection research project.*