# Parallel Training Setup Guide - RTX 2060 Laptop

**Date**: January 21, 2025  
**Purpose**: Set up second laptop for parallel YOLOv5n training  
**Hardware**: Intel i7-10750H, RTX 2060 (6GB), 128MB Intel UHD Graphics  
**Strategy**: Accelerate research timeline with dual-machine training

## Executive Summary

This guide enables parallel training across two machines to double research velocity. The RTX 2060 laptop will handle experimental trials while the RTX 3060 machine continues primary optimization work.

## Hardware Specifications Comparison

| Component | Primary Laptop (RTX 3060) | Secondary Laptop (RTX 2060) | Performance Ratio |
|-----------|---------------------------|----------------------------|-------------------|
| **GPU** | RTX 3060 (8GB) | RTX 2060 (6GB) | ~85% performance |
| **VRAM** | 8GB | 6GB | May require batch size reduction |
| **CUDA Cores** | 3584 | 1920 | ~54% raw compute |
| **Training Speed** | Baseline | ~80-90% of primary | Acceptable for parallel work |

## Step 1: Repository Setup

### 1.1 Clone Repository
```powershell
# Open PowerShell as Administrator on RTX 2060 laptop
cd "C:\Users\[YourUsername]\OneDrive\Desktop"
mkdir "Git Repos"
cd "Git Repos"

# Clone the repository
git clone https://github.com/[your-repo]/drone-obj-detection-lightweight-ai.git
# OR copy entire folder from primary laptop via network/USB
```

### 1.2 Verify Repository Structure
```powershell
cd "drone-obj-detection-lightweight-ai"
ls
# Should see: config/, data/, src/, runs/, documentations/, etc.
```

## Step 2: Environment Setup

### 2.1 Python Installation
- **Install Python 3.8-3.10** (same version as primary laptop)
- **Add Python to PATH** during installation
- **Verify installation**: `python --version`

### 2.2 Virtual Environment Creation
```powershell
# Navigate to repository root
cd "C:\Users\[YourUsername]\OneDrive\Desktop\Git Repos\drone-obj-detection-lightweight-ai"

# Create virtual environments directory
mkdir venvs

# Create YOLOv5n environment (primary for parallel training)
python -m venv venvs\yolov5n_env

# Activate environment
.\venvs\yolov5n_env\Scripts\Activate.ps1

# If execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2.3 Dependencies Installation
```powershell
# With yolov5n_env activated
cd src\models\YOLOv5

# Install PyTorch with CUDA support (RTX 2060 compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install YOLOv5 requirements
pip install -r requirements.txt

# Install additional packages
pip install tensorboard
pip install pandas
pip install matplotlib
pip install seaborn
```

### 2.4 CUDA Verification
```powershell
# Test CUDA availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Expected output:
# CUDA Available: True
# GPU Name: NVIDIA GeForce RTX 2060
```

## Step 3: Dataset Setup

### 3.1 Dataset Transfer Options

#### Option A: Network Copy (Recommended if both laptops on same network)
```powershell
# From primary laptop, share the data folder
# On RTX 2060 laptop:
robocopy "\\[PRIMARY-LAPTOP-IP]\shared\data" "C:\Users\[YourUsername]\OneDrive\Desktop\Git Repos\drone-obj-detection-lightweight-ai\data" /E

# Alternative: Use OneDrive sync if data folder is in OneDrive
```

#### Option B: External Drive Transfer
1. Copy `data/` folder to external drive from primary laptop
2. Transfer to RTX 2060 laptop
3. Place in same relative path: `drone-obj-detection-lightweight-ai/data/`

#### Option C: Fresh Download (If dataset source is available)
```powershell
# Download VisDrone dataset directly to RTX 2060 laptop
# Follow same data preparation steps as primary laptop
```

### 3.2 Verify Dataset Structure
```powershell
# Check dataset structure
ls data\my_dataset\visdrone\
# Should see: train\, val\, test\ folders

ls data\my_dataset\visdrone\train\
# Should see: images\, labels\ folders

# Verify file counts match primary laptop
Get-ChildItem data\my_dataset\visdrone\train\images\ | Measure-Object
Get-ChildItem data\my_dataset\visdrone\train\labels\ | Measure-Object
```

## Step 4: Configuration Adjustment for RTX 2060

### 4.1 Create RTX 2060 Optimized Hyperparameters
```powershell
# Copy existing hyperparameter file
copy config\visdrone\yolov5n_v1\hyp_visdrone_trial4.yaml config\visdrone\yolov5n_v1\hyp_visdrone_trial4_rtx2060.yaml
```

**Edit `hyp_visdrone_trial4_rtx2060.yaml`:**
```yaml
# Reduce batch size for 6GB VRAM
batch_size: 16          # Reduced from 18 (RTX 3060) to 16 (RTX 2060)

# All other parameters remain identical to ensure comparable results
lr0: 0.005
obj: 1.25
# ... (keep all other settings same as Trial-4)
```

### 4.2 Create RTX 2060 Training Script
```powershell
# Copy Trial-4 script
copy src\scripts\visdrone\YOLOv5n\Trial-4\run_trial4_conservative.ps1 src\scripts\visdrone\YOLOv5n\Trial-4\run_trial4_rtx2060.ps1
```

**Edit batch size in `run_trial4_rtx2060.ps1`:**
```powershell
# Change line:
"--batch-size", "18",
# To:
"--batch-size", "16",

# Update hyperparameter file reference:
"--hyp", "..\..\..\config\visdrone\yolov5n_v1\hyp_visdrone_trial4_rtx2060.yaml",

# Update run name to distinguish results:
$RunName = if ($FullTraining) { "yolov5n_trial4_100epochs_rtx2060" } else { "yolov5n_trial4_validation_rtx2060" }
```

## Step 5: Parallel Training Strategy

### 5.1 Training Assignment Matrix

| Machine | Primary Role | Current Trial | Next Trials |
|---------|-------------|---------------|-------------|
| **RTX 3060** | Main optimization | Trial-4 (running) | Trial-5, Trial-6 |
| **RTX 2060** | Experimental | Focal loss test | MobileNet-SSD, NanoDet |

### 5.2 Communication Protocol
- **Shared Results Folder**: Use OneDrive or network share for `runs/` folder
- **Daily Sync**: Merge results at end of each day
- **Documentation**: Each machine documents its trials separately

### 5.3 Trial Coordination
```powershell
# RTX 3060 (Primary) - Continue current work
# Trial-4 → Trial-5 optimization → Multi-model comparison

# RTX 2060 (Secondary) - Experimental work
# 1. Focal loss impact study (Trial-3 failure analysis)
# 2. Alternative augmentation strategies
# 3. MobileNet-SSD baseline training
```

## Step 6: Testing and Validation

### 6.1 Quick Validation Test
```powershell
# Activate environment
.\venvs\yolov5n_env\Scripts\Activate.ps1

# Navigate to Trial-4 directory
cd src\scripts\visdrone\YOLOv5n\Trial-4

# Run 1-epoch test to verify everything works
.\run_trial4_rtx2060.ps1 -Epochs 1

# Expected: Training should start without errors
# GPU utilization should show RTX 2060 usage
# Memory usage should be lower than RTX 3060 (~4-5GB vs 6-8GB)
```

### 6.2 Performance Benchmarking
```powershell
# Run 5-epoch benchmark to compare speeds
.\run_trial4_rtx2060.ps1 -Epochs 5

# Compare with RTX 3060 timings:
# RTX 3060: ~2.0-2.5s per batch
# RTX 2060: Expected ~2.5-3.0s per batch (acceptable)
```

## Step 7: Parallel Workflow Implementation

### 7.1 Daily Workflow
```
Morning:
- RTX 3060: Start primary trial (Trial-5)
- RTX 2060: Start experimental trial (focal loss study)

Evening:
- Sync results to shared location
- Analyze both trials' progress
- Plan next day's experiments
```

### 7.2 Results Management
```powershell
# Create shared results directory structure
mkdir results_sync\rtx3060\
mkdir results_sync\rtx2060\

# Daily sync script (run on both machines)
# RTX 2060:
robocopy runs\ results_sync\rtx2060\ /E /XD .git

# RTX 3060:
robocopy runs\ results_sync\rtx3060\ /E /XD .git
```

## Step 8: Troubleshooting

### 8.1 Common Issues and Solutions

#### VRAM Out of Memory
```yaml
# Reduce batch size further if needed
batch_size: 12    # or even 8 if necessary

# Enable gradient checkpointing (if available)
# Reduce image size temporarily
img_size: 512     # instead of 640
```

#### Slower Performance Than Expected
- **Check GPU utilization**: Task Manager → Performance → GPU
- **Verify CUDA installation**: `nvidia-smi`
- **Monitor thermals**: RTX 2060 may throttle if overheating

#### Path Resolution Issues
- **Use same PowerShell script patterns** documented in CLAUDE.md
- **Verify all relative paths** resolve correctly
- **Test with simple 1-epoch run** before full training

### 8.2 Performance Optimization
```powershell
# Windows power settings
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c  # High Performance

# NVIDIA Control Panel
# Set Power Management Mode to "Prefer Maximum Performance"
# Set CUDA GPUs to all available

# Close unnecessary applications during training
# Ensure adequate cooling/ventilation
```

## Step 9: Results Comparison Protocol

### 9.1 Performance Normalization
Since RTX 2060 uses different batch size (16 vs 18), normalize results:
- **Focus on mAP@0.5** (batch-size independent)
- **Compare learning curves** (epochs to convergence)
- **Validate hyperparameter effects** are consistent across GPUs

### 9.2 Documentation Standards
- **Separate documentation files** for each machine
- **Clear naming convention**: `trial_X_rtx2060_results.md`
- **Hardware specification notes** in all result files

## Timeline and Benefits

### Setup Time Investment
- **Initial setup**: 2-3 hours
- **First test run**: 1 hour
- **Total investment**: Half day

### Research Acceleration
- **Training velocity**: 2x (parallel trials)
- **Experiment coverage**: Double the trial count
- **Risk mitigation**: Backup training capability
- **Thesis timeline**: Significant advantage with 40 days remaining

### Expected Outcomes
- **Week 1**: RTX 2060 running focal loss experiments
- **Week 2**: RTX 2060 training MobileNet-SSD baseline
- **Week 3-4**: Both machines running different model architectures
- **Result**: Complete multi-model comparison in parallel

## Success Metrics

### Technical Validation
- ✅ RTX 2060 achieves >80% of RTX 3060 training speed
- ✅ Comparable mAP@0.5 results with adjusted batch size
- ✅ Stable training without VRAM overflow

### Research Impact
- ✅ Double the number of experiments completed
- ✅ Parallel A/B testing of hyperparameter changes
- ✅ Accelerated multi-model comparison phase
- ✅ Enhanced thesis methodology and results coverage

---

**Status**: READY FOR IMPLEMENTATION  
**Priority**: HIGH (significant timeline advantage)  
**Estimated Setup Time**: 2-3 hours  
**Research Acceleration**: 2x trial velocity