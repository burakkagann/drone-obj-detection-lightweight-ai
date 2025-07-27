# YOLOv5n Trial-1 (Phase 2) Training Script - VisDrone Dataset
# Master's Thesis: Robust Object Detection for Surveillance Drones
# 
# This PowerShell script executes YOLOv5n Trial-1 (Phase 2) training with 
# environmental augmentation to establish robustness performance.
#
# Author: Burak Kağan Yılmazer
# Date: July 2025
# Environment: yolov5n_env
# Protocol: Version 2.0 - True Baseline Framework

param(
    [int]$Epochs = 50,
    [switch]$QuickTest,
    [switch]$Help
)

function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Write-Success { Write-ColorOutput Green $args }
function Write-Warning { Write-ColorOutput Yellow $args }
function Write-Error { Write-ColorOutput Red $args }
function Write-Info { Write-ColorOutput Cyan $args }

if ($Help) {
    Write-Host "YOLOv5n Trial-1 (Phase 2) Training Script" -ForegroundColor Green
    Write-Host "=========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "PROTOCOL:" -ForegroundColor Yellow
    Write-Host "  Version 2.0 - True Baseline Framework"
    Write-Host "  Phase 2: Environmental Robustness"
    Write-Host ""
    Write-Host "USAGE:" -ForegroundColor Yellow
    Write-Host "  .\run_yolov5n_trial1.ps1 [-Epochs 100] [-QuickTest] [-Help]"
    Write-Host ""
    Write-Host "PARAMETERS:" -ForegroundColor Yellow
    Write-Host "  -Epochs      Number of training epochs (default: 50)"
    Write-Host "  -QuickTest   Run quick validation (20 epochs)"
    Write-Host "  -Help        Show this help message"
    Write-Host ""
    Write-Host "EXAMPLES:" -ForegroundColor Yellow
    Write-Host "  .\run_yolov5n_trial1.ps1                    # Standard 50-epoch training"
    Write-Host "  .\run_yolov5n_trial1.ps1 -Epochs 100        # Extended 100-epoch training"
    Write-Host "  .\run_yolov5n_trial1.ps1 -QuickTest         # Quick 20-epoch validation"
    Write-Host ""
    Write-Host "PHASE 2 FEATURES:" -ForegroundColor Yellow
    Write-Host "  Environmental robustness training:"
    Write-Host "  • Environmental augmented dataset (if available)"
    Write-Host "  • Real-time augmentation enabled"
    Write-Host "  • Optimized hyperparameters for robustness"
    Write-Host "  • Multi-scale training enabled"
    Write-Host "  • Complete methodology demonstration"
    Write-Host ""
    Write-Host "EXPECTED PERFORMANCE:" -ForegroundColor Yellow
    Write-Host "  Target: >25% mAP@0.5 (+7pp from baseline)"
    Write-Host "  Methodology: Demonstrate complete research impact"
    Write-Host "  Thesis value: Show environmental robustness improvement"
    exit 0
}

Write-Host "================================================================" -ForegroundColor Green
Write-Host "YOLOv5n Trial-1 (Phase 2) Training - VisDrone Dataset" -ForegroundColor Green
Write-Host "PROTOCOL: Version 2.0 - Environmental Robustness Framework" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""

# Training configuration summary
Write-Info "[CONFIGURATION] Training Setup:"
Write-Host "  • Model: YOLOv5n (nano)"
Write-Host "  • Phase: 2 (Environmental Robustness)"
Write-Host "  • Protocol: Version 2.0 Framework"
Write-Host "  • Dataset: Environmental Augmented (if available)"
Write-Host "  • Epochs: $Epochs"
if ($QuickTest) {
    Write-Host "  • Mode: Quick Test (20 epochs)" -ForegroundColor Yellow
} else {
    Write-Host "  • Mode: Full Robustness Training"
}
Write-Host "  • Target: >25% mAP@0.5 (+7pp improvement)"
Write-Host "  • GPU: NVIDIA RTX 3060 Laptop (5GB)"
Write-Host "  • Environment: yolov5n_env"
Write-Host ""

Write-Info "[METHODOLOGY] Phase 2 Key Features:"
Write-Host "  • Environmental Dataset: Original + synthetic conditions"
Write-Host "  • Real-time Augmentation: Mosaic, mixup, HSV, geometric"
Write-Host "  • Multi-scale Training: Enhanced for robustness"
Write-Host "  • Optimized Hyperparameters: Reduced LR, balanced losses"
Write-Host "  • Batch Size: 16 (optimized for RTX 3060)"
Write-Host "  • Complete Methodology: Show total research impact"
Write-Host ""

Write-Info "[THESIS OBJECTIVES] Expected Outcomes:"
Write-Host "  • Target mAP@0.5: >25% (dramatic improvement from 18%)"
Write-Host "  • Model Size: <7MB (maintained edge readiness)"
Write-Host "  • Inference Speed: >20 FPS (maintained real-time)"
Write-Host "  • Research Impact: +7pp absolute improvement demonstration"
Write-Host "  • Methodology: Complete environmental robustness validation"
Write-Host ""

# Validate repository location
$expectedPath = "C:\Users\burak\OneDrive\Desktop\Git Repos\drone-obj-detection-lightweight-ai"
if ((Get-Location).Path -ne $expectedPath) {
    Write-Warning "[WARNING] Not in expected repository location"
    Write-Host "Expected: $expectedPath"
    Write-Host "Current:  $((Get-Location).Path)"
    Write-Host ""
    Write-Info "[NAVIGATION] Changing to repository root..."
    try {
        Set-Location $expectedPath
        Write-Success "[SUCCESS] Changed to repository root"
    } catch {
        Write-Error "[ERROR] Failed to change directory: $($_.Exception.Message)"
        exit 1
    }
}

# Environment validation - assumes manual activation  
Write-Info "[ENVIRONMENT] Using currently activated Python environment"
Write-Host "  • Ensure YOLOv5n VisDrone environment is activated before running this script"
Write-Host "  • Required: .\venvs\visdrone\yolov5n_visdrone_env\Scripts\Activate.ps1"

Write-Host ""
Write-Info "[VALIDATION] Environment Information:"
Write-Host "  • Python: $(python --version 2>$null)"
Write-Host "  • Location: $(Get-Location)"

# Validate PyTorch installation
Write-Info "[VALIDATION] Checking PyTorch installation..."
try {
    $torchCheck = python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('PyTorch ready!')" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "[READY] PyTorch installation validated"
        Write-Host $torchCheck
    } else {
        Write-Error "[ERROR] PyTorch validation failed"
        exit 1
    }
} catch {
    Write-Error "[ERROR] Failed to validate PyTorch: $($_.Exception.Message)"
    exit 1
}

# Validate datasets
Write-Info "[VALIDATION] Checking dataset availability..."

# Check for environmental augmented dataset
$envDatasetPath = ".\data\environmental_augmented_dataset\visdrone"
$origDatasetPath = ".\data\my_dataset\visdrone"

if (Test-Path $envDatasetPath) {
    Write-Success "[OPTIMAL] Environmental augmented dataset found:"
    Write-Host "  • Dataset path: $envDatasetPath"
    Write-Host "  • Contains: Original + synthetic environmental conditions"
    Write-Host "  • Training mode: Phase 2 optimal (environmental + real-time aug)"
} elseif (Test-Path $origDatasetPath) {
    Write-Warning "[FALLBACK] Using original dataset with enhanced augmentation:"
    Write-Host "  • Dataset path: $origDatasetPath"
    Write-Host "  • Contains: Original VisDrone images only"
    Write-Host "  • Training mode: Phase 2 fallback (enhanced real-time aug only)"
} else {
    Write-Error "[ERROR] No valid dataset found"
    Write-Host "Required: Either environmental augmented or original VisDrone dataset"
    exit 1
}

Write-Host ""
Write-Info "[TRAINING] Starting YOLOv5n Trial-1 (Phase 2) training..."
Write-Host "Training script: src\scripts\visdrone\YOLOv5n\trial-1\train_yolov5n_trial1.py"
if ($QuickTest) {
    Write-Host "Expected duration: 30-45 minutes (quick test - 20 epochs)"
} else {
    Write-Host "Expected duration: 2-3 hours (100 epochs)"
}
Write-Host ""

# Prepare Python arguments
$pythonArgs = @(
    "src\scripts\visdrone\YOLOv5n\trial-1\train_yolov5n_trial1.py",
    "--epochs", $Epochs.ToString()
)

if ($QuickTest) {
    $pythonArgs += "--quick-test"
}

# Execute training
Write-Info "[EXECUTION] Running Phase 2 environmental robustness training..."
Write-Host "Command: python $($pythonArgs -join ' ')"
Write-Host ""

try {
    $process = Start-Process -FilePath "python" -ArgumentList $pythonArgs -NoNewWindow -PassThru -Wait
    
    if ($process.ExitCode -eq 0) {
        Write-Host ""
        Write-Success "[SUCCESS] YOLOv5n Trial-1 (Phase 2) training completed successfully!"
        Write-Host ""
        Write-Info "[RESULTS] Training Summary:"
        Write-Host "  • Model: YOLOv5n Trial-1 (Phase 2)"
        Write-Host "  • Epochs: $Epochs"
        Write-Host "  • Dataset: Environmental (if available) + Real-time Aug"
        Write-Host "  • Augmentation: Full robustness suite enabled"
        Write-Host "  • Results: runs\train\yolov5n_trial1_*"
        Write-Host ""
        Write-Info "[METHODOLOGY COMPLIANCE] Phase Analysis:"
        Write-Host "  • Phase 2 Robustness: Environmental training complete"
        Write-Host "  • Target Performance: >25% mAP@0.5"
        Write-Host "  • Expected Improvement: +7pp from Phase 1 baseline"
        Write-Host "  • Research Impact: Complete methodology demonstration"
        Write-Host ""
        Write-Info "[COMPARATIVE ANALYSIS] Phase 1 vs Phase 2:"
        Write-Host "  1. Compare baseline (18% mAP) vs robustness (25%+ mAP)"
        Write-Host "  2. Quantify environmental robustness improvement"
        Write-Host "  3. Document complete methodology impact"
        Write-Host "  4. Generate thesis-quality comparative results"
        Write-Host "  5. Validate edge deployment readiness"
        
    } else {
        Write-Error "[ERROR] Training failed with exit code: $($process.ExitCode)"
        Write-Host "Check the training logs for detailed error information."
        exit 1
    }
    
} catch {
    Write-Error "[ERROR] Failed to execute training: $($_.Exception.Message)"
    exit 1
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Success "[COMPLETED] YOLOv5n Trial-1 (Phase 2) Training Session Finished"
Write-Host "================================================================" -ForegroundColor Green