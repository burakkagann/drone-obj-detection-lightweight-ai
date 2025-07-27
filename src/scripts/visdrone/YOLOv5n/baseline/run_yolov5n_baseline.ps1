# YOLOv5n Baseline (Phase 1) Training Script - VisDrone Dataset
# Master's Thesis: Robust Object Detection for Surveillance Drones
# 
# This PowerShell script executes YOLOv5n Baseline (Phase 1) training with 
# NO augmentation to establish true baseline performance.
#
# Author: Burak Kağan Yılmazer
# Date: July 2025
# Environment: yolov5n_env
# Protocol: Version 2.0 - True Baseline Framework

param(
    [int]$Epochs = 100,
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
    Write-Host "YOLOv5n Baseline (Phase 1) Training Script" -ForegroundColor Green
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "PROTOCOL:" -ForegroundColor Yellow
    Write-Host "  Version 2.0 - True Baseline Framework"
    Write-Host "  Phase 1: True Baseline (No Augmentation)"
    Write-Host ""
    Write-Host "USAGE:" -ForegroundColor Yellow
    Write-Host "  .\run_yolov5n_baseline.ps1 [-Epochs 100] [-QuickTest] [-Help]"
    Write-Host ""
    Write-Host "PARAMETERS:" -ForegroundColor Yellow
    Write-Host "  -Epochs      Number of training epochs (default: 100)"
    Write-Host "  -QuickTest   Run quick validation (20 epochs)"
    Write-Host "  -Help        Show this help message"
    Write-Host ""
    Write-Host "EXAMPLES:" -ForegroundColor Yellow
    Write-Host "  .\run_yolov5n_baseline.ps1                    # Standard 100-epoch baseline"
    Write-Host "  .\run_yolov5n_baseline.ps1 -Epochs 50         # Reduced 50-epoch training"
    Write-Host "  .\run_yolov5n_baseline.ps1 -QuickTest         # Quick 20-epoch validation"
    Write-Host ""
    Write-Host "PHASE 1 FEATURES:" -ForegroundColor Yellow
    Write-Host "  True baseline training with NO augmentation:"
    Write-Host "  • NO real-time augmentation (all disabled)"
    Write-Host "  • NO synthetic environmental augmentation"
    Write-Host "  • Original VisDrone dataset only"
    Write-Host "  • Minimal preprocessing (resize, normalize)"
    Write-Host "  • Pure model capability measurement"
    Write-Host ""
    Write-Host "EXPECTED PERFORMANCE:" -ForegroundColor Yellow
    Write-Host "  Target: >18% mAP@0.5 (protocol requirement)"
    Write-Host "  Methodology: Establish absolute reference point for Phase 2"
    Write-Host "  Thesis value: Show complete methodology impact (18% → 25%+)"
    exit 0
}

Write-Host "================================================================" -ForegroundColor Green
Write-Host "YOLOv5n Baseline (Phase 1) Training - VisDrone Dataset" -ForegroundColor Green
Write-Host "PROTOCOL: Version 2.0 - True Baseline Framework" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""

# Training configuration summary
Write-Info "[CONFIGURATION] Training Setup:"
Write-Host "  • Model: YOLOv5n (nano)"
Write-Host "  • Phase: 1 (True Baseline - No Augmentation)"
Write-Host "  • Protocol: Version 2.0 Framework"
Write-Host "  • Dataset: VisDrone (original only)"
Write-Host "  • Epochs: $Epochs"
if ($QuickTest) {
    Write-Host "  • Mode: Quick Test (20 epochs)" -ForegroundColor Yellow
} else {
    Write-Host "  • Mode: Full Baseline Training"
}
Write-Host "  • Target: >18% mAP@0.5 (thesis requirement)"
Write-Host "  • GPU: NVIDIA RTX 3060 Laptop (5GB)"
Write-Host "  • Environment: yolov5n_env"
Write-Host ""

Write-Info "[METHODOLOGY] Phase 1 Key Features:"
Write-Host "  • True Baseline: NO augmentation whatsoever"
Write-Host "  • Original Dataset: VisDrone images and labels only"
Write-Host "  • Minimal Processing: Resize (640x640) and normalize only"
Write-Host "  • YOLOv5n Architecture: Lightweight nano model"
Write-Host "  • Batch Size: 16 (optimized for RTX 3060)"
Write-Host "  • Pure Model Performance: Absolute reference point"
Write-Host ""

Write-Info "[THESIS OBJECTIVES] Expected Outcomes:"
Write-Host "  • Target mAP@0.5: >18% (minimum protocol requirement)"
Write-Host "  • Model Size: <7MB (edge deployment ready)"
Write-Host "  • Inference Speed: >20 FPS (real-time capability)"
Write-Host "  • Methodology: True baseline for Phase 2 comparison"
Write-Host "  • Research Value: Demonstrate complete methodology impact"
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

# Validate dataset
Write-Info "[VALIDATION] Checking VisDrone dataset..."
$datasetPath = ".\data\my_dataset\visdrone"
$trainPath = "$datasetPath\train\images"
$valPath = "$datasetPath\val\images"

if (-not (Test-Path $datasetPath)) {
    Write-Error "[ERROR] VisDrone dataset not found: $datasetPath"
    exit 1
}

if (-not (Test-Path $trainPath)) {
    Write-Error "[ERROR] Training images not found: $trainPath"
    exit 1
}

if (-not (Test-Path $valPath)) {
    Write-Error "[ERROR] Validation images not found: $valPath"
    exit 1
}

# Count dataset files
$trainCount = (Get-ChildItem "$trainPath\*.jpg").Count
$valCount = (Get-ChildItem "$valPath\*.jpg").Count
Write-Success "[READY] Dataset validated:"
Write-Host "  • Dataset path: $datasetPath"
Write-Host "  • Training images: $trainCount files"
Write-Host "  • Validation images: $valCount files"
Write-Host "  • Original dataset: No synthetic augmentation"

Write-Host ""
Write-Info "[TRAINING] Starting YOLOv5n Baseline (Phase 1) training..."
Write-Host "Training script: src\scripts\visdrone\YOLOv5n\baseline\train_yolov5n_baseline.py"
if ($QuickTest) {
    Write-Host "Expected duration: 30-45 minutes (quick test - 20 epochs)"
} else {
    Write-Host "Expected duration: 2-3 hours (100 epochs)"
}
Write-Host ""

# Prepare Python arguments
$pythonArgs = @(
    "src\scripts\visdrone\YOLOv5n\baseline\train_yolov5n_baseline.py",
    "--epochs", $Epochs.ToString()
)

if ($QuickTest) {
    $pythonArgs += "--quick-test"
}

# Execute training
Write-Info "[EXECUTION] Running Phase 1 baseline training..."
Write-Host "Command: python $($pythonArgs -join ' ')"
Write-Host ""

try {
    $process = Start-Process -FilePath "python" -ArgumentList $pythonArgs -NoNewWindow -PassThru -Wait
    
    if ($process.ExitCode -eq 0) {
        Write-Host ""
        Write-Success "[SUCCESS] YOLOv5n Baseline (Phase 1) training completed successfully!"
        Write-Host ""
        Write-Info "[RESULTS] Training Summary:"
        Write-Host "  • Model: YOLOv5n Baseline (Phase 1)"
        Write-Host "  • Epochs: $Epochs"
        Write-Host "  • Dataset: VisDrone (original only)"
        Write-Host "  • Augmentation: None (true baseline)"
        Write-Host "  • Results: runs\train\yolov5n_baseline_*"
        Write-Host ""
        Write-Info "[METHODOLOGY COMPLIANCE] Phase Analysis:"
        Write-Host "  • Phase 1 Baseline: TRUE baseline established"
        Write-Host "  • Target Performance: >18% mAP@0.5"
        Write-Host "  • Next Phase: Phase 2 environmental robustness training"
        Write-Host "  • Expected Improvement: 18% → 25%+ mAP@0.5 in Phase 2"
        Write-Host ""
        Write-Info "[NEXT STEPS] After Phase 1 completion:"
        Write-Host "  1. Evaluate baseline performance metrics"
        Write-Host "  2. Execute Phase 2 environmental robustness training"
        Write-Host "  3. Compare Phase 1 vs Phase 2 performance"
        Write-Host "  4. Demonstrate complete methodology impact"
        Write-Host "  5. Generate thesis-quality comparative analysis"
        
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
Write-Success "[COMPLETED] YOLOv5n Baseline (Phase 1) Training Session Finished"
Write-Host "================================================================" -ForegroundColor Green