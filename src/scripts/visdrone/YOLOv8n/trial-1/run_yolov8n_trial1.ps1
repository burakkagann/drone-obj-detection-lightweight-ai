# YOLOv8n Trial-1 Training Script - VisDrone Dataset
# Master's Thesis: Robust Object Detection for Surveillance Drones
# 
# This PowerShell script executes YOLOv8n Trial-1 training with hyperparameters
# adapted from successful YOLOv5n Trial-2 optimizations.
#
# Author: Burak Kağan Yılmazer
# Date: January 2025
# Environment: yolov8n-visdrone_venv

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
    Write-Host "YOLOv8n Trial-1 Training Script" -ForegroundColor Green
    Write-Host "===============================" -ForegroundColor Green
    Write-Host ""
    Write-Host "USAGE:" -ForegroundColor Yellow
    Write-Host "  .\run_yolov8n_trial1.ps1 [-Epochs 50] [-QuickTest] [-Help]"
    Write-Host ""
    Write-Host "PARAMETERS:" -ForegroundColor Yellow
    Write-Host "  -Epochs      Number of training epochs (default: 50)"
    Write-Host "  -QuickTest   Run quick validation (10 epochs, reduced settings)"
    Write-Host "  -Help        Show this help message"
    Write-Host ""
    Write-Host "EXAMPLES:" -ForegroundColor Yellow
    Write-Host "  .\run_yolov8n_trial1.ps1                      # Standard 50-epoch Trial-1"
    Write-Host "  .\run_yolov8n_trial1.ps1 -Epochs 100          # Extended 100-epoch training"
    Write-Host "  .\run_yolov8n_trial1.ps1 -QuickTest           # Quick 10-epoch validation"
    Write-Host ""
    Write-Host "OPTIMIZATIONS:" -ForegroundColor Yellow
    Write-Host "  Hyperparameters adapted from YOLOv5n Trial-2 success:"
    Write-Host "  • Reduced learning rate (0.005) for small object detection"
    Write-Host "  • Extended warmup (5 epochs) for training stability"
    Write-Host "  • Enabled mosaic (0.8) and mixup (0.4) augmentation"
    Write-Host "  • Optimized augmentation settings for drone imagery"
    Write-Host "  • Higher resolution (640px) for small objects"
    Write-Host "  • Optimized batch size (16) for stable gradients"
    Write-Host ""
    Write-Host "EXPECTED PERFORMANCE:" -ForegroundColor Yellow
    Write-Host "  Target: >25% mAP@0.5 (improvement over baseline)"
    Write-Host "  Based on YOLOv5n Trial-2 success (23.557% mAP@0.5)"
    exit 0
}

Write-Host "================================================================" -ForegroundColor Green
Write-Host "YOLOv8n Trial-1 Training - VisDrone Dataset" -ForegroundColor Green
Write-Host "Hyperparameters adapted from YOLOv5n Trial-2 success" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""

# Training configuration summary
Write-Info "[CONFIGURATION] Training Setup:"
Write-Host "  • Model: YOLOv8n (Trial-1 optimized hyperparameters)"
Write-Host "  • Dataset: VisDrone (10 classes)"
Write-Host "  • Epochs: $Epochs"
if ($QuickTest) {
    Write-Host "  • Mode: Quick Test (10 epochs, reduced settings)" -ForegroundColor Yellow
} else {
    Write-Host "  • Mode: Full Optimized Training"
}
Write-Host "  • GPU: NVIDIA RTX 3060 Laptop (5GB)"
Write-Host "  • Environment: yolov8n-visdrone_venv"
Write-Host ""

Write-Info "[OPTIMIZATIONS] Key Trial-1 Adaptations:"
Write-Host "  • Learning Rate: 0.005 (reduced for small objects)"
Write-Host "  • Warmup: 5 epochs (extended for stability)"
Write-Host "  • Augmentation: Mosaic 0.8, Mixup 0.4 (enabled)"
Write-Host "  • Resolution: 640px (higher for small objects)"
Write-Host "  • Batch Size: 16 (optimized for RTX 3060)"
Write-Host "  • Loss Weights: Adapted for YOLOv8 architecture"
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

# Validate virtual environment
$venvPath = ".\venvs\yolov8n-visdrone_venv"
if (-not (Test-Path $venvPath)) {
    Write-Error "[ERROR] Virtual environment not found: $venvPath"
    Write-Host "Please create the environment first using:"
    Write-Host "  python -m venv .\venvs\yolov8n-visdrone_venv"
    exit 1
}

Write-Info "[ACTIVATION] Activating YOLOv8n VisDrone environment..."

# Activate virtual environment
try {
    & ".\venvs\yolov8n-visdrone_venv\Scripts\Activate.ps1"
    Write-Success "[SUCCESS] Virtual environment activated"
} catch {
    Write-Error "[ERROR] Failed to activate virtual environment: $($_.Exception.Message)"
    exit 1
}

Write-Host ""
Write-Info "[VALIDATION] Environment Information:"
Write-Host "  • Python: $(python --version 2>$null)"
Write-Host "  • Location: $(Get-Location)"

# Validate YOLOv8 installation
Write-Info "[VALIDATION] Checking YOLOv8 installation..."
try {
    $yoloCheck = python -c "from ultralytics import YOLO; import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('YOLOv8 ready!')" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "[READY] YOLOv8 installation validated"
        Write-Host $yoloCheck
    } else {
        Write-Error "[ERROR] YOLOv8 validation failed"
        exit 1
    }
} catch {
    Write-Error "[ERROR] Failed to validate YOLOv8: $($_.Exception.Message)"
    exit 1
}

Write-Host ""
Write-Info "[TRAINING] Starting YOLOv8n Trial-1 training..."
Write-Host "Training script: src\scripts\visdrone\YOLOv8n\trial-1\train_yolov8n_trial1.py"
if ($QuickTest) {
    Write-Host "Expected duration: 20-30 minutes (quick test)"
} else {
    Write-Host "Expected duration: 1-2 hours (50 epochs)"
}
Write-Host ""

# Prepare Python arguments
$pythonArgs = @(
    "src\scripts\visdrone\YOLOv8n\trial-1\train_yolov8n_trial1.py",
    "--epochs", $Epochs.ToString()
)

if ($QuickTest) {
    $pythonArgs += "--quick-test"
}

# Execute training
Write-Info "[EXECUTION] Running Trial-1 optimized training..."
Write-Host "Command: python $($pythonArgs -join ' ')"
Write-Host ""

try {
    $process = Start-Process -FilePath "python" -ArgumentList $pythonArgs -NoNewWindow -PassThru -Wait
    
    if ($process.ExitCode -eq 0) {
        Write-Host ""
        Write-Success "[SUCCESS] YOLOv8n Trial-1 training completed successfully!"
        Write-Host ""
        Write-Info "[RESULTS] Training Summary:"
        Write-Host "  • Model: YOLOv8n Trial-1 (optimized)"
        Write-Host "  • Epochs: $Epochs"
        Write-Host "  • Dataset: VisDrone"
        Write-Host "  • Results: runs\train\yolov8n_trial1_*"
        Write-Host ""
        Write-Info "[PERFORMANCE ANALYSIS] Expected Improvements:"
        Write-Host "  • Target mAP@0.5: >25% (vs baseline)"
        Write-Host "  • Key factors: Enhanced augmentation, optimized LR"
        Write-Host "  • Based on YOLOv5n Trial-2 success (23.557% mAP@0.5)"
        Write-Host ""
        Write-Info "[NEXT STEPS] After Trial-1 completion:"
        Write-Host "  1. Compare Trial-1 vs Baseline performance"
        Write-Host "  2. Analyze mAP@0.5, precision, recall metrics"
        Write-Host "  3. Document hyperparameter impact for thesis"
        Write-Host "  4. Consider Trial-2 if further optimization needed"
        Write-Host "  5. Compare with YOLOv5n performance benchmarks"
        
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
Write-Success "[COMPLETED] YOLOv8n Trial-1 Training Session Finished"
Write-Host "================================================================" -ForegroundColor Green