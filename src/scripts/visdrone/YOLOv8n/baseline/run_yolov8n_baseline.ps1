# YOLOv8n Baseline Training Script - VisDrone Dataset
# Master's Thesis: Robust Object Detection for Surveillance Drones
# 
# This PowerShell script executes YOLOv8n baseline training with proper environment setup
# and comprehensive logging for thesis research documentation.
#
# Author: Burak Kağan Yılmazer
# Date: January 2025
# Environment: yolov8n-visdrone_venv

param(
    [int]$Epochs = 20,
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
    Write-Host "YOLOv8n Baseline Training Script" -ForegroundColor Green
    Write-Host "===============================" -ForegroundColor Green
    Write-Host ""
    Write-Host "USAGE:" -ForegroundColor Yellow
    Write-Host "  .\run_yolov8n_baseline.ps1 [-Epochs 20] [-QuickTest] [-Help]"
    Write-Host ""
    Write-Host "PARAMETERS:" -ForegroundColor Yellow
    Write-Host "  -Epochs      Number of training epochs (default: 20)"
    Write-Host "  -QuickTest   Run quick validation (5 epochs, reduced settings)"
    Write-Host "  -Help        Show this help message"
    Write-Host ""
    Write-Host "EXAMPLES:" -ForegroundColor Yellow
    Write-Host "  .\run_yolov8n_baseline.ps1                    # Standard 20-epoch baseline"
    Write-Host "  .\run_yolov8n_baseline.ps1 -Epochs 50         # Extended 50-epoch training"
    Write-Host "  .\run_yolov8n_baseline.ps1 -QuickTest         # Quick 5-epoch validation"
    Write-Host ""
    Write-Host "PURPOSE:" -ForegroundColor Yellow
    Write-Host "  Establish YOLOv8n baseline performance on VisDrone dataset"
    Write-Host "  for comparison with optimized Trial-1 results."
    exit 0
}

Write-Host "================================================================" -ForegroundColor Green
Write-Host "YOLOv8n Baseline Training - VisDrone Dataset" -ForegroundColor Green
Write-Host "Master's Thesis: Robust Object Detection for Surveillance Drones" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""

# Training configuration summary
Write-Info "[CONFIGURATION] Training Setup:"
Write-Host "  • Model: YOLOv8n (baseline hyperparameters)"
Write-Host "  • Dataset: VisDrone (10 classes)"
Write-Host "  • Epochs: $Epochs"
if ($QuickTest) {
    Write-Host "  • Mode: Quick Test (reduced settings)" -ForegroundColor Yellow
} else {
    Write-Host "  • Mode: Full Training"
}
Write-Host "  • GPU: NVIDIA RTX 3060 Laptop (5GB)"
Write-Host "  • Environment: yolov8n-visdrone_venv"
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
Write-Info "[TRAINING] Starting YOLOv8n baseline training..."
Write-Host "Training script: src\scripts\visdrone\YOLOv8n\baseline\train_yolov8n_baseline.py"
Write-Host "Expected duration: 20-40 minutes (depending on epochs)"
Write-Host ""

# Prepare Python arguments
$pythonArgs = @(
    "src\scripts\visdrone\YOLOv8n\baseline\train_yolov8n_baseline.py",
    "--epochs", $Epochs.ToString()
)

if ($QuickTest) {
    $pythonArgs += "--quick-test"
}

# Execute training
Write-Info "[EXECUTION] Running baseline training..."
Write-Host "Command: python $($pythonArgs -join ' ')"
Write-Host ""

try {
    $process = Start-Process -FilePath "python" -ArgumentList $pythonArgs -NoNewWindow -PassThru -Wait
    
    if ($process.ExitCode -eq 0) {
        Write-Host ""
        Write-Success "[SUCCESS] YOLOv8n baseline training completed successfully!"
        Write-Host ""
        Write-Info "[RESULTS] Training Summary:"
        Write-Host "  • Model: YOLOv8n baseline"
        Write-Host "  • Epochs: $Epochs"
        Write-Host "  • Dataset: VisDrone"
        Write-Host "  • Results: runs\train\yolov8n_baseline_*"
        Write-Host ""
        Write-Info "[NEXT STEPS] After baseline completion:"
        Write-Host "  1. Review baseline performance metrics"
        Write-Host "  2. Compare with YOLOv5n baseline (if available)"
        Write-Host "  3. Run Trial-1 optimization: .\run_yolov8n_trial1.ps1"
        Write-Host "  4. Document results for thesis analysis"
        
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
Write-Success "[COMPLETED] YOLOv8n Baseline Training Session Finished"
Write-Host "================================================================" -ForegroundColor Green