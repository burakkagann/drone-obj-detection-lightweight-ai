# YOLOv5n Trial-2 Hyperparameter Optimization Training Script
# Optimized for VisDrone dataset with research-backed settings
# 
# Key Optimizations Applied:
# 1. Enabled mosaic and mixup augmentation (critical for performance)
# 2. Increased image resolution from 416 to 640 pixels
# 3. Reduced learning rate for small object detection
# 4. Increased batch size for stable gradients
# 5. Optimized loss function weights for small objects
#
# Expected Improvements:
# - Baseline: 17.80% mAP@0.5
# - Target: 22-25% mAP@0.5 (+3-5% improvement)
# - Success threshold: >18.8% mAP@0.5 (+1% minimum)

param(
    [int]$Epochs = 100,
    [switch]$QuickTest,
    [switch]$Help
)

# Color functions for better output
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

# Help function
function Show-Help {
    Write-Host "YOLOv5n Trial-2 Hyperparameter Optimization Training" -ForegroundColor Green
    Write-Host "====================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "USAGE:" -ForegroundColor Yellow
    Write-Host "  .\run_trial2_hyperopt.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "OPTIONS:" -ForegroundColor Yellow
    Write-Host "  -Epochs <number>     Number of training epochs (default: 100)"
    Write-Host "  -QuickTest          Run 20-epoch validation test first"
    Write-Host "  -Help               Show this help message"
    Write-Host ""
    Write-Host "EXAMPLES:" -ForegroundColor Yellow
    Write-Host "  .\run_trial2_hyperopt.ps1 -QuickTest"
    Write-Host "  .\run_trial2_hyperopt.ps1 -Epochs 100"
    Write-Host ""
    Write-Host "OPTIMIZATIONS APPLIED:" -ForegroundColor Cyan
    Write-Host "  [CHECK] Enabled mosaic augmentation (0.0 → 0.8)"
    Write-Host "  [CHECK] Enabled mixup augmentation (0.0 → 0.4)"
    Write-Host "  [CHECK] Increased image resolution (416 → 640)"
    Write-Host "  [CHECK] Reduced learning rate (0.01 → 0.005)"
    Write-Host "  [CHECK] Increased batch size (8 → 16)"
    Write-Host "  [CHECK] Extended warmup epochs (3.0 → 5.0)"
    Write-Host "  [CHECK] Optimized loss function weights"
    Write-Host ""
    Write-Host "EXPECTED RESULTS:" -ForegroundColor Cyan
    Write-Host "  Baseline mAP@0.5: 17.80%"
    Write-Host "  Target mAP@0.5: 22-25% (+3-5% improvement)"
    Write-Host "  Success threshold: >18.8% mAP@0.5 (+1% minimum)"
    exit 0
}

# Show help if requested
if ($Help) {
    Show-Help
}

# Header
Write-Host "================================================================" -ForegroundColor Green
Write-Host "YOLOv5n Trial-2 Hyperparameter Optimization Training" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""

# Configuration
Write-Info "[CONFIG] Training Configuration:"
Write-Host "  • Epochs: $Epochs"
Write-Host "  • Quick Test: $QuickTest"
Write-Host "  • Batch Size: 16 (optimized)"
Write-Host "  • Image Size: 640 (optimized)"
Write-Host "  • Learning Rate: 0.005 (optimized)"
Write-Host "  • Augmentation: Mosaic + Mixup enabled"
Write-Host ""

# Check if quick test is enabled
if ($QuickTest) {
    $Epochs = 20
    Write-Warning "[QUICK TEST] Running 20-epoch validation test first"
    Write-Host ""
}

# Environment validation
Write-Info "[VALIDATION] Checking environment..."

# Check if virtual environment is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Warning "[WARNING] Virtual environment not detected"
    Write-Host "Please activate yolov5n_env first:"
    Write-Host "  venvs\yolov5n_env\Scripts\activate"
    Write-Host ""
}

# Check CUDA availability
Write-Info "[GPU] Checking GPU availability..."
try {
    $cudaResult = & nvidia-smi 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "[GPU] GPU detected and available"
    } else {
        Write-Warning "[GPU] GPU not detected, will use CPU"
    }
} catch {
    Write-Warning "[GPU] nvidia-smi not found, will use CPU"
}

# Check if hyperparameter file exists
$hypFile = "..\..\..\..\..\config\visdrone\yolov5n_v1\hyp_visdrone_trial-2_optimized.yaml"
if (Test-Path $hypFile) {
    Write-Success "[CONFIG] Hyperparameter file found"
} else {
    Write-Error "[ERROR] Hyperparameter file not found: $hypFile"
    exit 1
}

# Check if dataset config exists
$dataFile = "..\..\..\..\..\config\visdrone\yolov5n_v1\yolov5n_visdrone_config.yaml"
if (Test-Path $dataFile) {
    Write-Success "[CONFIG] Dataset config file found"
} else {
    Write-Error "[ERROR] Dataset config file not found: $dataFile"
    exit 1
}

Write-Host ""

# Training execution
Write-Info "[TRAINING] Starting Trial-2 hyperparameter optimization..."
Write-Host ""

# Prepare arguments
$pythonArgs = @(
    "train_yolov5n_trial2_hyperopt.py",
    "--epochs", $Epochs.ToString()
)

if ($QuickTest) {
    $pythonArgs += "--quick-test"
}

# Execute training
Write-Host "Executing: python $($pythonArgs -join ' ')" -ForegroundColor Cyan
Write-Host ""

try {
    $process = Start-Process -FilePath "python" -ArgumentList $pythonArgs -NoNewWindow -PassThru -Wait
    
    if ($process.ExitCode -eq 0) {
        Write-Host ""
        Write-Success "[SUCCESS] Trial-2 training completed successfully!"
        Write-Host ""
        
        # Show expected results reminder
        Write-Info "[RESULTS] Expected Performance Targets:"
        Write-Host "  • Minimum: >18.8% mAP@0.5 (+1% improvement)"
        Write-Host "  • Target: >21% mAP@0.5 (+3% improvement)"
        Write-Host "  • Excellent: >23% mAP@0.5 (+5% improvement)"
        Write-Host ""
        
        # Show next steps
        Write-Info "[NEXT STEPS] Recommendations:"
        Write-Host "  • Check training logs for final mAP@0.5 results"
        Write-Host "  • Compare against baseline (17.80% mAP@0.5)"
        Write-Host "  • If improvement >3%, proceed to full 100-epoch training"
        Write-Host "  • If improvement 1-3%, consider Phase 2 optimizations"
        Write-Host "  • If improvement <1%, debug and try alternative approaches"
        Write-Host ""
        
    } else {
        Write-Error "[ERROR] Training failed with exit code: $($process.ExitCode)"
        exit 1
    }
    
} catch {
    Write-Error "[ERROR] Failed to execute training: $($_.Exception.Message)"
    exit 1
}

Write-Host "================================================================" -ForegroundColor Green
Write-Host "Trial-2 Hyperparameter Optimization Complete" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green 