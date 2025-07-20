# YOLOv5n Trial-3A: Pedestrian Detection Fix Training Script
# Focus: Address class imbalance and pedestrian detection failure
# 
# Key Optimizations Applied:
# 1. Focal Loss implementation for class imbalance (alpha=0.25, gamma=2.0)
# 2. Class-specific loss weights (pedestrian=5.0, people=1.0)
# 3. Pedestrian-specific augmentation and oversampling (factor=3.0)
# 4. Anchor optimization for small objects
# 5. Enhanced data loading for balanced training
#
# Expected Improvements:
# - Baseline: 22.6% mAP@0.5 (Trial-2 result)
# - Target: 23-25% mAP@0.5 (+0.4-2.4% improvement)
# - Pedestrian mAP@0.5: >10% (vs current 1.25%)
# - Pedestrian Recall: >15% (vs current 0%)

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
    Write-Host "YOLOv5n Trial-3A: Pedestrian Detection Fix Training" -ForegroundColor Green
    Write-Host "=====================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "USAGE:" -ForegroundColor Yellow
    Write-Host "  .\run_trial3a_pedestrian_fix.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "OPTIONS:" -ForegroundColor Yellow
    Write-Host "  -Epochs <number>     Number of training epochs (default: 100)"
    Write-Host "  -QuickTest          Run 20-epoch validation test first"
    Write-Host "  -Help               Show this help message"
    Write-Host ""
    Write-Host "EXAMPLES:" -ForegroundColor Yellow
    Write-Host "  .\run_trial3a_pedestrian_fix.ps1 -QuickTest"
    Write-Host "  .\run_trial3a_pedestrian_fix.ps1 -Epochs 100"
    Write-Host ""
    Write-Host "CRITICAL ISSUES ADDRESSED:" -ForegroundColor Cyan
    Write-Host "  [FIX] Pedestrian detection failure (1.25% mAP@0.5)"
    Write-Host "  [FIX] Zero pedestrian recall (0%)"
    Write-Host "  [FIX] Class imbalance (96.5% people vs 3.5% pedestrian)"
    Write-Host "  [FIX] Poor small object detection"
    Write-Host ""
    Write-Host "OPTIMIZATIONS APPLIED:" -ForegroundColor Cyan
    Write-Host "  [NEW] Focal Loss implementation (alpha=0.25, gamma=2.0)"
    Write-Host "  [NEW] Class-specific loss weights (pedestrian=5.0, people=1.0)"
    Write-Host "  [NEW] Pedestrian-specific augmentation (factor=3.0)"
    Write-Host "  [NEW] Anchor optimization for small objects"
    Write-Host "  [NEW] Enhanced data loading for balanced training"
    Write-Host "  [NEW] Pedestrian oversampling strategy"
    Write-Host ""
    Write-Host "EXPECTED RESULTS:" -ForegroundColor Cyan
    Write-Host "  Baseline mAP@0.5: 22.6% (Trial-2 result)"
    Write-Host "  Target mAP@0.5: 23-25% (+0.4-2.4% improvement)"
    Write-Host "  Pedestrian mAP@0.5: >10% (vs current 1.25%)"
    Write-Host "  Pedestrian Recall: >15% (vs current 0%)"
    Write-Host "  Success threshold: >23% mAP@0.5 (+0.4% minimum)"
    exit 0
}

# Show help if requested
if ($Help) {
    Show-Help
}

# Header
Write-Host "================================================================" -ForegroundColor Green
Write-Host "YOLOv5n Trial-3A: Pedestrian Detection Fix Training" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""

# Configuration
Write-Info "[CONFIG] Trial 3A Configuration:"
Write-Host "  • Epochs: $Epochs"
Write-Host "  • Quick Test: $QuickTest"
Write-Host "  • Batch Size: 16 (optimized)"
Write-Host "  • Image Size: 640 (optimized)"
Write-Host "  • Learning Rate: 0.005 (optimized)"
Write-Host "  • Focal Loss: Enabled (alpha=0.25, gamma=2.0)"
Write-Host "  • Class Weights: [people=1.0, pedestrian=5.0]"
Write-Host "  • Pedestrian Augmentation: Factor 3.0"
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
Write-Info "[TRAINING] Starting Trial-3A Pedestrian Detection Fix..."
Write-Host ""

# Prepare arguments
$pythonArgs = @(
    "train_yolov5n_trial3a_pedestrian_fix.py",
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
        Write-Success "[SUCCESS] Trial-3A training completed successfully!"
        Write-Host ""
        
        # Show expected results reminder
        Write-Info "[RESULTS] Expected Performance Targets:"
        Write-Host "  • Minimum: >23% mAP@0.5 (+0.4% improvement)"
        Write-Host "  • Target: >24% mAP@0.5 (+1.4% improvement)"
        Write-Host "  • Excellent: >25% mAP@0.5 (+2.4% improvement)"
        Write-Host "  • Pedestrian mAP@0.5: >10% (vs current 1.25%)"
        Write-Host "  • Pedestrian Recall: >15% (vs current 0%)"
        Write-Host ""
        
        # Show next steps
        Write-Info "[NEXT STEPS] Recommendations:"
        Write-Host "  • Check training logs for final mAP@0.5 results"
        Write-Host "  • Compare pedestrian detection performance"
        Write-Host "  • If improvement >1.4%, proceed to Trial 3B (Recall Optimization)"
        Write-Host "  • If improvement 0.4-1.4%, refine focal loss parameters"
        Write-Host "  • If improvement <0.4%, investigate data quality issues"
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
Write-Host "Trial-3A Pedestrian Detection Fix Complete" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green 