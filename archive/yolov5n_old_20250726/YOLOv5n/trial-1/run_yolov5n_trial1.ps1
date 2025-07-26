# YOLOv5n Trial-1 (Phase 3) Training Script - VisDrone Dataset
# Master's Thesis: Robust Object Detection for Surveillance Drones
# 
# This PowerShell script executes YOLOv5n Trial-1 (Phase 3) training with 
# synthetic environmental augmentation for thesis methodology compliance.
#
# Author: Burak Kağan Yılmazer
# Date: January 2025
# Environment: yolov5n_env

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
    Write-Host "YOLOv5n Trial-1 (Phase 3) Training Script" -ForegroundColor Green
    Write-Host "=========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "METHODOLOGY:" -ForegroundColor Yellow
    Write-Host "  Phase 3: Synthetic Environmental Augmentation"
    Write-Host "  Compare against Phase 2 baseline (18.28% mAP@0.5)"
    Write-Host ""
    Write-Host "USAGE:" -ForegroundColor Yellow
    Write-Host "  .\run_yolov5n_trial1.ps1 [-Epochs 100] [-QuickTest] [-Help]"
    Write-Host ""
    Write-Host "PARAMETERS:" -ForegroundColor Yellow
    Write-Host "  -Epochs      Number of training epochs (default: 100)"
    Write-Host "  -QuickTest   Run quick validation (20 epochs, reduced settings)"
    Write-Host "  -Help        Show this help message"
    Write-Host ""
    Write-Host "EXAMPLES:" -ForegroundColor Yellow
    Write-Host "  .\run_yolov5n_trial1.ps1                      # Standard 100-epoch Trial-1"
    Write-Host "  .\run_yolov5n_trial1.ps1 -Epochs 150          # Extended 150-epoch training"
    Write-Host "  .\run_yolov5n_trial1.ps1 -QuickTest           # Quick 20-epoch validation"
    Write-Host ""
    Write-Host "PHASE 3 FEATURES:" -ForegroundColor Yellow
    Write-Host "  Enhanced synthetic environmental augmentation:"
    Write-Host "  • Fog simulation (light, medium, heavy)"
    Write-Host "  • Night conditions (low-light with noise)"
    Write-Host "  • Motion blur (camera/object movement)"
    Write-Host "  • Rain/snow effects (visibility reduction)"
    Write-Host "  • Enhanced standard augmentation (mosaic, mixup, HSV)"
    Write-Host ""
    Write-Host "EXPECTED PERFORMANCE:" -ForegroundColor Yellow
    Write-Host "  Target: >20% mAP@0.5 (vs 18.28% Phase 2 baseline)"
    Write-Host "  Methodology: Quantify synthetic augmentation benefits"
    Write-Host "  Thesis value: Phase 2 vs Phase 3 comparison analysis"
    exit 0
}

Write-Host "================================================================" -ForegroundColor Green
Write-Host "YOLOv5n Trial-1 (Phase 3) Training - VisDrone Dataset" -ForegroundColor Green
Write-Host "METHODOLOGY: Synthetic Environmental Augmentation" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""

# Training configuration summary
Write-Info "[CONFIGURATION] Training Setup:"
Write-Host "  • Model: YOLOv5n (Trial-1 Phase 3 optimized)"
Write-Host "  • Phase: 3 (Synthetic Environmental Augmentation)"
Write-Host "  • Dataset: VisDrone (10 classes)"
Write-Host "  • Epochs: $Epochs"
if ($QuickTest) {
    Write-Host "  • Mode: Quick Test (20 epochs, reduced settings)" -ForegroundColor Yellow
} else {
    Write-Host "  • Mode: Full Optimized Training"
}
Write-Host "  • Baseline: Phase 2 (18.28% mAP@0.5)"
Write-Host "  • GPU: NVIDIA RTX 3060 Laptop (5GB)"
Write-Host "  • Environment: yolov5n_env"
Write-Host ""

Write-Info "[METHODOLOGY] Phase 3 Key Features:"
Write-Host "  • Synthetic Environmental Augmentation: Fog, night, blur, rain"
Write-Host "  • Enhanced Standard Augmentation: Mosaic 0.8, Mixup 0.4"
Write-Host "  • Optimized Hyperparameters: Learning rate 0.005, Warmup 5.0"
Write-Host "  • Higher Resolution: 640px for small object detection"
Write-Host "  • Batch Size: 16 (optimized for stable gradients)"
Write-Host "  • Loss Weights: Optimized for small objects (obj: 1.2)"
Write-Host ""

Write-Info "[THESIS OBJECTIVES] Expected Outcomes:"
Write-Host "  • Target mAP@0.5: >20% (improvement over 18.28% baseline)"
Write-Host "  • Robustness: Enhanced performance in low-visibility conditions"
Write-Host "  • Methodology: Quantified synthetic augmentation benefits"
Write-Host "  • Comparison: Phase 2 vs Phase 3 analysis for thesis"
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
$venvPath = ".\venvs\yolov5n_env"
if (-not (Test-Path $venvPath)) {
    Write-Error "[ERROR] Virtual environment not found: $venvPath"
    Write-Host "Please create the environment first or use existing YOLOv5n environment"
    exit 1
}

Write-Info "[ACTIVATION] Activating YOLOv5n environment..."

# Activate virtual environment
try {
    & ".\venvs\yolov5n_env\Scripts\Activate.ps1"
    Write-Success "[SUCCESS] Virtual environment activated"
} catch {
    Write-Error "[ERROR] Failed to activate virtual environment: $($_.Exception.Message)"
    exit 1
}

Write-Host ""
Write-Info "[VALIDATION] Environment Information:"
Write-Host "  • Python: $(python --version 2>$null)"
Write-Host "  • Location: $(Get-Location)"

# Validate YOLOv5 installation and dependencies
Write-Info "[VALIDATION] Checking YOLOv5 installation..."
try {
    $yoloCheck = python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); if torch.cuda.is_available(): print('GPU:', torch.cuda.get_device_name(0)); print('YOLOv5 ready!')" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "[READY] YOLOv5 installation validated"
        Write-Host $yoloCheck
    } else {
        Write-Error "[ERROR] YOLOv5 validation failed"
        exit 1
    }
} catch {
    Write-Error "[ERROR] Failed to validate YOLOv5: $($_.Exception.Message)"
    exit 1
}

# Validate dataset and configuration files
Write-Info "[VALIDATION] Checking dataset and configurations..."
$datasetPath = ".\data\my_dataset\visdrone"
$configPath = ".\config\visdrone\yolov5n_v1\yolov5n_visdrone_config.yaml"
$modelPath = ".\src\models\YOLOv5\models\yolov5n.yaml"

if (-not (Test-Path $datasetPath)) {
    Write-Error "[ERROR] Dataset not found: $datasetPath"
    exit 1
}

if (-not (Test-Path $configPath)) {
    Write-Error "[ERROR] Dataset config not found: $configPath"
    exit 1
}

if (-not (Test-Path $modelPath)) {
    Write-Error "[ERROR] Model config not found: $modelPath"
    exit 1
}

Write-Success "[READY] Dataset and configurations validated"

Write-Host ""
Write-Info "[TRAINING] Starting YOLOv5n Trial-1 (Phase 3) training..."
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
Write-Info "[EXECUTION] Running Trial-1 (Phase 3) optimized training..."
Write-Host "Command: python $($pythonArgs -join ' ')"
Write-Host ""

try {
    $process = Start-Process -FilePath "python" -ArgumentList $pythonArgs -NoNewWindow -PassThru -Wait
    
    if ($process.ExitCode -eq 0) {
        Write-Host ""
        Write-Success "[SUCCESS] YOLOv5n Trial-1 (Phase 3) training completed successfully!"
        Write-Host ""
        Write-Info "[RESULTS] Training Summary:"
        Write-Host "  • Model: YOLOv5n Trial-1 (Phase 3 - Synthetic Augmentation)"
        Write-Host "  • Epochs: $Epochs"
        Write-Host "  • Dataset: VisDrone"
        Write-Host "  • Results: runs\train\yolov5n_trial1_phase3_*"
        Write-Host ""
        Write-Info "[METHODOLOGY COMPLIANCE] Phase Analysis:"
        Write-Host "  • Phase 2 Baseline: 18.28% mAP@0.5 (established)"
        Write-Host "  • Phase 3 Target: >20% mAP@0.5 (synthetic augmentation)"
        Write-Host "  • Comparison: Quantified synthetic data benefits"
        Write-Host "  • Thesis Value: Environmental robustness demonstration"
        Write-Host ""
        Write-Info "[NEXT STEPS] After Trial-1 completion:"
        Write-Host "  1. Compare Trial-1 vs Baseline performance (Phase 2 vs Phase 3)"
        Write-Host "  2. Analyze mAP@0.5, precision, recall improvements"
        Write-Host "  3. Document synthetic augmentation impact for thesis"
        Write-Host "  4. Generate comprehensive comparison analysis"
        Write-Host "  5. Prepare for multi-model framework replication"
        
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
Write-Success "[COMPLETED] YOLOv5n Trial-1 (Phase 3) Training Session Finished"
Write-Host "================================================================" -ForegroundColor Green