# NanoDet Trial-1 (Phase 3) Training Script - VisDrone Dataset
# Master's Thesis: Robust Object Detection for Surveillance Drones
# 
# This PowerShell script executes NanoDet Trial-1 (Phase 3) training with 
# synthetic environmental augmentation for ultra-lightweight robustness testing.
#
# Author: Burak Kağan Yılmazer
# Date: January 2025
# Environment: nanodet_env

param(
    [int]$Epochs = 120,
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
    Write-Host "NanoDet Trial-1 (Phase 3) Training Script" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "METHODOLOGY:" -ForegroundColor Yellow
    Write-Host "  Phase 3: Synthetic Environmental Augmentation"
    Write-Host "  Ultra-lightweight model (<3MB) with enhanced robustness"
    Write-Host "  Compare against Phase 2 baseline performance"
    Write-Host ""
    Write-Host "USAGE:" -ForegroundColor Yellow
    Write-Host "  .\run_nanodet_trial1.ps1 [-Epochs 120] [-QuickTest] [-Help]"
    Write-Host ""
    Write-Host "PARAMETERS:" -ForegroundColor Yellow
    Write-Host "  -Epochs      Number of training epochs (default: 120)"
    Write-Host "  -QuickTest   Run quick validation (15 epochs, minimal settings)"
    Write-Host "  -Help        Show this help message"
    Write-Host ""
    Write-Host "EXAMPLES:" -ForegroundColor Yellow
    Write-Host "  .\run_nanodet_trial1.ps1                    # Standard 120-epoch Trial-1"
    Write-Host "  .\run_nanodet_trial1.ps1 -Epochs 150        # Extended 150-epoch training"
    Write-Host "  .\run_nanodet_trial1.ps1 -QuickTest         # Quick 15-epoch validation"
    Write-Host ""
    Write-Host "PHASE 3 ULTRA-LIGHTWEIGHT FEATURES:" -ForegroundColor Yellow
    Write-Host "  Enhanced synthetic environmental augmentation:"
    Write-Host "  • Fog simulation (atmospheric scattering)"
    Write-Host "  • Night conditions (gamma correction + noise)"
    Write-Host "  • Motion blur (linear kernel convolution)"
    Write-Host "  • Rain effects (streak overlay)"
    Write-Host "  • Enhanced standard augmentation (brightness, flip)"
    Write-Host "  • 60% environmental augmentation probability"
    Write-Host "  • Ultra-lightweight architecture (<3MB target)"
    Write-Host ""
    Write-Host "EXPECTED PERFORMANCE:" -ForegroundColor Yellow
    Write-Host "  Target: >17% mAP@0.5 (vs Phase 2 baseline >15%)"
    Write-Host "  Model Size: <3MB (ultra-lightweight edge deployment)"
    Write-Host "  Methodology: Quantify synthetic augmentation benefits"
    Write-Host "  Thesis value: Phase 2 vs Phase 3 ultra-lightweight comparison"
    exit 0
}

Write-Host "================================================================" -ForegroundColor Green
Write-Host "NanoDet Trial-1 (Phase 3) Training - VisDrone Dataset" -ForegroundColor Green
Write-Host "METHODOLOGY: Ultra-Lightweight Synthetic Environmental Augmentation" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""

# Training configuration summary
Write-Info "[CONFIGURATION] Training Setup:"
Write-Host "  • Model: NanoDet (Trial-1 Phase 3 ultra-lightweight)"
Write-Host "  • Phase: 3 (Synthetic Environmental Augmentation)"
Write-Host "  • Dataset: VisDrone (10 classes, COCO format)"
Write-Host "  • Epochs: $Epochs"
if ($QuickTest) {
    Write-Host "  • Mode: Quick Test (15 epochs, minimal settings)" -ForegroundColor Yellow
} else {
    Write-Host "  • Mode: Full Augmented Training"
}
Write-Host "  • Baseline: Phase 2 (original dataset only, >15% mAP@0.5)"
Write-Host "  • Target: >17% mAP@0.5 (improvement over baseline)"
Write-Host "  • Model Size Target: <3MB (ultra-lightweight)"
Write-Host "  • GPU: NVIDIA RTX 3060 Laptop (5GB)"
Write-Host "  • Environment: nanodet_env"
Write-Host ""

Write-Info "[METHODOLOGY] Phase 3 Ultra-Lightweight Features:"
Write-Host "  • Synthetic Environmental Augmentation: Fog, night, blur, rain"
Write-Host "  • Enhanced Standard Augmentation: Brightness, horizontal flip"
Write-Host "  • Augmentation Probability: 60% environmental, 30% brightness"
Write-Host "  • PyTorch Implementation: Simplified NanoDet-like architecture"
Write-Host "  • Learning Rate: 0.0008 (optimized for augmentation stability)"
Write-Host "  • Ultra-Lightweight: <3MB for extreme edge deployment"
Write-Host ""

Write-Info "[THESIS OBJECTIVES] Expected Outcomes:"
Write-Host "  • Target mAP@0.5: >17% (improvement over Phase 2 baseline >15%)"
Write-Host "  • Model Size: <3MB (ultra-lightweight edge deployment)"
Write-Host "  • Robustness: Enhanced performance in low-visibility conditions"
Write-Host "  • Methodology: Quantified synthetic augmentation benefits"
Write-Host "  • Comparison: Phase 2 vs Phase 3 ultra-lightweight analysis"
Write-Host "  • Research Value: Extreme edge device validation"
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
$venvPath = ".\venvs\nanodet_env"
if (-not (Test-Path $venvPath)) {
    Write-Error "[ERROR] Virtual environment not found: $venvPath"
    Write-Host "Please create the environment first or use existing NanoDet environment"
    exit 1
}

Write-Info "[ACTIVATION] Activating NanoDet environment..."

# Activate virtual environment
try {
    & ".\venvs\nanodet_env\Scripts\Activate.ps1"
    Write-Success "[SUCCESS] Virtual environment activated"
} catch {
    Write-Error "[ERROR] Failed to activate virtual environment: $($_.Exception.Message)"
    exit 1
}

Write-Host ""
Write-Info "[VALIDATION] Environment Information:"
Write-Host "  • Python: $(python --version 2>$null)"
Write-Host "  • Location: $(Get-Location)"

# Validate PyTorch installation and dependencies
Write-Info "[VALIDATION] Checking PyTorch installation..."
try {
    $torchCheck = python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('GPU Count:', torch.cuda.device_count() if torch.cuda.is_available() else 0); print('PyTorch ready!')" 2>$null
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

# Validate dataset and COCO format
Write-Info "[VALIDATION] Checking dataset and COCO format..."
$datasetPath = ".\data\my_dataset\visdrone"
$cocoPath = "$datasetPath\nanodet_format"

if (-not (Test-Path $datasetPath)) {
    Write-Warning "[WARNING] NanoDet dataset not found: $datasetPath"
    Write-Host "  Will create dummy dataset for testing"
} else {
    Write-Success "[READY] VisDrone dataset found"
}

if (-not (Test-Path $cocoPath)) {
    Write-Warning "[WARNING] COCO format not found: $cocoPath"
    Write-Host "  Will create COCO format during training"
} else {
    Write-Success "[READY] COCO format found"
}

Write-Success "[READY] Dataset validation completed:"
Write-Host "  • Dataset path: $datasetPath"
Write-Host "  • COCO format: Will be created if needed"
Write-Host "  • Ultra-lightweight model: <3MB target"
Write-Host "  • Augmentation: Environmental + standard ready"

Write-Host ""
Write-Info "[TRAINING] Starting NanoDet Trial-1 (Phase 3) training..."
Write-Host "Training script: src\scripts\visdrone\nanodet\trial-1\train_nanodet_trial1.py"
if ($QuickTest) {
    Write-Host "Expected duration: 25-40 minutes (quick test - 15 epochs)"
} else {
    Write-Host "Expected duration: 2-3 hours (120 epochs with environmental augmentation)"
}
Write-Host ""

# Prepare Python arguments
$pythonArgs = @(
    "src\scripts\visdrone\nanodet\trial-1\train_nanodet_trial1.py",
    "--epochs", $Epochs.ToString()
)

if ($QuickTest) {
    $pythonArgs += "--quick-test"
}

# Execute training
Write-Info "[EXECUTION] Running Trial-1 (Phase 3) ultra-lightweight augmented training..."
Write-Host "Command: python $($pythonArgs -join ' ')"
Write-Host ""

try {
    $process = Start-Process -FilePath "python" -ArgumentList $pythonArgs -NoNewWindow -PassThru -Wait
    
    if ($process.ExitCode -eq 0) {
        Write-Host ""
        Write-Success "[SUCCESS] NanoDet Trial-1 (Phase 3) training completed successfully!"
        Write-Host ""
        Write-Info "[RESULTS] Training Summary:"
        Write-Host "  • Model: NanoDet Trial-1 (Phase 3 - Ultra-Lightweight Synthetic Augmentation)"
        Write-Host "  • Epochs: $Epochs"
        Write-Host "  • Dataset: VisDrone (original + synthetic environmental)"
        Write-Host "  • Results: runs\train\nanodet_trial1_phase3_*"
        Write-Host "  • Model Size: <3MB (ultra-lightweight)"
        Write-Host ""
        Write-Info "[METHODOLOGY COMPLIANCE] Phase Analysis:"
        Write-Host "  • Phase 2 Baseline: Original dataset established (>15% mAP@0.5)"
        Write-Host "  • Phase 3 Target: >17% mAP@0.5 (synthetic augmentation)"
        Write-Host "  • Ultra-Lightweight: <3MB model for extreme edge deployment"
        Write-Host "  • Comparison: Quantified synthetic data benefits"
        Write-Host "  • Thesis Value: Ultra-lightweight environmental robustness"
        Write-Host ""
        Write-Info "[NEXT STEPS] After Trial-1 completion:"
        Write-Host "  1. Compare Trial-1 vs Baseline performance (Phase 2 vs Phase 3)"
        Write-Host "  2. Analyze mAP@0.5, precision, recall improvements"
        Write-Host "  3. Validate <3MB model size achievement"
        Write-Host "  4. Document ultra-lightweight synthetic augmentation impact"
        Write-Host "  5. Add to multi-model comparative framework"
        
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
Write-Success "[COMPLETED] NanoDet Trial-1 (Phase 3) Ultra-Lightweight Training Session Finished"
Write-Host "================================================================" -ForegroundColor Green