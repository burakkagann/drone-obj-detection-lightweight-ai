# MobileNet-SSD Baseline (Phase 2) Training Script - VisDrone Dataset
# Master's Thesis: Robust Object Detection for Surveillance Drones
# 
# This PowerShell script executes MobileNet-SSD Baseline (Phase 2) training with 
# original dataset only and minimal augmentation for true baseline performance.
#
# Author: Burak Kağan Yılmazer
# Date: January 2025
# Environment: mobilenet_ssd_env

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
    Write-Host "MobileNet-SSD Baseline (Phase 2) Training Script" -ForegroundColor Green
    Write-Host "================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "METHODOLOGY:" -ForegroundColor Yellow
    Write-Host "  Phase 2: Original Dataset Only with Minimal Augmentation"
    Write-Host "  True baseline performance for comparison with Phase 3"
    Write-Host ""
    Write-Host "USAGE:" -ForegroundColor Yellow
    Write-Host "  .\run_mobilenet_ssd_baseline.ps1 [-Epochs 50] [-QuickTest] [-Help]"
    Write-Host ""
    Write-Host "PARAMETERS:" -ForegroundColor Yellow
    Write-Host "  -Epochs      Number of training epochs (default: 50)"
    Write-Host "  -QuickTest   Run quick validation (10 epochs, 100 samples)"
    Write-Host "  -Help        Show this help message"
    Write-Host ""
    Write-Host "EXAMPLES:" -ForegroundColor Yellow
    Write-Host "  .\run_mobilenet_ssd_baseline.ps1                    # Standard 50-epoch baseline"
    Write-Host "  .\run_mobilenet_ssd_baseline.ps1 -Epochs 100        # Extended 100-epoch training"
    Write-Host "  .\run_mobilenet_ssd_baseline.ps1 -QuickTest         # Quick 10-epoch validation"
    Write-Host ""
    Write-Host "PHASE 2 FEATURES:" -ForegroundColor Yellow
    Write-Host "  Original dataset training with minimal augmentation:"
    Write-Host "  • NO synthetic environmental augmentation"
    Write-Host "  • NO enhanced standard augmentation"
    Write-Host "  • Resize and normalize only"
    Write-Host "  • TensorFlow MobileNet-SSD implementation"
    Write-Host "  • VOC format annotation support"
    Write-Host ""
    Write-Host "EXPECTED PERFORMANCE:" -ForegroundColor Yellow
    Write-Host "  Target: >18% mAP@0.5 (protocol requirement)"
    Write-Host "  Methodology: Establish true baseline for Phase 3 comparison"
    Write-Host "  Thesis value: Phase 2 vs Phase 3 comparative analysis"
    exit 0
}

Write-Host "================================================================" -ForegroundColor Green
Write-Host "MobileNet-SSD Baseline (Phase 2) Training - VisDrone Dataset" -ForegroundColor Green
Write-Host "METHODOLOGY: Original Dataset Only, Minimal Augmentation" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""

# Training configuration summary
Write-Info "[CONFIGURATION] Training Setup:"
Write-Host "  • Model: MobileNet-SSD (TensorFlow-based)"
Write-Host "  • Phase: 2 (Baseline - Original Dataset Only)"
Write-Host "  • Dataset: VisDrone (10 classes, VOC format)"
Write-Host "  • Epochs: $Epochs"
if ($QuickTest) {
    Write-Host "  • Mode: Quick Test (10 epochs, 100 samples)" -ForegroundColor Yellow
} else {
    Write-Host "  • Mode: Full Baseline Training"
}
Write-Host "  • Target: >18% mAP@0.5 (thesis requirement)"
Write-Host "  • GPU: NVIDIA RTX 3060 Laptop (5GB)"
Write-Host "  • Environment: mobilenet_ssd_env"
Write-Host ""

Write-Info "[METHODOLOGY] Phase 2 Key Features:"
Write-Host "  • Original Dataset Only: No synthetic environmental data"
Write-Host "  • Minimal Augmentation: Resize (300x300) and normalize only"
Write-Host "  • TensorFlow Implementation: MobileNetV2 + SSD detection heads"
Write-Host "  • VOC Format Support: Existing converted VisDrone annotations"
Write-Host "  • Batch Size: 16 (optimized for stable gradients)"
Write-Host "  • Architecture: Lightweight edge-ready model"
Write-Host ""

Write-Info "[THESIS OBJECTIVES] Expected Outcomes:"
Write-Host "  • Target mAP@0.5: >18% (minimum protocol requirement)"
Write-Host "  • Model Size: <10MB (edge deployment ready)"
Write-Host "  • Inference Speed: >15 FPS (real-time capability)"
Write-Host "  • Methodology: True baseline for Phase 3 comparison"
Write-Host "  • Research Value: Quantify synthetic augmentation benefits"
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
$venvPath = ".\venvs\mobilenet_ssd_env"
if (-not (Test-Path $venvPath)) {
    Write-Error "[ERROR] Virtual environment not found: $venvPath"
    Write-Host "Please create the environment first or use existing MobileNet-SSD environment"
    exit 1
}

Write-Info "[ACTIVATION] Activating MobileNet-SSD environment..."

# Activate virtual environment
try {
    & ".\venvs\mobilenet_ssd_env\Scripts\Activate.ps1"
    Write-Success "[SUCCESS] Virtual environment activated"
} catch {
    Write-Error "[ERROR] Failed to activate virtual environment: $($_.Exception.Message)"
    exit 1
}

Write-Host ""
Write-Info "[VALIDATION] Environment Information:"
Write-Host "  • Python: $(python --version 2>$null)"
Write-Host "  • Location: $(Get-Location)"

# Validate TensorFlow installation and dependencies
Write-Info "[VALIDATION] Checking TensorFlow installation..."
try {
    $tfCheck = python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0); print('Keras:', tf.keras.__version__); print('TensorFlow ready!')" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "[READY] TensorFlow installation validated"
        Write-Host $tfCheck
    } else {
        Write-Error "[ERROR] TensorFlow validation failed"
        exit 1
    }
} catch {
    Write-Error "[ERROR] Failed to validate TensorFlow: $($_.Exception.Message)"
    exit 1
}

# Validate dataset and VOC format
Write-Info "[VALIDATION] Checking dataset and VOC annotations..."
$datasetPath = ".\data\my_dataset\visdrone\mobilenet-ssd"
$vocPath = "$datasetPath\voc_format\train"
$imagesPath = "$datasetPath\images"

if (-not (Test-Path $datasetPath)) {
    Write-Error "[ERROR] MobileNet-SSD dataset not found: $datasetPath"
    exit 1
}

if (-not (Test-Path $vocPath)) {
    Write-Error "[ERROR] VOC annotations not found: $vocPath"
    exit 1
}

if (-not (Test-Path $imagesPath)) {
    Write-Error "[ERROR] Images directory not found: $imagesPath"
    exit 1
}

# Count annotations and images
$xmlCount = (Get-ChildItem "$vocPath\*.xml").Count
Write-Success "[READY] Dataset validated:"
Write-Host "  • Dataset path: $datasetPath"
Write-Host "  • XML annotations: $xmlCount files"
Write-Host "  • VOC format: Properly structured"
Write-Host "  • Images directory: Available"

Write-Host ""
Write-Info "[TRAINING] Starting MobileNet-SSD Baseline (Phase 2) training..."
Write-Host "Training script: src\scripts\visdrone\MobileNet-SSD\baseline\train_mobilenet_ssd_baseline.py"
if ($QuickTest) {
    Write-Host "Expected duration: 15-30 minutes (quick test - 10 epochs)"
} else {
    Write-Host "Expected duration: 1-2 hours (50 epochs)"
}
Write-Host ""

# Prepare Python arguments
$pythonArgs = @(
    "src\scripts\visdrone\MobileNet-SSD\baseline\train_mobilenet_ssd_baseline.py",
    "--epochs", $Epochs.ToString()
)

if ($QuickTest) {
    $pythonArgs += "--quick-test"
}

# Execute training
Write-Info "[EXECUTION] Running Baseline (Phase 2) optimized training..."
Write-Host "Command: python $($pythonArgs -join ' ')"
Write-Host ""

try {
    $process = Start-Process -FilePath "python" -ArgumentList $pythonArgs -NoNewWindow -PassThru -Wait
    
    if ($process.ExitCode -eq 0) {
        Write-Host ""
        Write-Success "[SUCCESS] MobileNet-SSD Baseline (Phase 2) training completed successfully!"
        Write-Host ""
        Write-Info "[RESULTS] Training Summary:"
        Write-Host "  • Model: MobileNet-SSD Baseline (Phase 2)"
        Write-Host "  • Epochs: $Epochs"
        Write-Host "  • Dataset: VisDrone (original only)"
        Write-Host "  • Results: runs\train\mobilenet_ssd_baseline_*"
        Write-Host ""
        Write-Info "[METHODOLOGY COMPLIANCE] Phase Analysis:"
        Write-Host "  • Phase 2 Baseline: TRUE baseline established"
        Write-Host "  • Target Performance: >18% mAP@0.5"
        Write-Host "  • Next Phase: Phase 3 synthetic augmentation training"
        Write-Host "  • Comparison: Baseline vs augmented performance analysis"
        Write-Host ""
        Write-Info "[NEXT STEPS] After Baseline completion:"
        Write-Host "  1. Evaluate baseline performance metrics"
        Write-Host "  2. Execute Phase 3 synthetic augmentation training"
        Write-Host "  3. Compare Phase 2 vs Phase 3 performance"
        Write-Host "  4. Generate comprehensive analysis for thesis"
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
Write-Success "[COMPLETED] MobileNet-SSD Baseline (Phase 2) Training Session Finished"
Write-Host "================================================================" -ForegroundColor Green