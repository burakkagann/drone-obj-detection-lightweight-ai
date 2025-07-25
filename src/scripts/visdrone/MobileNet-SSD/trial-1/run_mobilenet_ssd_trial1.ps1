# MobileNet-SSD Trial-1 (Phase 3) Training Script - VisDrone Dataset
# Master's Thesis: Robust Object Detection for Surveillance Drones
# 
# This PowerShell script executes MobileNet-SSD Trial-1 (Phase 3) training with 
# synthetic environmental augmentation for enhanced robustness testing.
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
    Write-Host "MobileNet-SSD Trial-1 (Phase 3) Training Script" -ForegroundColor Green
    Write-Host "===============================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "METHODOLOGY:" -ForegroundColor Yellow
    Write-Host "  Phase 3: Synthetic Environmental Augmentation"
    Write-Host "  Compare against Phase 2 baseline performance"
    Write-Host ""
    Write-Host "USAGE:" -ForegroundColor Yellow
    Write-Host "  .\run_mobilenet_ssd_trial1.ps1 [-Epochs 50] [-QuickTest] [-Help]"
    Write-Host ""
    Write-Host "PARAMETERS:" -ForegroundColor Yellow
    Write-Host "  -Epochs      Number of training epochs (default: 50)"
    Write-Host "  -QuickTest   Run quick validation (10 epochs, 100 samples)"
    Write-Host "  -Help        Show this help message"
    Write-Host ""
    Write-Host "EXAMPLES:" -ForegroundColor Yellow
    Write-Host "  .\run_mobilenet_ssd_trial1.ps1                    # Standard 50-epoch Trial-1"
    Write-Host "  .\run_mobilenet_ssd_trial1.ps1 -Epochs 75         # Extended 75-epoch training"
    Write-Host "  .\run_mobilenet_ssd_trial1.ps1 -QuickTest         # Quick 10-epoch validation"
    Write-Host ""
    Write-Host "PHASE 3 FEATURES:" -ForegroundColor Yellow
    Write-Host "  Enhanced synthetic environmental augmentation:"
    Write-Host "  • Fog simulation (light, medium, heavy)"
    Write-Host "  • Night conditions (low-light with noise)"
    Write-Host "  • Motion blur (camera/object movement)"
    Write-Host "  • Rain effects (visibility reduction)"
    Write-Host "  • Enhanced standard augmentation (brightness, flip)"
    Write-Host "  • 60% environmental augmentation probability"
    Write-Host ""
    Write-Host "EXPECTED PERFORMANCE:" -ForegroundColor Yellow
    Write-Host "  Target: >20% mAP@0.5 (vs baseline performance)"
    Write-Host "  Methodology: Quantify synthetic augmentation benefits"
    Write-Host "  Thesis value: Phase 2 vs Phase 3 comparison analysis"
    exit 0
}

Write-Host "================================================================" -ForegroundColor Green
Write-Host "MobileNet-SSD Trial-1 (Phase 3) Training - VisDrone Dataset" -ForegroundColor Green
Write-Host "METHODOLOGY: Synthetic Environmental Augmentation" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""

# Training configuration summary
Write-Info "[CONFIGURATION] Training Setup:"
Write-Host "  • Model: MobileNet-SSD (Trial-1 Phase 3 optimized)"
Write-Host "  • Phase: 3 (Synthetic Environmental Augmentation)"
Write-Host "  • Dataset: VisDrone (10 classes, VOC format)"
Write-Host "  • Epochs: $Epochs"
if ($QuickTest) {
    Write-Host "  • Mode: Quick Test (10 epochs, 100 samples)" -ForegroundColor Yellow
} else {
    Write-Host "  • Mode: Full Augmented Training"
}
Write-Host "  • Baseline: Phase 2 (original dataset only)"
Write-Host "  • Target: >20% mAP@0.5 (improvement over baseline)"
Write-Host "  • GPU: NVIDIA RTX 3060 Laptop (5GB)"
Write-Host "  • Environment: mobilenet_ssd_env"
Write-Host ""

Write-Info "[METHODOLOGY] Phase 3 Key Features:"
Write-Host "  • Synthetic Environmental Augmentation: Fog, night, blur, rain"
Write-Host "  • Enhanced Standard Augmentation: Brightness, horizontal flip"
Write-Host "  • Augmentation Probability: 60% environmental, 30% brightness"
Write-Host "  • TensorFlow Implementation: MobileNetV2 + SSD detection heads"
Write-Host "  • Learning Rate: 0.0005 (reduced for augmentation stability)"
Write-Host "  • Early Stopping: 15 epochs patience (increased for robustness)"
Write-Host ""

Write-Info "[THESIS OBJECTIVES] Expected Outcomes:"
Write-Host "  • Target mAP@0.5: >20% (improvement over Phase 2 baseline)"
Write-Host "  • Robustness: Enhanced performance in low-visibility conditions"
Write-Host "  • Methodology: Quantified synthetic augmentation benefits"
Write-Host "  • Comparison: Phase 2 vs Phase 3 analysis for thesis"
Write-Host "  • Research Value: Environmental adaptation validation"
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
Write-Host "  • Augmentation: Environmental + standard ready"

Write-Host ""
Write-Info "[TRAINING] Starting MobileNet-SSD Trial-1 (Phase 3) training..."
Write-Host "Training script: src\scripts\visdrone\MobileNet-SSD\trial-1\train_mobilenet_ssd_trial1.py"
if ($QuickTest) {
    Write-Host "Expected duration: 20-35 minutes (quick test - 10 epochs)"
} else {
    Write-Host "Expected duration: 1.5-2.5 hours (50 epochs with augmentation)"
}
Write-Host ""

# Prepare Python arguments
$pythonArgs = @(
    "src\scripts\visdrone\MobileNet-SSD\trial-1\train_mobilenet_ssd_trial1.py",
    "--epochs", $Epochs.ToString()
)

if ($QuickTest) {
    $pythonArgs += "--quick-test"
}

# Execute training
Write-Info "[EXECUTION] Running Trial-1 (Phase 3) augmented training..."
Write-Host "Command: python $($pythonArgs -join ' ')"
Write-Host ""

try {
    $process = Start-Process -FilePath "python" -ArgumentList $pythonArgs -NoNewWindow -PassThru -Wait
    
    if ($process.ExitCode -eq 0) {
        Write-Host ""
        Write-Success "[SUCCESS] MobileNet-SSD Trial-1 (Phase 3) training completed successfully!"
        Write-Host ""
        Write-Info "[RESULTS] Training Summary:"
        Write-Host "  • Model: MobileNet-SSD Trial-1 (Phase 3 - Synthetic Augmentation)"
        Write-Host "  • Epochs: $Epochs"
        Write-Host "  • Dataset: VisDrone (original + synthetic environmental)"
        Write-Host "  • Results: runs\train\mobilenet_ssd_trial1_phase3_*"
        Write-Host ""
        Write-Info "[METHODOLOGY COMPLIANCE] Phase Analysis:"
        Write-Host "  • Phase 2 Baseline: Original dataset established"
        Write-Host "  • Phase 3 Target: >20% mAP@0.5 (synthetic augmentation)"
        Write-Host "  • Comparison: Quantified synthetic data benefits"
        Write-Host "  • Thesis Value: Environmental robustness demonstration"
        Write-Host ""
        Write-Info "[NEXT STEPS] After Trial-1 completion:"
        Write-Host "  1. Compare Trial-1 vs Baseline performance (Phase 2 vs Phase 3)"
        Write-Host "  2. Analyze mAP@0.5, precision, recall improvements"
        Write-Host "  3. Document synthetic augmentation impact for thesis"
        Write-Host "  4. Generate comprehensive comparison analysis"
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
Write-Success "[COMPLETED] MobileNet-SSD Trial-1 (Phase 3) Training Session Finished"
Write-Host "================================================================" -ForegroundColor Green