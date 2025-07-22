# YOLOv5n TRUE BASELINE Training Script
# Purpose: Establish raw YOLOv5n performance using default hyperparameters
# Expected Performance: 15-18% mAP@0.5 (unoptimized)
# Scientific Value: Shows optimization impact potential

param(
    [int]$Epochs = 20,
    [switch]$FullTraining,
    [switch]$Help
)

# Color functions
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
    Write-Host "YOLOv5n TRUE BASELINE Training (Raw Performance)" -ForegroundColor Green
    Write-Host "===============================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "PURPOSE:" -ForegroundColor Yellow
    Write-Host "  Establish raw YOLOv5n performance using default hyperparameters"
    Write-Host "  Shows unoptimized model capability for measuring optimization impact"
    Write-Host ""
    Write-Host "USAGE:" -ForegroundColor Yellow
    Write-Host "  .\run_true_baseline.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "OPTIONS:" -ForegroundColor Yellow
    Write-Host "  -Epochs <number>     Number of training epochs (default: 20 for validation)"
    Write-Host "  -FullTraining        Run full 100-epoch training (after 20-epoch validation)"
    Write-Host "  -Help               Show this help message"
    Write-Host ""
    Write-Host "TRAINING PROTOCOL:" -ForegroundColor Cyan
    Write-Host "  Phase 1: 20-epoch validation (default)"
    Write-Host "  Phase 2: 100-epoch full training (only if validation successful)"
    Write-Host ""
    Write-Host "PERFORMANCE CONTEXT:" -ForegroundColor Cyan
    Write-Host "  True Baseline: 15-18% mAP@0.5 (this script - YOLOv5 defaults)"
    Write-Host "  Optimized Baseline: 23-25% mAP@0.5 (Trial-2 optimized)"
    Write-Host "  Optimization Impact: +8-10% mAP@0.5 potential improvement"
    Write-Host ""
    Write-Host "HYPERPARAMETER DIFFERENCES:" -ForegroundColor Cyan
    Write-Host "  Batch Size: 8 (vs optimized: 16)"
    Write-Host "  Image Size: 416px (vs optimized: 640px)"
    Write-Host "  Learning Rate: 0.01 (vs optimized: 0.005)"
    Write-Host "  Mixup: DISABLED (vs optimized: 0.4)"
    Write-Host "  Multi-scale: DISABLED (vs optimized: ENABLED)"
    Write-Host "  Object Loss: 1.0 (vs optimized: 1.2)"
    exit 0
}

# Show help if requested
if ($Help) {
    Show-Help
}

# Determine training configuration
if ($FullTraining) {
    $Epochs = 100
    $TrainingType = "Full TRUE Baseline Training"
    $RunName = "yolov5n_true_baseline_100epochs"
} else {
    $Epochs = 20
    $TrainingType = "TRUE Baseline Validation"
    $RunName = "yolov5n_true_baseline_validation"
}

# Header
Write-Host "================================================================" -ForegroundColor Green
Write-Host "YOLOv5n TRUE BASELINE Training (Raw Performance)" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""

# Configuration
Write-Info "[CONFIG] TRUE Baseline Configuration:"
Write-Host "  • Training Type: $TrainingType"
Write-Host "  • Epochs: $Epochs"
Write-Host "  • Hyperparameters: YOLOv5 DEFAULTS (unoptimized)"
Write-Host "  • Batch Size: 8 (YOLOv5 default)"
Write-Host "  • Image Size: 416px (YOLOv5 default)"
Write-Host "  • Learning Rate: 0.01 (YOLOv5 default)"
Write-Host "  • Multi-scale: DISABLED (YOLOv5 default)"
Write-Host "  • Mixup: DISABLED (YOLOv5 default)"
Write-Host ""

Write-Info "[SCIENTIFIC CONTEXT] Performance Expectations:"
Write-Host "  • TRUE Baseline (this): 15-18% mAP@0.5"
Write-Host "  • Optimized Baseline: 23-25% mAP@0.5"
Write-Host "  • Optimization Potential: +8-10% mAP@0.5"
Write-Host "  • Purpose: Show raw model capability"
Write-Host ""

# Environment validation
Write-Info "[VALIDATION] Checking environment..."

# Check if virtual environment is activated
if (-not $env:VIRTUAL_ENV) {
    Write-Warning "[WARNING] Virtual environment not detected"
    Write-Host "Please activate yolov5n_env first:"
    Write-Host "  .\venvs\yolov5n_env\Scripts\Activate.ps1"
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

# Check if true baseline hyperparameter file exists
$hypFile = "..\..\..\..\..\config\visdrone\yolov5n_v1\hyp_visdrone_true_baseline.yaml"
if (Test-Path $hypFile) {
    Write-Success "[CONFIG] TRUE Baseline hyperparameter file found"
} else {
    Write-Warning "[INFO] TRUE Baseline hyperparameter file will be created automatically"
}

# Check if dataset config exists
$dataFile = "..\..\..\..\..\config\visdrone\yolov5n_v1\baseline_dataset_config.yaml"
if (Test-Path $dataFile) {
    Write-Success "[CONFIG] Baseline dataset config file found"
} else {
    Write-Error "[ERROR] Baseline dataset config file not found: $dataFile"
    exit 1
}

Write-Host ""

# Training execution
Write-Info "[TRAINING] Starting TRUE Baseline training with YOLOv5 defaults..."
Write-Host ""

# Navigate to YOLOv5 directory and execute training directly
$yolov5Path = "..\..\..\..\..\src\models\YOLOv5"
if (Test-Path $yolov5Path) {
    Write-Success "[PATH] YOLOv5 directory found"
    Push-Location $yolov5Path
    
    try {
        # Prepare training command
        # Use robust Python wrapper (proven architecture)
        $pythonScript = "`"C:\Users\burak\OneDrive\Desktop\Git Repos\drone-obj-detection-lightweight-ai\src\scripts\visdrone\YOLOv5n\Baseline\train_true_baseline_yolov5n.py`""
        $pythonArgs = @($pythonScript)
        
        # Add epoch parameter
        $pythonArgs += "--epochs", $Epochs.ToString()
        
        # Add quick test flag if validation mode
        if (-not $FullTraining) {
            $pythonArgs += "--quick-test"
        }
        
        Write-Host "Executing: python $($pythonArgs -join ' ')" -ForegroundColor Cyan
        Write-Host ""
        
        $process = Start-Process -FilePath "python" -ArgumentList $pythonArgs -NoNewWindow -PassThru -Wait
        
        if ($process.ExitCode -eq 0) {
            Write-Host ""
            Write-Success "[SUCCESS] TRUE Baseline $TrainingType completed successfully!"
            Write-Host ""
            
            if ($FullTraining) {
                Write-Info "[RESULTS] TRUE Baseline Full Training Completed:"
                Write-Host "  • Check final mAP@0.5 in results.csv"
                Write-Host "  • Expected Range: 15-18% mAP@0.5"
                Write-Host "  • Compare with optimized baseline later"
                Write-Host ""
                
                Write-Info "[NEXT STEPS] Optimization Impact Analysis:"
                Write-Host "  • Run optimized baseline (Trial-2 hyperparameters)"
                Write-Host "  • Calculate optimization improvement"
                Write-Host "  • Run environmental augmentation experiments"
                Write-Host ""
            } else {
                Write-Info "[VALIDATION] TRUE Baseline 20-Epoch Validation Results:"
                Write-Host "  • Check runs/train/$RunName/results.csv"
                Write-Host "  • Expected Range: 15-18% mAP@0.5"
                Write-Host "  • If performance acceptable, proceed with full training:"
                Write-Host "    .\run_true_baseline.ps1 -FullTraining"
                Write-Host ""
                
                Write-Info "[SCIENTIFIC VALUE] This Establishes:"
                Write-Host "  • Raw YOLOv5n capability without optimization"
                Write-Host "  • Baseline for measuring optimization impact"
                Write-Host "  • Complete context for thesis analysis"
                Write-Host ""
            }
            
        } else {
            Write-Error "[ERROR] TRUE Baseline training failed with exit code: $($process.ExitCode)"
            exit 1
        }
        
    } catch {
        Write-Error "[ERROR] Failed to execute TRUE baseline training: $($_.Exception.Message)"
        exit 1
    } finally {
        Pop-Location
    }
    
} else {
    Write-Error "[ERROR] YOLOv5 directory not found: $yolov5Path"
    exit 1
}

Write-Host "================================================================" -ForegroundColor Green
Write-Host "TRUE Baseline Training Complete" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green