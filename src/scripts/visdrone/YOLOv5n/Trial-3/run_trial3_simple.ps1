# YOLOv5n Trial-3 Training Script
# Based on successful Trial-2 results (23.557% mAP@0.5)
# Target: Push beyond 25% mAP@0.5 threshold for thesis excellence

param(
    [int]$Epochs = 100,
    [switch]$QuickTest,
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
    Write-Host "YOLOv5n Trial-3 Optimization Training" -ForegroundColor Green
    Write-Host "=====================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "USAGE:" -ForegroundColor Yellow
    Write-Host "  .\run_trial3_simple.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "OPTIONS:" -ForegroundColor Yellow
    Write-Host "  -Epochs <number>     Number of training epochs (default: 100)"
    Write-Host "  -QuickTest          Run 20-epoch validation test first"
    Write-Host "  -Help               Show this help message"
    Write-Host ""
    Write-Host "BASELINE:" -ForegroundColor Cyan
    Write-Host "  Trial-2 mAP@0.5: 23.557%"
    Write-Host "  Target mAP@0.5: 25%+ (thesis excellence)"
    exit 0
}

# Show help if requested
if ($Help) {
    Show-Help
}

# Header
Write-Host "================================================================" -ForegroundColor Green
Write-Host "YOLOv5n Trial-3 Optimization Training" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""

# Configuration
Write-Info "[CONFIG] Training Configuration:"
Write-Host "  • Epochs: $Epochs"
Write-Host "  • Quick Test: $QuickTest"
Write-Host "  • Batch Size: 20 (enhanced)"
Write-Host "  • Image Size: 640 (high resolution)"
Write-Host "  • Baseline: 23.557% mAP@0.5 (Trial-2)"
Write-Host "  • Target: 25%+ mAP@0.5"
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
$hypFile = "..\..\..\..\..\config\visdrone\yolov5n_v1\hyp_visdrone_trial3.yaml"
if (Test-Path $hypFile) {
    Write-Success "[CONFIG] Trial-3 hyperparameter file found"
} else {
    Write-Error "[ERROR] Trial-3 hyperparameter file not found: $hypFile"
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
Write-Info "[TRAINING] Starting Trial-3 optimization..."
Write-Host ""

# Navigate to YOLOv5 directory and execute training directly
$yolov5Path = "..\..\..\..\..\src\models\YOLOv5"
if (Test-Path $yolov5Path) {
    Write-Success "[PATH] YOLOv5 directory found"
    Push-Location $yolov5Path
    
    try {
        # Prepare training command
        $runName = if ($QuickTest) { "yolov5n_trial3_quicktest" } else { "yolov5n_trial3_100epochs" }
        
        $command = @(
            "train.py",
            "--data", "..\..\..\config\visdrone\yolov5n_v1\yolov5n_visdrone_config.yaml",
            "--cfg", "models\yolov5n.yaml",
            "--hyp", "..\..\..\config\visdrone\yolov5n_v1\hyp_visdrone_trial3.yaml",
            "--epochs", $Epochs.ToString(),
            "--batch-size", "20",
            "--img-size", "640",
            "--device", "0",
            "--workers", "4",
            "--name", $runName,
            "--save-period", "5",
            "--patience", "15",
            "--project", "..\..\..\runs\train",
            "--exist-ok"
        )
        
        Write-Host "Executing: python $($command -join ' ')" -ForegroundColor Cyan
        Write-Host ""
        
        $process = Start-Process -FilePath "python" -ArgumentList $command -NoNewWindow -PassThru -Wait
        
        if ($process.ExitCode -eq 0) {
            Write-Host ""
            Write-Success "[SUCCESS] Trial-3 training completed successfully!"
            Write-Host ""
            
            Write-Info "[RESULTS] Performance Targets:"
            Write-Host "  • Baseline: 23.557% mAP@0.5 (Trial-2)"
            Write-Host "  • Target: >25% mAP@0.5 (thesis excellence)"
            Write-Host "  • Outstanding: >26% mAP@0.5"
            Write-Host ""
            
            Write-Info "[NEXT STEPS] Check results:"
            Write-Host "  • Check runs/train/$runName/results.csv"
            Write-Host "  • Compare final mAP@0.5 against 23.557% baseline"
            Write-Host "  • Document results in Trial-3 folder"
            Write-Host ""
            
        } else {
            Write-Error "[ERROR] Training failed with exit code: $($process.ExitCode)"
            exit 1
        }
        
    } catch {
        Write-Error "[ERROR] Failed to execute training: $($_.Exception.Message)"
        exit 1
    } finally {
        Pop-Location
    }
    
} else {
    Write-Error "[ERROR] YOLOv5 directory not found: $yolov5Path"
    exit 1
}

Write-Host "================================================================" -ForegroundColor Green
Write-Host "Trial-3 Optimization Complete" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green