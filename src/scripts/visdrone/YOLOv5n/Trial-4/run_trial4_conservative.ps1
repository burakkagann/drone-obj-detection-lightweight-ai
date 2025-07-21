# YOLOv5n Trial-4 Conservative Training Script
# Based on proven Trial-2 results (23.557% mAP@0.5)
# Conservative optimization strategy after Trial-3 failure analysis
# Target: 24-25% mAP@0.5 with minimal risk

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
    Write-Host "YOLOv5n Trial-4 Conservative Training" -ForegroundColor Green
    Write-Host "====================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "USAGE:" -ForegroundColor Yellow
    Write-Host "  .\run_trial4_conservative.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "OPTIONS:" -ForegroundColor Yellow
    Write-Host "  -Epochs <number>     Number of training epochs (default: 20 for validation)"
    Write-Host "  -FullTraining        Run full 100-epoch training (only after 20-epoch validation)"
    Write-Host "  -Help               Show this help message"
    Write-Host ""
    Write-Host "TRAINING PROTOCOL:" -ForegroundColor Cyan
    Write-Host "  Phase 1: 20-epoch validation (default)"
    Write-Host "  Phase 2: 100-epoch full training (only if validation successful)"
    Write-Host ""
    Write-Host "BASELINE:" -ForegroundColor Cyan
    Write-Host "  Trial-2 mAP@0.5: 23.557% (proven baseline)"
    Write-Host "  Trial-3 mAP@0.5: 0.002% (catastrophic failure)"
    Write-Host "  Trial-4 Target: 24-25% mAP@0.5 (conservative improvement)"
    exit 0
}

# Show help if requested
if ($Help) {
    Show-Help
}

# Determine training configuration
if ($FullTraining) {
    $Epochs = 100
    $TrainingType = "Full Training"
    $RunName = "yolov5n_trial4_100epochs"
} else {
    $Epochs = 20
    $TrainingType = "Validation"
    $RunName = "yolov5n_trial4_validation"
}

# Header
Write-Host "================================================================" -ForegroundColor Green
Write-Host "YOLOv5n Trial-4 Conservative Training" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""

# Configuration
Write-Info "[CONFIG] Trial-4 Configuration:"
Write-Host "  • Training Type: $TrainingType"
Write-Host "  • Epochs: $Epochs"
Write-Host "  • Batch Size: 18 (modest increase from Trial-2's 16)"
Write-Host "  • Image Size: 640 (same as Trial-2)"
Write-Host "  • Object Loss: 1.25 (slight increase from Trial-2's 1.2)"
Write-Host "  • Focal Loss: DISABLED (critical lesson from Trial-3)"
Write-Host ""

Write-Info "[BASELINE] Performance Context:"
Write-Host "  • Trial-2 Baseline: 23.557% mAP@0.5 ✅"
Write-Host "  • Trial-3 Failure: 0.002% mAP@0.5 ❌ (focal loss activation)"
Write-Host "  • Trial-4 Target: 24-25% mAP@0.5 🎯"
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

# Check if hyperparameter file exists
$hypFile = "..\..\..\..\..\config\visdrone\yolov5n_v1\hyp_visdrone_trial4.yaml"
if (Test-Path $hypFile) {
    Write-Success "[CONFIG] Trial-4 hyperparameter file found"
} else {
    Write-Error "[ERROR] Trial-4 hyperparameter file not found: $hypFile"
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
Write-Info "[TRAINING] Starting Trial-4 conservative optimization..."
Write-Host ""

# Navigate to YOLOv5 directory and execute training directly
$yolov5Path = "..\..\..\..\..\src\models\YOLOv5"
if (Test-Path $yolov5Path) {
    Write-Success "[PATH] YOLOv5 directory found"
    Push-Location $yolov5Path
    
    try {
        # Prepare training command
        # Use relative paths like working Trial-3 script
        $command = @(
            "train.py",
            "--data", "..\..\..\config\visdrone\yolov5n_v1\yolov5n_visdrone_config.yaml",
            "--cfg", "models\yolov5n.yaml",
            "--hyp", "..\..\..\config\visdrone\yolov5n_v1\hyp_visdrone_trial4.yaml",
            "--epochs", $Epochs.ToString(),
            "--batch-size", "18",
            "--img-size", "640",
            "--device", "0",
            "--workers", "4",
            "--name", $RunName,
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
            Write-Success "[SUCCESS] Trial-4 $TrainingType completed successfully!"
            Write-Host ""
            
            if ($FullTraining) {
                Write-Info "[RESULTS] Full Training Completed:"
                Write-Host "  • Check final mAP@0.5 in results.csv"
                Write-Host "  • Compare against Trial-2 baseline (23.557%)"
                Write-Host "  • Target achieved if >24% mAP@0.5"
                Write-Host ""
            } else {
                Write-Info "[VALIDATION] 20-Epoch Validation Results:"
                Write-Host "  • Check runs/train/$RunName/results.csv"
                Write-Host "  • Compare final mAP@0.5 against Trial-2 baseline"
                Write-Host "  • If improvement shown, proceed with full training:"
                Write-Host "    .\run_trial4_conservative.ps1 -FullTraining"
                Write-Host ""
            }
            
            Write-Info "[NEXT STEPS]:"
            Write-Host "  • Analyze results in runs/train/$RunName/"
            Write-Host "  • Document performance in Trial-4 folder"
            Write-Host "  • Compare against Trial-2 baseline (23.557%)"
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
Write-Host "Trial-4 Conservative Training Complete" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green