#!/usr/bin/env pwsh
<#
.SYNOPSIS
    YOLOv8n Trial-2 Training Execution Script
    
.DESCRIPTION
    This script executes YOLOv8n Trial-2 training with Enhanced Small Object Detection strategy.
    Implements higher resolution (832px), optimized augmentation, and refined hyperparameters
    for improved small object detection performance, targeting 30-32% mAP@0.5.
    
    Trial-2 Strategy Focus:
    - Enhanced Small Object Detection (832px resolution)
    - Optimized augmentation (mosaic 0.9, copy-paste 0.4)
    - Refined hyperparameters (lr0: 0.003, extended warmup)
    - Multi-scale training for scale invariance
    
.PARAMETER Epochs
    Number of training epochs (default: 40)
    
.PARAMETER QuickTest
    Run quick test with reduced settings (15 epochs, 640px)
    
.EXAMPLE
    .\run_yolov8n_trial2.ps1
    .\run_yolov8n_trial2.ps1 -Epochs 60
    .\run_yolov8n_trial2.ps1 -QuickTest
    
.NOTES
    Master's Thesis: Robust Object Detection for Surveillance Drones
    Author: Burak Kağan Yılmazer
    Date: January 2025
    Environment: yolov8n-visdrone_venv
#>

param(
    [Parameter(Mandatory=$false)]
    [int]$Epochs = 50,
    
    [Parameter(Mandatory=$false)]
    [switch]$QuickTest
)

# Set strict error handling
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Color functions for output
function Write-Info { param($Message) Write-Host "[INFO] $Message" -ForegroundColor Cyan }
function Write-Success { param($Message) Write-Host "[SUCCESS] $Message" -ForegroundColor Green }
function Write-Warning { param($Message) Write-Host "[WARNING] $Message" -ForegroundColor Yellow }
function Write-Error { param($Message) Write-Host "[ERROR] $Message" -ForegroundColor Red }

Write-Host "=" * 80 -ForegroundColor Blue
Write-Host "YOLOv8n Trial-2 Training - Enhanced Small Object Detection" -ForegroundColor White
Write-Host "=" * 80 -ForegroundColor Blue
Write-Info "Strategy: Enhanced Small Object Detection (832px, optimized augmentation)"
Write-Info "Target: 30-32% mAP@0.5 (+1-3% over Trial-1)"
Write-Info "Expected: Improved pedestrian class performance"
Write-Info "Epochs: $Epochs"
if ($QuickTest) { Write-Info "Mode: Quick Test (15 epochs, reduced settings)" }
Write-Info "Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "=" * 80 -ForegroundColor Blue

try {
    # Step 1: Validate Repository Structure and Navigate to Root
    Write-Info "Validating repository structure..."
    
    $RepoRoot = "C:\Users\burak\OneDrive\Desktop\Git Repos\drone-obj-detection-lightweight-ai"
    
    if (-not (Test-Path $RepoRoot)) {
        throw "Repository not found: $RepoRoot"
    }
    
    # Navigate to repository root (CRITICAL for relative path resolution)
    $currentLocation = (Get-Location).Path
    if ($currentLocation -ne $RepoRoot) {
        Write-Info "Changing to repository root..."
        Set-Location $RepoRoot
        Write-Success "Changed to repository root: $RepoRoot"
    }
    
    # Validate training script using relative path
    $TrainingScript = "src\scripts\visdrone\YOLOv8n\trial-2\train_yolov8n_trial2.py"
    if (-not (Test-Path $TrainingScript)) {
        throw "Training script not found: $TrainingScript"
    }
    
    Write-Success "Repository structure validated"
    
    # Step 2: Virtual Environment Activation
    Write-Info "Activating YOLOv8n virtual environment..."
    
    # Check if already in virtual environment
    $CurrentEnv = $env:VIRTUAL_ENV
    if ($CurrentEnv -and $CurrentEnv.Contains("yolov8n")) {
        Write-Success "YOLOv8n environment already active: $CurrentEnv"
    } else {
        # Try multiple possible environment paths
        $EnvPaths = @(
            ".\venvs\yolov8n-visdrone_venv\Scripts\Activate.ps1",
            ".\venvs\visdrone\yolov8n_visdrone_env\Scripts\Activate.ps1",
            ".\venvs\yolov8n_env\Scripts\Activate.ps1"
        )
        
        $EnvActivated = $false
        foreach ($EnvPath in $EnvPaths) {
            $FullEnvPath = Join-Path $RepoRoot $EnvPath
            if (Test-Path $FullEnvPath) {
                Write-Info "Activating environment: $FullEnvPath"
                & $FullEnvPath
                if ($LASTEXITCODE -eq 0) {
                    $EnvActivated = $true
                    Write-Success "Virtual environment activated successfully"
                    break
                }
            }
        }
        
        if (-not $EnvActivated) {
            Write-Warning "No YOLOv8n environment found, attempting to use current environment"
        }
    }
    
    # Step 3: Environment Validation
    Write-Info "Validating Python environment..."
    
    # Validate Python and CUDA
    python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) {
        throw "PyTorch validation failed"
    }
    
    # Validate Ultralytics
    python -c "from ultralytics import YOLO; print('Ultralytics YOLO imported successfully')"
    if ($LASTEXITCODE -ne 0) {
        throw "Ultralytics validation failed"
    }
    
    Write-Success "Python environment validated"
    
    # Step 4: Dataset Validation
    Write-Info "Validating VisDrone dataset..."
    
    $DatasetPath = "data\my_dataset\visdrone"
    $ConfigPath = "config\visdrone\yolov8n_v1\yolov8n_visdrone_config.yaml"
    
    if (-not (Test-Path $DatasetPath)) {
        throw "Dataset not found: $DatasetPath"
    }
    
    if (-not (Test-Path $ConfigPath)) {
        throw "Dataset config not found: $ConfigPath"
    }
    
    # Check dataset subsets
    $Subsets = @("train", "val", "test")
    foreach ($Subset in $Subsets) {
        $SubsetPath = "$DatasetPath\$Subset\images"
        if (-not (Test-Path $SubsetPath)) {
            throw "Dataset subset not found: $SubsetPath"
        }
    }
    
    Write-Success "VisDrone dataset validated"
    
    # Step 5: Training Execution
    Write-Info "Starting YOLOv8n Trial-2 training..."
    Write-Info "Strategy: Enhanced Small Object Detection"
    Write-Info "Key optimizations: 832px resolution, mosaic 0.9, lr0 0.003"
    Write-Host ""
    
    # Build Python command arguments using relative path
    $PythonArgs = @(
        $TrainingScript,
        "--epochs", $Epochs.ToString()
    )
    
    if ($QuickTest) {
        $PythonArgs += "--quick-test"
        Write-Info "Quick test mode enabled"
    }
    
    # Execute training with proper error handling
    Write-Info "Executing: python $($PythonArgs -join ' ')"
    Write-Host ""
    
    $process = Start-Process -FilePath "python" -ArgumentList $PythonArgs -NoNewWindow -PassThru -Wait
    
    if ($process.ExitCode -eq 0) {
        Write-Host ""
        Write-Host "=" * 80 -ForegroundColor Green
        Write-Success "YOLOv8n Trial-2 Training Completed Successfully!"
        Write-Success "Strategy: Enhanced Small Object Detection executed"
        Write-Success "Expected improvements: +1-3% mAP@0.5 over Trial-1"
        Write-Success "Key focus: Improved small object detection performance"
        Write-Host "=" * 80 -ForegroundColor Green
        
        # Step 6: Results Location Information
        Write-Host ""
        Write-Info "Training results saved in: runs/train/yolov8n_trial2_*"
        Write-Info "Key files to check:"
        Write-Info "  • weights/best.pt - Best model weights"
        Write-Info "  • results.csv - Training metrics"
        Write-Info "  • evaluation/ - Comprehensive evaluation results"
        Write-Info "  • yolov8n_trial2_hyperparameters.yaml - Trial-2 configuration"
        
        # Step 7: Next Steps Guidance
        Write-Host ""
        Write-Info "Next steps for thesis completion:"
        Write-Info "  1. Analyze Trial-2 results vs Trial-1 comparison"
        Write-Info "  2. Create comprehensive Trial-2 documentation"
        Write-Info "  3. Validate 30-32% mAP@0.5 target achievement"
        Write-Info "  4. Proceed with MobileNet-SSD and NanoDet frameworks"
        
    } else {
        throw "Training process failed with exit code: $($process.ExitCode)"
    }
    
} catch {
    Write-Host ""
    Write-Host "=" * 80 -ForegroundColor Red
    Write-Error "YOLOv8n Trial-2 Training Failed!"
    Write-Error "Error: $($_.Exception.Message)"
    Write-Host "=" * 80 -ForegroundColor Red
    
    Write-Host ""
    Write-Info "Troubleshooting steps:"
    Write-Info "  1. Verify virtual environment activation"
    Write-Info "  2. Check CUDA availability: python -c 'import torch; print(torch.cuda.is_available())'"
    Write-Info "  3. Validate dataset paths in config/visdrone/yolov8n_v1/"
    Write-Info "  4. Check disk space for training outputs"
    Write-Info "  5. Review error logs above for specific issues"
    
    exit 1
} finally {
    # Script cleanup completed
    Write-Info "Script execution completed"
}

Write-Host ""
Write-Info "YOLOv8n Trial-2 execution script completed"
Write-Info "Check training logs for detailed progress and results"