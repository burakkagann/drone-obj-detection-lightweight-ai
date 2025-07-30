# NanoDet Trial-1 Training Script for VisDrone
# Uses the fixed implementation with proper loss functions

param(
    [int]$Epochs = 100,
    [switch]$QuickTest
)

# Script configuration
$ErrorActionPreference = "Stop"
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Get-Item $scriptPath).Parent.Parent.Parent.Parent.Parent.FullName

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "NanoDet Trial-1 Training (Fixed)" -ForegroundColor Cyan
Write-Host "Protocol: Version 2.0 Compliant" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Project Root: $projectRoot"
Write-Host "Script Path: $scriptPath"
Write-Host "Epochs: $Epochs"
Write-Host "Quick Test: $QuickTest"
Write-Host ""

# Check if running in virtual environment
$venvPath = "$projectRoot\venvs\nanodet_env"
if (-not (Test-Path $venvPath)) {
    Write-Host "[ERROR] NanoDet virtual environment not found at: $venvPath" -ForegroundColor Red
    Write-Host "[INFO] Please create it first using:" -ForegroundColor Yellow
    Write-Host "python -m venv $venvPath" -ForegroundColor Yellow
    Write-Host "Then install requirements: pip install torch torchvision opencv-python pycocotools albumentations" -ForegroundColor Yellow
    exit 1
}

# Check if environment is activated
$pythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source
if ($pythonPath -and $pythonPath.Contains("nanodet_env")) {
    Write-Host "[INFO] NanoDet environment is active" -ForegroundColor Green
} else {
    Write-Host "[WARNING] NanoDet environment not active. Please activate it:" -ForegroundColor Yellow
    Write-Host "& '$venvPath\Scripts\Activate.ps1'" -ForegroundColor Yellow
    Write-Host ""
    
    # Try to activate it
    Write-Host "[INFO] Attempting to activate environment..." -ForegroundColor Cyan
    & "$venvPath\Scripts\Activate.ps1"
}

# Navigate to project root
Set-Location $projectRoot
Write-Host "[INFO] Working directory: $(Get-Location)" -ForegroundColor Gray

# Check dataset exists
$datasetPath = "$projectRoot\data\my_dataset\visdrone\nanodet_format"
if (-not (Test-Path $datasetPath)) {
    Write-Host "[ERROR] Dataset not found at: $datasetPath" -ForegroundColor Red
    Write-Host "[INFO] Please run the COCO format conversion first" -ForegroundColor Yellow
    exit 1
}

# Prepare arguments - Quote the script path to handle spaces
$trainScript = "`"$scriptPath\train_nanodet_simple_trial1.py`""
$pythonArgs = @(
    $trainScript,
    "--epochs", $Epochs.ToString(),
    "--batch-size", "16",
    "--lr", "0.001"
)

if ($QuickTest) {
    $pythonArgs += "--quick-test"
}

Write-Host ""
Write-Host "[INFO] Starting NanoDet Trial-1 training..." -ForegroundColor Green
Write-Host "[INFO] Command: python $($pythonArgs -join ' ')" -ForegroundColor Gray
Write-Host ""

# Run training
try {
    $process = Start-Process -FilePath "python" -ArgumentList $pythonArgs -NoNewWindow -PassThru -Wait
    
    if ($process.ExitCode -eq 0) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Green
        Write-Host "[SUCCESS] Training completed successfully!" -ForegroundColor Green
        Write-Host "========================================" -ForegroundColor Green
        
        # Find the latest run directory
        $runsDir = "$projectRoot\runs\train"
        $latestRun = Get-ChildItem $runsDir -Directory | Where-Object { $_.Name -like "nanodet_trial1_fixed_*" } | Sort-Object CreationTime -Descending | Select-Object -First 1
        
        if ($latestRun) {
            Write-Host "Results saved to: $($latestRun.FullName)" -ForegroundColor Cyan
            Write-Host ""
            Write-Host "Key files:" -ForegroundColor Yellow
            Write-Host "  - Best model: $($latestRun.FullName)\best_model.pth" -ForegroundColor Gray
            Write-Host "  - Final model: $($latestRun.FullName)\final_model.pth" -ForegroundColor Gray
            Write-Host "  - Training log: $($latestRun.FullName)\training.log" -ForegroundColor Gray
            Write-Host "  - History: $($latestRun.FullName)\history.json" -ForegroundColor Gray
        }
    } else {
        Write-Host "[ERROR] Training failed with exit code: $($process.ExitCode)" -ForegroundColor Red
    }
} catch {
    Write-Host "[ERROR] Failed to run training: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[INFO] Trial-1 Features:" -ForegroundColor Cyan
Write-Host "  - Proper Focal Loss for classification" -ForegroundColor Gray
Write-Host "  - Stable SmoothL1 Loss for bounding box regression" -ForegroundColor Gray
Write-Host "  - Real data loading (not random noise)" -ForegroundColor Gray
Write-Host "  - Environmental augmentation (fog, rain, blur)" -ForegroundColor Gray
Write-Host "  - Lightweight architecture (<3MB)" -ForegroundColor Gray
Write-Host "  - Numerical stability safeguards" -ForegroundColor Gray
Write-Host ""
Write-Host "[NEXT] Run evaluation metrics after training completes" -ForegroundColor Yellow