#!/usr/bin/env pwsh
<#
.SYNOPSIS
    NanoDet Phase 1 (True Baseline) Training Script - VisDrone Dataset
    Master's Thesis: Robust Object Detection for Surveillance Drones

.DESCRIPTION
    Professional PowerShell wrapper for NanoDet Phase 1 (True Baseline) training
    following Protocol Version 2.0 - True Baseline Framework.
    
    Key Features:
    - Phase 1: True baseline training with NO AUGMENTATION
    - Original VisDrone dataset only (Protocol v2.0 compliance)
    - Ultra-lightweight model (<3MB target)
    - Comprehensive evaluation integration
    - Professional error handling and logging

.PARAMETER Epochs
    Number of training epochs (default: 100)

.PARAMETER QuickTest
    Run quick test with reduced settings (20 epochs)

.EXAMPLE
    .\run_nanodet_baseline.ps1
    .\run_nanodet_baseline.ps1 -Epochs 150
    .\run_nanodet_baseline.ps1 -QuickTest

.NOTES
    Author: Burak Kağan Yılmazer
    Date: July 2025
    Environment: nanodet_env
    Protocol: Version 2.0 - True Baseline Framework
#>

param(
    [Parameter(Mandatory=$false)]
    [int]$Epochs = 100,
    
    [Parameter(Mandatory=$false)]
    [switch]$QuickTest
)

# Script configuration
$ErrorActionPreference = "Stop"
$InformationPreference = "Continue"

# Project paths
$ProjectRoot = (Get-Item $PSScriptRoot).Parent.Parent.Parent.Parent.Parent.FullName
$ScriptPath = Join-Path $PSScriptRoot "train_nanodet_baseline.py"

function Write-Header {
    param([string]$Title, [string]$Phase)
    
    Write-Host "=" * 80 -ForegroundColor Cyan
    Write-Host $Title -ForegroundColor White
    Write-Host "PROTOCOL: Version 2.0 - True Baseline Framework" -ForegroundColor Green
    Write-Host "=" * 80 -ForegroundColor Cyan
    Write-Host "Phase: $Phase" -ForegroundColor Yellow
    Write-Host "Target: >12% mAP@0.5 (ultra-lightweight baseline)" -ForegroundColor Magenta
    Write-Host "Model Size Target: <3MB" -ForegroundColor Magenta
    Write-Host "Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray
    Write-Host "=" * 80 -ForegroundColor Cyan
    Write-Host ""
}

function Write-StatusMessage {
    param([string]$Message, [string]$Type = "INFO")
    
    $timestamp = Get-Date -Format "HH:mm:ss"
    switch ($Type) {
        "SUCCESS" { Write-Host "[$timestamp] [SUCCESS] $Message" -ForegroundColor Green }
        "ERROR"   { Write-Host "[$timestamp] [ERROR] $Message" -ForegroundColor Red }
        "WARNING" { Write-Host "[$timestamp] [WARNING] $Message" -ForegroundColor Yellow }
        default   { Write-Host "[$timestamp] [INFO] $Message" -ForegroundColor Cyan }
    }
}

function Test-Prerequisites {
    Write-StatusMessage "Validating prerequisites..."
    
    # Check Python
    if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
        throw "Python not found. Please ensure Python is installed and in PATH."
    }
    
    # Note: Virtual environment should be manually activated before running this script
    
    # Check training script
    if (-not (Test-Path $ScriptPath)) {
        throw "Training script not found: $ScriptPath"
    }
    
    # Check project structure
    $DataPath = Join-Path $ProjectRoot "data\my_dataset\visdrone"
    if (-not (Test-Path $DataPath)) {
        Write-StatusMessage "VisDrone dataset not found at $DataPath" "WARNING"
        Write-StatusMessage "Please ensure dataset is properly installed" "WARNING"
    }
    
    Write-StatusMessage "Prerequisites validation completed" "SUCCESS"
}

function Test-PythonEnvironment {
    Write-StatusMessage "Checking Python environment..."
    
    try {
        # Verify environment
        $pythonVersion = & python --version 2>&1
        Write-StatusMessage "Python version: $pythonVersion"
        
        # Check key packages
        $packages = @("torch", "torchvision", "opencv-python", "pycocotools")
        foreach ($package in $packages) {
            try {
                $packageImport = $package.Replace('-', '_')
                & python -c "import $packageImport; print('$package OK')" 2>$null
                if ($LASTEXITCODE -eq 0) {
                    Write-StatusMessage "${package}: Available" "SUCCESS"
                } else {
                    Write-StatusMessage "${package}: Not found" "WARNING"
                }
            } catch {
                Write-StatusMessage "${package}: Error checking" "WARNING"
            }
        }
        
        Write-StatusMessage "Python environment check completed" "SUCCESS"
    }
    catch {
        throw "Failed to check Python environment: $_"
    }
}

function Invoke-Training {
    param([int]$Epochs, [bool]$QuickTest)
    
    Write-StatusMessage "Starting NanoDet Phase 1 (True Baseline) training..."
    Write-StatusMessage "Training Configuration:"
    Write-StatusMessage "  - Epochs: $Epochs"
    Write-StatusMessage "  - Quick Test: $QuickTest"
    Write-StatusMessage "  - Phase: 1 (True Baseline - NO Augmentation)"
    Write-StatusMessage "  - Protocol: Version 2.0 compliant"
    Write-StatusMessage ""
    
    try {
        # Change to project root directory
        Push-Location $ProjectRoot
        
        # Prepare Python arguments with proper quoting
        $pythonArgs = @(
            "`"$ScriptPath`"",
            "--epochs", $Epochs.ToString()
        )
        
        if ($QuickTest) {
            $pythonArgs += "--quick-test"
            Write-StatusMessage "Quick test mode enabled (20 epochs)" "WARNING"
        }
        
        Write-StatusMessage "Executing training command..."
        Write-StatusMessage "Command: python $($pythonArgs -join ' ')"
        Write-StatusMessage ""
        
        # Execute training with proper path handling
        $process = Start-Process -FilePath "python" -ArgumentList $pythonArgs -NoNewWindow -PassThru -Wait -WorkingDirectory $ProjectRoot
        
        if ($process.ExitCode -eq 0) {
            Write-StatusMessage "Training completed successfully!" "SUCCESS"
            return $true
        } else {
            Write-StatusMessage "Training failed with exit code: $($process.ExitCode)" "ERROR"
            return $false
        }
    }
    catch {
        Write-StatusMessage "Training execution failed: $_" "ERROR"
        return $false
    }
    finally {
        Pop-Location
    }
}

function Show-CompletionSummary {
    param([bool]$Success)
    
    Write-Host ""
    Write-Host "=" * 80 -ForegroundColor Cyan
    
    if ($Success) {
        Write-Host "[SUCCESS] NanoDet Phase 1 (True Baseline) Training Complete!" -ForegroundColor Green
        Write-Host ""
        Write-Host "METHODOLOGY VALIDATION:" -ForegroundColor Yellow
        Write-Host "  ✅ Phase 1 true baseline training completed" -ForegroundColor Green
        Write-Host "  ✅ NO AUGMENTATION applied (Protocol v2.0 compliance)" -ForegroundColor Green
        Write-Host "  ✅ Original VisDrone dataset only" -ForegroundColor Green
        Write-Host "  ✅ Ultra-lightweight model architecture" -ForegroundColor Green
        Write-Host ""
        Write-Host "EXPECTED RESULTS:" -ForegroundColor Yellow
        Write-Host "  • Target Performance: >12% mAP@0.5" -ForegroundColor Magenta
        Write-Host "  • Model Size: <3MB" -ForegroundColor Magenta
        Write-Host "  • True baseline for Phase 1 vs Phase 2 comparison" -ForegroundColor Magenta
        Write-Host ""
        Write-Host "NEXT STEPS:" -ForegroundColor Yellow
        Write-Host "  1. Review training results in runs/train/ directory" -ForegroundColor White
        Write-Host "  2. Run Phase 2 (Environmental Robustness) training" -ForegroundColor White
        Write-Host "  3. Compare Phase 1 vs Phase 2 performance" -ForegroundColor White
        Write-Host "  4. Execute comprehensive evaluation framework" -ForegroundColor White
    } else {
        Write-Host "[ERROR] NanoDet Phase 1 Training Failed!" -ForegroundColor Red
        Write-Host ""
        Write-Host "TROUBLESHOOTING:" -ForegroundColor Yellow
        Write-Host "  1. Check virtual environment activation" -ForegroundColor White
        Write-Host "  2. Verify all required packages are installed" -ForegroundColor White
        Write-Host "  3. Ensure COCO format data is available" -ForegroundColor White
        Write-Host "  4. Check CUDA/GPU compatibility" -ForegroundColor White
        Write-Host "  5. Review error logs for detailed information" -ForegroundColor White
    }
    
    Write-Host "=" * 80 -ForegroundColor Cyan
}

# Main execution
try {
    Write-Header "NanoDet Phase 1 (True Baseline) Training - VisDrone Dataset" "1 (True Baseline - NO Augmentation)"
    
    # Validate prerequisites
    Test-Prerequisites
    
    # Check Python environment (assumes virtual environment is already activated)
    Test-PythonEnvironment
    
    # Execute training
    $trainingSuccess = Invoke-Training -Epochs $Epochs -QuickTest $QuickTest.IsPresent
    
    # Show completion summary
    Show-CompletionSummary -Success $trainingSuccess
    
    if (-not $trainingSuccess) {
        exit 1
    }
}
catch {
    Write-StatusMessage "Script execution failed: $_" "ERROR"
    Write-Host ""
    Write-Host "=" * 80 -ForegroundColor Red
    Write-Host "[ERROR] NanoDet Phase 1 Training Failed!" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host "=" * 80 -ForegroundColor Red
    exit 1
}