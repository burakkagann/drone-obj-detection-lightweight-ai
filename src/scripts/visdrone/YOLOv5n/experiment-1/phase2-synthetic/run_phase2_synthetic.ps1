# YOLOv5n Phase 2 (Synthetic Environmental Augmentation) Training Wrapper Script
# Master's Thesis: Robust Object Detection for Surveillance Drones
# Protocol: Version 2.0 - Phase 2 Synthetic Environmental Augmentation
#
# This script provides a robust wrapper for Phase 2 synthetic augmentation training
# with proper environment activation, error handling, and logging.
#
# Author: Burak Kağan Yılmazer
# Date: July 2025
# Environment: yolov5n_visdrone_env

param(
    [int]$Epochs = 100,
    [switch]$QuickTest,
    [switch]$Help,
    [string]$Config = "",
    [switch]$Verbose
)

# Display help information
if ($Help) {
    Write-Host "YOLOv5n Phase 2 (Synthetic Environmental Augmentation) Training Wrapper" -ForegroundColor Cyan
    Write-Host "PROTOCOL: Version 2.0 - Phase 2 Synthetic Environmental Augmentation" -ForegroundColor Green
    Write-Host ""
    Write-Host "USAGE:" -ForegroundColor Yellow
    Write-Host "  # 1. Activate virtual environment first"
    Write-Host "  .\venvs\visdrone\yolov5n_visdrone_env\Scripts\Activate.ps1"
    Write-Host ""
    Write-Host "  # 2. Run training script"
    Write-Host "  .\run_phase2_synthetic.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "OPTIONS:" -ForegroundColor Yellow
    Write-Host "  -Epochs <int>     Number of training epochs (default: 100)"
    Write-Host "  -QuickTest        Run quick test with 20 epochs"
    Write-Host "  -Config <path>    Path to custom configuration file"
    Write-Host "  -Verbose          Enable detailed logging"
    Write-Host "  -Help             Show this help message"
    Write-Host ""
    Write-Host "EXAMPLES:" -ForegroundColor Yellow
    Write-Host "  # Standard Phase 2 training (after activating venv)"
    Write-Host "  .\run_phase2_synthetic.ps1"
    Write-Host ""
    Write-Host "  # Quick test (20 epochs)"
    Write-Host "  .\run_phase2_synthetic.ps1 -QuickTest"
    Write-Host ""
    Write-Host "  # Custom epoch count"
    Write-Host "  .\run_phase2_synthetic.ps1 -Epochs 150"
    Write-Host ""
    Write-Host "PHASE 2 FEATURES:" -ForegroundColor Magenta
    Write-Host "  - Synthetic environmental augmentation (fog, night, blur)"
    Write-Host "  - Enhanced standard augmentation (mosaic, HSV, geometric)"
    Write-Host "  - Optimized hyperparameters (AdamW, cosine LR, mixed precision)"
    Write-Host "  - Baseline comparison with Phase 1 (24.9% mAP@0.5)"
    Write-Host "  - Target: >5% improvement in adverse conditions"
    Write-Host ""
    exit 0
}

# Script configuration
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Project paths
$ProjectRoot = (Get-Item $PSScriptRoot).Parent.Parent.Parent.Parent.Parent.Parent.FullName
$TrainingScript = Join-Path $PSScriptRoot "train_phase2_synthetic.py"
$LogDir = Join-Path $ProjectRoot "logs\phase2_synthetic"

# Ensure log directory exists
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

# Generate timestamp for session
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$SessionLog = Join-Path $LogDir "phase2_synthetic_session_$Timestamp.log"

# Logging function
function Write-Log {
    param(
        [string]$Message,
        [string]$Level = "INFO",
        [string]$Color = "White"
    )
    
    $LogEntry = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [$Level] $Message"
    Write-Host $LogEntry -ForegroundColor $Color
    Add-Content -Path $SessionLog -Value $LogEntry
}

# Error handling function
function Handle-Error {
    param([string]$ErrorMessage)
    
    Write-Log "CRITICAL ERROR: $ErrorMessage" -Level "ERROR" -Color "Red"
    Write-Log "Training session failed - check logs for details" -Level "ERROR" -Color "Red"
    
    Write-Host ""
    Write-Host "[ERROR] Phase 2 training failed!" -ForegroundColor Red
    Write-Host "Error: $ErrorMessage" -ForegroundColor Red
    Write-Host "Session Log: $SessionLog" -ForegroundColor Yellow
    Write-Host ""
    
    exit 1
}

# Main execution
try {
    # Header
    Write-Host "=" * 80 -ForegroundColor Cyan
    Write-Host "YOLOv5n Phase 2 (Synthetic Environmental Augmentation) Training - VisDrone Dataset" -ForegroundColor Cyan
    Write-Host "PROTOCOL: Version 2.0 - Phase 2 Synthetic Environmental Augmentation" -ForegroundColor Green
    Write-Host "=" * 80 -ForegroundColor Cyan
    
    Write-Log "Starting Phase 2 (Synthetic Environmental Augmentation) training session" -Color "Green"
    Write-Log "Protocol: Version 2.0 - Phase 2 Synthetic Environmental Augmentation" -Color "Green"
    Write-Log "Session ID: $Timestamp" -Color "Cyan"
    
    # Validate paths
    Write-Log "Validating environment and paths..." -Color "Yellow"
    
    if (-not (Test-Path $ProjectRoot)) {
        Handle-Error "Project root not found: $ProjectRoot"
    }
    
    if (-not (Test-Path $TrainingScript)) {
        Handle-Error "Training script not found: $TrainingScript"
    }
    
    Write-Log "Path validation successful" -Color "Green"
    
    # Display configuration
    Write-Log "Configuration Summary:" -Color "Cyan"
    Write-Log "  • Project Root: $ProjectRoot" -Color "White"
    Write-Log "  • Training Script: train_phase2_synthetic.py" -Color "White"
    Write-Log "  • Epochs: $Epochs" -Color "White"
    Write-Log "  • Quick Test: $QuickTest" -Color "White"
    Write-Log "  • Phase: 2 (Synthetic Environmental Augmentation)" -Color "White"
    Write-Log "  • Baseline: Phase 1 - 24.9% mAP@0.5" -Color "White"
    Write-Log "  • Target: >5% improvement in adverse conditions" -Color "White"
    Write-Log "  • Session Log: $SessionLog" -Color "White"
    
    # Phase 2 methodology compliance
    Write-Log "" -Color "White"
    Write-Log "[PHASE-2] Synthetic Environmental Augmentation Features:" -Color "Magenta"
    Write-Log "  - SYNTHETIC AUGMENTATION: Fog, night, sensor distortions enabled" -Color "White"
    Write-Log "  - ENHANCED STANDARD AUGMENTATION: Mosaic, HSV, geometric transforms" -Color "White"
    Write-Log "  - OPTIMIZED HYPERPARAMETERS: AdamW, cosine LR, mixed precision" -Color "White"
    Write-Log "  - METHODOLOGY COMPLIANCE: Protocol v2.0 Phase 2" -Color "White"
    Write-Log "  - PURPOSE: Improve robustness over Phase 1 baseline" -Color "White"
    Write-Log "  - EXPECTED: Superior performance under adverse conditions" -Color "White"
    Write-Log "" -Color "White"
    
    # Navigate to project root
    Write-Log "Navigating to project root..." -Color "Yellow"
    Set-Location $ProjectRoot
    
    # Verify Python environment (assuming venv is pre-activated)
    Write-Log "Verifying Python environment (assuming venv is activated)..." -Color "Yellow"
    
    try {
        $PythonVersion = python --version 2>&1
        Write-Log "Python Version: $PythonVersion" -Color "Green"
        
        # Check PyTorch
        $TorchCheck = python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>&1
        Write-Log "Environment Check: $TorchCheck" -Color "Green"
        
        # Check augmentation modules
        $AugCheck = python -c "from augmentation_pipeline.augment_scripts import fog, night, sensor_distortions; print('Augmentation modules: OK')" 2>&1
        Write-Log "Augmentation Check: $AugCheck" -Color "Green"
    }
    catch {
        Handle-Error "Python environment verification failed - ensure virtual environment is activated: $($_.Exception.Message)"
    }
    
    # Prepare training arguments
    $TrainingArgs = @()
    
    if ($Epochs -ne 100) {
        $TrainingArgs += "--epochs", $Epochs.ToString()
    }
    
    if ($QuickTest) {
        $TrainingArgs += "--quick-test"
        Write-Log "Quick test mode enabled - training will use 20 epochs" -Color "Yellow"
    }
    
    if ($Config) {
        $TrainingArgs += "--config", $Config
        Write-Log "Using custom configuration: $Config" -Color "Yellow"
    }
    
    if ($Verbose) {
        Write-Log "Verbose mode enabled - detailed logging active" -Color "Yellow"
    }
    
    # Execute training
    Write-Log "" -Color "White"
    Write-Log "[TRAINING] Starting Phase 2 synthetic augmentation training..." -Color "Green"
    Write-Log "Command: python $TrainingScript $($TrainingArgs -join ' ')" -Color "Cyan"
    Write-Log "Expected Duration: $(if ($QuickTest) { '30-45 minutes' } else { '3-4 hours' })" -Color "Yellow"
    Write-Log "Expected Features:" -Color "Yellow"
    Write-Log "  • Synthetic environmental effects during training" -Color "White"
    Write-Log "  • Enhanced standard augmentation (mosaic, HSV, etc.)" -Color "White"
    Write-Log "  • AdamW optimizer with cosine learning rate" -Color "White"
    Write-Log "  • Mixed precision training for efficiency" -Color "White"
    Write-Log "  • Comparison with Phase 1 baseline (24.9% mAP@0.5)" -Color "White"
    Write-Log "" -Color "White"
    
    $TrainingStartTime = Get-Date
    
    try {
        if ($TrainingArgs.Count -gt 0) {
            & python $TrainingScript @TrainingArgs
        } else {
            & python $TrainingScript
        }
        
        if ($LASTEXITCODE -ne 0) {
            Handle-Error "Training process failed with exit code $LASTEXITCODE"
        }
    }
    catch {
        Handle-Error "Training execution failed: $($_.Exception.Message)"
    }
    
    $TrainingEndTime = Get-Date
    $TrainingDuration = $TrainingEndTime - $TrainingStartTime
    
    # Success summary
    Write-Log "" -Color "White"
    Write-Log "[SUCCESS] Phase 2 training completed successfully!" -Color "Green"
    Write-Log "Training Duration: $($TrainingDuration.ToString('hh\:mm\:ss'))" -Color "Green"
    Write-Log "Session Log: $SessionLog" -Color "Cyan"
    
    Write-Host ""
    Write-Host "=" * 80 -ForegroundColor Green
    Write-Host "[SUCCESS] YOLOv5n Phase 2 (Synthetic Environmental Augmentation) Training Complete!" -ForegroundColor Green
    Write-Host "Training Duration: $($TrainingDuration.ToString('hh\:mm\:ss'))" -ForegroundColor Green
    Write-Host "Expected: Enhanced robustness over Phase 1 baseline" -ForegroundColor Yellow
    Write-Host "Target: >5% improvement in adverse conditions" -ForegroundColor Yellow
    Write-Host "Next Step: Comparative evaluation on synthetic test sets" -ForegroundColor Cyan
    Write-Host "Baseline Comparison: Phase 1 - 24.9% mAP@0.5" -ForegroundColor Cyan
    Write-Host "Session Log: $SessionLog" -ForegroundColor Cyan
    Write-Host "=" * 80 -ForegroundColor Green
    
    Write-Log "Phase 2 synthetic augmentation training session completed successfully" -Color "Green"
    Write-Log "Expected: Superior performance under adverse conditions" -Color "Green"
    Write-Log "Next: Comparative evaluation against Phase 1 baseline" -Color "Cyan"
    
}
catch {
    Handle-Error "Unexpected error in training wrapper: $($_.Exception.Message)"
}
finally {
    # Cleanup
    Write-Log "Training session finished at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -Color "Cyan"
    
    # Return to original location if needed
    if (Test-Path $PSScriptRoot) {
        Set-Location $PSScriptRoot
    }
}