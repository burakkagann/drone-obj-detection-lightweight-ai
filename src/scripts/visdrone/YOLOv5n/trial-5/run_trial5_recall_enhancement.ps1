# YOLOv5n Trial-5 "Recall Enhancement" Training Script
# Strategic optimization targeting persistent low recall issue (20% → 25-30%)
# Based on comprehensive analysis of Trial-2 (23.557%), Trial-4 (23.70%) performance patterns
# Target: 24.5-26% mAP@0.5 with improved precision-recall balance

param(
    [int]$Epochs = 25,
    [switch]$FullTraining,
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
    Write-Host "YOLOv5n Trial-5 'Recall Enhancement' Training" -ForegroundColor Green
    Write-Host "=============================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "STRATEGY:" -ForegroundColor Yellow
    Write-Host "  Address persistent precision-recall imbalance observed in all successful trials"
    Write-Host "  Current pattern: High precision (~80%) but low recall (~20%)"
    Write-Host "  Target: Improve recall to 25-30% while maintaining 75-80% precision"
    Write-Host ""
    Write-Host "USAGE:" -ForegroundColor Yellow
    Write-Host "  .\run_trial5_recall_enhancement.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "OPTIONS:" -ForegroundColor Yellow
    Write-Host "  -Epochs <number>     Number of training epochs (default: 25 for full training)"
    Write-Host "  -QuickTest          Run 20-epoch validation test first (recommended)"
    Write-Host "  -FullTraining       Run full 25-epoch training (use after validation)"
    Write-Host "  -Help              Show this help message"
    Write-Host ""
    Write-Host "TRAINING PROTOCOL:" -ForegroundColor Cyan
    Write-Host "  Phase 1: 20-epoch validation test (recommended first step)"
    Write-Host "  Phase 2: 25-epoch full training (if validation shows improvement)"
    Write-Host ""
    Write-Host "PERFORMANCE CONTEXT:" -ForegroundColor Cyan
    Write-Host "  Trial-2 Baseline: 23.557% mAP@0.5, 81.08% precision, 19.71% recall"
    Write-Host "  Trial-4 Baseline: 23.70% mAP@0.5, 81.15% precision, 19.97% recall"
    Write-Host "  Trial-5 Target: 24.5-26% mAP@0.5, 75-80% precision, 25-30% recall"
    Write-Host ""
    Write-Host "KEY MODIFICATIONS:" -ForegroundColor Cyan
    Write-Host "  • IoU threshold: 0.15 → 0.10 (more lenient matching)"
    Write-Host "  • Anchor threshold: 4.0 → 3.5 (more anchor assignments)"
    Write-Host "  • Object loss: 1.25 → 1.5 (stronger objectness)"
    Write-Host "  • Box loss: 0.03 → 0.025 (less restrictive)"
    Write-Host "  • Extended training: 25 epochs (vs 20)"
    Write-Host "  • Extended warmup: 6 epochs (vs 5)"
    exit 0
}

# Show help if requested
if ($Help) {
    Show-Help
}

# Determine training configuration
if ($QuickTest) {
    $Epochs = 20
    $TrainingType = "Validation Test"
    $RunName = "yolov5n_trial5_validation"
} elseif ($FullTraining) {
    $Epochs = 25
    $TrainingType = "Full Recall Enhancement Training"
    $RunName = "yolov5n_trial5_full_training"
} else {
    # Default to validation test
    $Epochs = 20
    $TrainingType = "Validation Test"
    $RunName = "yolov5n_trial5_validation"
    Write-Warning "[DEFAULT] Running validation test first (recommended approach)"
}

# Header
Write-Host "================================================================" -ForegroundColor Green
Write-Host "YOLOv5n Trial-5 'Recall Enhancement' Training" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""

# Strategy overview
Write-Info "[STRATEGY] Trial-5 Recall Enhancement Strategy:"
Write-Host "  • Primary Goal: Address precision-recall imbalance"
Write-Host "  • Current Issue: High precision (~80%) but low recall (~20%)"
Write-Host "  • Approach: Reduce detection thresholds, enhance objectness"
Write-Host "  • Expected: 25-30% recall, 75-80% precision, 24.5-26% mAP@0.5"
Write-Host ""

# Configuration
Write-Info "[CONFIG] Trial-5 Configuration:"
Write-Host "  • Training Type: $TrainingType"
Write-Host "  • Epochs: $Epochs"
Write-Host "  • Batch Size: 16 (proven from Trial-2/4)"
Write-Host "  • Image Size: 640 (proven optimal)"
Write-Host "  • IoU Threshold: 0.10 (reduced from 0.15 for recall)"
Write-Host "  • Anchor Threshold: 3.5 (reduced from 4.0 for recall)"
Write-Host "  • Object Loss Weight: 1.5 (increased from 1.25 for objectness)"
Write-Host "  • Box Loss Weight: 0.025 (reduced from 0.03 for flexibility)"
Write-Host "  • Focal Loss: DISABLED (critical lesson from Trial-3)"
Write-Host ""

Write-Info "[BASELINE] Performance Context:"
Write-Host "  • Trial-2: 23.557% mAP@0.5, 81.08% precision, 19.71% recall ✅"
Write-Host "  • Trial-4: 23.70% mAP@0.5, 81.15% precision, 19.97% recall ✅"
Write-Host "  • Trial-3: 0.002% mAP@0.5 (focal loss disaster) ❌"
Write-Host "  • Trial-5 Target: 24.5-26% mAP@0.5, 75-80% precision, 25-30% recall 🎯"
Write-Host ""

# Key modifications explanation
Write-Info "[MODIFICATIONS] Key Changes from Trial-4:"
Write-Host "  • IoU Threshold: 0.15 → 0.10 (Expected: +2-4% recall)"
Write-Host "  • Anchor Threshold: 4.0 → 3.5 (Expected: +1-3% recall)"
Write-Host "  • Object Loss: 1.25 → 1.5 (Expected: +1-2% recall)"
Write-Host "  • Box Loss: 0.03 → 0.025 (Less restrictive regression)"
Write-Host "  • Training Length: 20 → 25 epochs (More convergence time)"
Write-Host "  • Combined Expected: +4-9% recall improvement"
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
$hypFile = "..\..\..\..\..\config\visdrone\yolov5n_v1\hyp_visdrone_trial5.yaml"
if (Test-Path $hypFile) {
    Write-Success "[CONFIG] Trial-5 hyperparameter file found"
} else {
    Write-Error "[ERROR] Trial-5 hyperparameter file not found: $hypFile"
    Write-Host "Expected location: config/visdrone/yolov5n_v1/hyp_visdrone_trial5.yaml"
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
Write-Info "[TRAINING] Starting Trial-5 Recall Enhancement training..."
Write-Host ""

# Navigate to YOLOv5 directory and execute training
$yolov5Path = "..\..\..\..\..\src\models\YOLOv5"
if (Test-Path $yolov5Path) {
    Write-Success "[PATH] YOLOv5 directory found"
    Push-Location $yolov5Path
    
    try {
        # Prepare training command using Trial-2 proven Python wrapper architecture
        $pythonScript = "`"C:\Users\burak\OneDrive\Desktop\Git Repos\drone-obj-detection-lightweight-ai\src\scripts\visdrone\YOLOv5n\trial-5\train_yolov5n_trial5.py`""
        $pythonArgs = @($pythonScript)
        
        # Add epoch parameter
        $pythonArgs += "--epochs", $Epochs.ToString()
        
        # Add quick test flag if validation mode
        if ($QuickTest -or (-not $FullTraining)) {
            $pythonArgs += "--quick-test"
        }
        
        Write-Host "Executing: python $($pythonArgs -join ' ')" -ForegroundColor Cyan
        Write-Host ""
        
        $process = Start-Process -FilePath "python" -ArgumentList $pythonArgs -NoNewWindow -PassThru -Wait
        
        if ($process.ExitCode -eq 0) {
            Write-Host ""
            Write-Success "[SUCCESS] Trial-5 $TrainingType completed successfully!"
            Write-Host ""
            
            # Results analysis
            Write-Info "[RESULTS] Training Results Analysis:"
            if ($FullTraining) {
                Write-Host "  • Full 25-epoch recall enhancement training completed"
                Write-Host "  • Check final mAP@0.5 and recall metrics in results.csv"
                Write-Host "  • Compare against Trial-4 baseline:"
                Write-Host "    - mAP@0.5: Target >24.5% (vs 23.70% Trial-4)"
                Write-Host "    - Recall: Target >25% (vs 19.97% Trial-4)"
                Write-Host "    - Precision: Acceptable >75% (vs 81.15% Trial-4)"
                Write-Host ""
                Write-Info "[SUCCESS CRITERIA]:"
                Write-Host "  • Excellent: >25.5% mAP@0.5, >28% recall"
                Write-Host "  • Target: >24.5% mAP@0.5, >25% recall"
                Write-Host "  • Minimum: >23.8% mAP@0.5, >21% recall"
                Write-Host ""
            } else {
                Write-Host "  • Validation test completed (20 epochs)"
                Write-Host "  • Check runs/train/$RunName/results.csv for:"
                Write-Host "    - mAP@0.5 trend (should show improvement trajectory)"
                Write-Host "    - Recall improvement (target: >21% vs 19.97% baseline)"
                Write-Host "    - Precision maintenance (should stay >75%)"
                Write-Host ""
                Write-Info "[NEXT STEP]:"
                Write-Host "  If validation shows recall improvement, proceed with full training:"
                Write-Host "    .\run_trial5_recall_enhancement.ps1 -FullTraining"
                Write-Host ""
            }
            
            Write-Info "[ANALYSIS CHECKLIST]:"
            Write-Host "  • Compare precision-recall balance with previous trials"
            Write-Host "  • Verify recall enhancement hypothesis (expected +4-9%)"
            Write-Host "  • Check if precision-recall ratio improved"
            Write-Host "  • Document results in Trial-5 progression notes"
            Write-Host "  • Evaluate success against thesis targets (25% mAP@0.5)"
            Write-Host ""
            
            Write-Info "[INTERPRETATION GUIDE]:"
            Write-Host "  Success Indicators:"
            Write-Host "    ✅ Recall >25% (major breakthrough)"
            Write-Host "    ✅ mAP@0.5 >24.5% (significant improvement)"
            Write-Host "    ✅ Precision >75% (acceptable trade-off)"
            Write-Host ""
            Write-Host "  Moderate Success:"
            Write-Host "    ⚠️ Recall >21% (meaningful improvement)"
            Write-Host "    ⚠️ mAP@0.5 >23.8% (beats Trial-4)"
            Write-Host ""
            Write-Host "  Further Optimization Needed:"
            Write-Host "    ❌ Recall <21% (strategy adjustment required)"
            Write-Host "    ❌ mAP@0.5 <23.8% (performance regression)"
            Write-Host ""
            
        } else {
            Write-Error "[ERROR] Training failed with exit code: $($process.ExitCode)"
            Write-Host ""
            Write-Warning "[TROUBLESHOOTING] Common Issues:"
            Write-Host "  • CUDA out of memory → Check GPU memory, reduce batch size"
            Write-Host "  • Dataset not found → Verify dataset paths in config files"
            Write-Host "  • Hyperparameter file missing → Check hyp_visdrone_trial5.yaml exists"
            Write-Host "  • Virtual environment → Ensure yolov5n_env is activated"
            Write-Host ""
            exit 1
        }
        
    } catch {
        Write-Error "[ERROR] Failed to execute training: $($_.Exception.Message)"
        Write-Host ""
        Write-Warning "[DEBUG] Check the following:"
        Write-Host "  • Python script path: train_yolov5n_trial5.py exists"
        Write-Host "  • Virtual environment activated (yolov5n_env)"
        Write-Host "  • All dependencies installed in virtual environment"
        Write-Host "  • Sufficient disk space for training outputs"
        Write-Host ""
        exit 1
    } finally {
        Pop-Location
    }
    
} else {
    Write-Error "[ERROR] YOLOv5 directory not found: $yolov5Path"
    exit 1
}

Write-Host "================================================================" -ForegroundColor Green
Write-Host "Trial-5 'Recall Enhancement' Training Complete" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""

# Final recommendations
Write-Info "[RECOMMENDATIONS] Next Steps Based on Results:"
Write-Host ""
Write-Host "IF RECALL ENHANCEMENT SUCCESSFUL (>25% recall, >24.5% mAP@0.5):"
Write-Host "  1. Document breakthrough in progression notes"
Write-Host "  2. Proceed with multi-model comparison (YOLOv8n, MobileNet-SSD, NanoDet)"
Write-Host "  3. Begin edge device simulation testing"
Write-Host "  4. Start thesis results analysis and writing"
Write-Host ""
Write-Host "IF MODERATE SUCCESS (>21% recall, >23.8% mAP@0.5):"
Write-Host "  1. Analyze parameter sensitivity for further optimization"
Write-Host "  2. Consider Trial-6 with refined thresholds"
Write-Host "  3. Investigate data augmentation enhancements"
Write-Host "  4. Document lessons learned for thesis methodology"
Write-Host ""
Write-Host "IF INSUFFICIENT IMPROVEMENT (<21% recall, <23.8% mAP@0.5):"
Write-Host "  1. Debug configuration differences with Trial-4"
Write-Host "  2. Consider alternative recall enhancement strategies"
Write-Host "  3. Investigate model architecture modifications"
Write-Host "  4. Review literature for additional optimization techniques"
Write-Host ""

Write-Success "[COMPLETE] Trial-5 Recall Enhancement training workflow completed!"
Write-Host "Check the training results and follow the recommendations above." -ForegroundColor White