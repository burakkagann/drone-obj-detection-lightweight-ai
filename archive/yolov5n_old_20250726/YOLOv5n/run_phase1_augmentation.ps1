# Phase 1: Synthetic Data Augmentation Pipeline Execution Script
# For YOLOv5n + VisDrone Methodology Implementation

# Ensure the script stops on errors
$ErrorActionPreference = "Stop"

# Configuration
$CONFIG = @{
    SOURCE_DATASET_PATH = "..\..\..\..\data\my_dataset\visdrone"
    OUTPUT_DATASET_PATH = "..\..\..\..\data\my_dataset\visdrone_augmented"
    VALIDATION_OUTPUT_DIR = "validation_results"
    SEED = 42
}

# Function to print colored output
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

# Note: Virtual environment should be activated manually before running this script

# Function to verify prerequisites
function Test-Prerequisites {
    Write-ColorOutput "[CHECK] Verifying prerequisites..." "Cyan"
    
    # Check if source dataset exists
    if (-not (Test-Path $CONFIG.SOURCE_DATASET_PATH)) {
        Write-ColorOutput "[ERROR] Source dataset not found at: $($CONFIG.SOURCE_DATASET_PATH)" "Red"
        Write-ColorOutput "   Please ensure VisDrone dataset is properly prepared" "Yellow"
        exit 1
    }
    
    # Check required Python packages
    $requiredPackages = @("opencv-python", "numpy", "matplotlib", "seaborn", "scikit-image", "pandas", "pyyaml")
    
    foreach ($package in $requiredPackages) {
        $result = python -c "import $($package.Replace('-', '_')); print('OK')" 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-ColorOutput "[ERROR] Required package not found: $package" "Red"
            Write-ColorOutput "   Installing $package..." "Yellow"
            pip install $package
            if ($LASTEXITCODE -ne 0) {
                Write-ColorOutput "[ERROR] Failed to install $package" "Red"
                exit 1
            }
        }
    }
    
    # Check dataset structure
    $requiredDirs = @("train\images", "val\images", "train\labels", "val\labels")
    foreach ($dir in $requiredDirs) {
        $fullPath = Join-Path $CONFIG.SOURCE_DATASET_PATH $dir
        if (-not (Test-Path $fullPath)) {
            Write-ColorOutput "[WARNING] Directory not found: $fullPath" "Yellow"
        }
    }
    
    Write-ColorOutput "[SUCCESS] Prerequisites verified" "Green"
}

# Function to create output directories
function Initialize-OutputDirectories {
    Write-ColorOutput "[SETUP] Creating output directories..." "Cyan"
    
    # Create output dataset directory
    if (-not (Test-Path $CONFIG.OUTPUT_DATASET_PATH)) {
        New-Item -ItemType Directory -Path $CONFIG.OUTPUT_DATASET_PATH -Force | Out-Null
    }
    
    # Create validation output directory
    if (-not (Test-Path $CONFIG.VALIDATION_OUTPUT_DIR)) {
        New-Item -ItemType Directory -Path $CONFIG.VALIDATION_OUTPUT_DIR -Force | Out-Null
    }
    
    Write-ColorOutput "[SUCCESS] Output directories created" "Green"
}

# Function to execute Phase 1
function Invoke-Phase1Pipeline {
    Write-ColorOutput "[START] Starting Phase 1: Synthetic Data Augmentation Pipeline" "Magenta"
    Write-ColorOutput "=" * 80 "Magenta"
    
    # Display configuration
    Write-ColorOutput "[CONFIG] Configuration:" "Cyan"
    Write-ColorOutput "   Source Dataset: $($CONFIG.SOURCE_DATASET_PATH)" "White"
    Write-ColorOutput "   Output Dataset: $($CONFIG.OUTPUT_DATASET_PATH)" "White"
    Write-ColorOutput "   Validation Output: $($CONFIG.VALIDATION_OUTPUT_DIR)" "White"
    Write-ColorOutput "   Random Seed: $($CONFIG.SEED)" "White"
    Write-ColorOutput ""
    
    # Execute Phase 1 orchestrator
    $startTime = Get-Date
    
    try {
        python phase1_synthetic_augmentation_orchestrator.py `
            --source $CONFIG.SOURCE_DATASET_PATH `
            --output $CONFIG.OUTPUT_DATASET_PATH `
            --validation-output $CONFIG.VALIDATION_OUTPUT_DIR `
            --seed $CONFIG.SEED
        
        if ($LASTEXITCODE -eq 0) {
            $endTime = Get-Date
            $duration = $endTime - $startTime
            
            Write-ColorOutput "=" * 80 "Green"
            Write-ColorOutput "[SUCCESS] Phase 1 Successfully Completed!" "Green"
            Write-ColorOutput "[TIME] Duration: $($duration.ToString('hh\:mm\:ss'))" "Green"
            Write-ColorOutput "[OUTPUT] Results saved to: $($CONFIG.OUTPUT_DATASET_PATH)" "Green"
            Write-ColorOutput "=" * 80 "Green"
            
            # Show summary
            Show-ExecutionSummary
            
        } else {
            Write-ColorOutput "[ERROR] Phase 1 execution failed" "Red"
            exit 1
        }
        
    } catch {
        Write-ColorOutput "[ERROR] Error during Phase 1 execution: $($_.Exception.Message)" "Red"
        exit 1
    }
}

# Function to show execution summary
function Show-ExecutionSummary {
    Write-ColorOutput "[SUMMARY] Execution Summary:" "Cyan"
    
    # Check if results file exists
    $resultsFile = Join-Path $CONFIG.OUTPUT_DATASET_PATH "phase1_execution_results.json"
    if (Test-Path $resultsFile) {
        try {
            $results = Get-Content $resultsFile | ConvertFrom-Json
            
            if ($results.results.stratification.statistics) {
                $stats = $results.results.stratification.statistics
                Write-ColorOutput "   Total Images Generated: $($stats.total_images)" "White"
                Write-ColorOutput "   Original Images: $($stats.original_count)" "White"
                Write-ColorOutput "   Light Augmented: $($stats.light_count)" "White"
                Write-ColorOutput "   Medium Augmented: $($stats.medium_count)" "White"
                Write-ColorOutput "   Heavy Augmented: $($stats.heavy_count)" "White"
                
                if ($results.results.stratification.methodology_compliance) {
                    Write-ColorOutput "   Methodology Compliance: [PASS]" "Green"
                } else {
                    Write-ColorOutput "   Methodology Compliance: [FAIL]" "Red"
                }
            }
            
        } catch {
            Write-ColorOutput "   [WARNING] Could not parse results file" "Yellow"
        }
    }
    
    # Show available reports
    $reportsDir = Join-Path $CONFIG.OUTPUT_DATASET_PATH "reports"
    if (Test-Path $reportsDir) {
        Write-ColorOutput "[REPORTS] Available Reports:" "Cyan"
        Get-ChildItem $reportsDir -Filter "*.md" | ForEach-Object {
            Write-ColorOutput "   - $($_.Name)" "White"
        }
    }
}

# Function to show next steps
function Show-NextSteps {
    Write-ColorOutput "[NEXT] Next Steps:" "Cyan"
    Write-ColorOutput "   1. Review quality validation results in: $($CONFIG.VALIDATION_OUTPUT_DIR)" "White"
    Write-ColorOutput "   2. Check augmentation samples in: $($CONFIG.VALIDATION_OUTPUT_DIR)\samples" "White"
    Write-ColorOutput "   3. Review methodology compliance in reports" "White"
    Write-ColorOutput "   4. Proceed to Phase 2: Enhanced Training Pipeline" "White"
    Write-ColorOutput "   5. Update training script to use augmented dataset" "White"
}

# Main execution
function Main {
    Write-ColorOutput "[MAIN] YOLOv5n + VisDrone Phase 1: Synthetic Data Augmentation" "Magenta"
    Write-ColorOutput "[INFO] Methodology Implementation Framework" "Magenta"
    Write-ColorOutput ""
    
    try {
        # Step 1: Verify prerequisites
        Test-Prerequisites
        
        # Step 2: Initialize output directories
        Initialize-OutputDirectories
        
        # Step 3: Execute Phase 1 pipeline
        Invoke-Phase1Pipeline
        
        # Step 4: Show next steps
        Show-NextSteps
        
    } catch {
        Write-ColorOutput "[ERROR] Fatal error: $($_.Exception.Message)" "Red"
        Write-ColorOutput "[HELP] Check the error messages above for troubleshooting" "Yellow"
        exit 1
    }
}

# Execute main function
Main 