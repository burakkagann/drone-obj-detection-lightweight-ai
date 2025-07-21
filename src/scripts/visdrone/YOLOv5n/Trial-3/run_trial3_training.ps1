# YOLOv5n Trial-3 Training Script
# Based on successful Trial-2 results (23.557% mAP@0.5)
# Target: Push beyond 25% mAP@0.5 threshold for thesis excellence

# Ensure the script stops on errors
$ErrorActionPreference = "Stop"

# Function to check CUDA availability
function Test-CUDAAvailability {
    $pythonCmd = "import torch; print('0' if torch.cuda.is_available() else 'cpu')"
    $result = python -c "$pythonCmd"
    return $result.Trim()
}

# Configuration - Trial-3 optimization
$CONFIG = @{
    MODEL_NAME = "yolov5n"
    TRIAL_NAME = "trial3"
    DATASET = "visdrone"
    CONFIG_PATH = "../../../../../config/visdrone/yolov5n_v1/yolov5n_visdrone_config.yaml"
    MODEL_CONFIG_PATH = "../../../../../src/models/YOLOv5/models/yolov5n.yaml"
    HYP_CONFIG_PATH = "../../../../../config/visdrone/yolov5n_v1/hyp_visdrone_trial3.yaml"
    DATASET_PATH = "../../../../../data/my_dataset/visdrone"
    BATCH_SIZE = 20
    EPOCHS = 100
    IMAGE_SIZE = 640
    SAVE_PERIOD = 5
    PATIENCE = 15
    DEVICE = "0"
    WORKERS = 4
    QUICK_TEST = $false
}

# Step 1: Function to start virtual environment
function Start-Venv {
    Write-Host "Starting YOLOv5 virtual environment for Trial-3..."
    $env:VIRTUAL_ENV = ".\venvs\yolov5n_env"
    $env:Path = "$env:VIRTUAL_ENV\Scripts;$env:Path"
    
    Write-Host "Activating virtual environment at: $env:VIRTUAL_ENV"
    & "$env:VIRTUAL_ENV\Scripts\Activate.ps1"
    Write-Host "Activated virtual environment: $env:VIRTUAL_ENV"
}

# Step 2: Function to verify prerequisites
function Test-Prerequisites {
    Write-Host "Verifying Trial-3 prerequisites..."
    
    # Check if config files exist
    $requiredConfigs = @(
        $CONFIG.CONFIG_PATH,
        $CONFIG.MODEL_CONFIG_PATH,
        $CONFIG.HYP_CONFIG_PATH
    )
    
    foreach ($config in $requiredConfigs) {
        if (-not (Test-Path $config)) {
            throw "Configuration file not found at: $config"
        }
    }
    Write-Host "✓ All configuration files found"
    
    # Verify hyperparameter config is Trial-3
    $hypContent = Get-Content $CONFIG.HYP_CONFIG_PATH -Raw
    if ($hypContent -match "Trial-3") {
        Write-Host "✓ Using Trial-3 hyperparameters"
    } else {
        throw "Hyperparameter file is not the Trial-3 version"
    }
    
    # Check dataset directories
    $requiredPaths = @(
        "$($CONFIG.DATASET_PATH)/train/images",
        "$($CONFIG.DATASET_PATH)/val/images", 
        "$($CONFIG.DATASET_PATH)/train/labels",
        "$($CONFIG.DATASET_PATH)/val/labels"
    )
    
    foreach ($path in $requiredPaths) {
        if (-not (Test-Path $path)) {
            throw "Required dataset directory not found at: $path"
        }
    }
    Write-Host "✓ Dataset directories verified"
    
    # Check CUDA and set device
    Write-Host "Checking CUDA availability..."
    $deviceResult = Test-CUDAAvailability
    $CONFIG.DEVICE = $deviceResult
    Write-Host "✓ Using device: $($CONFIG.DEVICE)"
    
    Write-Host "All prerequisites verified successfully."
}

# Step 3: Enhanced cache cleanup
function Clean-Cache {
    Write-Host "Performing cache cleanup for Trial-3..."
    
    $cachePatterns = @("*.cache", "*.cache.npy", "*.cache.pkl")
    $directories = @(
        "$($CONFIG.DATASET_PATH)/labels/train",
        "$($CONFIG.DATASET_PATH)/labels/val",
        "$($CONFIG.DATASET_PATH)/labels"
    )
    
    foreach ($dir in $directories) {
        if (Test-Path $dir) {
            Write-Host "Cleaning cache in: $dir"
            foreach ($pattern in $cachePatterns) {
                $files = Get-ChildItem -Path $dir -Filter $pattern -ErrorAction SilentlyContinue
                foreach ($file in $files) {
                    try {
                        Remove-Item -Path $file.FullName -Force
                        Write-Host "Removed: $($file.Name)"
                    }
                    catch {
                        Write-Host "Warning: Could not remove $($file.Name)" -ForegroundColor Yellow
                    }
                }
            }
        }
    }
    Write-Host "Cache cleanup completed."
}

# Step 4: Trial-3 training function
function Train-Trial3-YOLOv5 {
    Write-Host "=== Starting Trial-3 Training ===" -ForegroundColor Green
    Write-Host "Baseline: 23.557% mAP@0.5 (Trial-2) | Target: >25% mAP@0.5"
    
    # Set training parameters based on quick test flag
    $epochs = if ($CONFIG.QUICK_TEST) { 20 } else { $CONFIG.EPOCHS }
    $runName = if ($CONFIG.QUICK_TEST) { 
        "yolov5n_trial3_quicktest" 
    } else { 
        "yolov5n_trial3_100epochs" 
    }
    
    Write-Host "Training Configuration:"
    Write-Host "  - Trial: $($CONFIG.TRIAL_NAME)"
    Write-Host "  - Epochs: $epochs"
    Write-Host "  - Batch Size: $($CONFIG.BATCH_SIZE)"
    Write-Host "  - Image Size: $($CONFIG.IMAGE_SIZE)"
    Write-Host "  - Device: $($CONFIG.DEVICE)"
    Write-Host "  - Hyperparameters: Trial-3 optimized"
    Write-Host "  - Quick Test: $($CONFIG.QUICK_TEST)"
    
    # Clean cache before training
    Clean-Cache
    
    # Set memory optimization for CUDA
    if ($CONFIG.DEVICE -eq "cuda") {
        $env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512"
    }
    
    # Navigate to YOLOv5 directory
    $yolov5Path = "../../../../../src/models/YOLOv5"
    if (-not (Test-Path $yolov5Path)) {
        throw "YOLOv5 directory not found at: $yolov5Path"
    }
    
    Write-Host "Navigating to YOLOv5 directory: $yolov5Path"
    Push-Location $yolov5Path
    
    try {
        # Execute training with Trial-3 hyperparameters
        $command = @(
            "python", "train.py",
            "--data", "../../../config/visdrone/yolov5n_v1/yolov5n_visdrone_config.yaml",
            "--cfg", "models/yolov5n.yaml",
            "--hyp", "../../../config/visdrone/yolov5n_v1/hyp_visdrone_trial3.yaml",
            "--epochs", "$epochs",
            "--batch-size", "$($CONFIG.BATCH_SIZE)",
            "--img-size", "$($CONFIG.IMAGE_SIZE)",
            "--device", "$($CONFIG.DEVICE)",
            "--workers", "$($CONFIG.WORKERS)",
            "--name", "$runName",
            "--save-period", "$($CONFIG.SAVE_PERIOD)",
            "--patience", "$($CONFIG.PATIENCE)",
            "--project", "../../../runs/train",
            "--exist-ok"
        )
        
        Write-Host "Executing command: $($command -join ' ')" -ForegroundColor Cyan
        & $command[0] $command[1..($command.Length-1)]
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "=== Trial-3 Training Completed Successfully ===" -ForegroundColor Green
            Write-Host "Results saved to: runs/train/$runName"
            
            # Quick validation summary
            $resultsPath = "../../../runs/train/$runName/results.csv"
            if (Test-Path $resultsPath) {
                Write-Host "Checking final mAP@0.5 results..." -ForegroundColor Yellow
                $lastLine = Get-Content $resultsPath | Select-Object -Last 1
                if ($lastLine -match "(\d+\.\d+).*?(\d+\.\d+).*?(\d+\.\d+)") {
                    Write-Host "Final training complete - check results.csv for mAP@0.5" -ForegroundColor Green
                }
            }
        } else {
            throw "Training failed with exit code: $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}

# Main execution function
function Run-Trial3-Training {
    try {
        Write-Host "=== YOLOv5n Trial-3 Training Pipeline ===" -ForegroundColor Magenta
        Write-Host "Baseline: 23.557% mAP@0.5 | Target: >25% mAP@0.5"
        
        # Verify prerequisites
        Test-Prerequisites
        
        # Execute Trial-3 training
        Train-Trial3-YOLOv5
        
        Write-Host "=== Trial-3 Training Pipeline Completed ===" -ForegroundColor Green
        Write-Host "Next steps:"
        Write-Host "1. Check results.csv for final mAP@0.5"
        Write-Host "2. Compare against Trial-2 baseline (23.557%)"
        Write-Host "3. Document results in Trial-3 folder"
        
    }
    catch {
        Write-Host "Error in Trial-3 Training: $_" -ForegroundColor Red
        exit 1
    }
}

# Execute Trial-3 training
Run-Trial3-Training