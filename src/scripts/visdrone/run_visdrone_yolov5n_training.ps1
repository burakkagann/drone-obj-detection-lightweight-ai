# Ensure the script stops on errors
$ErrorActionPreference = "Stop"

# Function to check CUDA availability
function Test-CUDAAvailability {
    $pythonCmd = "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"
    $result = python -c "$pythonCmd"
    return $result.Trim()
}

# Configuration
$CONFIG = @{
    MODEL_NAME = "yolov5n"
    DATASET = "visdrone"
    CONFIG_PATH = "config/visdrone/yolov5n_v1/visdrone_yolov5n.yaml"
    MODEL_CONFIG_PATH = "config/visdrone/yolov5n_v1/model_yolov5n.yaml"
    DATASET_PATH = "data/my_dataset/visdrone"
    BATCH_SIZE = 8  # Reduced from 16 for memory optimization
    EPOCHS = 100  # Increased for better convergence
    IMAGE_SIZE = 416  # Reduced from 640 for memory optimization
    SAVE_PERIOD = 5
    PATIENCE = 10
    DEVICE = "0"  # Use first CUDA device if available, otherwise CPU
    WORKERS = 2  # Reduced from 4 for memory optimization
}

# Step 1: Function to start virtual environment
function Start-Venv {
    Write-Host "Starting YOLOv5 virtual environment..."
    $env:VIRTUAL_ENV = ".\venvs\yolov5n_env"  # Relative path to virtual environment
    $env:Path = "$env:VIRTUAL_ENV\Scripts;$env:Path"
    
    # Check if the environment is activated correctly
    Write-Host "Activating virtual environment at: $env:VIRTUAL_ENV"
    & "$env:VIRTUAL_ENV\Scripts\Activate.ps1"
    Write-Host "Activated virtual environment: $env:VIRTUAL_ENV"
}

# Step 2: Function to verify dataset, config, and CUDA
function Test-Prerequisites {
    Write-Host "Verifying prerequisites..."
    
    # Check if config file exists
    if (-not (Test-Path $CONFIG.CONFIG_PATH)) {
        throw "Configuration file not found at: $($CONFIG.CONFIG_PATH)"
    }
    
    # Check if model config exists
    if (-not (Test-Path $CONFIG.MODEL_CONFIG_PATH)) {
        throw "Model configuration file not found at: $($CONFIG.MODEL_CONFIG_PATH)"
    }
    
    # Check if dataset directories exist
    $requiredPaths = @(
        "$($CONFIG.DATASET_PATH)/images/train",
        "$($CONFIG.DATASET_PATH)/images/val",
        "$($CONFIG.DATASET_PATH)/labels/train",
        "$($CONFIG.DATASET_PATH)/labels/val"
    )
    
    foreach ($path in $requiredPaths) {
        if (-not (Test-Path $path)) {
            throw "Required dataset directory not found at: $path"
        }
    }
    
    # Check CUDA availability and set device
    Write-Host "Checking CUDA availability..."
    $CONFIG.DEVICE = Test-CUDAAvailability
    Write-Host "Using device: $($CONFIG.DEVICE)"
    
    Write-Host "Prerequisites verified successfully."
}

# Step 3: Function to clean up cache files
function Clean-Cache {
    Write-Host "Cleaning up cache files..."
    
    # Define cache patterns to clean
    $cachePatterns = @(
        "*.cache",
        "*.cache.npy",
        "*.cache.pkl"
    )
    
    # Clean cache in train and val directories
    $directories = @(
        "$($CONFIG.DATASET_PATH)/labels/train",
        "$($CONFIG.DATASET_PATH)/labels/val",
        "$($CONFIG.DATASET_PATH)/labels"  # Also clean root labels directory
    )
    
    foreach ($dir in $directories) {
        if (Test-Path $dir) {
            Write-Host "Cleaning cache in: $dir"
            foreach ($pattern in $cachePatterns) {
                $files = Get-ChildItem -Path $dir -Filter $pattern -ErrorAction SilentlyContinue
                if ($files) {
                    foreach ($file in $files) {
                        try {
                            # Take ownership and grant full permissions before deleting
                            $acl = Get-Acl $file.FullName
                            $identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
                            $fileSystemRights = [System.Security.AccessControl.FileSystemRights]::FullControl
                            $type = [System.Security.AccessControl.AccessControlType]::Allow
                            $fileSystemAccessRule = New-Object System.Security.AccessControl.FileSystemAccessRule($identity, $fileSystemRights, $type)
                            $acl.SetAccessRule($fileSystemAccessRule)
                            Set-Acl $file.FullName $acl
                            
                            # Now try to delete the file
                            Remove-Item -Path $file.FullName -Force
                            Write-Host "Successfully removed: $($file.Name)"
                        }
                        catch {
                            Write-Host "Warning: Could not remove $($file.Name): $_" -ForegroundColor Yellow
                        }
                    }
                }
            }
        }
    }
    
    Write-Host "Cache cleanup completed."
}

# Step 4: Function to train YOLOv5
function Train-YOLOv5 {
    Write-Host "Starting $($CONFIG.MODEL_NAME) training on $($CONFIG.DATASET) dataset..."
    Write-Host "Using device: $($CONFIG.DEVICE)"
    
    # Clean cache before training
    Write-Host "Performing pre-training cache cleanup..."
    Clean-Cache
    
    # Ensure we're in the correct directory
    Set-Location -Path "src/models/YOLOv5"
    
    # Set environment variables to limit memory usage
    if ($CONFIG.DEVICE -eq "cuda") {
        $env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512"
    }
    
    # Build the command as a string first
    $trainCmd = "python train.py " + `
        "--img $($CONFIG.IMAGE_SIZE) " + `
        "--batch $($CONFIG.BATCH_SIZE) " + `
        "--epochs $($CONFIG.EPOCHS) " + `
        "--data '../../../$($CONFIG.CONFIG_PATH)' " + `
        "--cfg '../../../$($CONFIG.MODEL_CONFIG_PATH)' " + `
        "--weights '$($CONFIG.MODEL_NAME).pt' " + `
        "--name '$($CONFIG.MODEL_NAME)_$($CONFIG.DATASET)' " + `
        "--patience $($CONFIG.PATIENCE) " + `
        "--save-period $($CONFIG.SAVE_PERIOD) " + `
        "--device 0 " + `  # Explicitly use first GPU
        "--workers $($CONFIG.WORKERS) " + `
        "--project '../../../runs/train' " + `
        "--exist-ok " + `
        "--cache False"
    
    # Execute the command
    Invoke-Expression $trainCmd
        
    # Return to original directory
    Set-Location -Path "../../.."
}

# Main script execution
function Run-Training {
    try {
        Write-Host "=== Starting YOLOv5 Training Pipeline ==="
        
        # Start virtual environment
        Start-Venv
        
        # Verify prerequisites
        Test-Prerequisites
        
        # Train YOLOv5n model
        Train-YOLOv5
        
        Write-Host "=== Training Pipeline Completed Successfully ==="
    }
    catch {
        Write-Host "Error: $_" -ForegroundColor Red
        exit 1
    }
}

# Run the training process
Run-Training
