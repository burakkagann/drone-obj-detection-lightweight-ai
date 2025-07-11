# Ensure the script stops on errors
$ErrorActionPreference = "Stop"

# Step 1: Function to start virtual environment
function Start-Venv {
    Write-Host "Starting YOLOv5 virtual environment..."
    $env:VIRTUAL_ENV = "C:\Users\burak\OneDrive\Desktop\Git Repos\drone-obj-detection-lightweight-ai\venvs\yolov5n_env"  # Path to your YOLOv5 virtual environment
    $env:Path = "$env:VIRTUAL_ENV\Scripts;$env:Path"
    
    # Check if the environment is activated correctly
    Write-Host "Activating virtual environment at: $env:VIRTUAL_ENV"
    & "$env:VIRTUAL_ENV\Scripts\Activate.ps1"
    Write-Host "Activated virtual environment: $env:VIRTUAL_ENV"
}

# Step 3: Function to clean up cache files
function Clean-Cache {
    Write-Host "Cleaning up cache files..."
    Remove-Item -Recurse -Force "data\my_dataset\visdrone\labels\train.cache" -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force "data\my_dataset\visdrone\labels\train.cache.npy" -ErrorAction SilentlyContinue
}

# Step 4: Function to train YOLOv5
function Train-YOLOv5 {
    Write-Host "Starting YOLOv5n training with CUDA (RTX 3060 6GB optimized)..."
    
    # Change to the YOLOv5 directory
    Set-Location -Path "src\models\YOLOv5"
    
    # Run training optimized for RTX 3060 6GB VRAM
    python train.py `
        --img 640 `
        --batch 16 `
        --epochs 50 `
        --data ..\..\..\config\visdrone\yolov5n_v1\yolov5n_visdrone_config.yaml `
        --weights yolov5n.pt `
        --name yolo5n_visdrone_cuda `
        --patience 10 `
        --save-period 5 `
        --device 0 `
        --workers 4 `
        --project ..\..\..\runs\train `
        --exist-ok `
        --cache
    
    # Return to the original directory
    Set-Location -Path "..\..\.."
}

# Main script execution
function Run-Training {
    # Start virtual environment
    Start-Venv

    # Clean up any existing cache
    Clean-Cache

    # Train YOLOv5n model
    Train-YOLOv5
}

# Run the training process
Run-Training
