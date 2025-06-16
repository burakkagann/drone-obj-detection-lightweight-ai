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
    Remove-Item -Recurse -Force "..\data\my_dataset\visdrone\labels\train.cache" -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force "..\data\my_dataset\visdrone\labels\train.cache.npy" -ErrorAction SilentlyContinue
}

# Step 4: Function to train YOLOv5
function Train-YOLOv5 {
    Write-Host "Starting YOLOv5n training..."
    python ..\..\..\src\models\YOLOv5\train.py `
        --img 640 `
        --batch 16 `
        --epochs 50 `
        --data ..\..\..\config\my_dataset.yaml `
        --weights yolov5n.pt `
        --name yolo5n_baseline `
        --patience 10 `
        --save-period 5 `
        --device cpu `
        --workers 4 `
        --project runs/train `
        --exist-ok
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
