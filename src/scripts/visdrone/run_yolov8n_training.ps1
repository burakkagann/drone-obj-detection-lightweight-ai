# Ensure the script stops on errors
$ErrorActionPreference = "Stop"

# Step 1: Function to start virtual environment
function Start-Venv {
    Write-Host "Starting YOLOv8 virtual environment..."
    $env:VIRTUAL_ENV = "C:\Users\burak\OneDrive\Desktop\Git Repos\drone-obj-detection-lightweight-ai\venvs\yolov8n_env"  # Path to your YOLOv8 virtual environment
    $env:Path = "$env:VIRTUAL_ENV\Scripts;$env:Path"
    & "$env:VIRTUAL_ENV\Scripts\Activate.ps1"
}

# Step 2: Function to stop virtual environment
function Stop-Venv {
    Write-Host "Stopping virtual environment..."
    deactivate
}

# Step 3: Function to clean up cache files
function Clean-Cache {
    Write-Host "Cleaning up cache files..."
    Remove-Item -Recurse -Force "..\data\my_dataset\visdrone\labels\train.cache" -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force "..\data\my_dataset\visdrone\labels\train.cache.npy" -ErrorAction SilentlyContinue
}

# Step 4: Function to train YOLOv8
function Train-YOLOv8 {
    Write-Host "Starting YOLOv8n training..."
    python ..\src\models\YOLOv8\train.py --img 640 --batch 16 --epochs 5 --data ..\config\my_dataset.yaml --weights yolov8n.yaml --name yolo8n_baseline
}

# Main script execution
function Run-Training {
    # Start virtual environment
    Start-Venv

    # Clean up any existing cache
    Clean-Cache

    # Train YOLOv8n model
    Train-YOLOv8

    # Stop virtual environment
    Stop-Venv
}

# Run the training process
Run-Training
