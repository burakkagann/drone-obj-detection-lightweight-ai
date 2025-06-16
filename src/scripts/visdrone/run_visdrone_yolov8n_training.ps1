# Ensure the script stops on errors
$ErrorActionPreference = "Stop"

# Step 1: Function to start virtual environment
function Start-Venv {
    Write-Host "Starting YOLOv8 virtual environment..."
    $env:VIRTUAL_ENV = "C:\Users\burak\OneDrive\Desktop\Git Repos\drone-obj-detection-lightweight-ai\venvs\yolov8n_env"  # Path to your YOLOv8 virtual environment
    $env:Path = "$env:VIRTUAL_ENV\Scripts;$env:Path"
    & "$env:VIRTUAL_ENV\Scripts\Activate.ps1"
    Write-Host "Activated virtual environment: $env:VIRTUAL_ENV"
}

# Step 2: Function to clean up cache files
function Clean-Cache {
    Write-Host "Cleaning up cache files..."
    Remove-Item -Recurse -Force "..\data\my_dataset\visdrone\labels\train.cache" -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force "..\data\my_dataset\visdrone\labels\train.cache.npy" -ErrorAction SilentlyContinue
}

# Step 3: Function to train YOLOv8
function Train-YOLOv8 {
    Write-Host "Starting YOLOv8n training..."
    yolo train `
        model=yolov8n.pt `
        data="C:\Users\burak\OneDrive\Desktop\Git Repos\drone-obj-detection-lightweight-ai\config\my_dataset.yaml" `
        epochs=50 `
        batch=32 `
        imgsz=640 `
        name=yolov8n_baseline `
        patience=10 `
        save_period=5 `
        device=cpu `
        workers=4 `
        project=runs/train `
        exist_ok
}


# Main script execution
function Run-Training {
    # Start virtual environment
    Start-Venv

    # Clean up any existing cache
    Clean-Cache

    # Train YOLOv8n model
    Train-YOLOv8
}

# Run the training process
Run-Training
