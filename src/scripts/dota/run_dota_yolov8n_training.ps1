# Ensure the script stops on errors
$ErrorActionPreference = "Stop"

# Step 1: Function to start the virtual environment
function Start-Venv {
    Write-Host "Starting YOLOv8 virtual environment for DOTA..."
    $env:VIRTUAL_ENV = "C:\Users\burak\OneDrive\Desktop\Git Repos\drone-obj-detection-lightweight-ai\venvs\yolov8n_dota_env"  # Path to your YOLOv8 virtual environment
    $env:Path = "$env:VIRTUAL_ENV\Scripts;$env:Path"
    
    # Check if the environment is activated correctly
    Write-Host "Activating virtual environment at: $env:VIRTUAL_ENV"
    & "$env:VIRTUAL_ENV\Scripts\Activate.ps1"
    Write-Host "Activated virtual environment: $env:VIRTUAL_ENV"
}

# Step 2: Function to clean up cache files
function Clean-Cache {
    Write-Host "Cleaning up cache files..."
    Remove-Item -Recurse -Force "..\data\my_dataset\dota\dota-v1.0\labels\train.cache" -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force "..\data\my_dataset\dota\dota-v1.0\labels\train.cache.npy" -ErrorAction SilentlyContinue
}

# Step 3: Function to train YOLOv8 on DOTA dataset
function Train-YOLOv8 {
    Write-Host "Starting YOLOv8n training on DOTA..."
    python ..\..\..\src\models\YOLOv8\train.py --img 1024 --batch 16 --epochs 50 --data ..\..\..\config\dota.yaml --weights yolov8n.pt --name yolo_dota_baseline
}

# Main script execution
function Run-Training {
    # Start virtual environment
    Start-Venv

    # Clean up any existing cache
    Clean-Cache

    # Train YOLOv8n model on DOTA dataset
    Train-YOLOv8
}

# Run the training process
Run-Training
