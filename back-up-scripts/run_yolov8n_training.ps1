# This script is designed to run YOLOv5n training on a Windows machine using PowerShell.
# Ensure the script stops on errors
$ErrorActionPreference = "Stop"

# Step 1: Activate the virtual environment
Write-Host "Activating YOLOv8 virtual environment..."
$env:VIRTUAL_ENV = "C:\path\to\yolov8-env"  # Update this with the path to your YOLOv8 virtual environment
$env:Path = "$env:VIRTUAL_ENV\Scripts;$env:Path"
& "$env:VIRTUAL_ENV\Scripts\Activate.ps1"

# Step 2: Clear any existing cache files in the dataset
Write-Host "Cleaning up cache files..."
Remove-Item -Recurse -Force "C:\Users\burak\OneDrive\Desktop\Git Repos\drone-obj-detection-lightweight-ai\data\my_dataset\visdrone\labels\train.cache" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "C:\Users\burak\OneDrive\Desktop\Git Repos\drone-obj-detection-lightweight-ai\data\my_dataset\visdrone\labels\val.cache" -ErrorAction SilentlyContinue

# Step 3: Run the YOLOv8 training command
Write-Host "Starting YOLOv8n training..."
yolo train model=yolov8n.yaml data=config/my_dataset.yaml epochs=5 batch=16 imgsz=640

# Step 4: Deactivate the virtual environment after training completes
Write-Host "Training complete. Deactivating virtual environment..."
deactivate
