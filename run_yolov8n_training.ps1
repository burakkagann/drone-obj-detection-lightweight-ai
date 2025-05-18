# Ensure the script stops on errors
$ErrorActionPreference = "Stop"

# Step 1: Function to activate virtual environment
function Activate_Venv {
    Write-Host "Activating YOLOv8 virtual environment..."
    $env:VIRTUAL_ENV = "yolov8-env"  # Path to your YOLOv8 virtual environment
    $env:Path = "$env:VIRTUAL_ENV\Scripts;$env:Path"
    & "$env:VIRTUAL_ENV\Scripts\Activate.ps1"
}

# Step 2: Function to deactivate virtual environment
function Deactivate_Venv {
    Write-Host "Deactivating virtual environment..."
    deactivate
}

# Step 3: Function to clean up cache files
function Clean_Cache {
    Write-Host "Cleaning up cache files..."
    Remove-Item -Recurse -Force "C:\Users\burak\OneDrive\Desktop\Git Repos\drone-obj-detection-lightweight-ai\data\my_dataset\visdrone\labels\train.cache" -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force "C:\Users\burak\OneDrive\Desktop\Git Repos\drone-obj-detection-lightweight-ai\data\my_dataset\visdrone\labels\val.cache" -ErrorAction SilentlyContinue
}

# Step 4: Function to run YOLOv8 training
function Train_YOLOv8 {
    Write-Host "Starting YOLOv8n training..."
    yolo train model=yolov8n.yaml data=config/my_dataset.yaml epochs=5 batch=16 imgsz=640
}

# Step 5: Function to export YOLOv8 model
function Export_YOLOv8 {
    Write-Host "Exporting YOLOv8n model to ONNX, TensorFlow, TorchScript..."
    yolo export model=runs/train/yolo8n_baseline/weights/best.pt format=onnx
    yolo export model=runs/train/yolo8n_baseline/weights/best.pt format=tensorflow
    yolo export model=runs/train/yolo8n_baseline/weights/best.pt format=torchscript
}

# Main script execution
function Run_Training {
    # Activate virtual environment
    Activate-Venv

    # Clean up any existing cache
    Clean-Cache

    # Train YOLOv8n model
    Train-YOLOv8

    # Export YOLOv8 model
    Export-YOLOv8

    # Deactivate virtual environment
    Deactivate-Venv
}

# Run the training and export process
Run-Training
