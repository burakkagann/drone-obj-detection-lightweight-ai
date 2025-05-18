# Ensure the script stops on errors
$ErrorActionPreference = "Stop"

# Step 1: Function to activate virtual environment
function Activate-Venv {
    Write-Host "Activating YOLOv5 virtual environment..."
    $env:VIRTUAL_ENV = "yolov5-env"  # Path to your YOLOv5 virtual environment
    $env:Path = "$env:VIRTUAL_ENV\Scripts;$env:Path"
    & "$env:VIRTUAL_ENV\Scripts\Activate.ps1"
}

# Step 2: Function to deactivate virtual environment
function Deactivate-Venv {
    Write-Host "Deactivating virtual environment..."
    deactivate
}

# Step 3: Function to clean up cache files
function Clean-Cache {
    Write-Host "Cleaning up cache files..."
    Remove-Item -Recurse -Force "C:\Users\burak\OneDrive\Desktop\Git Repos\drone-obj-detection-lightweight-ai\data\my_dataset\visdrone\labels\train.cache" -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force "C:\Users\burak\OneDrive\Desktop\Git Repos\drone-obj-detection-lightweight-ai\data\my_dataset\visdrone\labels\val.cache" -ErrorAction SilentlyContinue
}

# Step 4: Function to run YOLOv5 training
function Train-YOLOv5 {
    Write-Host "Starting YOLOv5n training..."
    python src\models\YOLOv5\train.py --img 640 --batch 16 --epochs 5 --data config/my_dataset.yaml --weights yolov5n.pt --name yolo5n_baseline
}

# Step 5: Function to export YOLOv5 model
function Export-YOLOv5 {
    Write-Host "Exporting YOLOv5n model to ONNX, TensorFlow, TorchScript..."
    python src\models\YOLOv5\export.py --weights runs/train/yolo5n_baseline/weights/best.pt --img 640 --batch 16 --include onnx,torchscript,tensorflow
}

# Main script execution
function Run-Training {
    # Activate virtual environment
    Activate-Venv

    # Clean up any existing cache
    Clean-Cache

    # Train YOLOv5n model
    Train-YOLOv5

    # Export YOLOv5 model
    Export-YOLOv5

    # Deactivate virtual environment
    Deactivate-Venv
}

# Run the training and export process
Run-Training
