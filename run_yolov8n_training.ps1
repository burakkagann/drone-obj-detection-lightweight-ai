# Ensure the script stops on errors
$ErrorActionPreference = "Stop"

# Activate virtual environment for YOLOv8n
Write-Host "Activating YOLOv8n virtual environment..."
.\yolov8n_env\Scripts\Activate.ps1

# Delete cache files before training
Write-Host "Cleaning up cache files..."
Remove-Item -Path "data/my_dataset/visdrone/labels/train.cache" -ErrorAction SilentlyContinue
Remove-Item -Path "data/my_dataset/visdrone/labels/train.cache.npy" -ErrorAction SilentlyContinue

# Run YOLOv8n training
Write-Host "Starting YOLOv8n training..."
python src/models/YOLOv8/train.py --img 640 --batch 16 --epochs 5 --data config/my_dataset.yaml --weights yolov8n.pt --name yolov8n_baseline

# Deactivate the virtual environment after training completes
Write-Host "Training complete. Deactivating virtual environment..."
deactivate
