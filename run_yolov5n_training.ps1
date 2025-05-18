# run_training.ps1

# Ensure the script stops on errors
$ErrorActionPreference = "Stop"

# Activate virtual environment
Write-Host "Activating YOLOv8 virtual environment..."
.\yolov5-env\Scripts\Activate.ps1

# Delete cache files before training
Write-Host "Cleaning up cache files..."
Remove-Item -Path "data/my_dataset/visdrone/labels/train.cache" -ErrorAction SilentlyContinue
Remove-Item -Path "data/my_dataset/visdrone/labels/train.cache.npy" -ErrorAction SilentlyContinue

# Run training
Write-Host "Starting YOLOv8n training..."
python src/models/YOLOv5/train.py --img 640 --batch 16 --epochs 5 --data config/my_dataset.yaml --weights yolov5n.pt --name yolo5n_baseline

# Deactivate the virtual environment after training completes
Write-Host "Training complete. Deactivating virtual environment..."
deactivate
