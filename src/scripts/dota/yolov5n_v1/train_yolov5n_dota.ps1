# PowerShell script for training YOLOv5n on DOTA v1.0 dataset

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
.\venvs\dota\venvs\yolov5n_dota_env\Scripts\Activate.ps1

# Install requirements
Write-Host "Installing requirements..." -ForegroundColor Green
pip install -r requirements/dota/requirements_yolov5n.txt

# Convert DOTA dataset to YOLOv5 format
Write-Host "Converting DOTA dataset to YOLOv5 format..." -ForegroundColor Green
python src/scripts/dota/yolov5n_v1/convert_dota_to_yolov5.py

# Start training
Write-Host "Starting YOLOv5n training..." -ForegroundColor Green
python src/scripts/dota/yolov5n_v1/train_yolov5n_dota.py

Write-Host "Training pipeline completed!" -ForegroundColor Green 