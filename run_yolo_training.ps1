# run_training.ps1

# Activate virtual environment
.\yolov5-env\Scripts\Activate.ps1

# Delete cache files before training
Remove-Item -Path "data/my_dataset/visdrone/labels/train.cache" -ErrorAction SilentlyContinue
Remove-Item -Path "data/my_dataset/visdrone/labels/train.cache.npy" -ErrorAction SilentlyContinue

# Run training
python src/models/YOLOv5/train.py --img 640 --batch 16 --epochs 5 --data config/my_dataset.yaml --weights yolov5n.pt --name yolo5n_baseline
