# YOLOv8 training script
# Requires ultralytics package installed via pip

from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Load pre-trained model
model.train(data='../data.yaml', epochs=50, imgsz=640, batch=16)
