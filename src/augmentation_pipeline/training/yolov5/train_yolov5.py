# YOLOv5 training script
# Ensure you have the official YOLOv5 repository cloned
# Place your dataset in the format: images/train, images/val, labels/train, labels/val

import os
os.system('git clone https://github.com/ultralytics/yolov5')
os.chdir('yolov5')
os.system('pip install -r requirements.txt')
os.system('python train.py --img 640 --batch 16 --epochs 50 --data ../data.yaml --weights yolov5n.pt')
