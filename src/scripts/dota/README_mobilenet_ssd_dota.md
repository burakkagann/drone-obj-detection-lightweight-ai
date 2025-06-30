# Training MobileNet-SSD on DOTA Dataset

This guide explains how to train MobileNet-SSD on the DOTA (Dataset for Object deTection in Aerial images) dataset.

## Dataset Preparation

1. Download the DOTA dataset from the official website:
   - Visit [DOTA Dataset Website](https://captain-whu.github.io/DOTA/dataset.html)
   - Download the following files:
     - Training set: `DOTA-v1.0_train.zip`
     - Validation set: `DOTA-v1.0_val.zip`
     - (Optional) Test set: `DOTA-v1.0_test.zip`

2. Extract the downloaded files and organize them in the following structure:
```
data/my_dataset/dota/dota-v1.0/
├── images/
│   ├── train/
│   │   └── [training images]
│   └── val/
│       └── [validation images]
├── labels/
│   ├── train_original/
│   │   └── [training labels]
│   └── val_original/
│       └── [validation labels]
└── mobilenet-ssd/
    ├── train/
    │   └── Annotations/
    └── val/
        └── Annotations/
```

3. Convert DOTA format to VOC format:
```bash
python src/scripts/dota/convert_dota_to_mobilenet_ssd.py
```

## Training

1. Create and activate virtual environment:
```bash
python -m venv venvs/dota/venvs/mobilenet_ssd_dota_env
source venvs/dota/venvs/mobilenet_ssd_dota_env/Scripts/activate  # Linux/Mac
.\venvs\dota\venvs\mobilenet_ssd_dota_env\Scripts\Activate.ps1   # Windows
```

2. Install requirements:
```bash
pip install -r requirements/dota/requirements_mobilenet_ssd.txt
```

3. Start training:
```bash
python src/scripts/dota/train_mobilenet_ssd_dota.py
```

## Model Architecture

The MobileNet-SSD model uses:
- MobileNetV2 as the backbone network
- SSD (Single Shot Detector) for object detection
- Feature Pyramid Network for multi-scale detection
- Focal Loss for handling class imbalance

## Training Configuration

The default training configuration:
- Input size: 300x300
- Batch size: 32
- Learning rate: 0.001
- Number of epochs: 100
- Optimizer: Adam
- Learning rate scheduler: StepLR (step_size=30, gamma=0.1)

## Results

After training, you can find:
- Best model weights: `checkpoints/mobilenet_ssd_dota_best.pth`
- Checkpoints (every 10 epochs): `checkpoints/mobilenet_ssd_dota_epoch_*.pth`

## DOTA Classes

The model is trained to detect the following 15 classes:
1. plane
2. ship
3. storage-tank
4. baseball-diamond
5. tennis-court
6. basketball-court
7. ground-track-field
8. harbor
9. bridge
10. large-vehicle
11. small-vehicle
12. helicopter
13. roundabout
14. soccer-ball-field
15. swimming-pool

## Monitoring Training

The training script outputs the following metrics for each epoch:
- Training Loss
- Validation Loss

## Notes

- The training process includes data augmentation techniques suitable for aerial imagery
- The model architecture is MobileNetV2-SSD Lite, which provides a good balance between accuracy and inference speed
- Checkpoints are saved regularly to allow training resumption if needed 