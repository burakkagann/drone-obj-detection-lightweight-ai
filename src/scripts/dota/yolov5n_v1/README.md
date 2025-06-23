# YOLOv5n Training Pipeline for DOTA v1.0

This directory contains scripts and configurations for training YOLOv5n on the DOTA v1.0 dataset.

## Directory Structure

```
├── config/dota/yolov5n_v1/           # Configuration files
│   └── dota_v1.yaml                  # Dataset and model configuration
├── data/my_dataset/dota/dota-v1.0/   # Dataset directory
│   ├── images/                       # Original images
│   ├── labels/                       # Original annotations
│   └── yolov5n/                      # Converted YOLOv5 format data
└── src/scripts/dota/yolov5n_v1/      # Training scripts
    ├── convert_dota_to_yolov5.py     # Dataset conversion script
    ├── train_yolov5n_dota.py         # Training script
    └── train_yolov5n_dota.ps1        # PowerShell training pipeline
```

## Prerequisites

1. Python 3.8 or later
2. CUDA-capable GPU (recommended)
3. Virtual environment setup:
   ```powershell
   .\venvs\dota\venvs\yolov5n_dota_env\Scripts\Activate.ps1
   ```

## Dataset Preparation

1. Ensure DOTA v1.0 dataset is downloaded and organized in the following structure:
   ```
   data/my_dataset/dota/dota-v1.0/
   ├── images/
   │   ├── train/
   │   ├── val/
   │   └── test/
   └── labels/
       ├── train_original/
       ├── val_original/
       └── test_original/
   ```

2. Run the conversion script to prepare data for YOLOv5:
   ```powershell
   python src/scripts/dota/yolov5n_v1/convert_dota_to_yolov5.py
   ```

## Training

1. Review and adjust the configuration in `config/dota/yolov5n_v1/dota_v1.yaml`
2. Run the training pipeline:
   ```powershell
   .\src\scripts\dota\yolov5n_v1\train_yolov5n_dota.ps1
   ```

## Model Configuration

The YOLOv5n model is configured with:
- Input size: 640x640
- Batch size: 16
- Number of epochs: 100
- Number of classes: 15 (DOTA v1.0 classes)

## Output

Training outputs will be saved in:
- Checkpoints: `runs/train/yolov5n_dota_v1/weights/`
- Training logs: `runs/train/yolov5n_dota_v1/`

## Monitoring

Training progress can be monitored through:
- Terminal output
- TensorBoard logs in `runs/train/yolov5n_dota_v1/`

## References

- [DOTA Dataset](https://captain-whu.github.io/DOTA/)
- [YOLOv5 Repository](https://github.com/ultralytics/yolov5)
- [Ultralytics Documentation](https://docs.ultralytics.com/) 