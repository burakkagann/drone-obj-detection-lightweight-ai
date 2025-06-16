# Scripts

This directory contains training and utility scripts for the project.

## MobileNet-SSD Training Script

`train_mobilenet_ssd.py` is the main training script for MobileNet-SSD on VisDrone dataset.

### Features

- Command-line argument parsing
- GPU/CPU device selection
- TensorBoard integration
- Checkpoint management
- Early stopping and learning rate scheduling

### Usage

```bash
python train_mobilenet_ssd.py --config path/to/config.yaml --gpu 0
```

### Arguments

- `--config`: Path to configuration file (default: config/mobilenet_ssd_visdrone.yaml)
- `--gpu`: GPU device ID (default: 0, use -1 for CPU)

### Training Features

- Automatic checkpoint saving
- TensorBoard logging
- Early stopping on validation loss
- Learning rate reduction on plateau
- Progress bar with metrics

### Output Structure

```
├── checkpoints/
│   └── mobilenet_ssd_visdrone/
│       ├── model_final.h5
│       └── model_{epoch}_{val_loss}.h5
└── logs/
    └── mobilenet_ssd_visdrone/
        └── {timestamp}/
``` 