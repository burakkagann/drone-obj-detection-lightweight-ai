# Data Preparation

This directory contains data loaders and preprocessing utilities for various datasets.

## VisDrone Data Loader

`visdrone_loader.py` implements a data loader for the VisDrone dataset:

- Supports VOC format annotations
- Implements efficient data loading using tf.data API
- Handles image preprocessing for MobileNet-SSD
- Supports train/val/test splits

### Features

- Dynamic batch loading
- Parallel data processing
- Memory-efficient data pipeline
- Automatic data augmentation
- Support for multiple image formats (jpg, png)

### Usage

```python
from data_preparation.visdrone_loader import VisDroneDataLoader

# Initialize data loader with config
data_loader = VisDroneDataLoader(config)

# Load training dataset
train_dataset = data_loader.load_dataset('train')

# Load validation dataset
val_dataset = data_loader.load_dataset('val')
```

### Expected Data Structure

```
data/my_dataset/visdrone/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── voc_format/
    ├── train/
    ├── val/
    └── test/
```

### Preprocessing

- Resizes images to 300x300
- Normalizes pixel values
- Converts to BGR color space (if specified)
- Applies mean subtraction and scaling 