# Models

This directory contains model implementations for object detection.

## MobileNet-SSD

`mobilenet_ssd.py` implements the MobileNet-SSD architecture for efficient object detection:

- Uses MobileNet as the backbone network
- Implements SSD (Single Shot Detector) architecture
- Features multi-scale detection with feature pyramid
- Includes custom loss functions and metrics
- Supports transfer learning from ImageNet weights

### Architecture Details

- Input Shape: 300x300x3
- Backbone: MobileNet (pretrained on ImageNet)
- Feature Pyramid: Multiple detection heads at different scales
- Output: Classification and regression heads for object detection

### Usage

```python
from models.mobilenet_ssd import MobileNetSSD

# Initialize model with config
model = MobileNetSSD(config)

# Build model architecture
net = model.build_model()

# Get loss functions and metrics
losses = model.get_loss()
metrics = model.get_metrics()
```
