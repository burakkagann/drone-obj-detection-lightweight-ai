# MobileNet-SSD for VisDrone Object Detection

Basic instructions for the MobileNet-SSD model.

## Setup Instructions

1. **Create and activate virtual environment:**

```bash
# Create virtual environment
python -m venv venvs/mobilenet_ssd_env

# Activate (Windows PowerShell)
.\venvs\mobilenet_ssd_env\Scripts\Activate.ps1

# Activate (Windows CMD)
.\venvs\mobilenet_ssd_env\Scripts\activate.bat

# Install dependencies
pip install -r requirements/visdrone/requirements_mobilenet-ssd.txt
```

## Dataset Structure

The dataset should be organized as follows:
```
data/my_dataset/visdrone/
├── mobilenet-ssd/
│   └── voc_format/
│       ├── train/
│       └── val/
├── images/
│   ├── train/
│   └── val/
```

- `images/train/` and `images/val/`: Contains the original images
- `mobilenet-ssd/voc_format/train/` and `val/`: Contains XML annotations in VOC format

## Configuration

The model configuration is stored in `config/mobilenet_ssd_visdrone.yaml`:

```yaml
model:
  name: 'mobilenet_ssd'
  input_shape: [300, 300, 3]  # Input image size
  num_classes: 10             # Number of object classes
  pretrained: true           # Use pretrained MobileNet backbone
  batch_size: 16             # Training batch size
  epochs: 100
  learning_rate: 0.001

dataset:
  # Dataset paths and settings
  voc_format_dir: 'mobilenet-ssd/voc_format'

training:
  # Training settings like checkpoints, early stopping
  checkpoint_dir: 'checkpoints/mobilenet_ssd_visdrone'
  log_dir: 'logs/mobilenet_ssd_visdrone'
```

## Training

1. **Start Training:**

```bash
# Activate environment if not already activated
.\venvs\mobilenet_ssd_env\Scripts\Activate.ps1

# Run training script
python src/scripts/train_mobilenet_ssd.py
```

2. **Training Options:**

```bash
# Train with specific GPU
python src/scripts/train_mobilenet_ssd.py --gpu 0

# Use custom config
python src/scripts/train_mobilenet_ssd.py --config path/to/config.yaml
```

3. **Monitor Training:**
- Progress is displayed in console
- Logs are saved in `logs/mobilenet_ssd_visdrone`
- Model checkpoints in `checkpoints/mobilenet_ssd_visdrone`
- View with TensorBoard:
  ```bash
  tensorboard --logdir logs/mobilenet_ssd_visdrone
  ```

## Using the Model

1. **Load Model and Weights:**

```python
import tensorflow as tf
import yaml
from src.models.mobilenet_ssd import MobileNetSSD

# Load config
with open('config/mobilenet_ssd_visdrone.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize model
model = MobileNetSSD(config)
net = model.build_model()

# Load trained weights
net.load_weights('checkpoints/mobilenet_ssd_visdrone/model_final.h5')
```

2. **Perform Inference:**

```python
import cv2
import numpy as np

def preprocess_image(image_path, input_shape):
    """Preprocess image for inference."""
    image = cv2.imread(image_path)
    image = cv2.resize(image, input_shape[:2])
    image = image.astype(np.float32)
    image = (image - 127.5) * 0.007843  # Scale to [-1, 1]
    if image.shape[-1] == 3:  # Convert BGR to RGB
        image = image[..., ::-1]
    return image

def detect_objects(image_path, model, config):
    """Detect objects in an image."""
    # Preprocess image
    image = preprocess_image(image_path, config['model']['input_shape'])
    
    # Run inference
    cls_pred, reg_pred = model.predict(np.expand_dims(image, axis=0))
    
    # Post-process predictions
    # TODO: Implement non-maximum suppression and confidence thresholding
    return cls_pred, reg_pred

# Example usage
image_path = 'path/to/test_image.jpg'
cls_pred, reg_pred = detect_objects(image_path, net, config)
```

## Model Architecture

The MobileNet-SSD model consists of:
1. MobileNet backbone (pretrained on ImageNet)
2. SSD (Single Shot Detector) head with:
   - Multiple feature maps for detection
   - Classification branch for object class prediction
   - Regression branch for bounding box prediction

## Training Details

- Uses transfer learning with pretrained MobileNet backbone
- Early layers are frozen for faster training
- Trains on multiple feature maps for different scales
- Uses categorical cross-entropy for classification
- Uses Huber loss for bounding box regression
- Implements early stopping and learning rate reduction
- Saves best model based on validation loss

## Performance Considerations

- Batch size of 16 is recommended for 8GB GPU
- Reduce batch size if running out of memory
- Training time depends on dataset size and hardware
- CPU training is possible but significantly slower
- Consider using data augmentation for better results