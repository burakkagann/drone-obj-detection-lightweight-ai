# Configuration Files

This directory contains YAML configuration files for various models and datasets.

## MobileNet-SSD VisDrone Configuration

`mobilenet_ssd_visdrone.yaml` contains configuration for training MobileNet-SSD on VisDrone dataset:

### Configuration Sections

#### Model Configuration
- Architecture parameters
- Training hyperparameters
- Input specifications
- Optimization settings

#### Dataset Configuration
- Dataset paths
- Class definitions
- Data split locations
- Annotation format

#### Training Configuration
- Checkpoint settings
- Logging parameters
- Early stopping criteria
- Learning rate scheduling

#### Preprocessing Configuration
- Image normalization
- Color space settings
- Mean and scale values

### Usage

```python
import yaml

with open('config/mobilenet_ssd_visdrone.yaml', 'r') as f:
    config = yaml.safe_load(f)
```

### Example Configuration Structure

```yaml
model:
  name: 'mobilenet_ssd'
  input_shape: [300, 300, 3]
  # ... other model settings

dataset:
  name: 'visdrone'
  root_dir: 'data/my_dataset/visdrone'
  # ... other dataset settings

training:
  checkpoint_dir: 'checkpoints/mobilenet_ssd_visdrone'
  # ... other training settings

preprocessing:
  mean: [127.5, 127.5, 127.5]
  # ... other preprocessing settings
```
