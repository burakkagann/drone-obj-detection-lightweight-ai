# MobileNet-SSD Training on VisDrone Dataset

## Overview
This document details the process and structure for training the MobileNet-SSD object detection model on the VisDrone dataset. MobileNet-SSD is a lightweight, efficient model suitable for real-time object detection on resource-constrained devices.

## Project Structure
- **src/models/MobileNet-SSD/**: Contains the MobileNet-SSD model implementation and training scripts
- **src/scripts/visdrone/**: Contains PowerShell scripts for training automation
- **venvs/**: Python virtual environments for dependency management
- **requirements/visdrone/**: Requirements files for different models
- **data/my_dataset/visdrone/**: Directory for VisDrone dataset

## Model Architecture
MobileNet-SSD combines:
- MobileNet as the base network (lightweight CNN)
- Single Shot Detector (SSD) for object detection
- Key features:
  - Efficient for real-time detection
  - Suitable for resource-constrained devices
  - Good balance between speed and accuracy

## Training Pipeline
### 1. Environment Setup
```powershell
.\venvs\mobilenet_ssd_env\Scripts\Activate.ps1
```

### 2. Dependency Management
Install required packages:
```powershell
pip install -r requirements/visdrone/requirements_mobilenet-ssd.txt
```

### 3. Data Preparation
- Convert VisDrone dataset to VOC format
- Generate LMDB files for training
- Update class mappings in prototxt files

### 4. Training Process
1. **Data Conversion**
   - Convert VisDrone annotations to VOC format
   - Generate train/val splits
   - Create LMDB databases

2. **Model Configuration**
   - Update `train.prototxt` with VisDrone classes
   - Configure `solver_train.prototxt` for training parameters
   - Set appropriate learning rates and batch sizes

3. **Training**
   - Use pre-trained MobileNet weights
   - Fine-tune on VisDrone dataset
   - Monitor training progress and metrics

### 5. Training Parameters
- **Base Learning Rate**: 0.001
- **Batch Size**: 32
- **Max Iterations**: 120,000
- **Weight Decay**: 0.0005
- **Momentum**: 0.9
- **Input Size**: 300x300

### 6. Evaluation
- Use VisDrone validation set
- Calculate mAP (mean Average Precision)
- Monitor detection performance per class

## Implementation Details
1. **Data Format Conversion**
   - Convert VisDrone annotations to VOC format
   - Handle class mapping
   - Generate train/val splits

2. **Model Adaptation**
   - Update number of classes
   - Modify anchor boxes for VisDrone objects
   - Adjust network parameters

3. **Training Script**
   - Automated environment setup
   - Data preparation
   - Training execution
   - Progress monitoring

## References
- [MobileNet-SSD Documentation](https://docs.openvino.ai/2023.3/omz_models_model_mobilenet_ssd.html)
- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)
- [Practical Aspects of Model Selection](https://vdeepvision.medium.com/practical-aspects-to-select-a-model-for-object-detection-c704055ab325)

## Next Steps
1. Implement data conversion scripts
2. Create training automation script
3. Set up evaluation pipeline
4. Document performance metrics

---
This README serves as a reference for the MobileNet-SSD training pipeline and can be adapted for thesis documentation or further research reports. 