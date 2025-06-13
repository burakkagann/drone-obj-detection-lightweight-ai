# YOLOv8n Training on VisDrone Dataset

## Overview
This document details the process and structure for training the YOLOv8n object detection model on the VisDrone dataset within this project. The workflow is designed for reproducibility, clarity, and ease of experimentation, making it suitable for academic research and thesis documentation.

## Project Structure
- **src/scripts/visdrone/**: Contains PowerShell scripts for automating the training process for YOLO models.
- **venvs/**: Python virtual environments for isolated dependency management for each model.
- **requirements/visdrone/**: Requirements files for different models and frameworks used in VisDrone experiments.
- **data/my_dataset/visdrone/**: Directory for VisDrone dataset images and labels (not included in repo).
- **config/my_dataset.yaml**: Dataset configuration file for YOLO training.

## Training Pipeline
### 1. Environment Setup
- Each model (YOLOv5n, YOLOv8n, etc.) uses its own Python virtual environment for dependency isolation.
- The environment is activated at the start of each training script to ensure correct package versions are used.

### 2. Dependency Management
- All required Python packages are listed in `requirements/visdrone/requirements_yolov8n.txt`.
- Install dependencies with:
  ```powershell
  pip install -r requirements/visdrone/requirements_yolov8n.txt
  ```

### 3. Data Preparation
- The VisDrone dataset should be placed under `data/my_dataset/visdrone/` with the correct structure for YOLO training (images and labels in appropriate subfolders).
- The dataset configuration YAML (`config/my_dataset.yaml`) specifies paths and class information for training.

### 4. Training Script
- The main script for YOLOv8n training is `run_visdrone_yolov8n_training.ps1`.
- Key steps in the script:
  1. **Activate the virtual environment**
  2. **Clean up any existing cache files** to avoid stale data
  3. **Run the YOLOv8 training command** with standardized parameters

#### Example Command (from script):
```powershell
yolo train `
    model=yolov8n.pt `
    data="C:\Users\burak\OneDrive\Desktop\Git Repos\drone-obj-detection-lightweight-ai\config\my_dataset.yaml" `
    epochs=50 `
    batch=32 `
    imgsz=640 `
    name=yolov8n_baseline `
    patience=10 `
    save_period=5 `
    device=cpu `
    workers=4 `
    project=runs/train `
    exist_ok
```

### 5. Training Parameters
- **Image size**: 640x640
- **Batch size**: 32 (adjusted for CPU or GPU)
- **Epochs**: 50 (with early stopping if no improvement for 10 epochs)
- **Checkpoints**: Saved every 5 epochs
- **Device**: CPU (can be set to GPU if available)
- **Output**: Results and checkpoints are saved in `runs/train/yolov8n_baseline/`

### 6. Automation and Reproducibility
- The PowerShell script ensures that each run is clean (removes old cache), uses the correct environment, and applies consistent parameters.
- This structure supports reproducible experiments and easy parameter tuning.

## What Has Been Done So Far
- **Script created** for YOLOv8n training automation.
- **Standardized training parameters** across models for fair comparison.
- **Cache management** added to avoid data leakage or stale results.
- **Tested the YOLOv8n training script** on CPU, confirmed it runs and saves outputs as expected.
- **Requirements file** curated for YOLOv8n to ensure correct dependencies.

## Next Steps
- Analyze and compare results between YOLOv5n and YOLOv8n.
- Document findings and integrate insights into the thesis.

## References
- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)

---
This README serves as a reference for the YOLOv8n training pipeline and can be adapted for thesis documentation or further research reports. 