# YOLOv5n Training on VisDrone Dataset

## Overview
This document outlines the process and structure for training the YOLOv5n object detection model on the VisDrone dataset within this project. The workflow is designed for reproducibility, clarity, and ease of experimentation, and is suitable for academic and research purposes.

## Project Structure
- **src/scripts/visdrone/**: Contains PowerShell scripts for automating the training process for different YOLO models.
- **venvs/**: Contains Python virtual environments for isolated dependency management for each model.
- **requirements/visdrone/**: Contains requirements files for different models and frameworks used in VisDrone experiments.
- **data/my_dataset/visdrone/**: Directory for VisDrone dataset images and labels (not included in repo).
- **config/my_dataset.yaml**: Dataset configuration file for YOLO training.

## Training Pipeline
### 1. Environment Setup
- Each model (YOLOv5n, YOLOv8n, etc.) uses its own Python virtual environment for dependency isolation.
- The environment is activated at the start of each training script to ensure correct package versions are used.

### 2. Dependency Management
- All required Python packages are listed in `requirements/visdrone/requirements_yolov5n.txt`.
- Install dependencies with:
  ```powershell
  pip install -r ../../../requirements/visdrone/requirements_yolov5n.txt
  ```

### 3. Data Preparation
- The VisDrone dataset should be placed under `data/my_dataset/visdrone/` with the correct structure for YOLO training (images and labels in appropriate subfolders).
- The dataset configuration YAML (`config/my_dataset.yaml`) specifies paths and class information for training.

### 4. Training Script
- The main script for YOLOv5n training is `run_visdrone_yolov5n_training.ps1`.
- Key steps in the script:
  1. **Activate the virtual environment**
  2. **Clean up any existing cache files** to avoid stale data
  3. **Run the YOLOv5 training script** with standardized parameters

#### Example Command (from script):
```powershell
python ../../../src/models/YOLOv5/train.py `
    --img 640 `
    --batch 16 `
    --epochs 50 `
    --data ../../../config/my_dataset.yaml `
    --weights yolov5n.pt `
    --name yolo5n_baseline `
    --patience 10 `
    --save-period 5 `
    --device cpu `
    --workers 4 `
    --project runs/train `
    --exist-ok
```

### 5. Training Parameters
- **Image size**: 640x640
- **Batch size**: 16 (adjusted for CPU training)
- **Epochs**: 50 (with early stopping if no improvement for 10 epochs)
- **Checkpoints**: Saved every 5 epochs
- **Device**: CPU (can be set to GPU if available)
- **Output**: Results and checkpoints are saved in `runs/train/yolo5n_baseline/`

### 6. Automation and Reproducibility
- The PowerShell script ensures that each run is clean (removes old cache), uses the correct environment, and applies consistent parameters.
- This structure supports reproducible experiments and easy parameter tuning.

## What Has Been Done So Far
- **Scripts created** for YOLOv5n and YOLOv8n training automation.
- **Standardized training parameters** across models for fair comparison.
- **Cache management** added to avoid data leakage or stale results.
- **Tested the YOLOv5n training script** on CPU, confirmed it runs and saves outputs as expected.
- **Requirements files** curated for each model to ensure correct dependencies.

## Next Steps
- Test and validate the YOLOv8n training script using a similar process.
- Analyze and compare results between YOLOv5n and YOLOv8n.
- Document findings and integrate insights into the thesis.

## References
- [YOLOv5 GitHub](https://github.com/ultralytics/yolov5)
- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)

---
This README serves as a reference for the training pipeline and can be adapted for thesis documentation or further research reports. 