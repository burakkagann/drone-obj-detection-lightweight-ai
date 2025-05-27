# Augmentation Pipeline

This project implements a modular image augmentation and training pipeline for object detection using datasets like VisDrone, DOTA, and CIFAR.

## 📁 Folder Structure Overview

- **data/**: Place your raw and augmented image datasets here.
- **augment/**: Contains custom scripts to simulate fog, night, and sensor distortion effects.
- **utils/**: Utility functions for histogram visualization and augmentation evaluation using SSIM and PSNR.
- **training/**: YOLOv5, YOLOv8, and MobileNet-SSD training scripts.
- **evaluation/**: Compare baseline vs. augmented model performance.
- **main.py**: Run augmentations and evaluate visually/statistically.
- **requirements.txt**: All required dependencies for this project.

## 🧪 Augmentations
- `fog.py`: Simulate fog with adjustable opacity and depth.
- `night.py`: Apply gamma correction for night-time simulation.
- `sensor_distortions.py`: Add blur, noise, and chromatic aberration.

## 📊 Evaluation
Use SSIM, PSNR, and histogram comparisons to validate realism and impact of augmentations.

## 🧠 Training
Train and evaluate YOLOv5, YOLOv8, and MobileNet-SSD on original vs. augmented data to compare performance.

## 🚀 Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run `main.py` to test augmentations.
3. Use training templates in `training/` to evaluate model performance.

