# MobileNet-SSD Training on VisDrone Dataset

## Overview
This document details the process and structure for training the MobileNet-SSD object detection model on the VisDrone dataset. MobileNet-SSD is a lightweight, efficient model suitable for real-time object detection on resource-constrained devices.

## Project Structure
```
project_root/
├── data/
│   └── visdrone/
│       ├── images/
│       ├── annotations/
│       ├── voc_format/
│       └── lmdb/
├── models/
│   └── mobilenet_ssd/
│       ├── deploy.prototxt
│       ├── solver.prototxt
│       └── train.prototxt
├── scripts/
│   ├── data_preparation/
│   │   ├── visdrone_to_voc.py
│   │   └── create_lmdb.py
│   └── training/
│       └── train_mobilenet_ssd.py
└── requirements/
    └── requirements.txt
```

## Setup Process

### 1. Environment Setup
#### Windows Environment
- Python virtual environment: `mobilenet_ssd_env`
- Activation command: `.\venvs\mobilenet_ssd_env\Scripts\Activate.ps1`
- Core dependencies:
  - Python 3.10
  - OpenCV 4.11.0.86
  - NumPy 1.23.5
  - Matplotlib 3.10.3
  - Pandas 1.4.2
  - LMDB 1.4.1
  - Protobuf 3.20.3

#### WSL (Windows Subsystem for Linux) Setup
1. Install Ubuntu on WSL:
   ```powershell
   wsl --install -d Ubuntu
   ```
2. Create user account and set password
3. Install Caffe dependencies:
   ```bash
   sudo apt-get update
   sudo apt-get upgrade -y
   sudo apt-get install -y build-essential cmake git pkg-config
   sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev
   sudo apt-get install -y libhdf5-serial-dev protobuf-compiler
   sudo apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev
   sudo apt-get install -y libopencv-dev python3-dev python3-pip
   ```

### 2. Caffe Installation
#### Current Status
- Attempting to install Caffe in WSL Ubuntu environment
- Created installation script: `install_caffe_wsl.sh`
- Script includes:
  - System updates
  - Dependency installation
  - Caffe cloning and building
  - Python path configuration

#### Installation Steps
1. Clone Caffe:
   ```bash
   cd ~
   git clone https://github.com/BVLC/caffe.git
   ```
2. Install Python dependencies:
   ```bash
   cd caffe
   pip3 install -r python/requirements.txt
   ```
3. Build Caffe:
   ```bash
   mkdir build
   cd build
   cmake ..
   make -j$(nproc)
   make install
   ```

### 3. Data Preparation
#### VisDrone Dataset Structure
- Images in `data/visdrone/images/`
- Annotations in `data/visdrone/annotations/`
- 10 classes:
  1. pedestrian
  2. person
  3. bicycle
  4. car
  5. van
  6. truck
  7. tricycle
  8. awning-tricycle
  9. bus
  10. motor

#### Data Conversion Process
1. Convert VisDrone format to VOC format
2. Generate LMDB files for training
3. Create train/val splits

## Current State
1. **Environment Setup**:
   - ✅ Python virtual environment created
   - ✅ Basic Python dependencies installed
   - ⚠️ Caffe installation in progress

2. **Data Preparation**:
   - ✅ VisDrone dataset available
   - ⚠️ Data conversion scripts need testing
   - ❌ LMDB generation pending

3. **Model Setup**:
   - ❌ Caffe installation pending
   - ❌ Model configuration pending
   - ❌ Training setup pending

## Pain Points
1. **Caffe Installation**:
   - Caffe not available via pip on Windows
   - Requires WSL setup
   - Complex dependency management

2. **Data Conversion**:
   - Need to handle VisDrone's specific annotation format
   - Memory-intensive LMDB generation
   - Large dataset size

3. **Training Setup**:
   - GPU memory requirements
   - Long training times
   - Need for proper monitoring

## Next Steps
1. **Immediate Actions**:
   - Complete Caffe installation in WSL
   - Test data conversion scripts
   - Generate LMDB files

2. **Short-term Goals**:
   - Configure MobileNet-SSD for VisDrone
   - Set up training pipeline
   - Implement validation

3. **Long-term Objectives**:
   - Train and evaluate model
   - Optimize performance
   - Document results

## References
- [MobileNet-SSD Documentation](https://docs.openvino.ai/2023.3/omz_models_model_mobilenet_ssd.html)
- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)
- [Practical Aspects of Model Selection](https://vdeepvision.medium.com/practical-aspects-to-select-a-model-for-object-detection-c704055ab325)

## Notes
- Keep track of GPU memory usage during training
- Monitor training progress regularly
- Backup model checkpoints
- Document any issues and solutions

---
This README will be updated as we progress through the implementation. 