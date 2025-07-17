# YOLOv5n VisDrone Training Documentation

## 📋 Overview

This documentation provides a comprehensive guide for training YOLOv5n (nano) model on the VisDrone dataset with CUDA acceleration, thesis evaluation framework, and edge device deployment preparation.

**Research Context**: "Robust Object Detection for Surveillance Drones in Low-Visibility Environments Using Lightweight AI Models"

## 🎯 Project Objectives

- **Lightweight Model Training**: YOLOv5n optimized for surveillance drone applications
- **Comprehensive Evaluation**: Thesis-compliant metrics collection (mAP, FPS, power consumption)
- **Edge Device Readiness**: Optimization for NVIDIA Jetson Nano and Raspberry Pi 4
- **Performance Benchmarking**: Baseline establishment for model comparison

## 📁 Project Structure

```
progression-notes/yolov5n_visdrone/
├── README.md                           # This documentation
├── v2_yolov5n_visdrone-setup-notes.md  # Detailed setup and progression notes
├── to-do.md                            # Thesis roadmap and tasks
├── edge_device_testing_strategy.md     # Edge device deployment strategy
├── YOLOv5n_VisDrone_Implementation_Guide.md
├── yolov5n_visdrone_setup_and_training_log.md
└── progression-notes.md

Related Files:
├── src/evaluation/thesis_metrics.py                      # Comprehensive metrics framework
├── src/scripts/visdrone/train_yolov5n_with_thesis_metrics.py  # Integrated training script
├── config/visdrone/yolov5n_v1/yolov5n_visdrone_config.yaml   # Dataset configuration
└── src/models/YOLOv5/                                    # YOLOv5 implementation
```

## 🚀 Quick Start

### Prerequisites
- **GPU**: NVIDIA RTX 3060 (6GB VRAM) or equivalent
- **CUDA**: Version 12.9+
- **Python**: 3.8+
- **PyTorch**: 2.5.1+cu121 (CUDA-enabled)

### 1. Environment Setup
```powershell
# Navigate to project directory
cd "C:\Users\burak\OneDrive\Desktop\Git Repos\drone-obj-detection-lightweight-ai"

# Activate virtual environment (if using one)
# .\venvs\yolov5n_env\Scripts\activate

# Install CUDA-enabled PyTorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

### 2. Verify CUDA Setup
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

### 3. Run Training
```powershell
# Recommended: Comprehensive thesis training with metrics
cd src/scripts/visdrone
python train_yolov5n_with_thesis_metrics.py

# Alternative: Basic YOLOv5 training
cd src/models/YOLOv5
python train.py --data ../../../config/visdrone/yolov5n_v1/yolov5n_visdrone_config.yaml --cfg models/yolov5n.yaml --weights yolov5n.pt --batch-size 16 --epochs 50 --device 0 --workers 4 --cache
```

## 📊 Dataset Configuration

### VisDrone Dataset
- **Classes**: 10 object classes (pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor)
- **Location**: `data/my_dataset/visdrone/`
- **Format**: YOLO format with bounding boxes
- **Configuration**: `config/visdrone/yolov5n_v1/yolov5n_visdrone_config.yaml`

### Dataset Structure
```
data/my_dataset/visdrone/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── dataset.yaml
```

## 🧠 Model Configuration

### YOLOv5n Specifications
- **Model**: YOLOv5n (nano - lightweight version)
- **Input Size**: 640x640 pixels
- **Pretrained Weights**: YOLOv5n COCO pretrained
- **Transfer Learning**: 343/349 pretrained weights transferred
- **Optimization**: Memory-optimized for 6GB VRAM

### Training Parameters
```yaml
# Optimized for RTX 3060 6GB
batch_size: 16
epochs: 50
image_size: 640
workers: 4
device: 0  # GPU device
cache: true  # Enable dataset caching
```

## 🔧 Training Options

### Option 1: Comprehensive Thesis Training (Recommended)
**File**: `src/scripts/visdrone/train_yolov5n_with_thesis_metrics.py`

**Features**:
- ✅ Complete model training
- ✅ Automated metrics collection (mAP, Precision, Recall, FPS)
- ✅ GPU performance monitoring (power consumption, memory usage)
- ✅ Inference speed benchmarking
- ✅ Thesis-compliant report generation
- ✅ Edge device readiness assessment

**Usage**:
```powershell
cd src/scripts/visdrone
python train_yolov5n_with_thesis_metrics.py
```

### Option 2: Basic YOLOv5 Training
**File**: `src/models/YOLOv5/train.py`

**Features**:
- ✅ Standard YOLOv5 training
- ✅ Basic metrics (loss, mAP)
- ✅ TensorBoard logging

**Usage**:
```powershell
cd src/models/YOLOv5
python train.py --data ../../../config/visdrone/yolov5n_v1/yolov5n_visdrone_config.yaml --cfg models/yolov5n.yaml --weights yolov5n.pt --batch-size 16 --epochs 50 --device 0 --workers 4 --cache
```

## 📈 Thesis Evaluation Framework

### Comprehensive Metrics Collection
**Location**: `src/evaluation/thesis_metrics.py`

**Metrics Collected**:
- **Detection Accuracy**: mAP@0.5, mAP@0.5:0.95, Precision, Recall
- **Performance**: FPS (frames per second)
- **System Resources**: GPU memory usage, power consumption
- **Model Characteristics**: Parameter count, model size

### Automated Reporting
- **CSV Export**: Structured data for thesis analysis
- **Visualization**: Performance charts and graphs
- **Thesis Reports**: Academic-formatted evaluation documents

### Usage Example
```python
from src.evaluation.thesis_metrics import ThesisMetricsCollector

# Initialize metrics collector
metrics = ThesisMetricsCollector(
    model_name="YOLOv5n",
    dataset_name="VisDrone",
    device="RTX_3060"
)

# Collect comprehensive metrics
results = metrics.collect_all_metrics(
    model_path="path/to/model.pt",
    data_path="path/to/dataset.yaml",
    output_dir="results/"
)
```

## 🖥️ Edge Device Testing Strategy

### Phase 1: Simulation-Based Optimization (Immediate - $0 Cost)
**Objectives**:
- Model optimization (ONNX, TensorRT conversion)
- Performance constraint simulation
- Resource usage modeling

**Implementation**:
```python
# Model optimization pipeline
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

# Export to ONNX for edge deployment
torch.onnx.export(model, sample_input, "yolov5n_visdrone.onnx")

# TensorRT optimization (if available)
# trtexec --onnx=yolov5n_visdrone.onnx --saveEngine=yolov5n_visdrone.trt
```

### Phase 2: Physical Device Testing ($150-400 Budget)
**Target Devices**:
- **Primary**: NVIDIA Jetson Nano (4GB/2GB)
- **Secondary**: Raspberry Pi 4 (4GB/8GB)

**Performance Targets**:
- **FPS**: ≥10 FPS for real-time detection
- **Memory**: <2GB RAM usage
- **Power**: <15W consumption

### Alternative Testing Approaches
- **University Lab Access**: Use existing hardware
- **Cloud Simulation**: Google Colab with resource constraints
- **Community Collaboration**: Partner with other researchers

## 🔍 Performance Benchmarks

### RTX 3060 6GB Baseline (Achieved)
- **Training Speed**: Efficient with 1.97G/6G VRAM usage (33% utilization)
- **Memory Optimization**: Successful 6GB VRAM management
- **Model Convergence**: Healthy loss reduction confirmed
- **Transfer Learning**: 343/349 pretrained weights successfully transferred

### Training Metrics (Example)
```
Epoch 1/50 Results:
├── Box Loss: 0.1649 → 0.1563 (↓ decreasing ✅)
├── Class Loss: 0.06916 → 0.05203 (↓ decreasing ✅)
├── Object Loss: Stable progression
└── Precision/Recall: Improving trends
```

## ⚠️ Known Issues & Solutions

### 1. pkg_resources Deprecation Warning
**Issue**: setuptools 67.5.0+ deprecation warnings
**Impact**: ⚠️ Cosmetic only - no functional impact
**Solution**: 
```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message="pkg_resources is deprecated")
```

### 2. Autocast Deprecation Warning
**Issue**: PyTorch autocast syntax migration
**Impact**: ⚠️ Cosmetic only - forward compatibility notice
**Solution**: Use `torch.autocast("cuda")` instead of `torch.cuda.amp.autocast()`

### 3. Memory Management
**Issue**: CUDA out of memory errors
**Solutions**:
- Reduce batch size: `--batch-size 8` or `--batch-size 4`
- Enable gradient checkpointing
- Use mixed precision training

## 📚 Thesis Methodology Alignment

### Research Phases
- **✅ Phase 1**: Literature review and baseline model training (COMPLETE)
- **🔄 Phase 2**: Performance evaluation and benchmarking (IN PROGRESS)
- **📋 Phase 3**: Synthetic data augmentation pipeline (PLANNED)
- **📋 Phase 4**: Edge device deployment and testing (PLANNED)
- **📋 Phase 5**: Comparative analysis and thesis writing (PLANNED)

### Requirements Fulfilled
- **Models**: YOLOv5n (complete), YOLOv8n/MobileNet-SSD/NanoDet (framework ready)
- **Datasets**: VisDrone (complete), DOTA/CIFAR (framework ready)
- **Metrics**: mAP, Precision, Recall, FPS, memory usage, power consumption (implemented)
- **Edge Devices**: Jetson Nano, Raspberry Pi 4 (strategy documented)

## 🛠️ Troubleshooting

### Common Issues

#### CUDA Not Available
```python
# Check CUDA installation
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)

# Reinstall CUDA-enabled PyTorch
pip uninstall torch torchvision torchaudio
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

#### Dataset Loading Issues
```python
# Verify dataset structure
import os
print(os.listdir("data/my_dataset/visdrone/images/train/"))
print(os.listdir("data/my_dataset/visdrone/labels/train/"))

# Check configuration file
with open("config/visdrone/yolov5n_v1/yolov5n_visdrone_config.yaml", "r") as f:
    print(f.read())
```

#### Memory Issues
```powershell
# Monitor GPU memory usage
nvidia-smi

# Reduce batch size if needed
python train.py --batch-size 8  # or 4
```

## 📊 Results and Reports

### Generated Files
After training completion, you'll find:
```
results/
├── yolov5n_visdrone_metrics.csv      # Comprehensive metrics data
├── yolov5n_visdrone_report.md        # Thesis-formatted report
├── performance_charts/               # Visualization plots
├── model_weights/                    # Trained model checkpoints
└── inference_benchmarks/             # FPS and performance tests
```

### Key Metrics to Report
- **Detection Accuracy**: mAP@0.5, mAP@0.5:0.95
- **Performance**: Training FPS, Inference FPS
- **Resource Usage**: GPU memory, power consumption
- **Model Efficiency**: Parameters, model size, FLOPs

## 🔗 Related Documentation

- **Setup Notes**: [`v2_yolov5n_visdrone-setup-notes.md`](v2_yolov5n_visdrone-setup-notes.md)
- **Task Roadmap**: [`to-do.md`](to-do.md)
- **Edge Strategy**: [`edge_device_testing_strategy.md`](edge_device_testing_strategy.md)
- **Implementation Guide**: [`YOLOv5n_VisDrone_Implementation_Guide.md`](YOLOv5n_VisDrone_Implementation_Guide.md)

## 🎯 Next Steps

### Immediate Actions
1. **Run Comprehensive Training**: Execute thesis metrics collection
2. **Collect Baseline Performance**: Document RTX 3060 results
3. **Model Optimization**: Export to ONNX/TensorRT formats
4. **Edge Testing Preparation**: Simulate resource constraints

### Medium-Term Goals
- **Multi-Model Comparison**: Extend to YOLOv8n, MobileNet-SSD, NanoDet
- **Synthetic Augmentation**: Develop environmental condition simulation
- **Multiple Datasets**: Expand testing to DOTA and CIFAR
- **Physical Edge Testing**: Acquire and test on Jetson Nano

## 🏆 Achievements

### Technical Milestones
- ✅ **CUDA Acceleration**: Successfully enabled on RTX 3060
- ✅ **Memory Optimization**: Efficient 6GB VRAM usage
- ✅ **Dataset Integration**: VisDrone properly configured and cached
- ✅ **Training Validation**: Confirmed healthy learning progression
- ✅ **Comprehensive Metrics**: All thesis-required metrics implemented
- ✅ **Edge Device Strategy**: Complete testing framework designed

### Research Contributions
- **Performance Baseline**: RTX 3060 baseline established
- **Evaluation Framework**: Reusable system for model comparison
- **Edge Device Readiness**: Comprehensive deployment strategy
- **Thesis Alignment**: All methodology requirements addressed

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the detailed setup notes in `v2_yolov5n_visdrone-setup-notes.md`
3. Consult the thesis roadmap in `to-do.md`
4. Refer to the edge device strategy in `edge_device_testing_strategy.md`

## 📄 License

This project is developed for academic research purposes as part of the thesis "Robust Object Detection for Surveillance Drones in Low-Visibility Environments Using Lightweight AI Models".

---

*Last Updated: 2024-12-19*

**Status**: ✅ Training framework complete, Phase 1 accomplished, Phase 2 in progress 