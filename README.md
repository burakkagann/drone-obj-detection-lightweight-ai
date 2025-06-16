# Surveillance Drones Thesis Repository

This repository contains the code, configuration, and structure for the master's thesis project on robust object detection in low-visibility conditions using lightweight AI models.


## Setup Instructions

### Step 1: Create and Activate Virtual Environments

1. **Navigate to the project directory:**

    ```bash
    cd path/to/drone-obj-detection-lightweight-ai
    ```

2. **Create a virtual environment for YOLOv5n:**

    ```bash
    python -m venv venvs/yolov5n_env
    ```

3. **Activate the virtual environment:**

    - **Windows PowerShell:**
        ```bash
        .\venvs\yolov5n_env\Scripts\Activate.ps1
        ```

    - **Windows Command Prompt:**
        ```bash
        .\venvs\yolov5n_env\Scripts\activate.bat
        ```

4. **Install dependencies for YOLOv5n:**

    ```bash
    pip install -r requirements/requirements_yolov5n.txt
    ```

5. **Repeat the process for other models (YOLOv8n, MobileNet-SSD, TensorFlow).**

    For example, for MobileNet-SSD:

    ```bash
    python -m venv venvs/mobilenet_ssd_env
    .\venvs\mobilenet_ssd_env\Scripts\Activate.ps1
    pip install -r requirements/visdrone/requirements_mobilenet-ssd.txt
    ```

    For YOLOv8n:

    ```bash
    python -m venv venvs/yolov8n_env
    .\venvs\yolov8n_env\Scripts\Activate.ps1
    pip install -r requirements/requirements_yolov8n.txt
    ```

### Step 2: Training Models

#### **Training YOLOv5n**

1. **Navigate to the script folder:**

    ```bash
    cd src/scripts/visdrone
    ```

2. **Run the training script:**

    ```bash
    .\run_yolov5n_training.ps1
    ```

   This will:
   - Activate the virtual environment
   - Clean cache files
   - Start YOLOv5n training with the given dataset
   - Deactivate the virtual environment after completion

#### **Training YOLOv8n**

1. **Navigate to the script folder:**

    ```bash
    cd src/scripts/visdrone
    ```

2. **Run the training script for YOLOv8n:**

    ```bash
    .\run_yolov8n_training.ps1
    ```

   This will:
   - Activate the virtual environment for YOLOv8
   - Clean cache files
   - Start YOLOv8n training with the given dataset
   - Deactivate the virtual environment after completion

#### **Training MobileNet-SSD**

1. **Prepare the Dataset:**
   
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

2. **Configure Training:**
   
   Edit `config/mobilenet_ssd_visdrone.yaml` to adjust:
   - Model parameters (input size, batch size, learning rate)
   - Dataset paths
   - Training settings (epochs, early stopping)
   - Preprocessing options

3. **Start Training:**

    ```bash
    # Activate the virtual environment
    .\venvs\mobilenet_ssd_env\Scripts\Activate.ps1
    
    # Run the training script
    python src/scripts/train_mobilenet_ssd.py
    ```

   The script will:
   - Load and preprocess the dataset
   - Initialize the MobileNet-SSD model
   - Train using the specified configuration
   - Save checkpoints and logs

4. **Monitor Training:**
   - Training progress is displayed in the console
   - Logs are saved in `logs/mobilenet_ssd_visdrone`
   - Model checkpoints are saved in `checkpoints/mobilenet_ssd_visdrone`

5. **Training Options:**
   ```bash
   # Train with a specific GPU (if available)
   python src/scripts/train_mobilenet_ssd.py --gpu 0
   
   # Use a different config file
   python src/scripts/train_mobilenet_ssd.py --config path/to/config.yaml
   ```

### Step 3: Exporting Models

After training a model, you can export it into multiple formats (ONNX, TensorFlow, TorchScript) by running the following command:

    ```bash
python src/models/YOLOv5/export.py --weights runs/train/yolo5n_baseline/weights/best.pt --img 640 --batch 16 --include onnx,torchscript,tensorflow
    ```

Ensure to replace yolo5n_baseline with the appropriate name of the model you're working with (e.g., yolov8n_baseline).

# Directory Structure and File Handling

## Data Directory
Ensure that your dataset is correctly structured in the data/my_dataset/ folder, with the appropriate train and val subdirectories containing images and annotations.

## Model Weights
Model weights are saved after training in:
- YOLO models: `runs/train/<model_name>/weights/best.pt`
- MobileNet-SSD: `checkpoints/mobilenet_ssd_visdrone/model_final.h5` and epoch checkpoints

## Model Configuration
- MobileNet-SSD configuration: `config/mobilenet_ssd_visdrone.yaml`
- Includes model architecture, training parameters, and dataset settings

## Training Logs
- Training logs are saved in `logs/mobilenet_ssd_visdrone`
- Use TensorBoard to visualize training progress:
  ```bash
  tensorboard --logdir logs/mobilenet_ssd_visdrone
  ```

## Model Usage
To use the trained MobileNet-SSD model for inference:

1. **Load the Model:**
   ```python
   import tensorflow as tf
   from src.models.mobilenet_ssd import MobileNetSSD
   
   # Load configuration
   with open('config/mobilenet_ssd_visdrone.yaml', 'r') as f:
       config = yaml.safe_load(f)
   
   # Initialize model
   model = MobileNetSSD(config)
   net = model.build_model()
   
   # Load weights
   net.load_weights('checkpoints/mobilenet_ssd_visdrone/model_final.h5')
   ```

2. **Perform Inference:**
   ```python
   import cv2
   import numpy as np
   
   def preprocess_image(image_path, input_shape):
       image = cv2.imread(image_path)
       image = cv2.resize(image, input_shape[:2])
       image = image.astype(np.float32)
       image = (image - 127.5) * 0.007843
       return image
   
   # Preprocess image
   image = preprocess_image('path/to/image.jpg', config['model']['input_shape'])
   
   # Run inference
   cls_pred, reg_pred = net.predict(np.expand_dims(image, axis=0))
   
   # Post-process predictions
   # (Implement non-maximum suppression and confidence thresholding)
   ```

