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

    For example, for YOLOv8n:

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

1. **Create a script similar to the YOLO models with training instructions for MobileNet-SSD.**

2. **Run the MobileNet-SSD training script:**

    ```bash
    .\run_mobilenet_ssd_training.ps1
    ```

### Step 3: Exporting Models

After training a model, you can export it into multiple formats (ONNX, TensorFlow, TorchScript) by running the following command:

    ```bash
python src/models/YOLOv5/export.py --weights runs/train/yolo5n_baseline/weights/best.pt --img 640 --batch 16 --include onnx,torchscript,tensorflow
    ```

Ensure to replace yolo5n_baseline with the appropriate name of the model you're working with (e.g., yolov8n_baseline).

# Directory Structure and File Handling
Data Directory: Ensure that your dataset is correctly structured in the data/my_dataset/ folder, with the appropriate train and val subdirectories containing images and annotations.

Model Weights: Model weights are saved after training in the runs/train/<model_name>/weights/ directory. Example: runs/train/yolo5n_baseline/weights/best.pt.

