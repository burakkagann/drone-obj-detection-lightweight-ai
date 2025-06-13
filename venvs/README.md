# Virtual Environments Setup Guide
This directory contains the virtual environments for various models and frameworks used in the project. Each model has its own dedicated environment to manage dependencies efficiently.

# Folder Structure
- yolov5n_env/: Virtual environment for YOLOv5n.
- yolov8n_env/: Virtual environment for YOLOv8n.
- mobilenet_ssd_env/: Virtual environment for MobileNet-SSD.
- tensorflow_env/: Virtual environment for TensorFlow-based models.
- nanodet_env/: Virtual environment for NanoDet.

# Setup Instructions
**Step 1:** Create Virtual Environment for Each Model
Navigate to the root directory of the repo.

For YOLOv5n, YOLOv8n, MobileNet-SSD, TensorFlow, and NanoDet, run the following command for each model to create a dedicated virtual environment.

- python -m venv venvs/yolov5n_env
- python -m venv venvs/yolov8n_env
- python -m venv venvs/mobilenet_ssd_env
- python -m venv venvs/tensorflow_env
- python -m venv venvs/nanodet_env

**Step 2:** Activate the Virtual Environment and Install Dependencies
For YOLOv5n:
- Activate the virtual environment:

- On Windows:
.\venvs\yolov5n_env\Scripts\Activate.ps1
- On Linux/macOS:
source venvs/yolov5n_env/bin/activate

- Install dependencies:
pip install -r requirements/requirements_yolov5n.txt

- Deactivate the virtual environment:
deactivate

For YOLOv8n:

Activate the virtual environment:

On Windows:
.\venvs\yolov8n_env\Scripts\Activate.ps1

On Linux/macOS:
source venvs/yolov8n_env/bin/activate

Install dependencies:
pip install -r requirements/requirements_yolov8n.txt

Deactivate the virtual environment:
deactivate

For MobileNet-SSD:
Activate the virtual environment:

On Windows:
.\venvs\mobilenet_ssd_env\Scripts\Activate.ps1

On Linux/macOS:
source venvs/mobilenet_ssd_env/bin/activate

Install dependencies:
pip install -r requirements/requirements_mobilenet-ssd.txt

Deactivate the virtual environment:
deactivate

For TensorFlow:
Activate the virtual environment:

On Windows:
.\venvs\tensorflow_env\Scripts\Activate.ps1

On Linux/macOS:
source venvs/tensorflow_env/bin/activate

Install dependencies:
pip install -r requirements/requirements_tensorflow.txt

Deactivate the virtual environment:
deactivate

For NanoDet:

Activate the virtual environment:

On Windows:
.\venvs\nanodet_env\Scripts\Activate.ps1

On Linux/macOS:
source venvs/nanodet_env/bin/activate

Install dependencies:
pip install -r requirements/requirements_nanodet.txt

Deactivate the virtual environment:
deactivate