import cv2
import yaml
import torch
import torchvision
import numpy as np
from tqdm import __version__ as tqdm_version
import os
import sys

# Add src directory to Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(src_path)

def test_mobilenet_ssd_dota_setup():
    """
    Test script to verify the environment setup for MobileNet-SSD DOTA training
    Checks:
    1. Required package installations
    2. CUDA availability
    3. Dataset paths
    4. Model imports
    """
    print("Testing MobileNet-SSD DOTA environment setup:")
    
    # Test OpenCV
    print("\nOpenCV:")
    print(f"Version: {cv2.__version__}")
    
    # Test PyYAML
    print("\nPyYAML:")
    print(f"Version: {yaml.__version__}")
    
    # Test PyTorch
    print("\nPyTorch:")
    print(f"Version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Test TorchVision
    print("\nTorchVision:")
    print(f"Version: {torchvision.__version__}")
    
    # Test NumPy
    print("\nNumPy:")
    print(f"Version: {np.__version__}")
    
    # Test TQDM
    print("\nTQDM:")
    print(f"Version: {tqdm_version}")
    
    # Check dataset paths
    print("\nChecking DOTA dataset paths:")
    data_root = "data/my_dataset/dota/dota-v1.0/mobilenet-ssd"
    paths_to_check = [
        os.path.join(data_root, "train", "JPEGImages"),
        os.path.join(data_root, "train", "Annotations"),
        os.path.join(data_root, "val", "JPEGImages"),
        os.path.join(data_root, "val", "Annotations")
    ]
    
    for path in paths_to_check:
        exists = os.path.exists(path)
        print(f"{path}: {'✓' if exists else '✗'}")
    
    # Test model imports
    print("\nTesting model imports:")
    try:
        from models.mobilenet_ssd import create_mobilenetv2_ssd_lite
        print("MobileNetV2-SSD-Lite model import: ✓")
    except ImportError as e:
        print(f"MobileNetV2-SSD-Lite model import: ✗ ({str(e)})")
    
    print("\nSetup verification completed!")

if __name__ == "__main__":
    test_mobilenet_ssd_dota_setup() 