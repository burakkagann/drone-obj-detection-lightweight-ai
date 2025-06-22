"""Script to setup NanoDet model files and structure

This script clones the official NanoDet repository and organizes required files
into our project structure.
"""

import os
import shutil
import subprocess
import time
from pathlib import Path

def safe_remove_tree(path):
    """Safely remove a directory tree, handling Windows permission errors"""
    if not path.exists():
        return True
    
    max_retries = 3
    for i in range(max_retries):
        try:
            if path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path, ignore_errors=True)
            if not path.exists():  # Successfully removed
                return True
            time.sleep(1)
        except Exception as e:
            if i == max_retries - 1:
                print(f"Warning: Could not remove {path}. Please delete it manually.")
                return False
    return False

def setup_nanodet():
    """Setup NanoDet model files"""
    # Define paths
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent.parent
    nanodet_model_dir = project_root / "src" / "models" / "nanodet"
    temp_clone_dir = project_root / "temp_nanodet"

    # Create directories if they don't exist
    nanodet_model_dir.mkdir(parents=True, exist_ok=True)

    # Remove existing temp directory if it exists
    if temp_clone_dir.exists() and not safe_remove_tree(temp_clone_dir):
        print("Error: Could not remove temporary directory. Please delete it manually and try again.")
        return

    try:
        # Clone NanoDet repository
        print("Cloning NanoDet repository...")
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/RangiLyu/nanodet.git", str(temp_clone_dir)], 
                      check=True, capture_output=True, text=True)

        # Copy required files
        print("Copying model files...")
        
        # Copy the entire nanodet package
        if (temp_clone_dir / "nanodet").exists():
            if (nanodet_model_dir / "nanodet").exists():
                safe_remove_tree(nanodet_model_dir / "nanodet")
            shutil.copytree(temp_clone_dir / "nanodet", nanodet_model_dir / "nanodet")
            print("Core model files copied successfully")
        else:
            print("Warning: Core model directory not found")
        
        # Copy config examples
        config_dir = project_root / "config" / "nanodet"
        if (temp_clone_dir / "config").exists():
            shutil.copytree(temp_clone_dir / "config", config_dir, dirs_exist_ok=True)
            print("Config files copied successfully")
        else:
            print("Warning: Config directory not found")
        
        # Create setup.py
        setup_py_content = '''from setuptools import setup, find_packages

setup(
    name="nanodet",
    version="1.0.0",
    description="NanoDet object detection model",
    author="RangiLyu",
    author_email="lyuchqi@gmail.com",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "torchvision>=0.8.1",
        "numpy>=1.19.5",
        "opencv-python>=4.1.2",
        "termcolor>=1.1.0",
        "tensorboard>=2.4.1",
        "pytorch-lightning>=1.3.0",
        "pycocotools>=2.0.2",
        "PyYAML>=5.3.1",
        "tqdm>=4.60.0",
        "matplotlib>=3.4.1",
        "scipy>=1.6.0",
        "pandas>=1.2.4",
        "albumentations>=0.5.2",
    ],
)
'''
        with open(nanodet_model_dir / "setup.py", "w") as f:
            f.write(setup_py_content)
        
        # Install the package
        print("Installing NanoDet package...")
        subprocess.run(["pip", "install", "-e", str(nanodet_model_dir)], 
                      check=True, capture_output=True, text=True)
        
        print("NanoDet setup completed!")
    
    except subprocess.CalledProcessError as e:
        print(f"Error during command execution: {e.stderr}")
    except Exception as e:
        print(f"Error during setup: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("Cleaning up...")
        safe_remove_tree(temp_clone_dir)

if __name__ == "__main__":
    setup_nanodet() 