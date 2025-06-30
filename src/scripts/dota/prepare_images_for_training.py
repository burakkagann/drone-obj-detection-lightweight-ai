import os
import shutil
from pathlib import Path

def copy_images():
    # Base paths
    base_dir = Path("data/my_dataset/dota/dota-v1.0")
    source_train = base_dir / "images/train"
    source_val = base_dir / "images/val"
    
    target_train = base_dir / "mobilenet-ssd/train/JPEGImages"
    target_val = base_dir / "mobilenet-ssd/val/JPEGImages"
    
    # Create target directories
    target_train.mkdir(parents=True, exist_ok=True)
    target_val.mkdir(parents=True, exist_ok=True)
    
    # Copy training images
    print("Copying training images...")
    for img_file in source_train.glob("*.png"):
        shutil.copy2(img_file, target_train / img_file.name)
    
    # Copy validation images
    print("Copying validation images...")
    for img_file in source_val.glob("*.png"):
        shutil.copy2(img_file, target_val / img_file.name)
    
    print("Done!")

if __name__ == "__main__":
    copy_images() 