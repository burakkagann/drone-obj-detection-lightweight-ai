"""
Script to check VisDrone dataset structure and contents.
"""

import os
import yaml
import sys

def main():
    # Load config
    config_path = 'config/mobilenet_ssd_visdrone.yaml'
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get dataset paths
    data_root = config['dataset']['root_dir']
    train_dir = os.path.join(data_root, config['dataset']['train_dir'])
    val_dir = os.path.join(data_root, config['dataset']['val_dir'])
    voc_dir = os.path.join(data_root, 'voc_format')
    voc_train_dir = os.path.join(voc_dir, 'train')
    voc_val_dir = os.path.join(voc_dir, 'val')
    
    # Check directory existence
    print("\nChecking dataset directories:")
    print(f"Data root: {data_root}")
    print(f"  Exists: {os.path.exists(data_root)}")
    
    print(f"\nTrain images: {train_dir}")
    print(f"  Exists: {os.path.exists(train_dir)}")
    if os.path.exists(train_dir):
        images = [f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.png'))]
        print(f"  Number of images: {len(images)}")
        if images:
            print(f"  Sample images: {images[:5]}")
    
    print(f"\nVal images: {val_dir}")
    print(f"  Exists: {os.path.exists(val_dir)}")
    if os.path.exists(val_dir):
        images = [f for f in os.listdir(val_dir) if f.endswith(('.jpg', '.png'))]
        print(f"  Number of images: {len(images)}")
        if images:
            print(f"  Sample images: {images[:5]}")
    
    print(f"\nVOC format directory: {voc_dir}")
    print(f"  Exists: {os.path.exists(voc_dir)}")
    
    print(f"\nVOC train annotations: {voc_train_dir}")
    print(f"  Exists: {os.path.exists(voc_train_dir)}")
    if os.path.exists(voc_train_dir):
        annotations = [f for f in os.listdir(voc_train_dir) if f.endswith('.xml')]
        print(f"  Number of annotations: {len(annotations)}")
        if annotations:
            print(f"  Sample annotations: {annotations[:5]}")
    
    print(f"\nVOC val annotations: {voc_val_dir}")
    print(f"  Exists: {os.path.exists(voc_val_dir)}")
    if os.path.exists(voc_val_dir):
        annotations = [f for f in os.listdir(voc_val_dir) if f.endswith('.xml')]
        print(f"  Number of annotations: {len(annotations)}")
        if annotations:
            print(f"  Sample annotations: {annotations[:5]}")

if __name__ == '__main__':
    main() 