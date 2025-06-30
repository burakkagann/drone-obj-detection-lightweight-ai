import os
import sys
import shutil
import requests
from tqdm import tqdm
import zipfile

def download_file(url, filename):
    """Download a file from a URL with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def main():
    # Create data directories
    data_root = "data/my_dataset/dota/dota-v1.0"
    os.makedirs(data_root, exist_ok=True)
    os.makedirs(os.path.join(data_root, "images"), exist_ok=True)
    os.makedirs(os.path.join(data_root, "labels", "train_original"), exist_ok=True)
    os.makedirs(os.path.join(data_root, "labels", "val_original"), exist_ok=True)
    
    # Download DOTA dataset
    print("Downloading DOTA dataset...")
    dota_urls = {
        "train_images": "https://captain-whu.github.io/DOTA/dataset/DOTA-v1.0_train.zip",
        "val_images": "https://captain-whu.github.io/DOTA/dataset/DOTA-v1.0_val.zip",
        "test_images": "https://captain-whu.github.io/DOTA/dataset/DOTA-v1.0_test.zip",
    }
    
    for name, url in dota_urls.items():
        zip_file = f"{name}.zip"
        if not os.path.exists(zip_file):
            print(f"Downloading {name}...")
            download_file(url, zip_file)
        
        print(f"Extracting {name}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_root)
        
        # Clean up
        os.remove(zip_file)
    
    # Move files to correct locations
    print("Organizing files...")
    for split in ['train', 'val']:
        src_dir = os.path.join(data_root, f"DOTA-v1.0_{split}")
        if os.path.exists(src_dir):
            # Move images
            images_src = os.path.join(src_dir, "images")
            images_dst = os.path.join(data_root, "images", split)
            if os.path.exists(images_src):
                shutil.move(images_src, images_dst)
            
            # Move labels
            labels_src = os.path.join(src_dir, "labelTxt")
            labels_dst = os.path.join(data_root, "labels", f"{split}_original")
            if os.path.exists(labels_src):
                shutil.move(labels_src, labels_dst)
            
            # Clean up
            shutil.rmtree(src_dir)
    
    print("Dataset preparation complete!")

if __name__ == '__main__':
    main() 