import os
import shutil
from pathlib import Path
import logging
from typing import Tuple, List, Dict
import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_preparation.log'),
        logging.StreamHandler()
    ]
)

# DOTA class mapping based on config/DOTAv1.yaml
CLASS_MAPPING = {
    'plane': 0,
    'ship': 1,
    'storage tank': 2,
    'baseball diamond': 3,
    'tennis court': 4,
    'basketball court': 5,
    'ground track field': 6,
    'harbor': 7,
    'bridge': 8,
    'large vehicle': 9,
    'small vehicle': 10,
    'helicopter': 11,
    'roundabout': 12,
    'soccer ball field': 13,
    'swimming pool': 14
}

class DOTAPreprocessor:
    def __init__(
        self,
        source_path: str,
        target_path: str,
        patch_size: Tuple[int, int] = (1024, 1024),
        overlap: Tuple[int, int] = (200, 200)
    ):
        self.source_path = Path(source_path)
        self.target_path = Path(target_path)
        self.patch_size = patch_size
        self.overlap = overlap
        
        # Ensure target directories exist
        self.target_path.mkdir(parents=True, exist_ok=True)
        for split in ['train', 'val', 'test']:
            (self.target_path / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.target_path / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    def process_dataset(self):
        """Process all splits of the dataset."""
        for split in ['train', 'val', 'test']:
            image_dir = self.source_path / "images" / split
            if not image_dir.exists():
                logging.warning(f"Split directory not found: {image_dir}")
                continue
                
            image_files = list(image_dir.glob("*.png"))
            if not image_files:
                image_files = list(image_dir.glob("*.jpg"))
            
            logging.info(f"Processing {len(image_files)} {split} images...")
            self._process_split(image_files, split)
    
    def _process_split(self, image_files: List[Path], split: str):
        """Process a specific split (train/val/test)."""
        for img_path in tqdm(image_files, desc=f"Processing {split} split"):
            try:
                # Read image
                img = cv2.imread(str(img_path))
                if img is None:
                    logging.error(f"Failed to read image: {img_path}")
                    continue
                
                # Look for label file with different possible extensions
                label_path = None
                for ext in ['.txt', '.xml']:
                    temp_path = self.source_path / "labels" / split / f"{img_path.stem}{ext}"
                    if temp_path.exists():
                        label_path = temp_path
                        break
                
                if label_path is None:
                    logging.warning(f"Label file not found for: {img_path}")
                    continue
                
                # Split image into patches
                patches = self._split_image(img, str(img_path))
                
                # Process and save patches
                for idx, (patch_img, patch_coords) in enumerate(patches):
                    patch_name = f"{img_path.stem}_{idx}"
                    
                    # Save image patch
                    output_path = self.target_path / "images" / split / f"{patch_name}.png"
                    cv2.imwrite(str(output_path), patch_img)
                    
                    # Process and save corresponding labels
                    self._process_labels(
                        label_path,
                        patch_coords,
                        patch_name,
                        split,
                        patch_img.shape[:2]
                    )
                    
            except Exception as e:
                logging.error(f"Error processing {img_path}: {str(e)}")
    
    def _split_image(self, img: np.ndarray, img_name: str):
        """Split large image into overlapping patches."""
        h, w = img.shape[:2]
        patches = []
        
        for y in range(0, h - self.overlap[0], self.patch_size[0] - self.overlap[0]):
            if y + self.patch_size[0] > h:
                y = h - self.patch_size[0]
            
            for x in range(0, w - self.overlap[1], self.patch_size[1] - self.overlap[1]):
                if x + self.patch_size[1] > w:
                    x = w - self.patch_size[1]
                
                patch = img[y:y + self.patch_size[0], x:x + self.patch_size[1]]
                patches.append((patch, (x, y, x + self.patch_size[1], y + self.patch_size[0])))
                
                if x + self.patch_size[1] >= w:
                    break
            if y + self.patch_size[0] >= h:
                break
        
        return patches
    
    def _process_labels(
        self,
        label_path: Path,
        patch_coords: Tuple[int, int, int, int],
        patch_name: str,
        split: str,
        patch_shape: Tuple[int, int]
    ):
        """Process and transform labels for a specific patch."""
        x1, y1, x2, y2 = patch_coords
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        new_labels = []
        for line in lines:
            try:
                parts = line.strip().split()
                if len(parts) < 9:  # Skip metadata lines
                    continue
                
                # Parse coordinates and class name
                coords = [float(x) for x in parts[:8]]
                class_name = parts[8].lower().replace('-', ' ')  # Normalize class name
                
                # Skip if class not in mapping
                if class_name not in CLASS_MAPPING:
                    logging.warning(f"Unknown class {class_name} in {label_path}")
                    continue
                
                class_id = CLASS_MAPPING[class_name]
                
                # Convert coordinates to patch space
                transformed_coords = []
                for i in range(0, 8, 2):
                    x = coords[i] - x1
                    y = coords[i + 1] - y1
                    
                    # Skip if point is outside patch
                    if not (0 <= x <= patch_shape[1] and 0 <= y <= patch_shape[0]):
                        break
                    
                    # Normalize coordinates
                    x /= patch_shape[1]
                    y /= patch_shape[0]
                    
                    transformed_coords.extend([x, y])
                
                if len(transformed_coords) == 8:  # Only add if all points are inside
                    new_labels.append(f"{class_id} {' '.join(map(str, transformed_coords))}\n")
                    
            except Exception as e:
                logging.error(f"Error processing label in {label_path}: {str(e)}")
        
        # Save new labels if any valid ones exist
        if new_labels:
            with open(self.target_path / "labels" / split / f"{patch_name}.txt", 'w') as f:
                f.writelines(new_labels)

if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DOTAPreprocessor(
        source_path="data/my_dataset/dota/dota-v1.0",
        target_path="data/my_dataset/dota/dota-v1.0-split",
        patch_size=(1024, 1024),
        overlap=(200, 200)
    )
    
    # Process dataset
    preprocessor.process_dataset()
    logging.info("Dataset preparation completed!") 