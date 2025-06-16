import os
from pathlib import Path
import cv2
import numpy as np
from typing import List, Tuple
import yaml
import logging
import random
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

class DatasetVerifier:
    def __init__(self, dataset_path: str, yaml_path: str):
        self.dataset_path = Path(dataset_path)
        with open(yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.class_names = self.config['names']
    
    def verify_structure(self) -> bool:
        """Verify the dataset directory structure."""
        required_dirs = [
            'images/train', 'images/val',
            'labels/train', 'labels/val'
        ]
        
        for dir_path in required_dirs:
            full_path = self.dataset_path / dir_path
            if not full_path.exists():
                logging.error(f"Missing directory: {full_path}")
                return False
        return True
    
    def verify_labels(self, split: str = 'train') -> Tuple[bool, List[str]]:
        """Verify label format for a given split."""
        label_dir = self.dataset_path / 'labels' / split
        image_dir = self.dataset_path / 'images' / split
        errors = []
        
        for label_file in tqdm(list(label_dir.glob('*.txt')), desc=f"Verifying {split} labels"):
            # Check if corresponding image exists
            image_file = image_dir / f"{label_file.stem}.png"
            if not image_file.exists():
                errors.append(f"Missing image for label: {label_file}")
                continue
            
            # Read and verify label format
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    parts = line.strip().split()
                    
                    # Check basic format
                    if len(parts) != 9:  # class_id + 8 coordinates
                        errors.append(f"Invalid format in {label_file}:{line_num}")
                        continue
                    
                    # Verify class id
                    try:
                        class_id = int(parts[8])
                        if class_id not in self.class_names:
                            errors.append(f"Invalid class ID in {label_file}:{line_num}")
                    except ValueError:
                        errors.append(f"Invalid class ID format in {label_file}:{line_num}")
                    
                    # Verify coordinates are normalized [0-1]
                    try:
                        coords = [float(x) for x in parts[:8]]
                        if not all(0 <= x <= 1 for x in coords):
                            errors.append(f"Coordinates not normalized in {label_file}:{line_num}")
                    except ValueError:
                        errors.append(f"Invalid coordinate format in {label_file}:{line_num}")
                
            except Exception as e:
                errors.append(f"Error processing {label_file}: {str(e)}")
        
        return len(errors) == 0, errors
    
    def visualize_random_samples(self, split: str = 'train', num_samples: int = 5):
        """Visualize random samples from the dataset."""
        image_dir = self.dataset_path / 'images' / split
        label_dir = self.dataset_path / 'labels' / split
        
        image_files = list(image_dir.glob('*.png'))
        samples = random.sample(image_files, min(num_samples, len(image_files)))
        
        for img_path in samples:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                logging.error(f"Failed to read image: {img_path}")
                continue
            
            # Read corresponding label
            label_path = label_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                logging.error(f"Missing label file: {label_path}")
                continue
            
            # Draw annotations
            h, w = img.shape[:2]
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 9:
                        continue
                    
                    # Get class and coordinates
                    class_id = int(parts[8])
                    coords = np.array([float(x) for x in parts[:8]]).reshape(-1, 2)
                    
                    # Denormalize coordinates
                    coords[:, 0] *= w
                    coords[:, 1] *= h
                    coords = coords.astype(np.int32)
                    
                    # Draw polygon
                    color = tuple(np.random.randint(0, 255, 3).tolist())
                    cv2.polylines(img, [coords], True, color, 2)
                    
                    # Add class label
                    class_name = self.class_names[class_id]
                    cv2.putText(img, class_name, tuple(coords[0]),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Display image
            cv2.imshow(f"Sample from {split}", img)
            cv2.waitKey(0)
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    verifier = DatasetVerifier(
        dataset_path="data/my_dataset/dota/dota-v1.0-split",
        yaml_path="config/DOTAv1.yaml"
    )
    
    # Verify dataset structure
    logging.info("Verifying dataset structure...")
    if verifier.verify_structure():
        logging.info("Dataset structure is valid!")
    else:
        logging.error("Dataset structure verification failed!")
        exit(1)
    
    # Verify labels
    for split in ['train', 'val']:
        logging.info(f"Verifying {split} split labels...")
        is_valid, errors = verifier.verify_labels(split)
        if is_valid:
            logging.info(f"{split} split labels are valid!")
        else:
            logging.error(f"{split} split label verification failed!")
            for error in errors[:10]:  # Show first 10 errors
                logging.error(error)
    
    # Visualize random samples
    logging.info("Visualizing random samples...")
    verifier.visualize_random_samples('train', num_samples=5) 