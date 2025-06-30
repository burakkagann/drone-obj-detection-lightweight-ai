import os
import sys
import cv2
import xml.etree.ElementTree as ET
import random
from pathlib import Path
import numpy as np

# Add src directory to Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(src_path)

class DOTADatasetVerifier:
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.class_colors = {
            'plane': (255, 0, 0),
            'ship': (0, 255, 0),
            'storage tank': (0, 0, 255),
            'baseball diamond': (255, 255, 0),
            'tennis court': (255, 0, 255),
            'basketball court': (0, 255, 255),
            'ground track field': (128, 0, 0),
            'harbor': (0, 128, 0),
            'bridge': (0, 0, 128),
            'large vehicle': (128, 128, 0),
            'small vehicle': (128, 0, 128),
            'helicopter': (0, 128, 128),
            'roundabout': (255, 128, 0),
            'soccer ball field': (128, 255, 0),
            'swimming pool': (0, 255, 128)
        }
    
    def verify_dataset_statistics(self, split='train'):
        """Print dataset statistics"""
        print(f"\nAnalyzing {split} split:")
        
        annotations_dir = self.data_root / split / 'Annotations'
        images_dir = self.data_root / split / 'JPEGImages'
        
        # Count files
        xml_files = list(annotations_dir.glob('*.xml'))
        image_files = list(images_dir.glob('*.png'))
        
        print(f"Number of annotation files: {len(xml_files)}")
        print(f"Number of image files: {len(image_files)}")
        
        # Count objects per class
        class_counts = {class_name: 0 for class_name in self.class_colors.keys()}
        total_objects = 0
        
        for xml_file in xml_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name in class_counts:
                    class_counts[class_name] += 1
                    total_objects += 1
        
        print(f"\nTotal objects: {total_objects}")
        print("\nObjects per class:")
        for class_name, count in class_counts.items():
            percentage = (count / total_objects) * 100 if total_objects > 0 else 0
            print(f"{class_name}: {count} ({percentage:.1f}%)")
    
    def visualize_random_samples(self, split='train', num_samples=5):
        """Visualize random samples with bounding boxes"""
        annotations_dir = self.data_root / split / 'Annotations'
        images_dir = self.data_root / split / 'JPEGImages'
        
        xml_files = list(annotations_dir.glob('*.xml'))
        if not xml_files:
            print(f"No annotation files found in {annotations_dir}")
            return
        
        # Select random samples
        samples = random.sample(xml_files, min(num_samples, len(xml_files)))
        
        for xml_file in samples:
            # Load image
            image_file = images_dir / f"{xml_file.stem}.png"
            if not image_file.exists():
                print(f"Image not found: {image_file}")
                continue
            
            img = cv2.imread(str(image_file))
            if img is None:
                print(f"Failed to load image: {image_file}")
                continue
            
            # Parse XML
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Draw bounding boxes
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                bbox = obj.find('bndbox')
                xmin = int(float(bbox.find('xmin').text))
                ymin = int(float(bbox.find('ymin').text))
                xmax = int(float(bbox.find('xmax').text))
                ymax = int(float(bbox.find('ymax').text))
                
                color = self.class_colors.get(class_name, (255, 255, 255))
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
                
                # Add label
                label = f"{class_name}"
                cv2.putText(img, label, (xmin, ymin-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Display image
            window_name = f"Sample: {xml_file.stem}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1024, 768)
            cv2.imshow(window_name, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def main():
    data_root = "data/my_dataset/dota/dota-v1.0/mobilenet-ssd"
    verifier = DOTADatasetVerifier(data_root)
    
    # Verify training set
    print("\nVerifying training set:")
    verifier.verify_dataset_statistics('train')
    verifier.visualize_random_samples('train', num_samples=3)
    
    # Verify validation set
    print("\nVerifying validation set:")
    verifier.verify_dataset_statistics('val')
    verifier.visualize_random_samples('val', num_samples=2)

if __name__ == "__main__":
    main() 