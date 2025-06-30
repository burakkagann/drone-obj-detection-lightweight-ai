import os
import cv2
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon
import json
import shutil

class DOTAtoNanoDetConverter:
    def __init__(self, dota_base_path, output_base_path):
        self.dota_base_path = dota_base_path
        self.output_base_path = output_base_path
        self.categories = self._get_dota_categories()
        
    def _get_dota_categories(self):
        """DOTA v1.0 categories with numeric indices"""
        return {
            str(i): i for i in range(15)  # DOTA has 15 categories, 0-14
        }

    def _convert_polygon_to_bbox(self, points):
        """Convert polygon points to bbox [x_min, y_min, x_max, y_max]"""
        polygon = Polygon(points)
        bounds = polygon.bounds
        return [bounds[0], bounds[1], bounds[2], bounds[3]]

    def _prepare_output_dirs(self, split):
        """Create output directories for images and annotations"""
        os.makedirs(os.path.join(self.output_base_path, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(self.output_base_path, 'annotations'), exist_ok=True)

    def convert_split(self, split='train'):
        """Convert a dataset split (train/val/test)"""
        print(f"Converting {split} split...")
        
        self._prepare_output_dirs(split)
        images_path = os.path.join(self.dota_base_path, 'images', split)
        labels_path = os.path.join(self.dota_base_path, 'labels', f'{split}_original')
        
        annotations = {
            'images': [],
            'annotations': [],
            'categories': [{'id': i, 'name': f'category_{i}'} for i in range(15)]
        }
        
        image_id = 1
        ann_id = 1
        
        # Process each image and its annotations
        for img_file in tqdm(os.listdir(images_path)):
            if not img_file.endswith(('.jpg', '.png')):
                continue
                
            # Image info
            img_path = os.path.join(images_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue
            
            height, width = img.shape[:2]
            
            # Copy image to output directory
            shutil.copy2(img_path, os.path.join(self.output_base_path, 'images', split, img_file))
            
            # Add image info
            annotations['images'].append({
                'id': image_id,
                'file_name': img_file,
                'height': height,
                'width': width
            })
            
            # Process annotations if they exist
            label_file = os.path.join(labels_path, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))
            if os.path.exists(label_file):
                try:
                    with open(label_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) < 9:  # Skip invalid annotations
                            continue
                            
                        # Parse polygon points
                        try:
                            points = [[float(parts[i]), float(parts[i+1])] for i in range(0, 8, 2)]
                            category = parts[8]
                            difficult = int(parts[9]) if len(parts) > 9 else 0
                            
                            if category not in self.categories:
                                print(f"Warning: Unknown category {category} in {label_file}")
                                continue
                            
                            # Convert polygon to bbox
                            bbox = self._convert_polygon_to_bbox(points)
                            
                            # Add annotation
                            annotations['annotations'].append({
                                'id': ann_id,
                                'image_id': image_id,
                                'category_id': self.categories[category],
                                'bbox': bbox,
                                'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                                'iscrowd': 0,
                                'difficult': difficult
                            })
                            ann_id += 1
                        except Exception as e:
                            print(f"Warning: Error processing annotation in {label_file}: {e}")
                            continue
                except Exception as e:
                    print(f"Warning: Error reading label file {label_file}: {e}")
            else:
                print(f"Warning: No label file found for {img_file}")
            
            image_id += 1
        
        # Save annotations
        output_file = os.path.join(self.output_base_path, 'annotations', f'instances_{split}.json')
        with open(output_file, 'w') as f:
            json.dump(annotations, f)
        
        print(f"Converted {split} split: {image_id-1} images, {ann_id-1} annotations")

def main():
    # Paths
    dota_base = "data/my_dataset/dota/dota-v1.0"
    output_base = "data/my_dataset/dota/dota-v1.0/nano-det"
    
    # Create converter
    converter = DOTAtoNanoDetConverter(dota_base, output_base)
    
    # Convert splits
    converter.convert_split('train')
    converter.convert_split('val')
    
    print("Conversion complete!")

if __name__ == '__main__':
    main() 