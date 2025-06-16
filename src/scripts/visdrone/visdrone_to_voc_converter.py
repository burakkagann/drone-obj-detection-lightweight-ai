"""
Script to convert VisDrone annotations to VOC format XML files.
"""

import os
import cv2
import glob
from xml.etree import ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm

class VisDroneToVOCConverter:
    def __init__(self, split='train'):
        """
        Initialize converter for a specific split (train or val)
        Args:
            split: Dataset split ('train' or 'val')
        """
        self.split = split
        
        # Input directories
        self.input_labels_dir = os.path.join('data', 'my_dataset', 'visdrone', 'mobilenet-ssd', 'labels', split)
        self.input_images_dir = os.path.join('data', 'my_dataset', 'visdrone', 'images', split)
        
        # Output directories
        self.output_voc_dir = os.path.join('data', 'my_dataset', 'visdrone', 'mobilenet-ssd', 'voc_format', split)
        
        # Class mapping for VisDrone dataset
        self.class_mapping = {
            0: 'ignored regions',
            1: 'pedestrian',
            2: 'people',
            3: 'bicycle',
            4: 'car',
            5: 'van',
            6: 'truck',
            7: 'tricycle',
            8: 'awning-tricycle',
            9: 'bus',
            10: 'motor',
            11: 'others'
        }
        
        # Create output directories
        os.makedirs(self.output_voc_dir, exist_ok=True)
        
        print(f"\nInitialized converter for {split} split:")
        print(f"Input labels: {self.input_labels_dir}")
        print(f"Input images: {self.input_images_dir}")
        print(f"Output VOC: {self.output_voc_dir}")

    def create_xml_annotation(self, image_path, boxes, image_size):
        """Create XML annotation in VOC format."""
        root = ET.Element('annotation')
        
        # Add folder and filename
        folder = ET.SubElement(root, 'folder')
        folder.text = os.path.basename(os.path.dirname(image_path))
        filename = ET.SubElement(root, 'filename')
        filename.text = os.path.basename(image_path)
        
        # Add path
        path = ET.SubElement(root, 'path')
        path.text = image_path
        
        # Add source
        source = ET.SubElement(root, 'source')
        database = ET.SubElement(source, 'database')
        database.text = 'VisDrone'
        
        # Add size information
        size = ET.SubElement(root, 'size')
        width = ET.SubElement(size, 'width')
        height = ET.SubElement(size, 'height')
        depth = ET.SubElement(size, 'depth')
        width.text = str(image_size[1])
        height.text = str(image_size[0])
        depth.text = str(image_size[2])
        
        # Add segmented
        segmented = ET.SubElement(root, 'segmented')
        segmented.text = '0'
        
        # Add object information
        for box in boxes:
            obj = ET.SubElement(root, 'object')
            
            name = ET.SubElement(obj, 'name')
            name.text = self.class_mapping[box['class_id']]
            
            pose = ET.SubElement(obj, 'pose')
            pose.text = 'Unspecified'
            
            truncated = ET.SubElement(obj, 'truncated')
            truncated.text = '0'
            
            difficult = ET.SubElement(obj, 'difficult')
            difficult.text = '0'
            
            bndbox = ET.SubElement(obj, 'bndbox')
            
            xmin = ET.SubElement(bndbox, 'xmin')
            ymin = ET.SubElement(bndbox, 'ymin')
            xmax = ET.SubElement(bndbox, 'xmax')
            ymax = ET.SubElement(bndbox, 'ymax')
            
            xmin.text = str(int(box['x1']))
            ymin.text = str(int(box['y1']))
            xmax.text = str(int(box['x2']))
            ymax.text = str(int(box['y2']))
        
        # Convert to string with proper formatting
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent='    ')
        return xml_str

    def convert_annotation(self, txt_path, image_path):
        """Convert a single annotation from VisDrone format to VOC format."""
        # Read image to get size
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            return None
            
        height, width, depth = image.shape
        
        # Read VisDrone annotation
        boxes = []
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                    
                # VisDrone format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>
                x1 = float(parts[0])
                y1 = float(parts[1])
                w = float(parts[2])
                h = float(parts[3])
                class_id = int(parts[5])
                
                # Skip ignored regions and others
                if class_id in [0, 11]:
                    continue
                
                # Calculate bottom right coordinates
                x2 = x1 + w
                y2 = y1 + h
                
                boxes.append({
                    'class_id': class_id,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2
                })
        
        if not boxes:
            return None
            
        return self.create_xml_annotation(image_path, boxes, (height, width, depth))

    def convert_dataset(self):
        """Convert entire dataset from VisDrone to VOC format."""
        print(f"\nConverting {self.split} split annotations to VOC format...")
        
        # Get all txt files
        txt_files = glob.glob(os.path.join(self.input_labels_dir, '*.txt'))
        print(f"Found {len(txt_files)} annotation files")
        
        converted_count = 0
        for txt_path in tqdm(txt_files):
            # Get corresponding image path
            image_name = os.path.splitext(os.path.basename(txt_path))[0] + '.jpg'
            image_path = os.path.join(self.input_images_dir, image_name)
            
            if not os.path.exists(image_path):
                print(f"Warning: Image not found for {txt_path}")
                continue
            
            # Convert annotation
            xml_content = self.convert_annotation(txt_path, image_path)
            if xml_content is None:
                continue
            
            # Save XML file
            xml_path = os.path.join(self.output_voc_dir, 
                                  os.path.splitext(os.path.basename(txt_path))[0] + '.xml')
            with open(xml_path, 'w') as f:
                f.write(xml_content)
            converted_count += 1
        
        print(f"✅ Converted {converted_count} annotations to VOC format") 