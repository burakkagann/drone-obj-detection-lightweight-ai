import os
import cv2
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET

def create_voc_xml(image_path, boxes, labels, output_path):
    """Create VOC format XML file for MobileNet-SSD"""
    print(f"Creating XML for {image_path} with {len(boxes)} boxes")
    root = ET.Element("annotation")
    
    # Add image info
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return
    height, width, depth = img.shape
    
    ET.SubElement(root, "folder").text = "DOTA"
    ET.SubElement(root, "filename").text = os.path.basename(image_path)
    
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)
    
    # Add object info
    for box, label in zip(boxes, labels):
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = label
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        
        bbox = ET.SubElement(obj, "bndbox")
        xmin, ymin, xmax, ymax = box
        ET.SubElement(bbox, "xmin").text = str(int(xmin))
        ET.SubElement(bbox, "ymin").text = str(int(ymin))
        ET.SubElement(bbox, "xmax").text = str(int(xmax))
        ET.SubElement(bbox, "ymax").text = str(int(ymax))
    
    tree = ET.ElementTree(root)
    print(f"Writing XML to {output_path}")
    tree.write(output_path)

def convert_dota_to_voc(dota_path, output_path, split_type):
    """Convert DOTA format annotations to VOC format"""
    print(f"Converting annotations from {dota_path} to {output_path}")
    if not os.path.exists(output_path):
        print(f"Creating output directory: {output_path}")
        os.makedirs(output_path)
        
    # DOTA classes (in order of class IDs)
    classes = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court', 
               'basketball-court', 'ground-track-field', 'harbor', 'bridge', 
               'large-vehicle', 'small-vehicle', 'helicopter', 'roundabout', 
               'soccer-ball-field', 'swimming-pool']
    
    # Process each txt file
    txt_files = [f for f in os.listdir(dota_path) if f.endswith('.txt')]
    print(f"Found {len(txt_files)} text files to process")
    
    for txt_file in tqdm(txt_files):
        image_name = txt_file.replace('.txt', '.png')
        image_path = os.path.join(os.path.dirname(os.path.dirname(dota_path)), 'images', split_type, image_name)
        
        if not os.path.exists(image_path):
            image_path = image_path.replace('.png', '.jpg')
            if not os.path.exists(image_path):
                print(f"Warning: Could not find image for {txt_file}")
                continue
        
        boxes = []
        labels = []
        
        # Read DOTA annotation
        with open(os.path.join(dota_path, txt_file), 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 10:  # Changed from 9 to 10 since we expect class_id and difficulty
                print(f"Warning: Invalid line format in {txt_file}: {line.strip()}")
                continue
                
            # Get coordinates and class
            coords = list(map(float, parts[:8]))
            try:
                class_id = int(parts[8])
                if class_id < 0 or class_id >= len(classes):
                    print(f"Warning: Invalid class ID {class_id} in {txt_file}")
                    continue
                class_name = classes[class_id]
            except (ValueError, IndexError):
                print(f"Warning: Invalid class ID format in {txt_file}: {parts[8]}")
                continue
            
            # Convert polygon to bbox
            x_coords = coords[::2]
            y_coords = coords[1::2]
            xmin, xmax = min(x_coords), max(x_coords)
            ymin, ymax = min(y_coords), max(y_coords)
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(class_name)
        
        # Create XML file
        xml_path = os.path.join(output_path, txt_file.replace('.txt', '.xml'))
        create_voc_xml(image_path, boxes, labels, xml_path)

if __name__ == "__main__":
    # Convert training set
    train_dota_path = "data/my_dataset/dota/dota-v1.0/labels/train_original"
    train_voc_path = "data/my_dataset/dota/dota-v1.0/mobilenet-ssd/train/Annotations"
    convert_dota_to_voc(train_dota_path, train_voc_path, 'train')
    
    # Convert validation set
    val_dota_path = "data/my_dataset/dota/dota-v1.0/labels/val_original"
    val_voc_path = "data/my_dataset/dota/dota-v1.0/mobilenet-ssd/val/Annotations"
    convert_dota_to_voc(val_dota_path, val_voc_path, 'val') 