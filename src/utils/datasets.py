import os
import torch
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from typing import Tuple, List, Dict

class VOCDataset(Dataset):
    """VOC Detection Dataset"""
    def __init__(self, root: str, transform=None):
        """
        Args:
            root: Root directory path (should contain JPEGImages and Annotations subdirectories)
            transform: Optional transform to be applied on a sample
        """
        self.root = root
        self.transform = transform
        self.images = []
        self.annotations = []
        self.class_names = [
            'background',  # Always index 0
            'plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court',
            'basketball-court', 'ground-track-field', 'harbor', 'bridge',
            'large-vehicle', 'small-vehicle', 'helicopter', 'roundabout',
            'soccer-ball-field', 'swimming-pool'
        ]
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
        
        # Get all image and annotation files
        image_dir = os.path.join(root, 'JPEGImages')
        annot_dir = os.path.join(root, 'Annotations')
        
        if not os.path.exists(image_dir):
            raise ValueError(f"Could not find JPEGImages directory in {root}")
        if not os.path.exists(annot_dir):
            raise ValueError(f"Could not find Annotations directory in {root}")
        
        for filename in os.listdir(annot_dir):
            if filename.endswith('.xml'):
                image_name = filename.replace('.xml', '.png')
                image_file = os.path.join(image_dir, image_name)
                
                if not os.path.exists(image_file):
                    image_file = os.path.join(image_dir, filename.replace('.xml', '.jpg'))
                    if not os.path.exists(image_file):
                        continue
                
                annot_file = os.path.join(annot_dir, filename)
                self.images.append(image_file)
                self.annotations.append(annot_file)
        
        print(f"Found {len(self.images)} images in {root}")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Load image
        img = Image.open(self.images[idx]).convert('RGB')
        img_width = img.size[0]
        img_height = img.size[1]
        
        # Load annotation
        boxes = []
        labels = []
        
        tree = ET.parse(self.annotations[idx])
        root = tree.getroot()
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.class_dict:
                continue
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text) / img_width
            ymin = float(bbox.find('ymin').text) / img_height
            xmax = float(bbox.find('xmax').text) / img_width
            ymax = float(bbox.find('ymax').text) / img_height
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[name])
        
        # Convert to tensor
        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        return img, {'boxes': boxes, 'labels': labels}
    
    def collate_fn(self, batch):
        """Custom collate function for DataLoader"""
        images = []
        targets = []
        for img, target in batch:
            images.append(img)
            targets.append(target)
        images = torch.stack(images, 0)
        return images, targets 