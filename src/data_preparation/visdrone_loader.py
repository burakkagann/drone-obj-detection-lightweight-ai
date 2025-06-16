"""
VisDrone dataset loader for MobileNet-SSD training.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from xml.etree import ElementTree as ET

class VisDroneDataLoader:
    def __init__(self, config):
        self.config = config
        self.input_shape = tuple(config['model']['input_shape'])
        self.batch_size = config['model']['batch_size']
        self.class_names = config['dataset']['class_names']
        self.num_classes = len(self.class_names)
        self.class_map = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Calculate number of anchor boxes per feature map cell
        self.aspect_ratios = [1.0, 2.0, 0.5, 3.0, 1.0/3.0]
        self.num_anchors = len(self.aspect_ratios)
        self.num_feature_maps = 5  # Number of feature maps in SSD
        self.total_anchors = 2200  # Total number of anchor boxes
        
        # Calculate number of anchor boxes per feature map cell
        self.aspect_ratios = [1.0, 2.0, 0.5, 3.0, 1.0/3.0]
        self.num_anchors = len(self.aspect_ratios)
        self.num_feature_maps = 5  # Number of feature maps in SSD
        self.total_anchors = 2200  # Total number of anchor boxes
        
    def load_dataset(self, split='train'):
        """Load dataset for specified split."""
        if split == 'train':
            data_dir = os.path.join(self.config['dataset']['root_dir'], self.config['dataset']['train_dir'])
        elif split == 'val':
            data_dir = os.path.join(self.config['dataset']['root_dir'], self.config['dataset']['val_dir'])
        else:
            data_dir = os.path.join(self.config['dataset']['root_dir'], self.config['dataset']['test_dir'])
            
        # Get annotation directory
        annotation_dir = os.path.join(
            self.config['dataset']['root_dir'],
            self.config['dataset']['voc_format_dir'],
            split
        )
        
        # Verify directories exist
        if not os.path.exists(data_dir):
            raise ValueError(f"Image directory not found: {data_dir}")
        if not os.path.exists(annotation_dir):
            raise ValueError(f"Annotation directory not found: {annotation_dir}")
            
        print(f"\nLoading {split} dataset:")
        print(f"Images from: {data_dir}")
        print(f"Annotations from: {annotation_dir}")
        
        # Get image and annotation paths
        image_paths = []
        annotation_paths = []
        for img_name in os.listdir(data_dir):
            if img_name.endswith(('.jpg', '.png')):
                img_path = os.path.join(data_dir, img_name)
                ann_path = os.path.join(annotation_dir, img_name.rsplit('.', 1)[0] + '.xml')
                if os.path.exists(ann_path):
                    image_paths.append(img_path)
                    annotation_paths.append(ann_path)
        
        if not image_paths:
            raise ValueError(f"No valid image-annotation pairs found in {data_dir}")
            
        print(f"Found {len(image_paths)} valid image-annotation pairs")
        
        # Create tf.data.Dataset
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, annotation_paths))
        dataset = dataset.map(self._parse_data, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _parse_data(self, image_path, annotation_path):
        """Parse image and annotation data."""
        def _process_data(img_path, ann_path):
            # Convert tensor paths to strings
            img_path = img_path.numpy().decode('utf-8')
            ann_path = ann_path.numpy().decode('utf-8')
            
            # Read and preprocess image
            image = tf.io.read_file(img_path)
            image = tf.image.decode_image(image, channels=3)
            image = tf.cast(image, tf.float32)
            image = tf.image.resize(image, self.input_shape[:2])
            
            # Apply MobileNet-SSD preprocessing
            image = image - self.config['preprocessing']['mean']
            image = image * self.config['preprocessing']['scale']
            if self.config['preprocessing']['bgr']:
                image = tf.reverse(image, axis=[-1])
                
            # Parse annotation
            boxes, labels = self._parse_annotation(ann_path)
            
            # Convert to SSD format
            cls_targets = np.zeros((self.total_anchors, self.num_classes + 1))
            reg_targets = np.zeros((self.total_anchors, 4))
            
            # For now, just use dummy targets (this needs to be implemented properly)
            cls_targets[:, 0] = 1  # Background class
            
            return image, cls_targets, reg_targets
        
        # Call the function and unpack the results
        image, cls_targets, reg_targets = tf.py_function(
            _process_data,
            [image_path, annotation_path],
            [tf.float32, tf.float32, tf.float32]
        )
        
        # Set shapes
        image.set_shape(self.input_shape)
        cls_targets.set_shape([self.total_anchors, self.num_classes + 1])
        reg_targets.set_shape([self.total_anchors, 4])
        
        return image, {'classification': cls_targets, 'regression': reg_targets}
    
    def _parse_annotation(self, annotation_path):
        """Parse VOC format annotation file."""
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        boxes = []
        labels = []
        
        img_width = float(root.find('size/width').text)
        img_height = float(root.find('size/height').text)
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name in self.class_map:
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text) / img_width
                ymin = float(bbox.find('ymin').text) / img_height
                xmax = float(bbox.find('xmax').text) / img_width
                ymax = float(bbox.find('ymax').text) / img_height
                
                # Ensure valid box coordinates
                if xmin < xmax and ymin < ymax and xmin >= 0 and ymin >= 0 and xmax <= 1 and ymax <= 1:
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(self.class_map[class_name])
        
        # Convert to numpy arrays
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        
        # Handle empty annotations
        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int32)
            
        return boxes, labels 