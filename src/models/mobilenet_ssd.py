"""
MobileNet-SSD model implementation for object detection on drone imagery.
"""

import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.regularizers import l2

class MobileNetSSD:
    def __init__(self, config):
        self.config = config
        self.num_classes = config['model']['num_classes']
        self.input_shape = tuple(config['model']['input_shape'])
        
    def build_model(self):
        """Build MobileNet-SSD model architecture."""
        # Base MobileNet model without top layers
        base_model = MobileNet(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet' if self.config['model']['pretrained'] else None
        )
        
        # Freeze early layers for transfer learning
        for layer in base_model.layers[:20]:
            layer.trainable = False
            
        # SSD specific layers
        source_layers = [
            base_model.get_layer('conv_pw_11_relu').output,  # 19x19
            base_model.get_layer('conv_pw_13_relu').output   # 10x10
        ]
        
        # Additional feature layers
        x = source_layers[-1]
        source_layers.extend(self._build_additional_features(x))
        
        # Build multibox head
        cls_outputs = []
        reg_outputs = []
        
        for i, source in enumerate(source_layers):
            name = f"pred_{i}"
            cls_head, reg_head = self._build_multibox_head(source, name)
            cls_outputs.append(cls_head)
            reg_outputs.append(reg_head)
            
        # Concatenate all predictions
        classification = layers.Concatenate(axis=1, name='classification')(cls_outputs)
        regression = layers.Concatenate(axis=1, name='regression')(reg_outputs)
            
        model = Model(
            inputs=base_model.input,
            outputs=[classification, regression],
            name='mobilenet_ssd'
        )
        
        return model
    
    def _build_additional_features(self, x):
        """Build additional feature layers."""
        additional_layers = []
        
        # 5x5
        x = layers.Conv2D(256, 1, padding='same', kernel_regularizer=l2(self.config['model']['weight_decay']))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(512, 3, strides=2, padding='same', kernel_regularizer=l2(self.config['model']['weight_decay']))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        additional_layers.append(x)
        
        # 3x3
        x = layers.Conv2D(128, 1, padding='same', kernel_regularizer=l2(self.config['model']['weight_decay']))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(256, 3, strides=2, padding='same', kernel_regularizer=l2(self.config['model']['weight_decay']))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        additional_layers.append(x)
        
        # 1x1
        x = layers.Conv2D(128, 1, padding='same', kernel_regularizer=l2(self.config['model']['weight_decay']))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(256, 3, padding='valid', kernel_regularizer=l2(self.config['model']['weight_decay']))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        additional_layers.append(x)
        
        return additional_layers
    
    def _build_multibox_head(self, source_layer, name_prefix):
        """Build multibox head for classification and regression."""
        aspect_ratios = [1.0, 2.0, 0.5, 3.0, 1.0/3.0]
        num_priors = len(aspect_ratios)
        
        # Classification head
        cls_head = layers.Conv2D(
            num_priors * (self.num_classes + 1),
            3,
            padding='same',
            kernel_regularizer=l2(self.config['model']['weight_decay']),
            name=f'{name_prefix}_cls'
        )(source_layer)
        cls_head = layers.Reshape((-1, self.num_classes + 1), name=f'{name_prefix}_cls_reshape')(cls_head)
        cls_head = layers.Softmax(axis=-1, name=f'{name_prefix}_cls_softmax')(cls_head)
        
        # Regression head
        reg_head = layers.Conv2D(
            num_priors * 4,
            3,
            padding='same',
            kernel_regularizer=l2(self.config['model']['weight_decay']),
            name=f'{name_prefix}_reg'
        )(source_layer)
        reg_head = layers.Reshape((-1, 4), name=f'{name_prefix}_reg_reshape')(reg_head)
        
        return cls_head, reg_head
    
    def get_loss(self):
        """Get loss functions for training."""
        return {
            'classification': tf.keras.losses.CategoricalCrossentropy(),
            'regression': tf.keras.losses.Huber()
        }
        
    def get_metrics(self):
        """Get metrics for model evaluation."""
        return {
            'classification': 'accuracy',
            'regression': 'mse'
        } 