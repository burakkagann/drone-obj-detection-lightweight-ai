#!/usr/bin/env python3
"""
MobileNet-SSD Trial-1 Training Script for VisDrone Dataset
Master's Thesis: Robust Object Detection for Surveillance Drones

This script implements Phase 3 (Trial-1) training for MobileNet-SSD on the VisDrone dataset
using synthetic environmental augmentation for enhanced robustness.

Key features for Trial-1 (Phase 3):
- Synthetic environmental augmentation (fog, night, blur, rain)
- Enhanced standard augmentation pipeline
- TensorFlow-based MobileNet-SSD implementation
- Baseline vs augmented comparison for thesis methodology

Author: Burak Kağan Yılmazer
Date: January 2025
Environment: mobilenet_ssd_env
"""

import os
import sys
import logging
import argparse
import warnings
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Suppress TensorFlow warnings for clean output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=FutureWarning)

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[5]
sys.path.append(str(project_root))

try:
    import tensorflow as tf
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    import json
    import random
except ImportError as e:
    print(f"[ERROR] Failed to import required packages: {e}")
    print("Please ensure you're using the mobilenet_ssd_env environment")
    print("Activation: .\\venvs\\mobilenet_ssd_env\\Scripts\\Activate.ps1")
    sys.exit(1)

# VisDrone class configuration
VISDRONE_CLASSES = {
    'pedestrian': 0,
    'people': 1,
    'bicycle': 2,
    'car': 3,
    'van': 4,
    'truck': 5,
    'tricycle': 6,
    'awning-tricycle': 7,
    'bus': 8,
    'motor': 9
}

NUM_CLASSES = len(VISDRONE_CLASSES) + 1  # +1 for background

def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging configuration"""
    log_file = output_dir / f"mobilenet_ssd_trial1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def validate_environment() -> None:
    """Validate training environment and dependencies"""
    # Check TensorFlow and GPU
    print(f"[INFO] TensorFlow Version: {tf.__version__}")
    
    if tf.config.list_physical_devices('GPU'):
        gpu_devices = tf.config.list_physical_devices('GPU')
        for device in gpu_devices:
            print(f"[INFO] GPU Available: {device}")
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    else:
        print("[WARNING] No GPU available, training will use CPU")
    
    # Validate dataset paths
    dataset_path = project_root / "data" / "my_dataset" / "visdrone" / "mobilenet-ssd"
    if not dataset_path.exists():
        raise FileNotFoundError(f"MobileNet-SSD dataset not found: {dataset_path}")
    
    # Validate VOC format directories
    voc_path = dataset_path / "voc_format"
    if not voc_path.exists():
        raise FileNotFoundError(f"VOC format data not found: {voc_path}")
    
    train_xml_path = voc_path / "train"
    if not train_xml_path.exists():
        raise FileNotFoundError(f"Training XML annotations not found: {train_xml_path}")
    
    images_path = dataset_path / "images"
    if not images_path.exists():
        raise FileNotFoundError(f"Images directory not found: {images_path}")
    
    print(f"[INFO] Dataset validated at: {dataset_path}")
    print(f"[INFO] VOC annotations: {train_xml_path}")
    print(f"[INFO] Images directory: {images_path}")

def parse_voc_annotation(xml_file: Path, images_dir: Path) -> Optional[Dict]:
    """Parse VOC XML annotation file and return structured data"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get image information
        filename = root.find('filename').text
        image_path = images_dir / filename
        
        if not image_path.exists():
            # Try different image directory structures
            for subset in ['train', 'val']:
                alt_path = images_dir / subset / filename
                if alt_path.exists():
                    image_path = alt_path
                    break
        
        if not image_path.exists():
            print(f"[WARNING] Image not found: {filename}")
            return None
        
        size_elem = root.find('size')
        width = int(size_elem.find('width').text)
        height = int(size_elem.find('height').text)
        
        # Parse objects
        objects = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in VISDRONE_CLASSES:
                print(f"[WARNING] Unknown class: {class_name}")
                continue
            
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            # Convert to normalized coordinates [0, 1]
            norm_xmin = xmin / width
            norm_ymin = ymin / height
            norm_xmax = xmax / width
            norm_ymax = ymax / height
            
            objects.append({
                'class_id': VISDRONE_CLASSES[class_name],
                'class_name': class_name,
                'bbox': [norm_ymin, norm_xmin, norm_ymax, norm_xmax],  # TensorFlow format: [ymin, xmin, ymax, xmax]
                'area': (norm_xmax - norm_xmin) * (norm_ymax - norm_ymin)
            })
        
        return {
            'image_path': str(image_path),
            'width': width,
            'height': height,
            'objects': objects
        }
        
    except Exception as e:
        print(f"[ERROR] Failed to parse {xml_file}: {e}")
        return None

def load_dataset(voc_dir: Path, images_dir: Path, subset: str = 'train') -> List[Dict]:
    """Load dataset from VOC format annotations"""
    xml_dir = voc_dir / subset
    if not xml_dir.exists():
        raise FileNotFoundError(f"XML directory not found: {xml_dir}")
    
    dataset = []
    xml_files = list(xml_dir.glob('*.xml'))
    
    print(f"[INFO] Loading {len(xml_files)} annotations from {xml_dir}")
    
    for i, xml_file in enumerate(xml_files):
        if i % 100 == 0:
            print(f"[INFO] Processed {i}/{len(xml_files)} annotations")
        
        annotation = parse_voc_annotation(xml_file, images_dir)
        if annotation and annotation['objects']:  # Only include images with objects
            dataset.append(annotation)
    
    print(f"[INFO] Loaded {len(dataset)} valid annotations with objects")
    return dataset

def apply_fog_augmentation(image: np.ndarray, intensity: float = 0.3) -> np.ndarray:
    """Apply fog simulation to image"""
    # Create fog overlay
    fog = np.ones_like(image) * 0.8  # Light gray fog
    
    # Blend with original image
    fogged = cv2.addWeighted(image, 1 - intensity, fog, intensity, 0)
    
    return np.clip(fogged, 0, 1)

def apply_night_augmentation(image: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    """Apply night/low-light simulation to image"""
    # Gamma correction for darkness
    night_image = np.power(image, gamma)
    
    # Add some noise
    noise = np.random.normal(0, 0.02, image.shape)
    night_image = night_image + noise
    
    return np.clip(night_image, 0, 1)

def apply_motion_blur(image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Apply motion blur to image"""
    # Create motion blur kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    
    # Apply blur
    blurred = cv2.filter2D(image, -1, kernel)
    
    return blurred

def apply_rain_augmentation(image: np.ndarray, intensity: float = 0.2) -> np.ndarray:
    """Apply rain simulation to image"""
    # Create rain-like noise
    rain_drops = np.random.random(image.shape[:2])
    rain_mask = (rain_drops < 0.005).astype(np.float32)  # 0.5% rain drops
    
    # Apply rain effect
    rain_effect = np.stack([rain_mask] * 3, axis=-1) * intensity
    rainy_image = image + rain_effect
    
    return np.clip(rainy_image, 0, 1)

def apply_environmental_augmentation(image: np.ndarray) -> np.ndarray:
    """Apply random environmental augmentation"""
    augmentation_type = random.choice(['fog', 'night', 'blur', 'rain', 'none'])
    
    if augmentation_type == 'fog':
        intensity = random.uniform(0.1, 0.4)
        return apply_fog_augmentation(image, intensity)
    elif augmentation_type == 'night':
        gamma = random.uniform(0.3, 0.7)
        return apply_night_augmentation(image, gamma)
    elif augmentation_type == 'blur':
        kernel_size = random.choice([7, 11, 15])
        return apply_motion_blur(image, kernel_size)
    elif augmentation_type == 'rain':
        intensity = random.uniform(0.1, 0.3)
        return apply_rain_augmentation(image, intensity)
    else:
        return image

def preprocess_image_with_augmentation(image_path: str, target_size: Tuple[int, int] = (300, 300), 
                                     apply_augmentation: bool = True) -> np.ndarray:
    """Preprocess image for MobileNet-SSD input with optional augmentation"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to target size
    image = cv2.resize(image, target_size)
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Apply environmental augmentation (Phase 3 specific)
    if apply_augmentation and random.random() < 0.6:  # 60% chance of augmentation
        image = apply_environmental_augmentation(image)
    
    # Standard augmentation
    if apply_augmentation:
        # Random brightness
        if random.random() < 0.3:
            brightness_factor = random.uniform(0.8, 1.2)
            image = image * brightness_factor
        
        # Random horizontal flip
        if random.random() < 0.5:
            image = np.fliplr(image)
    
    # MobileNet preprocessing (normalize to [-1, 1])
    image = (image - 0.5) * 2.0
    
    return np.clip(image, -1, 1)

def create_tf_dataset_with_augmentation(annotations: List[Dict], batch_size: int = 16, 
                                      target_size: Tuple[int, int] = (300, 300), 
                                      apply_augmentation: bool = True) -> tf.data.Dataset:
    """Create TensorFlow dataset from annotations with augmentation"""
    
    def generator():
        for annotation in annotations:
            try:
                # Load and preprocess image with augmentation
                image = preprocess_image_with_augmentation(
                    annotation['image_path'], 
                    target_size, 
                    apply_augmentation
                )
                
                # Prepare ground truth data
                num_objects = len(annotation['objects'])
                if num_objects == 0:
                    continue
                
                # Create ground truth tensors
                boxes = np.array([obj['bbox'] for obj in annotation['objects']], dtype=np.float32)
                classes = np.array([obj['class_id'] + 1 for obj in annotation['objects']], dtype=np.int32)  # +1 for background
                
                yield image, {'boxes': boxes, 'classes': classes}
                
            except Exception as e:
                print(f"[WARNING] Error processing {annotation['image_path']}: {e}")
                continue
    
    # Create dataset
    output_signature = (
        tf.TensorSpec(shape=target_size + (3,), dtype=tf.float32),
        {
            'boxes': tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            'classes': tf.TensorSpec(shape=(None,), dtype=tf.int32)
        }
    )
    
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )
    
    # Batch and prefetch
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=(
            target_size + (3,),
            {
                'boxes': [None, 4],
                'classes': [None]
            }
        ),
        padding_values=(
            0.0,
            {
                'boxes': 0.0,
                'classes': 0
            }
        )
    )
    
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def create_mobilenet_ssd_model(num_classes: int = 11, input_shape: Tuple[int, int, int] = (300, 300, 3)) -> tf.keras.Model:
    """Create MobileNet-SSD model using TensorFlow/Keras"""
    
    # MobileNetV2 backbone
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        alpha=1.0
    )
    
    # Freeze backbone initially (for faster training)
    backbone.trainable = False
    
    # Get feature maps from different layers for multi-scale detection
    layer_names = [
        'block_6_expand_relu',   # 19x19
        'block_13_expand_relu',  # 10x10
        'out_relu'               # 10x10
    ]
    
    feature_maps = [backbone.get_layer(name).output for name in layer_names]
    
    # Create SSD detection heads
    predictions = []
    
    for i, feature_map in enumerate(feature_maps):
        # Classification head
        cls_head = tf.keras.layers.Conv2D(
            num_classes * 4,  # 4 default boxes per cell
            kernel_size=3,
            padding='same',
            activation='softmax',
            name=f'cls_head_{i}'
        )(feature_map)
        
        # Regression head  
        reg_head = tf.keras.layers.Conv2D(
            4 * 4,  # 4 coordinates * 4 default boxes
            kernel_size=3,
            padding='same',
            name=f'reg_head_{i}'
        )(feature_map)
        
        predictions.extend([cls_head, reg_head])
    
    # Create model
    model = tf.keras.Model(inputs=backbone.input, outputs=predictions)
    
    return model

def train_mobilenet_ssd_trial1(epochs: int = 50, quick_test: bool = False) -> Path:
    """
    Train MobileNet-SSD Trial-1 (Phase 3) model on VisDrone dataset
    Using synthetic environmental augmentation for enhanced robustness
    
    Args:
        epochs: Number of training epochs (default: 50)
        quick_test: If True, use minimal settings for quick validation
    
    Returns:
        Path to training results directory
    """
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "runs" / "train" / f"mobilenet_ssd_trial1_phase3_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("[START] MobileNet-SSD Trial-1 (Phase 3) Training Started")
    logger.info("METHODOLOGY: Phase 3 - Synthetic Environmental Augmentation")
    logger.info(f"Output Directory: {output_dir}")
    
    try:
        # Validate environment
        validate_environment()
        
        # Dataset paths
        dataset_path = project_root / "data" / "my_dataset" / "visdrone" / "mobilenet-ssd"
        voc_path = dataset_path / "voc_format"
        images_path = dataset_path / "images"
        
        # Load training data
        logger.info("[DATA] Loading training annotations...")
        train_annotations = load_dataset(voc_path, images_path, 'train')
        
        # Split for validation (80-20 split)
        train_data, val_data = train_test_split(
            train_annotations, 
            test_size=0.2, 
            random_state=42
        )
        
        logger.info(f"[DATA] Training samples: {len(train_data)}")
        logger.info(f"[DATA] Validation samples: {len(val_data)}")
        
        # Quick test adjustments
        if quick_test:
            train_data = train_data[:100]  # Use only 100 samples
            val_data = val_data[:50]       # Use only 50 samples
            epochs = 10
            logger.info("[INFO] Quick test mode enabled (100 train, 50 val samples, 10 epochs)")
        
        # Create TensorFlow datasets with augmentation
        batch_size = 8 if quick_test else 16
        
        logger.info("[DATA] Creating TensorFlow datasets with augmentation...")
        train_dataset = create_tf_dataset_with_augmentation(
            train_data, 
            batch_size=batch_size, 
            apply_augmentation=True
        )
        val_dataset = create_tf_dataset_with_augmentation(
            val_data, 
            batch_size=batch_size, 
            apply_augmentation=False  # No augmentation for validation
        )
        
        # Create model
        logger.info("[MODEL] Creating MobileNet-SSD model...")
        model = create_mobilenet_ssd_model(num_classes=NUM_CLASSES)
        
        # Compile model with optimized settings for Trial-1
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Slightly lower LR for stability
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"[MODEL] Model created with {model.count_params():,} parameters")
        
        # Training configuration
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=output_dir / 'best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,  # Increased patience for augmented training
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                output_dir / 'training_log.csv'
            )
        ]
        
        # Log Phase 3 specific features
        logger.info("[PHASE-3] Synthetic Environmental Augmentation Features:")
        logger.info("  - SYNTHETIC AUGMENTATION: Fog, night, blur, rain simulation")
        logger.info("  - ENHANCED STANDARD AUGMENTATION: Brightness, horizontal flip")
        logger.info("  - AUGMENTATION PROBABILITY: 60% environmental, 30% brightness, 50% flip")
        logger.info("  - METHODOLOGY COMPLIANCE: Phase 3 augmented vs Phase 2 baseline")
        logger.info("  - TARGET PERFORMANCE: >20% mAP@0.5 (improvement over baseline)")
        logger.info("")
        logger.info("[OPTIMIZATIONS] Key Trial-1 adaptations:")
        logger.info("  - Learning rate: 0.0005 (reduced for stability with augmentation)")
        logger.info("  - Early stopping: 15 epochs patience (increased for augmented data)")
        logger.info("  - Environmental augmentation: 60% probability per image")
        logger.info("  - Multi-environmental: Fog, night, motion blur, rain effects")
        logger.info("  - Augmentation intensity: Randomized within realistic ranges")
        
        # Start training
        logger.info("[TRAINING] Starting MobileNet-SSD Trial-1 (Phase 3) training...")
        
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        model.save(output_dir / 'final_model.h5')
        
        # Save training history
        history_dict = {
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'accuracy': history.history.get('accuracy', []),
            'val_accuracy': history.history.get('val_accuracy', [])
        }
        
        with open(output_dir / 'training_history.json', 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        # Training completed
        logger.info("[SUCCESS] Training completed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        
        # Log final metrics
        final_loss = history.history['val_loss'][-1]
        final_acc = history.history.get('val_accuracy', [0])[-1]
        
        logger.info("[TRIAL-1] Final Trial-1 Metrics:")
        logger.info(f"  • Validation Loss: {final_loss:.4f}")
        logger.info(f"  • Validation Accuracy: {final_acc:.4f}")
        logger.info(f"  • Model Parameters: {model.count_params():,}")
        logger.info(f"  • Training Epochs: {len(history.history['loss'])}")
        
        # Expected performance analysis
        logger.info("[ANALYSIS] Synthetic Augmentation Impact Analysis:")
        logger.info("  - Methodology compliance: Phase 3 augmentation vs Phase 2 baseline")
        logger.info("  - Expected improvement: >2% mAP@0.5 over baseline")
        logger.info("  - Key factors: Environmental robustness, enhanced augmentation")
        logger.info("  - Research value: Quantifies synthetic data benefits for thesis")
        logger.info("  - Next step: Compare against baseline performance")
        
        return output_dir
        
    except Exception as e:
        logger.error(f"[ERROR] Training failed: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="MobileNet-SSD Trial-1 (Phase 3) Training for VisDrone")
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of training epochs (default: 50)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test with reduced settings (10 epochs, 100 samples)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("MobileNet-SSD Trial-1 (Phase 3) Training - VisDrone Dataset")
    print("METHODOLOGY: Synthetic Environmental Augmentation")
    print("="*80)
    print(f"Epochs: {args.epochs}")
    print(f"Quick Test: {args.quick_test}")
    print(f"Phase: 3 (Synthetic Augmentation)")
    print(f"Baseline Comparison: Phase 2 performance")
    print(f"Target: >20% mAP@0.5 (improvement over baseline)")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    try:
        output_dir = train_mobilenet_ssd_trial1(
            epochs=args.epochs,
            quick_test=args.quick_test
        )
        
        print("\n" + "="*80)
        print("[SUCCESS] MobileNet-SSD Trial-1 (Phase 3) Training Complete!")
        print(f"Results: {output_dir}")
        print("Expected: Improved performance over Phase 2 baseline")
        print("Target: >20% mAP@0.5 with synthetic environmental augmentation")
        print("="*80)
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()