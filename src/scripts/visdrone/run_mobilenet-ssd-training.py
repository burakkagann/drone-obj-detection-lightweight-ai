import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.utils import to_categorical

# Paths to your VisDrone dataset
train_img_dir = 'C:/Users/burak/OneDrive/Desktop/Git Repos/drone-obj-detection-lightweight-ai/data/my_dataset/visdrone/images/train'
val_img_dir = 'C:/Users/burak/OneDrive/Desktop/Git Repos/drone-obj-detection-lightweight-ai/data/my_dataset/visdrone/images/val'
train_ann_dir = 'C:/Users/burak/OneDrive/Desktop/Git Repos/drone-obj-detection-lightweight-ai/data/my_dataset/visdrone/labels/train/ssd_format'
val_ann_dir = 'C:/Users/burak/OneDrive/Desktop/Git Repos/drone-obj-detection-lightweight-ai/data/my_dataset/visdrone/labels/val/ssd_format'

# Load and preprocess dataset
def load_dataset(image_paths, annotation_paths, num_classes=10):
    images = []
    labels = []
    for image_path, annotation_path in zip(image_paths, annotation_paths):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (300, 300))  # Resize to 300x300 for SSD model input
        images.append(image)
        label = parse_annotation(annotation_path)  # Parse the annotations into bounding boxes and class ids
        labels.append(label)

    # One-hot encode the labels
    labels = [to_categorical(label, num_classes=num_classes) for label in labels]

    return np.array(images), np.array(labels)

# Parse annotations (from YOLO format, adjust as needed for SSD)
def parse_annotation(txt_path):
    print(f"Parsing annotation file: {txt_path}")  # Debugging message
    boxes = []
    try:
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                class_id = int(parts[0])  # Assuming class_id is the first entry
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # MobileNet-SSD expects normalized bounding boxes
                boxes.append([class_id, x_center, y_center, width, height])
    except Exception as e:
        print(f"Error reading annotation {txt_path}: {e}")
    return np.array(boxes)

# Define the MobileNet-SSD model architecture
def build_model(input_shape=(300, 300, 3), num_classes=10):
    model = tf.keras.Sequential([
        # First Convolution Layer
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Second Convolution Layer
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten Layer
        tf.keras.layers.Flatten(),
        
        # Fully Connected Layer
        tf.keras.layers.Dense(128, activation='relu'),
        
        # Output Layer: 10 classes for VisDrone dataset (you can adjust the number of classes)
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Classifier for 10 classes
    ])
    return model

# Compile the model with Adam optimizer and categorical crossentropy loss
model = build_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Get image and annotation paths for training and validation
train_image_paths = [os.path.join(train_img_dir, f) for f in os.listdir(train_img_dir) if f.endswith('.jpg')]
train_annotation_paths = [os.path.join(train_ann_dir, f.replace('.jpg', '.txt')) for f in os.listdir(train_img_dir) if f.endswith('.jpg')]

val_image_paths = [os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir) if f.endswith('.jpg')]
val_annotation_paths = [os.path.join(val_ann_dir, f.replace('.jpg', '.txt')) for f in os.listdir(val_img_dir) if f.endswith('.jpg')]

# Load the dataset (train and validation)
train_images, train_labels = load_dataset(train_image_paths, train_annotation_paths)
val_images, val_labels = load_dataset(val_image_paths, val_annotation_paths)

# Train the model
model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=50, batch_size=32)
