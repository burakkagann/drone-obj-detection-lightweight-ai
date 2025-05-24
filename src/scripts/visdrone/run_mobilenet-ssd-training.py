import tensorflow as tf
import numpy as np
import cv2
import os

# Paths to your VisDrone dataset
train_img_dir = 'data/my_dataset/visdrone/images/train'
val_img_dir = 'data/my_dataset/visdrone/images/val'
train_ann_dir = 'data/my_dataset/visdrone/labels/train/ssd_format'
val_ann_dir = 'data/my_dataset/visdrone/labels/val/ssd_format'

# Load and preprocess dataset
def load_dataset(image_paths, annotation_paths):
    images = []
    labels = []
    for image_path, annotation_path in zip(image_paths, annotation_paths):
        image = cv2.imread(image_path)
        image = cv2.resize(image, (300, 300))  # Resize to 300x300
        images.append(image)
        label = parse_annotation(annotation_path)  # Parse your annotations
        labels.append(label)
    return np.array(images), np.array(labels)

# Parse annotations (Pascal VOC format)
def parse_annotation(txt_path):
    # This function should parse the txt annotations and return the bounding box info
    boxes = []
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            boxes.append([class_id, x_center, y_center, width, height])
    return np.array(boxes)

# Define the model architecture (MobileNet-SSD architecture)
def build_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(21, activation='softmax'))  # 21 classes (for VOC dataset)
    return model

# Train the model
model = build_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Get image and annotation paths for training and validation
train_image_paths = [os.path.join(train_img_dir, f) for f in os.listdir(train_img_dir) if f.endswith('.jpg')]
train_annotation_paths = [os.path.join(train_ann_dir, f.replace('.jpg', '.txt')) for f in os.listdir(train_img_dir) if f.endswith('.jpg')]

val_image_paths = [os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir) if f.endswith('.jpg')]
val_annotation_paths = [os.path.join(val_ann_dir, f.replace('.jpg', '.txt')) for f in os.listdir(val_img_dir) if f.endswith('.jpg')]

# Load the dataset
train_images, train_labels = load_dataset(train_image_paths, train_annotation_paths)
val_images, val_labels = load_dataset(val_image_paths, val_annotation_paths)

# Train the model
model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=50, batch_size=32)
