"""
Training script for MobileNet-SSD on VisDrone dataset.
"""

import os
import yaml
import argparse
import tensorflow as tf
from datetime import datetime

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mobilenet_ssd import MobileNetSSD
from data_preparation.visdrone_loader import VisDroneDataLoader

def parse_args():
    parser = argparse.ArgumentParser(description='Train MobileNet-SSD on VisDrone dataset')
    parser.add_argument('--config', type=str, default='config/mobilenet_ssd_visdrone.yaml',
                      help='Path to config file')
    parser.add_argument('--gpu', type=str, default='0',
                      help='GPU device ID')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Print dataset paths and check existence
    data_root = config['dataset']['root_dir']
    train_dir = os.path.join(data_root, config['dataset']['train_dir'])
    val_dir = os.path.join(data_root, config['dataset']['val_dir'])
    voc_train_dir = os.path.join(data_root, config['dataset']['voc_format_dir'], 'train')
    voc_val_dir = os.path.join(data_root, config['dataset']['voc_format_dir'], 'val')
    
    print("\nChecking dataset directories:")
    print(f"Data root: {data_root} (exists: {os.path.exists(data_root)})")
    print(f"Train images: {train_dir} (exists: {os.path.exists(train_dir)})")
    print(f"Val images: {val_dir} (exists: {os.path.exists(val_dir)})")
    print(f"Train annotations: {voc_train_dir} (exists: {os.path.exists(voc_train_dir)})")
    print(f"Val annotations: {voc_val_dir} (exists: {os.path.exists(voc_val_dir)})")
    
    if not all(os.path.exists(d) for d in [data_root, train_dir, val_dir, voc_train_dir, voc_val_dir]):
        print("\nERROR: Some required dataset directories are missing!")
        sys.exit(1)
    
    # Create output directories
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['training']['log_dir'], exist_ok=True)
    
    # Initialize data loader
    print("\nInitializing data loader...")
    data_loader = VisDroneDataLoader(config)
    
    print("\nLoading datasets...")
    train_dataset = data_loader.load_dataset('train')
    val_dataset = data_loader.load_dataset('val')
    
    # Initialize model
    print("\nInitializing model...")
    model = MobileNetSSD(config)
    net = model.build_model()
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config['model']['learning_rate']
    )
    
    losses = model.get_loss()
    metrics = model.get_metrics()
    
    net.compile(
        optimizer=optimizer,
        loss=losses,
        metrics=metrics
    )
    
    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                config['training']['checkpoint_dir'],
                'model_{epoch:02d}_{val_loss:.4f}.h5'
            ),
            save_best_only=config['training']['save_best_only'],
            monitor='val_loss',
            mode='min'
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(
                config['training']['log_dir'],
                datetime.now().strftime("%Y%m%d-%H%M%S")
            )
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['training']['early_stopping_patience'],
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=config['training']['reduce_lr_factor'],
            patience=config['training']['reduce_lr_patience'],
            min_lr=1e-6
        )
    ]
    
    print("\nStarting training...")
    # Train model
    net.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config['model']['epochs'],
        callbacks=callbacks
    )
    
    # Save final model
    final_model_path = os.path.join(config['training']['checkpoint_dir'], 'model_final.h5')
    net.save(final_model_path)
    print(f"\n✅ Training completed! Final model saved to: {final_model_path}")

if __name__ == '__main__':
    main() 