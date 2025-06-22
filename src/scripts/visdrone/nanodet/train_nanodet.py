"""Training script for NanoDet with VisDrone dataset"""

import os
import sys
import argparse
from pathlib import Path

def setup_paths():
    """Add necessary paths to system path"""
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent.parent
    model_dir = project_root / "src" / "models" / "nanodet"
    if not model_dir.exists():
        print(f"Error: NanoDet directory not found at {model_dir}")
        sys.exit(1)
    return project_root

def train(config_path):
    """Train NanoDet model"""
    try:
        # Try importing NanoDet modules
        print("Importing NanoDet modules...")
        from nanodet.trainer.task import TrainingTask
        from nanodet.util import mkdir, cfg, load_config
        print("NanoDet modules imported successfully")
        
        # Load config
        print(f"Loading config from {config_path}")
        load_config(cfg, str(config_path))
        print("Config loaded successfully")
        
        # Create save directory
        save_dir = Path(cfg.save_dir)
        print(f"Creating save directory at {save_dir}")
        mkdir(local_rank=0, path=str(save_dir))
        
        # Initialize training task
        print("Initializing training task...")
        task = TrainingTask(cfg)
        
        # Start training
        print("Starting training...")
        task.train()

    except ImportError as e:
        print(f"Error importing NanoDet modules: {e}")
        print("Please make sure setup_nanodet.py was run successfully")
        sys.exit(1)
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/nanodet/train_config_nanodet_visdrone.yaml',
                      help='path to config file')
    args = parser.parse_args()

    project_root = setup_paths()
    config_path = project_root / args.config

    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    train(config_path)

if __name__ == '__main__':
    main() 