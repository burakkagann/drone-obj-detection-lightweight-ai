import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Add src directory to Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(src_path)

from models.mobilenet_ssd import create_mobilenetv2_ssd_lite
from utils.datasets import VOCDataset

def test_basic_training():
    """Simple test to verify if the training pipeline works"""
    print("Starting basic training test...")
    
    # Basic configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data paths and transforms
    data_root = "data/my_dataset/dota/dota-v1.0/mobilenet-ssd"
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
    ])
    
    try:
        # Try loading the dataset
        print("\nTesting dataset loading...")
        train_dataset = VOCDataset(
            os.path.join(data_root, "train"),
            transform=transform
        )
        print(f"✓ Successfully loaded training dataset with {len(train_dataset)} samples")
        
        # Try creating the dataloader
        print("\nTesting dataloader creation...")
        train_loader = DataLoader(
            train_dataset, 
            batch_size=2,
            shuffle=True, 
            num_workers=0,
            collate_fn=train_dataset.collate_fn
        )
        print("✓ Successfully created dataloader")
        
        # Try creating the model
        print("\nTesting model creation...")
        model = create_mobilenetv2_ssd_lite(num_classes=15)
        model = model.to(device)
        print("✓ Successfully created model")
        
        # Create optimizer and criterion
        print("\nSetting up training components...")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loc_criterion = nn.SmoothL1Loss()  # For bounding box regression
        conf_criterion = nn.CrossEntropyLoss()  # For classification
        print("✓ Successfully created optimizer and loss functions")
        
        # Try a training iteration
        print("\nTesting training loop...")
        model.train()
        for batch_idx, (images, targets) in enumerate(train_loader):
            print(f"\nBatch {batch_idx + 1}")
            print("  Loading batch...")
            images = images.to(device)
            
            # Forward pass
            print("  Running forward pass...")
            loc_preds, conf_preds = model(images)
            print("  ✓ Forward pass successful")
            
            # Try backward pass
            print("  Running backward pass...")
            optimizer.zero_grad()
            
            # Dummy target tensors for testing
            dummy_loc_target = torch.randn_like(loc_preds)
            dummy_conf_target = torch.randint(0, 15, (conf_preds.size(0), conf_preds.size(1))).to(device)
            
            # Calculate loss
            loc_loss = loc_criterion(loc_preds, dummy_loc_target)
            conf_loss = conf_criterion(conf_preds.view(-1, 16), dummy_conf_target.view(-1))
            loss = loc_loss + conf_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            print(f"  ✓ Backward pass successful (Loss: {loss.item():.4f})")
            
            if batch_idx == 2:  # Test with 3 batches
                break
            
        print("\n✓ All training tests passed!")
        
    except Exception as e:
        print(f"\n✗ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_basic_training() 