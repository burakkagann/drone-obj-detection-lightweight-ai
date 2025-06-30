import os
import sys

# Add src directory to Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(src_path)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm

from models.mobilenet_ssd import create_mobilenetv2_ssd_lite
from utils.box_utils import SSDBoxCoder
from utils.datasets import VOCDataset

class MultiBoxLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions, targets):
        """Simple L1 loss for demonstration"""
        loc_preds, conf_preds = predictions
        total_loss = 0
        batch_size = loc_preds.size(0)
        
        for idx in range(batch_size):
            # Get target boxes and labels for this item
            target_boxes = targets[idx]['boxes']
            target_labels = targets[idx]['labels']
            
            if len(target_boxes) == 0:
                continue
            
            # Simple L1 loss for demonstration
            # In practice, you would use a proper MultiBox loss
            loc_loss = nn.functional.l1_loss(loc_preds[idx], target_boxes)
            conf_loss = nn.functional.cross_entropy(conf_preds[idx], target_labels)
            
            total_loss += loc_loss + conf_loss
        
        return total_loss / batch_size

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (images, targets) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            
    return total_loss / len(val_loader)

def main():
    # Configuration
    num_classes = 15  # DOTA has 15 classes
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data paths
    data_root = "data/my_dataset/dota/dota-v1.0/mobilenet-ssd"
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = VOCDataset(
        os.path.join(data_root, "train"),
        transform=transform
    )
    val_dataset = VOCDataset(
        os.path.join(data_root, "val"),
        transform=transform
    )
    
    print(f"Found {len(train_dataset)} training samples")
    print(f"Found {len(val_dataset)} validation samples")
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: No samples found in the dataset.")
        print(f"Please check if the following paths exist:")
        print(f"- {os.path.join(data_root, 'train/Annotations')}")
        print(f"- {os.path.join(data_root, 'train/JPEGImages')}")
        print(f"- {os.path.join(data_root, 'val/Annotations')}")
        print(f"- {os.path.join(data_root, 'val/JPEGImages')}")
        return
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=4)
    
    # Create model
    model = create_mobilenetv2_ssd_lite(num_classes)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = MultiBoxLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 
                      'checkpoints/mobilenet_ssd_dota_best.pth')
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, f'checkpoints/mobilenet_ssd_dota_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    main() 