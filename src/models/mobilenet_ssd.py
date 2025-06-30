import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from typing import List, Tuple

class MobileNetV2SSDLite(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        
        # Load pretrained MobileNetV2 backbone
        backbone = mobilenet_v2(pretrained=True)
        self.features = backbone.features
        
        # SSD additional feature layers
        self.additional_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1280, 512, kernel_size=1),  # MobileNetV2 last layer has 1280 channels
                nn.ReLU6(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                nn.ReLU6(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1),
                nn.ReLU6(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU6(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=1),
                nn.ReLU6(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU6(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=1),
                nn.ReLU6(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU6(inplace=True),
            ),
        ])
        
        # Regression heads
        self.loc_layers = nn.ModuleList([
            nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)
            for in_channels, num_anchors in [(96, 6), (1280, 6), (512, 6), (256, 6), (256, 6), (256, 6)]
        ])
        
        # Classification heads
        self.conf_layers = nn.ModuleList([
            nn.Conv2d(in_channels, num_anchors * (num_classes + 1), kernel_size=3, padding=1)
            for in_channels, num_anchors in [(96, 6), (1280, 6), (512, 6), (256, 6), (256, 6), (256, 6)]
        ])
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.additional_blocks.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
        for m in self.loc_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
        for m in self.conf_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Feature maps for detection
        detection_features = []
        
        # Get feature maps from backbone
        for i, layer in enumerate(self.features):
            x = layer(x)
            # Save feature maps for detection
            if i == 13:  # First detection layer from backbone (conv_13)
                detection_features.append(x)
        
        # Last feature map from backbone
        detection_features.append(x)
        
        # Get additional feature maps
        for layer in self.additional_blocks:
            x = layer(x)
            detection_features.append(x)
        
        # Apply regression heads
        loc = []
        for x, l in zip(detection_features, self.loc_layers):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        loc = loc.view(loc.size(0), -1, 4)
        
        # Apply classification heads
        conf = []
        for x, c in zip(detection_features, self.conf_layers):
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        conf = conf.view(conf.size(0), -1, self.num_classes + 1)
        
        return loc, conf

def create_mobilenetv2_ssd_lite(num_classes: int) -> MobileNetV2SSDLite:
    """Create MobileNetV2-SSD-Lite model
    
    Args:
        num_classes: Number of classes to detect
        
    Returns:
        MobileNetV2-SSD-Lite model
    """
    return MobileNetV2SSDLite(num_classes) 