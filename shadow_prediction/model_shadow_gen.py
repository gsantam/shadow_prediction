"""
U-Net model for shadow mask generation.
Input: image without shadows (1 channel) + light position (x,y,z)
Output: binary shadow mask (1 channel)
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two convolutions with BatchNorm and ReLU."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downsampling with maxpool then double conv."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling then double conv."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        Args:
            x1: upsampled features from decoder
            x2: skip connection features from encoder
        """
        x1 = self.up(x1)
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ShadowMaskPredictor(nn.Module):
    """
    U-Net architecture for shadow mask prediction.
    
    Input: (batch, 1, H, W) - binary image without shadows
           (batch, 3) - light position (x, y, z)
    Output: (batch, 1, H, W) - binary shadow mask
    """
    
    def __init__(self, base_channels=64):
        super().__init__()
        
        # Encoder (contracting path)
        self.inc = DoubleConv(1, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)
        
        # Bottleneck - fuse with light position
        # After 4 downsamples: 224 -> 112 -> 56 -> 28 -> 14
        # Bottleneck spatial size will be 14x14
        self.bottleneck_conv = DoubleConv(base_channels * 16 + 3, base_channels * 16)
        
        # Decoder (expanding path)
        self.up1 = Up(base_channels * 16, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels)
        
        # Output
        self.outc = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, image, light_pos):
        """
        Args:
            image: (batch, 1, H, W) - binary image without shadows
            light_pos: (batch, 3) - light position (x, y, z)
        
        Returns:
            (batch, 1, H, W) - predicted shadow mask (values in [0, 1])
        """
        # Encoder
        x1 = self.inc(image)      # (batch, 64, 224, 224)
        x2 = self.down1(x1)        # (batch, 128, 112, 112)
        x3 = self.down2(x2)        # (batch, 256, 56, 56)
        x4 = self.down3(x3)        # (batch, 512, 28, 28)
        x5 = self.down4(x4)        # (batch, 1024, 14, 14)
        
        # Bottleneck fusion with light position
        # Expand light_pos to spatial dimensions
        batch_size = image.shape[0]
        h, w = x5.shape[2], x5.shape[3]
        light_expanded = light_pos.unsqueeze(-1).unsqueeze(-1)  # (batch, 3, 1, 1)
        light_expanded = light_expanded.expand(batch_size, 3, h, w)  # (batch, 3, 14, 14)
        
        # Concatenate with bottleneck features
        x5_fused = torch.cat([x5, light_expanded], dim=1)  # (batch, 1027, 14, 14)
        x5_fused = self.bottleneck_conv(x5_fused)  # (batch, 1024, 14, 14)
        
        # Decoder with skip connections
        x = self.up1(x5_fused, x4)  # (batch, 512, 28, 28)
        x = self.up2(x, x3)          # (batch, 256, 56, 56)
        x = self.up3(x, x2)          # (batch, 128, 112, 112)
        x = self.up4(x, x1)          # (batch, 64, 224, 224)
        
        # Output
        x = self.outc(x)             # (batch, 1, 224, 224)
        x = self.sigmoid(x)          # Binary prediction in [0, 1]
        
        return x


def create_shadow_model(base_channels=64, device=None):
    """
    Create a shadow mask predictor model.
    
    Args:
        base_channels: Base number of channels (default 64)
        device: Device to move model to (cuda/mps/cpu). If None, auto-detect.
    
    Returns:
        model: ShadowMaskPredictor on specified device
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    model = ShadowMaskPredictor(base_channels=base_channels)
    model = model.to(device)
    
    print(f"Created ShadowMaskPredictor with {base_channels} base channels")
    print(f"Model moved to device: {device}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model, device
