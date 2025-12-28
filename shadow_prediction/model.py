"""
ResNet-18 model for predicting light source position from shadow images.
"""

import torch
import torch.nn as nn
from torchvision import models


class LightPositionPredictor(nn.Module):
    """
    ResNet-18 based model for predicting 3D light source position.
    Modified to accept 1-channel grayscale images and output 3D coordinates.
    """
    
    def __init__(self, pretrained=False):
        """
        Args:
            pretrained: Whether to use pretrained ImageNet weights
        """
        super(LightPositionPredictor, self).__init__()
        
        # Load ResNet-18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Modify first conv layer to accept 1-channel input instead of 3
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Get number of features from the last layer
        num_features = self.resnet.fc.in_features
        
        # Replace final fully connected layer to output 3 values (x, y, z)
        self.resnet.fc = nn.Linear(num_features, 3)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 1, H, W)
        
        Returns:
            Light position predictions of shape (batch_size, 3)
        """
        return self.resnet(x)


def create_model(pretrained=False, device=None):
    """
    Create and initialize the light position predictor model.
    
    Args:
        pretrained: Whether to use pretrained ImageNet weights
        device: Device to place the model on (cuda/mps/cpu)
    
    Returns:
        model: LightPositionPredictor instance
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    model = LightPositionPredictor(pretrained=pretrained)
    model = model.to(device)
    
    return model
