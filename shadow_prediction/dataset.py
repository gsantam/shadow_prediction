"""
Dataset for synthetic shadow images with light source positions.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.synthetic_data import generate_synthetic_scene


class SyntheticShadowDataset(Dataset):
    """
    Dataset that generates synthetic shadow images on-the-fly.
    Each sample is a (image, light_position) pair.
    """
    
    def __init__(self, num_samples=1000, img_width=224, img_height=224):
        """
        Args:
            num_samples: Number of samples in the dataset
            img_width: Width of generated images
            img_height: Height of generated images
        """
        self.num_samples = num_samples
        self.img_width = img_width
        self.img_height = img_height
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Generate a synthetic image and return (image, light_position).
        
        Returns:
            image: torch.Tensor of shape (1, H, W) with values in [0, 1]
            light_pos: torch.Tensor of shape (3,) with (x, y, z) coordinates
        """
        # Generate synthetic scene with light position
        image, scene_data = generate_synthetic_scene(
            random_projection=False,
            randomize_objects=True,
            add_shadows=True,
            img_width=self.img_width,
            img_height=self.img_height,
            return_scene_data=True
        )
        
        # Convert to torch tensors
        image_tensor = torch.from_numpy(image).float()  # (1, H, W)
        light_pos = torch.from_numpy(scene_data['light_pos']).float()  # (3,)
        
        return image_tensor, light_pos


def get_dataloaders(train_size=800, val_size=200, batch_size=16, 
                    img_width=224, img_height=224, num_workers=0):
    """
    Create training and validation dataloaders.
    
    Args:
        train_size: Number of training samples
        val_size: Number of validation samples
        batch_size: Batch size
        img_width: Image width
        img_height: Image height
        num_workers: Number of worker processes for data loading
    
    Returns:
        train_loader, val_loader
    """
    train_dataset = SyntheticShadowDataset(
        num_samples=train_size,
        img_width=img_width,
        img_height=img_height
    )
    
    val_dataset = SyntheticShadowDataset(
        num_samples=val_size,
        img_width=img_width,
        img_height=img_height
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader
