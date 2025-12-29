"""
Dataset for shadow mask generation task.
Input: image without shadows + light position
Output: binary shadow mask
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.synthetic_data import generate_synthetic_scene


class ShadowGenerationDataset(Dataset):
    """
    Dataset for shadow mask generation.
    Returns: (image_without_shadows, light_position) -> shadow_mask
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
        Generate a synthetic scene and return input/target for shadow generation.
        
        Returns:
            input_data: Dict with:
                - 'image': torch.Tensor of shape (1, H, W) - binary image without shadows
                - 'light_pos': torch.Tensor of shape (3,) - (x, y, z) light coordinates
            target: torch.Tensor of shape (1, H, W) - binary shadow mask
        """
        # Generate scene with separate masks
        image_without_shadows, shadow_mask, image_with_shadows, scene_data = generate_synthetic_scene(
            random_projection=False,
            randomize_objects=True,
            add_shadows=True,
            img_width=self.img_width,
            img_height=self.img_height,
            return_separate_masks=True
        )
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image_without_shadows).float()  # (1, H, W)
        light_pos = torch.from_numpy(scene_data['light_pos']).float()   # (3,)
        shadow_mask_tensor = torch.from_numpy(shadow_mask).float()      # (1, H, W)
        
        # Return as dict for clarity
        input_data = {
            'image': image_tensor,
            'light_pos': light_pos
        }
        
        return input_data, shadow_mask_tensor


def get_dataloaders_shadow_gen(train_size=10000, val_size=2000, batch_size=32, 
                                img_width=224, img_height=224, num_workers=0):
    """
    Create train and validation dataloaders for shadow generation.
    
    Args:
        train_size: Number of training samples
        val_size: Number of validation samples
        batch_size: Batch size
        img_width: Image width
        img_height: Image height
        num_workers: Number of data loading workers
    
    Returns:
        train_loader, val_loader
    """
    train_dataset = ShadowGenerationDataset(
        num_samples=train_size,
        img_width=img_width,
        img_height=img_height
    )
    
    val_dataset = ShadowGenerationDataset(
        num_samples=val_size,
        img_width=img_width,
        img_height=img_height
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader
