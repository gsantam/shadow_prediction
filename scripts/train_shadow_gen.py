"""
Training script for shadow mask generation model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from trainer_base import BaseTrainer
from shadow_prediction.dataset_shadow_gen import get_dataloaders_shadow_gen
from shadow_prediction.model_shadow_gen import create_shadow_model


class ShadowGenTrainer(BaseTrainer):
    """Trainer for shadow mask generation model."""
    
    def __init__(self, base_channels=64, **kwargs):
        """
        Initialize shadow generation trainer.
        
        Args:
            base_channels: Base number of channels in U-Net
            **kwargs: Arguments passed to BaseTrainer
        """
        kwargs.setdefault('task_name', 'Shadow Mask Generation Training')
        kwargs.setdefault('loss_name', 'Binary Cross Entropy')
        super().__init__(**kwargs)
        self.base_channels = base_channels
    
    def create_model(self):
        """Create shadow mask prediction model."""
        model, _ = create_shadow_model(base_channels=self.base_channels, device=self.device)
        return model
    
    def create_dataloaders(self):
        """Create dataloaders for shadow generation."""
        return get_dataloaders_shadow_gen(
            train_size=self.train_size,
            val_size=self.val_size,
            batch_size=self.batch_size,
            img_width=224,
            img_height=224
        )
    
    def create_criterion(self):
        """Create Binary Cross Entropy loss."""
        return nn.BCELoss()
    
    def forward_pass(self, batch):
        """Forward pass for shadow generation."""
        input_data, target = batch
        
        # Move to device
        image = input_data['image'].to(self.device)
        light_pos = input_data['light_pos'].to(self.device)
        target = target.to(self.device)
        
        # Forward
        output = self.model(image, light_pos)
        loss = self.criterion(output, target)
        
        return output, target, loss
    
    def get_extra_info_lines(self):
        """Add base_channels info to header."""
        return [f"Base channels: {self.base_channels}"]


def train(num_epochs=10, batch_size=32, learning_rate=1e-3,
          train_size=50000, val_size=5000, base_channels=64,
          eval_every_n_steps=None, device=None):
    """
    Train the shadow mask generation model.
    
    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        train_size: Number of training samples
        val_size: Number of validation samples
        base_channels: Base number of channels in U-Net
        eval_every_n_steps: If set, evaluate every N gradient steps instead of every epoch
        device: Device to train on (cuda/mps/cpu). If None, auto-detect.
    
    Returns:
        model: Trained model
        train_losses: List of training losses
        val_losses: List of validation losses
    """
    trainer = ShadowGenTrainer(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        train_size=train_size,
        val_size=val_size,
        base_channels=base_channels,
        eval_every_n_steps=eval_every_n_steps,
        save_path='shadow_predictor.pth',
        device=device
    )
    return trainer.train()


if __name__ == '__main__':
    # Train with default parameters
    model, train_losses, val_losses = train(
        num_epochs=5,
        batch_size=32,
        learning_rate=1e-3,
        train_size=50000,
        val_size=5000,
        base_channels=64,
        eval_every_n_steps=1000
    )
