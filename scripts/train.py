"""
Training script for light position prediction model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from trainer_base import BaseTrainer
from shadow_prediction.dataset import get_dataloaders
from shadow_prediction.model import create_model


class LightPredictionTrainer(BaseTrainer):
    """Trainer for light position prediction model."""
    
    def create_model(self):
        """Create ResNet-18 light prediction model."""
        return create_model(pretrained=False, device=self.device)
    
    def create_dataloaders(self):
        """Create dataloaders for light prediction."""
        return get_dataloaders(
            train_size=self.train_size,
            val_size=self.val_size,
            batch_size=self.batch_size,
            img_width=224,
            img_height=224,
            num_workers=0
        )
    
    def create_criterion(self):
        """Create MSE loss."""
        return nn.MSELoss()
    
    def forward_pass(self, batch):
        """Forward pass for light prediction."""
        images, light_positions = batch
        
        # Move to device
        images = images.to(self.device)
        light_positions = light_positions.to(self.device)
        
        # Forward
        outputs = self.model(images)
        loss = self.criterion(outputs, light_positions)
        
        return outputs, light_positions, loss
    
    def print_header(self):
        """Print training header with light prediction specifics."""
        print("="*50)
        print("Light Position Prediction Training")
        print("="*50)
        print(f"\nTrain samples: {self.train_size}, Val samples: {self.val_size}")
        print(f"Batch size: {self.batch_size}")
        print(f"Optimizer: Adam (lr={self.learning_rate})")
        print(f"Loss function: MSE")
        print(f"Device: {self.device}")
        if self.eval_every_n_steps:
            print(f"Evaluating every {self.eval_every_n_steps} steps")
        print("="*50)


def train(num_epochs=10, batch_size=16, learning_rate=1e-3, device=None,
          train_size=800, val_size=200, save_path='light_predictor.pth',
          eval_every_n_steps=None, plot_curves=True):
    """
    Main training function.
    
    Args:
        num_epochs: Number of epochs to train
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on (cuda/mps/cpu)
        train_size: Number of training samples
        val_size: Number of validation samples
        save_path: Path to save the best model
        eval_every_n_steps: If set, evaluate every N training steps instead of every epoch
        plot_curves: Whether to plot and save training curves
        
    Returns:
        model: Trained model
        train_losses: List of training losses
        val_losses: List of validation losses
    """
    trainer = LightPredictionTrainer(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        train_size=train_size,
        val_size=val_size,
        eval_every_n_steps=eval_every_n_steps,
        save_path=save_path,
        device=device
    )
    
    model, train_losses, val_losses = trainer.train()
    
    # Plot training curves
    if plot_curves:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Evaluation Step')
        plt.ylabel('Loss (MSE)')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_curve.png')
        print('Training curve saved to training_curve.png')
    
    return model, train_losses, val_losses


if __name__ == '__main__':
    # Train the model
    model, train_losses, val_losses = train(
        num_epochs=10,
        batch_size=16,
        learning_rate=1e-3,
        train_size=800,
        val_size=200
    )
