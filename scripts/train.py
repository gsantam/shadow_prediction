"""
Training script for light position prediction model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from shadow_prediction.dataset import get_dataloaders
from shadow_prediction.model import create_model


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
    
    Returns:
        Average training loss for the epoch
    """
    model.train()
    running_loss = 0.0
    
    for images, light_positions in tqdm(train_loader, desc='Training'):
        images = images.to(device)
        light_positions = light_positions.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, light_positions)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def validate(model, val_loader, criterion, device):
    """
    Validate the model.
    
    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
    
    Returns:
        Average validation loss
    """
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for images, light_positions in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            light_positions = light_positions.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, light_positions)
            
            running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(val_loader.dataset)
    return epoch_loss


def train(num_epochs=10, batch_size=16, learning_rate=1e-3, device=None,
          train_size=800, val_size=200, save_path='light_predictor.pth'):
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
    """
    # Setup device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    print(f'Using device: {device}')
    
    # Create dataloaders
    print('Creating dataloaders...')
    train_loader, val_loader = get_dataloaders(
        train_size=train_size,
        val_size=val_size,
        batch_size=batch_size,
        img_width=224,
        img_height=224,
        num_workers=0
    )
    
    # Create model
    print('Creating model...')
    model = create_model(pretrained=False, device=device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f'Model saved to {save_path}')
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
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
