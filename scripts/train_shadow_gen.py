"""
Training script for shadow mask generation model.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from shadow_prediction.dataset_shadow_gen import get_dataloaders_shadow_gen
from shadow_prediction.model_shadow_gen import create_shadow_model


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train for one epoch.
    
    Returns:
        average_loss: Average loss over the epoch
    """
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (input_data, target) in enumerate(pbar):
        # Move to device
        image = input_data['image'].to(device)
        light_pos = input_data['light_pos'].to(device)
        target = target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(image, light_pos)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        running_loss += loss.item()
        avg_loss = running_loss / (batch_idx + 1)
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return running_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """
    Validate the model.
    
    Returns:
        average_loss: Average validation loss
    """
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for input_data, target in pbar:
            # Move to device
            image = input_data['image'].to(device)
            light_pos = input_data['light_pos'].to(device)
            target = target.to(device)
            
            # Forward pass
            output = model(image, light_pos)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            pbar.set_postfix({'val_loss': f'{running_loss / (pbar.n + 1):.4f}'})
    
    return running_loss / len(val_loader)


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
    print("="*50)
    print("Shadow Mask Generation Training")
    print("="*50)
    
    # Create model
    model, device = create_shadow_model(base_channels=base_channels, device=device)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = get_dataloaders_shadow_gen(
        train_size=train_size,
        val_size=val_size,
        batch_size=batch_size,
        img_width=224,
        img_height=224
    )
    print(f"Train samples: {train_size}, Val samples: {val_size}")
    print(f"Batch size: {batch_size}")
    
    # Loss and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy for binary shadow mask
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\nOptimizer: Adam (lr={learning_rate})")
    print(f"Loss function: Binary Cross Entropy")
    print(f"Device: {device}")
    print("="*50)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    if eval_every_n_steps is not None:
        print(f"\nEvaluating every {eval_every_n_steps} steps")
        global_step = 0
        
        for epoch in range(1, num_epochs + 1):
            model.train()
            running_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}')
            for batch_idx, (input_data, target) in enumerate(pbar):
                # Move to device
                image = input_data['image'].to(device)
                light_pos = input_data['light_pos'].to(device)
                target = target.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                output = model(image, light_pos)
                loss = criterion(output, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update metrics
                running_loss += loss.item()
                global_step += 1
                
                # Evaluate every N steps
                if global_step % eval_every_n_steps == 0:
                    avg_train_loss = running_loss / (batch_idx + 1)
                    train_losses.append(avg_train_loss)
                    
                    val_loss = validate(model, val_loader, criterion, device)
                    val_losses.append(val_loss)
                    
                    print(f"\nStep {global_step}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
                    model.train()  # Back to train mode
                
                pbar.set_postfix({
                    'loss': f'{running_loss / (batch_idx + 1):.4f}',
                    'step': global_step
                })
    else:
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            
            # Train
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
            train_losses.append(train_loss)
            
            # Validate
            val_loss = validate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    # Save model
    save_path = 'shadow_predictor.pth'
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Model saved to {save_path}")
    
    return model, train_losses, val_losses


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
