"""
Base trainer class for common training logic.
"""

import torch
from tqdm import tqdm


class BaseTrainer:
    """
    Base trainer with common training loop logic.
    Subclass this and override model/data/loss creation methods.
    """
    
    def __init__(self, num_epochs=10, batch_size=32, learning_rate=1e-3,
                 train_size=50000, val_size=5000, eval_every_n_steps=None,
                 save_path='model.pth', device=None):
        """
        Initialize trainer.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            train_size: Number of training samples
            val_size: Number of validation samples
            eval_every_n_steps: If set, evaluate every N steps instead of every epoch
            save_path: Path to save best model checkpoint
            device: Device to train on (cuda/mps/cpu). If None, auto-detect.
        """
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_size = train_size
        self.val_size = val_size
        self.eval_every_n_steps = eval_every_n_steps
        self.save_path = save_path
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        self.device = device
        
        # Will be set during training
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
    
    def create_model(self):
        """Create and return the model. Must be overridden."""
        raise NotImplementedError("Subclass must implement create_model()")
    
    def create_dataloaders(self):
        """Create and return (train_loader, val_loader). Must be overridden."""
        raise NotImplementedError("Subclass must implement create_dataloaders()")
    
    def create_criterion(self):
        """Create and return the loss function. Must be overridden."""
        raise NotImplementedError("Subclass must implement create_criterion()")
    
    def create_optimizer(self):
        """Create and return the optimizer. Can be overridden."""
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
    
    def forward_pass(self, batch):
        """
        Perform forward pass on a batch.
        Must be overridden to handle specific input/output format.
        
        Args:
            batch: Batch from dataloader
            
        Returns:
            (output, target, loss): Model output, ground truth, and loss value
        """
        raise NotImplementedError("Subclass must implement forward_pass()")
    
    def validate(self):
        """Run validation loop."""
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for batch in pbar:
                _, _, loss = self.forward_pass(batch)
                running_loss += loss.item()
                pbar.set_postfix({'val_loss': f'{running_loss / (pbar.n + 1):.4f}'})
        
        return running_loss / len(self.val_loader)
    
    def save_if_best(self, val_loss, best_val_loss):
        """Save model if validation loss improved."""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(self.model.state_dict(), self.save_path)
            print(f"💾 Saved best model (val_loss: {val_loss:.4f})")
        return best_val_loss
    
    def print_header(self):
        """Print training header. Can be overridden."""
        print("="*50)
        print("Training")
        print("="*50)
        print(f"\nTrain samples: {self.train_size}, Val samples: {self.val_size}")
        print(f"Batch size: {self.batch_size}")
        print(f"Optimizer: Adam (lr={self.learning_rate})")
        print(f"Device: {self.device}")
        if self.eval_every_n_steps:
            print(f"Evaluating every {self.eval_every_n_steps} steps")
        print("="*50)
    
    def train(self):
        """
        Main training loop.
        
        Returns:
            model: Trained model
            train_losses: List of training losses
            val_losses: List of validation losses
        """
        # Setup
        self.print_header()
        
        print("\nCreating model...")
        self.model = self.create_model()
        
        print("Creating dataloaders...")
        self.train_loader, self.val_loader = self.create_dataloaders()
        
        print("Creating optimizer and loss function...")
        self.criterion = self.create_criterion()
        self.optimizer = self.create_optimizer()
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        global_step = 0
        
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            running_loss = 0.0
            
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.num_epochs}')
            for batch_idx, batch in enumerate(pbar):
                # Forward pass
                self.optimizer.zero_grad()
                output, target, loss = self.forward_pass(batch)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                running_loss += loss.item()
                global_step += 1
                avg_train_loss = running_loss / (batch_idx + 1)
                
                # Evaluate every N steps or at end of epoch
                is_last_batch = (batch_idx == len(self.train_loader) - 1)
                should_eval = (self.eval_every_n_steps is not None and global_step % self.eval_every_n_steps == 0) or \
                             (self.eval_every_n_steps is None and is_last_batch)
                
                if should_eval:
                    train_losses.append(avg_train_loss)
                    val_loss = self.validate()
                    val_losses.append(val_loss)
                    
                    eval_type = "Step" if not is_last_batch else "Epoch"
                    eval_id = global_step if not is_last_batch else epoch
                    print(f"\n{eval_type} {eval_id}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
                    best_val_loss = self.save_if_best(val_loss, best_val_loss)
                    
                    self.model.train()  # Back to train mode
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{avg_train_loss:.4f}',
                    'step': global_step
                })
        
        print(f"\n✅ Training complete. Best model saved to {self.save_path} (val_loss: {best_val_loss:.4f})")
        
        return self.model, train_losses, val_losses
