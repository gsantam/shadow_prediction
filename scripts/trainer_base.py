"""
Base trainer class for common training logic.
"""

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import os
import json

class BaseTrainer:
    """
    Base trainer with common training loop logic.
    Subclass this and override model/data/loss creation methods.
    """
    
    def __init__(self, num_epochs=10, batch_size=32, learning_rate=1e-3,
                 train_size=50000, val_size=5000, eval_every_n_steps=None,
                 save_path='model.pth', device=None, minimize_metric=True,
                 plot_curves=False, task_name='Training', loss_name='Loss',
                 run_id=None, resume_from_checkpoint=None):
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
            minimize_metric: If True, lower metric is better. If False, higher is better.
            plot_curves: Whether to plot and save training curves after training
            task_name: Name of the training task (for header)
            loss_name: Name of the loss function (for header)
            run_id: Identifier for this training run (e.g., timestamp or experiment name)
            resume_from_checkpoint: Path to checkpoint to resume from (within the run folder)
        """
        self.resume_from_checkpoint = resume_from_checkpoint
        
        # If resuming, load run_id from checkpoint
        if resume_from_checkpoint:
            checkpoint_data = torch.load(resume_from_checkpoint, map_location='cpu')
            self.run_id = checkpoint_data.get('run_id', self._generate_run_id())
            print(f"📂 Resuming from checkpoint: {resume_from_checkpoint}")
            print(f"   Run ID: {self.run_id}")
        else:
            self.run_id = run_id if run_id is not None else self._generate_run_id()
        
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.train_size = train_size
        self.val_size = val_size
        self.eval_every_n_steps = eval_every_n_steps
        self.minimize_metric = minimize_metric
        self.plot_curves = plot_curves
        self.task_name = task_name
        self.loss_name = loss_name
        
        # Create run folder in models_checkpoints directory
        base_name = os.path.splitext(os.path.basename(save_path))[0]
        checkpoints_dir = 'models_checkpoints'
        os.makedirs(checkpoints_dir, exist_ok=True)
        self.run_folder = os.path.join(checkpoints_dir, f"{base_name}_{self.run_id}")
        os.makedirs(self.run_folder, exist_ok=True)
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
    
    def _generate_run_id(self):
        """Generate a unique run ID based on current timestamp."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load checkpoint and restore training state.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary with 'epoch', 'global_step', 'train_losses', 'val_losses', 'best_val_loss'
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return {
            'epoch': checkpoint['epoch'],
            'global_step': checkpoint['global_step'],
            'train_losses': checkpoint.get('train_losses', []),
            'val_losses': checkpoint.get('val_losses', []),
            'best_val_loss': checkpoint['val_loss']
        }
    
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

    def get_progress_metrics(self):
        """Return extra scalar metrics for progress bars. Override if needed."""
        return {}
    
    def validate(self):
        """Run validation loop."""
        self.model.eval()
        running_loss = 0.0
        metric_sums = {}
        metric_count = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for batch in pbar:
                _, _, loss = self.forward_pass(batch)
                running_loss += loss.item()
                metrics = self.get_progress_metrics()
                for name, value in metrics.items():
                    metric_sums[name] = metric_sums.get(name, 0.0) + float(value)
                metric_count += 1

                postfix = {'val_loss': f'{running_loss / (pbar.n + 1):.4f}'}
                for name, value in metrics.items():
                    postfix[name] = f'{float(value):.4f}'
                pbar.set_postfix(postfix)
        
        self.latest_val_metrics = {
            name: value / max(metric_count, 1)
            for name, value in metric_sums.items()
        }
        return running_loss / len(self.val_loader)
    
    def save_checkpoint(self, epoch, global_step, train_loss, val_loss, train_losses, val_losses, is_best=False):
        """
        Save checkpoint with model and metrics.
        
        Args:
            epoch: Current epoch number
            global_step: Global training step
            train_loss: Current training loss
            val_loss: Current validation loss
            train_losses: Full training loss history
            val_losses: Full validation loss history
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'global_step': global_step,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'run_id': self.run_id,
            'hyperparameters': {
                'num_epochs': self.num_epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'train_size': self.train_size,
                'val_size': self.val_size,
            }
        }
        
        # Save in run folder with step number
        checkpoint_path = os.path.join(self.run_folder, f"checkpoint_step{global_step}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as "best" if applicable
        if is_best:
            best_path = os.path.join(self.run_folder, "checkpoint_best.pth")
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model: {best_path} (val_metric: {val_loss:.4f})")
        else:
            print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    def save_if_best(self, val_loss, best_val_loss):
        """Check if validation metric improved."""
        is_better = (val_loss < best_val_loss) if self.minimize_metric else (val_loss > best_val_loss)
        if is_better:
            return val_loss, True
        return best_val_loss, False
    
    def get_extra_info_lines(self):
        """Get extra info lines to print in header. Override to add custom info."""
        return []
    
    def print_header(self):
        """Print training header."""
        print("="*50)
        print(self.task_name)
        print("="*50)
        print(f"\nRun ID: {self.run_id}")
        print(f"Run folder: {self.run_folder}")
        print(f"Train samples: {self.train_size}, Val samples: {self.val_size}")
        print(f"Batch size: {self.batch_size}")
        print(f"Optimizer: Adam (lr={self.learning_rate})")
        print(f"Loss function: {self.loss_name}")
        
        # Add any extra info from subclass
        for line in self.get_extra_info_lines():
            print(line)
        
        print(f"Device: {self.device}")
        if self.eval_every_n_steps:
            print(f"Evaluating every {self.eval_every_n_steps} steps")
        print("="*50)
    
    def plot_training_curves(self, train_losses, val_losses, save_path=None):
        """
        Plot and save training curves.
        Can be overridden to customize plot.
        
        Args:
            train_losses: List of training losses
            val_losses: List of validation losses
            save_path: Path to save the plot (if None, saves to run folder)
        """
        if save_path is None:
            save_path = os.path.join(self.run_folder, 'training_curve.png')
        
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Evaluation Step')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        print(f'Training curve saved to {save_path}')
    
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
        print(f"Using device: {self.device}")
        
        print("Creating dataloaders...")
        self.train_loader, self.val_loader = self.create_dataloaders()
        
        print("Creating optimizer and loss function...")
        self.criterion = self.create_criterion()
        self.optimizer = self.create_optimizer()
        
        # Training loop state
        train_losses = []
        val_losses = []
        best_val_loss = float('inf') if self.minimize_metric else float('-inf')
        global_step = 0
        start_epoch = 1
        
        # Resume from checkpoint if specified
        if self.resume_from_checkpoint:
            state = self.load_checkpoint(self.resume_from_checkpoint)
            start_epoch = state['epoch'] + 1
            global_step = state['global_step']
            train_losses = state['train_losses']
            val_losses = state['val_losses']
            best_val_loss = state['best_val_loss']
            print(f"\n✅ Resumed from epoch {state['epoch']}, step {global_step}")
            print(f"   Best val loss so far: {best_val_loss:.4f}")
        
        # Save run metadata
        metadata = {
            'run_id': self.run_id,
            'task_name': self.task_name,
            'hyperparameters': {
                'num_epochs': self.num_epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'train_size': self.train_size,
                'val_size': self.val_size,
                'eval_every_n_steps': self.eval_every_n_steps,
            },
            'resumed_from': self.resume_from_checkpoint,
            'start_time': datetime.now().isoformat()
        }
        with open(os.path.join(self.run_folder, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        for epoch in range(start_epoch, self.num_epochs + 1):
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
                    val_metrics = getattr(self, 'latest_val_metrics', {})
                    val_metric_text = ""
                    if val_metrics:
                        metric_parts = [
                            f"Val {name} = {value:.4f}"
                            for name, value in val_metrics.items()
                        ]
                        val_metric_text = ", " + ", ".join(metric_parts)
                    print(f"\n{eval_type} {eval_id}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}{val_metric_text}")
                    
                    # Check if best and save checkpoint
                    new_best, is_best = self.save_if_best(val_loss, best_val_loss)
                    self.save_checkpoint(epoch, global_step, avg_train_loss, val_loss, train_losses, val_losses, is_best)
                    best_val_loss = new_best
                    
                    self.model.train()  # Back to train mode
                
                # Update progress bar
                postfix = {
                    'loss': f'{avg_train_loss:.4f}',
                    'step': global_step
                }
                for name, value in self.get_progress_metrics().items():
                    postfix[name] = f'{float(value):.4f}'
                pbar.set_postfix(postfix)
        
        print(f"\n✅ Training complete. Best model saved in {self.run_folder} (val_loss: {best_val_loss:.4f})")
        
        # Plot training curves if requested
        if self.plot_curves:
            self.plot_training_curves(train_losses, val_losses)
        
        return self.model, train_losses, val_losses
