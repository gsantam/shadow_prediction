"""
Training script for a latent shadow world model.

Learns:
    scene_with_shadows_start + (sun_end - sun_start) -> scene_with_shadows_end

The model predicts the transition in latent space and decodes the predicted
latent state back into an image.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts.trainer_base import BaseTrainer
from shadow_prediction.dataset_shadow_transition import get_dataloaders_shadow_sequence
from shadow_prediction.model_shadow_world import create_shadow_world_model


class ShadowWorldTrainer(BaseTrainer):
    """Trainer for the latent shadow world model."""

    def __init__(self, latent_dim=128, base_channels=32,
                 latent_weight=1.0, image_weight=1.0, recon_weight=0.5,
                 sigreg_weight=0.1, sequence_length=4,
                 random_projection=False, **kwargs):
        kwargs.setdefault("task_name", "Latent Shadow World Model Training")
        kwargs.setdefault("loss_name", "latent MSE + image MSE + recon MSE + SIGReg")
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.latent_weight = latent_weight
        self.image_weight = image_weight
        self.recon_weight = recon_weight
        self.sigreg_weight = sigreg_weight
        self.sequence_length = sequence_length
        self.random_projection = random_projection
        self.latest_loss_parts = {}

    def create_model(self):
        model, _ = create_shadow_world_model(
            latent_dim=self.latent_dim,
            base_channels=self.base_channels,
            device=self.device,
        )
        return model

    def create_dataloaders(self):
        return get_dataloaders_shadow_sequence(
            train_size=self.train_size,
            val_size=self.val_size,
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            img_width=224,
            img_height=224,
            random_projection=self.random_projection,
        )

    def create_criterion(self):
        return None

    def forward_pass(self, batch):
        input_data, _ = batch

        images = input_data["images"].to(self.device)
        actions = input_data["actions"].to(self.device)

        output = self.model.forward_sequence(images, actions)
        loss, parts = self.model.sequence_loss(
            output,
            images,
            latent_weight=self.latent_weight,
            image_weight=self.image_weight,
            recon_weight=self.recon_weight,
            sigreg_weight=self.sigreg_weight,
        )
        self.latest_loss_parts = {key: float(value.cpu()) for key, value in parts.items()}

        return output["pred_images"][:, -1], images[:, -1], loss

    def get_extra_info_lines(self):
        return [
            f"Latent dim: {self.latent_dim}",
            f"Base channels: {self.base_channels}",
            f"Sequence length T: {self.sequence_length}",
            f"Loss weights: latent={self.latent_weight}, image={self.image_weight}, "
            f"recon={self.recon_weight}, sigreg={self.sigreg_weight}",
            f"Random projection: {self.random_projection}",
            "State: start shadowed scene; Action: sun_end - sun_start; Target: end shadowed scene",
        ]

    def get_progress_metrics(self):
        if not self.latest_loss_parts:
            return {}
        return {
            "pred": self.latest_loss_parts["latent_loss"],
            "sigreg": self.latest_loss_parts["sigreg_loss"],
            "image": self.latest_loss_parts["image_loss"],
            "recon": self.latest_loss_parts["recon_loss"],
        }


def train(num_epochs=5, batch_size=32, learning_rate=1e-3,
          train_size=50000, val_size=5000, latent_dim=128, base_channels=32,
          latent_weight=1.0, image_weight=1.0, recon_weight=0.5,
          sigreg_weight=0.1, sequence_length=4, random_projection=False,
          eval_every_n_steps=None, device=None, plot_curves=False):
    trainer = ShadowWorldTrainer(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        train_size=train_size,
        val_size=val_size,
        latent_dim=latent_dim,
        base_channels=base_channels,
        latent_weight=latent_weight,
        image_weight=image_weight,
        recon_weight=recon_weight,
        sigreg_weight=sigreg_weight,
        sequence_length=sequence_length,
        random_projection=random_projection,
        eval_every_n_steps=eval_every_n_steps,
        save_path="shadow_world_model.pth",
        device=device,
        plot_curves=plot_curves,
    )
    return trainer.train()


if __name__ == "__main__":
    model, train_losses, val_losses = train(
        num_epochs=5,
        batch_size=32,
        learning_rate=1e-3,
        train_size=50000,
        val_size=5000,
        latent_dim=128,
        base_channels=32,
        sequence_length=4,
        eval_every_n_steps=1000,
    )
