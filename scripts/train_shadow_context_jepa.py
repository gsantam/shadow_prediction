"""
Train a contextual shadow-only JEPA/world-model stage.

Learns:
    shadow_encoder(shadow_t) -> z_shadow_t
    scene_encoder(scene_without_shadows) -> z_scene
    predictor(z_shadow_t, z_scene, sun_{t+1} - sun_t) -> z_shadow_{t+1}

Loss:
    MSE(z_pred, z_target) + lambda * SIGReg(z_shadow)
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts.trainer_base import BaseTrainer
from shadow_prediction.dataset_shadow_transition import get_dataloaders_shadow_mask_sequence
from shadow_prediction.model_shadow_world import create_shadow_context_jepa_model


class ShadowContextJEPATrainer(BaseTrainer):
    """Trainer for shadow-only states with fixed scene geometry as context."""

    def __init__(self, latent_dim=128, base_channels=32, sigreg_weight=0.1,
                 sequence_length=4, random_projection=False, **kwargs):
        kwargs.setdefault("task_name", "Contextual Shadow JEPA Stage-1 Training")
        kwargs.setdefault("loss_name", "latent prediction MSE + SIGReg")
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.sigreg_weight = sigreg_weight
        self.sequence_length = sequence_length
        self.random_projection = random_projection
        self.latest_loss_parts = {}

    def create_model(self):
        model, _ = create_shadow_context_jepa_model(
            latent_dim=self.latent_dim,
            base_channels=self.base_channels,
            device=self.device,
        )
        return model

    def create_dataloaders(self):
        return get_dataloaders_shadow_mask_sequence(
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

        scene = input_data["scene"].to(self.device)
        shadows = input_data["shadows"].to(self.device)
        actions = input_data["actions"].to(self.device)

        output = self.model.forward_sequence(scene, shadows, actions)
        loss, parts = self.model.loss(output, sigreg_weight=self.sigreg_weight)
        self.latest_loss_parts = {key: float(value.cpu()) for key, value in parts.items()}

        return output["z_pred"], output["z_target"], loss

    def get_extra_info_lines(self):
        return [
            f"Latent dim: {self.latent_dim}",
            f"Base channels: {self.base_channels}",
            f"Sequence length T: {self.sequence_length}",
            f"SIGReg weight: {self.sigreg_weight}",
            f"Random projection: {self.random_projection}",
            "State: shadow mask; Context: shadow-free scene; Action: sun_{t+1} - sun_t",
        ]

    def get_progress_metrics(self):
        if not self.latest_loss_parts:
            return {}
        return {
            "pred": self.latest_loss_parts["pred_loss"],
            "sigreg": self.latest_loss_parts["sigreg_loss"],
        }


def train(num_epochs=10, batch_size=16, learning_rate=1e-3,
          train_size=50000, val_size=5000, latent_dim=64, base_channels=16,
          sigreg_weight=0.1, sequence_length=4, random_projection=False,
          eval_every_n_steps=None, device=None, plot_curves=False,
          resume_from_checkpoint=None):
    trainer = ShadowContextJEPATrainer(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        train_size=train_size,
        val_size=val_size,
        latent_dim=latent_dim,
        base_channels=base_channels,
        sigreg_weight=sigreg_weight,
        sequence_length=sequence_length,
        random_projection=random_projection,
        eval_every_n_steps=eval_every_n_steps,
        save_path="shadow_context_jepa.pth",
        device=device,
        plot_curves=plot_curves,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    return trainer.train()


if __name__ == "__main__":
    model, train_losses, val_losses = train(
        num_epochs=10,
        batch_size=16,
        learning_rate=1e-3,
        train_size=50000,
        val_size=5000,
        latent_dim=64,
        base_channels=16,
        sigreg_weight=0.1,
        sequence_length=4,
        eval_every_n_steps=1000,
    )
