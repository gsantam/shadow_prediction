"""
Train a standalone decoder from frozen JEPA latents to scene pixels.

Stage 1:
    encoder + predictor trained with pred + SIGReg.

Stage 2, this file:
    freeze encoder
    train decoder(encoder(scene_t)) -> scene_t
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn

from scripts.trainer_base import BaseTrainer
from shadow_prediction.dataset_shadow_transition import get_dataloaders_shadow_sequence
from shadow_prediction.model_shadow_world import (
    create_shadow_jepa_model,
    create_shadow_latent_decoder,
)


class ShadowDecoderTrainer(BaseTrainer):
    """Trainer for a standalone latent-to-pixel decoder."""

    def __init__(self, jepa_checkpoint, latent_dim=64, jepa_base_channels=16,
                 decoder_base_channels=16, sequence_length=4,
                 random_projection=False, **kwargs):
        kwargs.setdefault("task_name", "Shadow Latent Decoder Training")
        kwargs.setdefault("loss_name", "MSE reconstruction")
        super().__init__(**kwargs)
        self.jepa_checkpoint = jepa_checkpoint
        self.latent_dim = latent_dim
        self.jepa_base_channels = jepa_base_channels
        self.decoder_base_channels = decoder_base_channels
        self.sequence_length = sequence_length
        self.random_projection = random_projection
        self.encoder = None
        self.latest_loss_parts = {}

    def create_model(self):
        jepa, _ = create_shadow_jepa_model(
            latent_dim=self.latent_dim,
            base_channels=self.jepa_base_channels,
            device=self.device,
        )
        checkpoint = torch.load(self.jepa_checkpoint, map_location=self.device)
        jepa.load_state_dict(checkpoint["model_state_dict"])
        jepa.eval()
        for param in jepa.parameters():
            param.requires_grad_(False)
        self.encoder = jepa.encoder

        decoder, _ = create_shadow_latent_decoder(
            latent_dim=self.latent_dim,
            base_channels=self.decoder_base_channels,
            device=self.device,
        )
        return decoder

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
        return nn.MSELoss()

    def forward_pass(self, batch):
        input_data, _ = batch
        images = input_data["images"].to(self.device)
        batch_size, sequence_length = images.shape[:2]

        flat_images = images.reshape(batch_size * sequence_length, *images.shape[2:])
        with torch.no_grad():
            latents = self.encoder(flat_images)

        recon = self.model(latents)
        loss = self.criterion(recon, flat_images)
        self.latest_loss_parts = {"recon": float(loss.detach().cpu())}

        return recon, flat_images, loss

    def get_extra_info_lines(self):
        return [
            f"JEPA checkpoint: {self.jepa_checkpoint}",
            f"Latent dim: {self.latent_dim}",
            f"JEPA base channels: {self.jepa_base_channels}",
            f"Decoder base channels: {self.decoder_base_channels}",
            f"Sequence length T: {self.sequence_length}",
            "Encoder is frozen; only decoder weights are optimized",
        ]

    def get_progress_metrics(self):
        return self.latest_loss_parts


def train(jepa_checkpoint="models_checkpoints/shadow_jepa_20260712_100230/checkpoint_best.pth",
          num_epochs=5, batch_size=16, learning_rate=1e-3,
          train_size=50000, val_size=5000, latent_dim=64,
          jepa_base_channels=16, decoder_base_channels=16, sequence_length=4,
          random_projection=False, eval_every_n_steps=None, device=None,
          plot_curves=False, resume_from_checkpoint=None):
    trainer = ShadowDecoderTrainer(
        jepa_checkpoint=jepa_checkpoint,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        train_size=train_size,
        val_size=val_size,
        latent_dim=latent_dim,
        jepa_base_channels=jepa_base_channels,
        decoder_base_channels=decoder_base_channels,
        sequence_length=sequence_length,
        random_projection=random_projection,
        eval_every_n_steps=eval_every_n_steps,
        save_path="shadow_latent_decoder.pth",
        device=device,
        plot_curves=plot_curves,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    return trainer.train()


if __name__ == "__main__":
    model, train_losses, val_losses = train(
        num_epochs=5,
        batch_size=16,
        learning_rate=1e-3,
        train_size=50000,
        val_size=5000,
        latent_dim=64,
        jepa_base_channels=16,
        decoder_base_channels=16,
        sequence_length=4,
        eval_every_n_steps=1000,
    )
