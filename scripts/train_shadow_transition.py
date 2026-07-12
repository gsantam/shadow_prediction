"""
Training script for shadow transition prediction.

Learns:
    scene_with_shadows_start + (sun_end - sun_start) -> scene_with_shadows_end
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch.nn as nn

from scripts.trainer_base import BaseTrainer
from shadow_prediction.dataset_shadow_transition import get_dataloaders_shadow_transition
from shadow_prediction.model_shadow_gen import create_shadow_model


class ShadowTransitionTrainer(BaseTrainer):
    """Trainer for shadowed-scene transition prediction."""

    def __init__(self, base_channels=64, random_projection=False, **kwargs):
        kwargs.setdefault("task_name", "Shadow Transition Training")
        kwargs.setdefault("loss_name", "MSE")
        super().__init__(**kwargs)
        self.base_channels = base_channels
        self.random_projection = random_projection

    def create_model(self):
        """Reuse the existing U-Net: image + 3-vector -> image."""
        model, _ = create_shadow_model(base_channels=self.base_channels, device=self.device)
        return model

    def create_dataloaders(self):
        return get_dataloaders_shadow_transition(
            train_size=self.train_size,
            val_size=self.val_size,
            batch_size=self.batch_size,
            img_width=224,
            img_height=224,
            random_projection=self.random_projection,
        )

    def create_criterion(self):
        return nn.MSELoss()

    def forward_pass(self, batch):
        input_data, target = batch

        image = input_data["image"].to(self.device)
        sun_delta = input_data["sun_delta"].to(self.device)
        target = target.to(self.device)

        output = self.model(image, sun_delta)
        loss = self.criterion(output, target)

        return output, target, loss

    def get_extra_info_lines(self):
        return [
            f"Base channels: {self.base_channels}",
            f"Random projection: {self.random_projection}",
            "State: start shadowed scene; Action: sun_end - sun_start; Target: end shadowed scene",
        ]


def train(num_epochs=5, batch_size=32, learning_rate=1e-3,
          train_size=50000, val_size=5000, base_channels=64,
          random_projection=False, eval_every_n_steps=None, device=None,
          plot_curves=False):
    trainer = ShadowTransitionTrainer(
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        train_size=train_size,
        val_size=val_size,
        base_channels=base_channels,
        random_projection=random_projection,
        eval_every_n_steps=eval_every_n_steps,
        save_path="shadow_transition_predictor.pth",
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
        base_channels=64,
        eval_every_n_steps=1000,
    )
