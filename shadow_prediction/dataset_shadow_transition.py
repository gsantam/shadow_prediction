"""
Dataset for shadow transition prediction.

Input: scene with shadows at a start sun position + sun movement vector.
Output: same scene with shadows at the end sun position.
"""

import os
import sys

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.synthetic_data import (
    generate_synthetic_shadow_mask_sequence,
    generate_synthetic_shadow_sequence,
    generate_synthetic_shadow_transition,
)


class ShadowTransitionDataset(Dataset):
    """
    Generates paired views of the same synthetic scene under two sun positions.

    State: start image with shadows.
    Action: sun_end - sun_start.
    Target: end image with shadows.
    """

    def __init__(self, num_samples=1000, img_width=224, img_height=224,
                 random_projection=False):
        self.num_samples = num_samples
        self.img_width = img_width
        self.img_height = img_height
        self.random_projection = random_projection

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_image, sun_delta, end_image = generate_synthetic_shadow_transition(
            random_projection=self.random_projection,
            randomize_objects=True,
            img_width=self.img_width,
            img_height=self.img_height,
        )

        input_data = {
            "image": torch.from_numpy(start_image).float(),
            "sun_delta": torch.from_numpy(sun_delta).float(),
        }
        target = torch.from_numpy(end_image).float()

        return input_data, target


def get_dataloaders_shadow_transition(train_size=10000, val_size=2000, batch_size=32,
                                      img_width=224, img_height=224, num_workers=0,
                                      random_projection=False):
    """Create train and validation dataloaders for shadow transition prediction."""
    train_dataset = ShadowTransitionDataset(
        num_samples=train_size,
        img_width=img_width,
        img_height=img_height,
        random_projection=random_projection,
    )
    val_dataset = ShadowTransitionDataset(
        num_samples=val_size,
        img_width=img_width,
        img_height=img_height,
        random_projection=random_projection,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader


class ShadowSequenceDataset(Dataset):
    """
    Generates a sun-motion sequence for the same synthetic scene.

    States: images[0:T] are shadowed scenes.
    Actions: actions[0:T-1] are sun_{t+1} - sun_t.
    """

    def __init__(self, num_samples=1000, sequence_length=4, img_width=224,
                 img_height=224, random_projection=False):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.img_width = img_width
        self.img_height = img_height
        self.random_projection = random_projection

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        images, actions = generate_synthetic_shadow_sequence(
            num_steps=self.sequence_length,
            random_projection=self.random_projection,
            randomize_objects=True,
            img_width=self.img_width,
            img_height=self.img_height,
        )

        input_data = {
            "images": torch.from_numpy(images).float(),
            "actions": torch.from_numpy(actions).float(),
        }

        return input_data, torch.from_numpy(images).float()


def get_dataloaders_shadow_sequence(train_size=10000, val_size=2000, batch_size=32,
                                    sequence_length=4, img_width=224, img_height=224,
                                    num_workers=0, random_projection=False):
    """Create train and validation dataloaders for sequence world-model training."""
    train_dataset = ShadowSequenceDataset(
        num_samples=train_size,
        sequence_length=sequence_length,
        img_width=img_width,
        img_height=img_height,
        random_projection=random_projection,
    )
    val_dataset = ShadowSequenceDataset(
        num_samples=val_size,
        sequence_length=sequence_length,
        img_width=img_width,
        img_height=img_height,
        random_projection=random_projection,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader


class ShadowMaskSequenceDataset(Dataset):
    """
    Generates shadow-only states with fixed scene geometry as context.

    Context: scene is the shadow-free object layout.
    States: shadows[0:T] are binary visible-shadow masks.
    Actions: actions[0:T-1] are sun_{t+1} - sun_t.
    """

    def __init__(self, num_samples=1000, sequence_length=4, img_width=224,
                 img_height=224, random_projection=False):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.img_width = img_width
        self.img_height = img_height
        self.random_projection = random_projection

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        scene, shadows, actions = generate_synthetic_shadow_mask_sequence(
            num_steps=self.sequence_length,
            random_projection=self.random_projection,
            randomize_objects=True,
            img_width=self.img_width,
            img_height=self.img_height,
        )

        input_data = {
            "scene": torch.from_numpy(scene).float(),
            "shadows": torch.from_numpy(shadows).float(),
            "actions": torch.from_numpy(actions).float(),
        }

        return input_data, torch.from_numpy(shadows).float()


def get_dataloaders_shadow_mask_sequence(train_size=10000, val_size=2000,
                                         batch_size=32, sequence_length=4,
                                         img_width=224, img_height=224,
                                         num_workers=0, random_projection=False):
    """Create dataloaders for contextual shadow-mask world-model training."""
    train_dataset = ShadowMaskSequenceDataset(
        num_samples=train_size,
        sequence_length=sequence_length,
        img_width=img_width,
        img_height=img_height,
        random_projection=random_projection,
    )
    val_dataset = ShadowMaskSequenceDataset(
        num_samples=val_size,
        sequence_length=sequence_length,
        img_width=img_width,
        img_height=img_height,
        random_projection=random_projection,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader
