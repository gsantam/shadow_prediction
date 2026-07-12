"""
Latent world model for shadow transitions.

State: start scene with shadows.
Action: sun_end - sun_start.
Target: same scene with shadows at sun_end.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGReg(nn.Module):
    """
    Sketched isotropic Gaussian regularizer.

    This follows the LeWorldModel idea: project a batch of latent embeddings onto
    random 1D directions and penalize deviations from a standard normal.
    """

    def __init__(self, knots=17, num_proj=128):
        super().__init__()
        self.num_proj = num_proj

        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, z):
        """
        Args:
            z: latent embeddings with shape (T, B, D) or (B, D)
        """
        if z.ndim == 2:
            z = z.unsqueeze(0)

        projections = torch.randn(z.size(-1), self.num_proj, device=z.device)
        projections = projections / projections.norm(p=2, dim=0, keepdim=True).clamp_min(1e-8)

        projected = z @ projections
        x_t = projected.unsqueeze(-1) * self.t
        err = (x_t.cos().mean(dim=1) - self.phi).square()
        err = err + x_t.sin().mean(dim=1).square()
        statistic = (err @ self.weights) * z.size(1)
        return statistic.mean()


class ConvEncoder(nn.Module):
    """Small CNN encoder that maps a 1-channel image to a latent vector."""

    def __init__(self, latent_dim=128, base_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 8, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(base_channels * 8 * 7 * 7, latent_dim)

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1)
        return self.fc(x)


class ActionPredictor(nn.Module):
    """Predict the next latent state from current latent state and sun movement."""

    def __init__(self, latent_dim=128, action_dim=3, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z, action):
        delta_z = self.net(torch.cat([z, action.float()], dim=-1))
        return z + delta_z


class ContextActionPredictor(nn.Module):
    """Predict the next shadow latent from shadow state, static scene context, and sun movement."""

    def __init__(self, latent_dim=128, action_dim=3, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2 + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_shadow, z_scene, action):
        features = torch.cat([z_shadow, z_scene, action.float()], dim=-1)
        delta_z = self.net(features)
        return z_shadow + delta_z


class ConvDecoder(nn.Module):
    """Small deconvolutional decoder that maps a latent vector to an image."""

    def __init__(self, latent_dim=128, base_channels=32):
        super().__init__()
        self.base_channels = base_channels
        self.fc = nn.Linear(latent_dim, base_channels * 8 * 7 * 7)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), self.base_channels * 8, 7, 7)
        return self.net(x)


class ShadowWorldModel(nn.Module):
    """
    JEPA-style latent world model with a decoder for image-space supervision.
    """

    def __init__(self, latent_dim=128, base_channels=32, action_dim=3,
                 predictor_hidden_dim=256, sigreg_num_proj=128):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim=latent_dim, base_channels=base_channels)
        self.predictor = ActionPredictor(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=predictor_hidden_dim,
        )
        self.decoder = ConvDecoder(latent_dim=latent_dim, base_channels=base_channels)
        self.sigreg = SIGReg(num_proj=sigreg_num_proj)

    def forward(self, image, action, target=None):
        z_start = self.encoder(image)
        z_pred = self.predictor(z_start, action)
        pred_image = self.decoder(z_pred)

        output = {
            "pred_image": pred_image,
            "z_start": z_start,
            "z_pred": z_pred,
        }

        if target is not None:
            z_target = self.encoder(target)
            output.update({
                "z_target": z_target,
                "start_recon": self.decoder(z_start),
                "target_recon": self.decoder(z_target),
            })

        return output

    def forward_sequence(self, images, actions):
        """
        Teacher-forced latent prediction over a sequence.

        Args:
            images: (B, T, 1, H, W)
            actions: (B, T - 1, 3)
        """
        batch_size, sequence_length = images.shape[:2]

        flat_images = images.reshape(batch_size * sequence_length, *images.shape[2:])
        z = self.encoder(flat_images).reshape(batch_size, sequence_length, -1)

        z_start = z[:, :-1]
        z_target = z[:, 1:]
        flat_z_start = z_start.reshape(batch_size * (sequence_length - 1), -1)
        flat_actions = actions.reshape(batch_size * (sequence_length - 1), -1)

        z_pred = self.predictor(flat_z_start, flat_actions)
        z_pred = z_pred.reshape(batch_size, sequence_length - 1, -1)

        pred_images = self.decoder(z_pred.reshape(batch_size * (sequence_length - 1), -1))
        pred_images = pred_images.reshape(batch_size, sequence_length - 1, *images.shape[2:])

        recon_images = self.decoder(z.reshape(batch_size * sequence_length, -1))
        recon_images = recon_images.reshape(batch_size, sequence_length, *images.shape[2:])

        return {
            "z": z,
            "z_start": z_start,
            "z_target": z_target,
            "z_pred": z_pred,
            "pred_images": pred_images,
            "recon_images": recon_images,
        }

    def loss(self, output, image, target, latent_weight=1.0, image_weight=1.0,
             recon_weight=0.5, sigreg_weight=0.1):
        latent_loss = F.mse_loss(output["z_pred"], output["z_target"])
        image_loss = F.mse_loss(output["pred_image"], target)
        recon_loss = F.mse_loss(output["start_recon"], image)
        recon_loss = recon_loss + F.mse_loss(output["target_recon"], target)

        z = torch.stack([output["z_start"], output["z_target"]], dim=0)
        sigreg_loss = self.sigreg(z)

        total = (
            latent_weight * latent_loss
            + image_weight * image_loss
            + recon_weight * recon_loss
            + sigreg_weight * sigreg_loss
        )

        return total, {
            "latent_loss": latent_loss.detach(),
            "image_loss": image_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "sigreg_loss": sigreg_loss.detach(),
        }

    def sequence_loss(self, output, images, latent_weight=1.0, image_weight=1.0,
                      recon_weight=0.5, sigreg_weight=0.1):
        target_images = images[:, 1:]

        latent_loss = F.mse_loss(output["z_pred"], output["z_target"])
        image_loss = F.mse_loss(output["pred_images"], target_images)
        recon_loss = F.mse_loss(output["recon_images"], images)

        sigreg_loss = self.sigreg(output["z"].transpose(0, 1))

        total = (
            latent_weight * latent_loss
            + image_weight * image_loss
            + recon_weight * recon_loss
            + sigreg_weight * sigreg_loss
        )

        return total, {
            "latent_loss": latent_loss.detach(),
            "image_loss": image_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "sigreg_loss": sigreg_loss.detach(),
        }


def create_shadow_world_model(latent_dim=128, base_channels=32, device=None):
    """Create a ShadowWorldModel on the requested device."""
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model = ShadowWorldModel(latent_dim=latent_dim, base_channels=base_channels)
    model = model.to(device)

    print(f"Created ShadowWorldModel with latent_dim={latent_dim}, base_channels={base_channels}")
    print(f"Model moved to device: {device}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model, device


class ShadowJEPAModel(nn.Module):
    """
    Paper-faithful stage-1 model: encoder + action-conditioned predictor only.
    """

    def __init__(self, latent_dim=128, base_channels=32, action_dim=3,
                 predictor_hidden_dim=256, sigreg_num_proj=128):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim=latent_dim, base_channels=base_channels)
        self.predictor = ActionPredictor(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=predictor_hidden_dim,
        )
        self.sigreg = SIGReg(num_proj=sigreg_num_proj)

    def forward_sequence(self, images, actions):
        """
        Teacher-forced latent prediction over a sequence.

        Args:
            images: (B, T, 1, H, W)
            actions: (B, T - 1, 3)
        """
        batch_size, sequence_length = images.shape[:2]

        flat_images = images.reshape(batch_size * sequence_length, *images.shape[2:])
        z = self.encoder(flat_images).reshape(batch_size, sequence_length, -1)

        z_start = z[:, :-1]
        z_target = z[:, 1:]
        flat_z_start = z_start.reshape(batch_size * (sequence_length - 1), -1)
        flat_actions = actions.reshape(batch_size * (sequence_length - 1), -1)

        z_pred = self.predictor(flat_z_start, flat_actions)
        z_pred = z_pred.reshape(batch_size, sequence_length - 1, -1)

        return {
            "z": z,
            "z_start": z_start,
            "z_target": z_target,
            "z_pred": z_pred,
        }

    def loss(self, output, sigreg_weight=0.1):
        pred_loss = F.mse_loss(output["z_pred"], output["z_target"])
        sigreg_loss = self.sigreg(output["z"].transpose(0, 1))
        total = pred_loss + sigreg_weight * sigreg_loss

        return total, {
            "pred_loss": pred_loss.detach(),
            "sigreg_loss": sigreg_loss.detach(),
        }


class ShadowContextJEPAModel(nn.Module):
    """
    Decoder-free contextual shadow world model.

    State: binary shadow mask at time t.
    Context: shadow-free scene/object geometry, shared across the transition.
    Action: sun_{t+1} - sun_t.
    Target: binary shadow mask at time t+1.
    """

    def __init__(self, latent_dim=128, base_channels=32, action_dim=3,
                 predictor_hidden_dim=256, sigreg_num_proj=128):
        super().__init__()
        self.shadow_encoder = ConvEncoder(latent_dim=latent_dim, base_channels=base_channels)
        self.scene_encoder = ConvEncoder(latent_dim=latent_dim, base_channels=base_channels)
        self.predictor = ContextActionPredictor(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=predictor_hidden_dim,
        )
        self.sigreg = SIGReg(num_proj=sigreg_num_proj)

    def forward_sequence(self, scene, shadows, actions):
        """
        Teacher-forced latent prediction over a shadow-mask sequence.

        Args:
            scene: (B, 1, H, W), fixed shadow-free object geometry.
            shadows: (B, T, 1, H, W), binary visible-shadow masks.
            actions: (B, T - 1, 3), sun movement vectors.
        """
        batch_size, sequence_length = shadows.shape[:2]

        z_scene = self.scene_encoder(scene)
        flat_shadows = shadows.reshape(batch_size * sequence_length, *shadows.shape[2:])
        z_shadow = self.shadow_encoder(flat_shadows).reshape(batch_size, sequence_length, -1)

        z_start = z_shadow[:, :-1]
        z_target = z_shadow[:, 1:]
        flat_z_start = z_start.reshape(batch_size * (sequence_length - 1), -1)
        flat_actions = actions.reshape(batch_size * (sequence_length - 1), -1)
        flat_z_scene = (
            z_scene[:, None, :]
            .expand(batch_size, sequence_length - 1, z_scene.size(-1))
            .reshape(batch_size * (sequence_length - 1), -1)
        )

        z_pred = self.predictor(flat_z_start, flat_z_scene, flat_actions)
        z_pred = z_pred.reshape(batch_size, sequence_length - 1, -1)

        return {
            "z_scene": z_scene,
            "z_shadow": z_shadow,
            "z_start": z_start,
            "z_target": z_target,
            "z_pred": z_pred,
        }

    def loss(self, output, sigreg_weight=0.1):
        pred_loss = F.mse_loss(output["z_pred"], output["z_target"])
        sigreg_loss = self.sigreg(output["z_shadow"].transpose(0, 1))
        total = pred_loss + sigreg_weight * sigreg_loss

        return total, {
            "pred_loss": pred_loss.detach(),
            "sigreg_loss": sigreg_loss.detach(),
        }


def create_shadow_jepa_model(latent_dim=128, base_channels=32, device=None):
    """Create a decoder-free ShadowJEPAModel on the requested device."""
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model = ShadowJEPAModel(latent_dim=latent_dim, base_channels=base_channels)
    model = model.to(device)

    print(f"Created ShadowJEPAModel with latent_dim={latent_dim}, base_channels={base_channels}")
    print(f"Model moved to device: {device}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model, device


def create_shadow_context_jepa_model(latent_dim=128, base_channels=32, device=None):
    """Create a decoder-free contextual shadow JEPA model on the requested device."""
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model = ShadowContextJEPAModel(latent_dim=latent_dim, base_channels=base_channels)
    model = model.to(device)

    print(f"Created ShadowContextJEPAModel with latent_dim={latent_dim}, base_channels={base_channels}")
    print(f"Model moved to device: {device}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model, device


def create_shadow_latent_decoder(latent_dim=128, base_channels=32, device=None):
    """Create a standalone latent-to-image decoder."""
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    decoder = ConvDecoder(latent_dim=latent_dim, base_channels=base_channels)
    decoder = decoder.to(device)

    print(f"Created Shadow latent decoder with latent_dim={latent_dim}, base_channels={base_channels}")
    print(f"Decoder moved to device: {device}")
    total_params = sum(p.numel() for p in decoder.parameters())
    trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return decoder, device
