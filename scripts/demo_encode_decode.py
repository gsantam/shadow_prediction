"""
Demo: encode one shadowed scene into a JEPA latent and decode it back to pixels.

This is the stage-2 reconstruction path only:
    scene_t -> frozen encoder -> z_t -> decoder -> scene_t_recon
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import numpy as np
import torch

from shadow_prediction.model_shadow_world import (
    create_shadow_jepa_model,
    create_shadow_latent_decoder,
)
from utils.synthetic_data import generate_synthetic_shadow_transition


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_encoder(jepa_checkpoint, latent_dim, base_channels, device):
    jepa, _ = create_shadow_jepa_model(
        latent_dim=latent_dim,
        base_channels=base_channels,
        device=device,
    )
    checkpoint = torch.load(jepa_checkpoint, map_location=device)
    jepa.load_state_dict(checkpoint["model_state_dict"])
    jepa.eval()
    return jepa.encoder


def load_decoder(decoder_checkpoint, latent_dim, base_channels, device):
    decoder, _ = create_shadow_latent_decoder(
        latent_dim=latent_dim,
        base_channels=base_channels,
        device=device,
    )
    checkpoint = torch.load(decoder_checkpoint, map_location=device)
    decoder.load_state_dict(checkpoint["model_state_dict"])
    decoder.eval()
    return decoder


def save_comparison(scene, reconstruction, output_path):
    error = np.abs(scene - reconstruction)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))

    axes[0].imshow(scene, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("input scene")

    axes[1].imshow(reconstruction, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("decoded reconstruction")

    axes[2].imshow(error, cmap="magma", vmin=0.0, vmax=max(float(error.max()), 1e-6))
    axes[2].set_title("absolute error")

    for axis in axes:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Encode and decode one synthetic shadow scene.")
    parser.add_argument(
        "--jepa-checkpoint",
        default="models_checkpoints/shadow_jepa_20260712_100230/checkpoint_best.pth",
    )
    parser.add_argument(
        "--decoder-checkpoint",
        default="models_checkpoints/shadow_latent_decoder_20260712_101317/checkpoint_best.pth",
    )
    parser.add_argument("--output", default="outputs/encode_decode_demo.png")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--jepa-base-channels", type=int, default=16)
    parser.add_argument("--decoder-base-channels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()

    encoder = load_encoder(
        args.jepa_checkpoint,
        latent_dim=args.latent_dim,
        base_channels=args.jepa_base_channels,
        device=device,
    )
    decoder = load_decoder(
        args.decoder_checkpoint,
        latent_dim=args.latent_dim,
        base_channels=args.decoder_base_channels,
        device=device,
    )

    scene, _, _ = generate_synthetic_shadow_transition(
        random_projection=False,
        randomize_objects=True,
        img_width=224,
        img_height=224,
    )
    scene_tensor = torch.from_numpy(scene).unsqueeze(0).to(device)

    with torch.no_grad():
        latent = encoder(scene_tensor)
        reconstruction_tensor = decoder(latent)

    reconstruction = reconstruction_tensor.squeeze(0).squeeze(0).cpu().numpy()
    scene_image = scene.squeeze(0)
    mse = float(np.mean((reconstruction - scene_image) ** 2))

    save_comparison(scene_image, reconstruction, args.output)

    print(f"Device: {device}")
    print(f"Input scene tensor: {tuple(scene_tensor.shape)}")
    print(f"Latent tensor: {tuple(latent.shape)}")
    print(f"Reconstruction tensor: {tuple(reconstruction_tensor.shape)}")
    print(f"Pixel MSE: {mse:.6f}")
    print(f"Saved comparison: {args.output}")


if __name__ == "__main__":
    main()
