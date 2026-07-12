"""
Demo: encode one shadowed scene and decode only its shadow mask.

This uses the shadow-mask decoder:
    scene_t -> frozen encoder -> z_t -> decoder -> shadow_probability_t
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


def save_comparison(scene, shadow_target, shadow_prob, output_path, threshold):
    shadow_pred = shadow_prob >= threshold

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4))

    axes[0].imshow(scene, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("input scene")

    axes[1].imshow(shadow_target, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("target shadow")

    axes[2].imshow(shadow_prob, cmap="magma", vmin=0.0, vmax=1.0)
    axes[2].set_title("shadow probability")

    axes[3].imshow(shadow_pred, cmap="gray", vmin=0.0, vmax=1.0)
    axes[3].set_title(f"threshold >= {threshold:.2f}")

    for axis in axes:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def shadow_metrics(shadow_target, shadow_prob, valid_mask, threshold):
    shadow_pred = shadow_prob >= threshold
    true_shadow = shadow_target.astype(bool)
    valid = valid_mask.astype(bool)

    tp = np.logical_and.reduce([shadow_pred, true_shadow, valid]).sum()
    fp = np.logical_and.reduce([shadow_pred, ~true_shadow, valid]).sum()
    fn = np.logical_and.reduce([~shadow_pred, true_shadow, valid]).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    return precision, recall, iou


def main():
    parser = argparse.ArgumentParser(description="Decode one synthetic scene into a shadow mask.")
    parser.add_argument(
        "--jepa-checkpoint",
        default="models_checkpoints/shadow_jepa_20260712_100230/checkpoint_best.pth",
    )
    parser.add_argument(
        "--decoder-checkpoint",
        default="models_checkpoints/shadow_mask_decoder_20260712_103202/checkpoint_best.pth",
    )
    parser.add_argument("--output", default="outputs/shadow_mask_decode_demo.png")
    parser.add_argument("--threshold", type=float, default=0.5)
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
        shadow_prob_tensor = decoder(latent)

    scene_image = scene.squeeze(0)
    shadow_target = ((scene_image > 0.25) & (scene_image < 0.75)).astype(np.float32)
    valid_mask = (scene_image > 0.25).astype(np.float32)
    shadow_prob = shadow_prob_tensor.squeeze(0).squeeze(0).cpu().numpy()
    precision, recall, iou = shadow_metrics(
        shadow_target,
        shadow_prob,
        valid_mask,
        args.threshold,
    )

    save_comparison(
        scene_image,
        shadow_target,
        shadow_prob,
        args.output,
        args.threshold,
    )

    print(f"Device: {device}")
    print(f"Input scene tensor: {tuple(scene_tensor.shape)}")
    print(f"Latent tensor: {tuple(latent.shape)}")
    print(f"Shadow probability tensor: {tuple(shadow_prob_tensor.shape)}")
    print(f"Threshold: {args.threshold:.2f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"IoU: {iou:.4f}")
    print(f"Saved comparison: {args.output}")


if __name__ == "__main__":
    main()
