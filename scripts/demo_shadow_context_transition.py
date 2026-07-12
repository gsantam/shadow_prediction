"""
Demo a contextual shadow world-model transition.

Generates:
    scene_without_shadows, shadow_t, sun_delta, shadow_{t+1}

Runs:
    shadow_t + scene + sun_delta -> predicted next shadow latent -> decoder
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import matplotlib.pyplot as plt
import numpy as np
import torch

from shadow_prediction.model_shadow_world import (
    create_shadow_context_jepa_model,
    create_shadow_latent_decoder,
)
from utils.synthetic_data import generate_synthetic_shadow_mask_sequence


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_context_jepa(checkpoint_path, latent_dim, base_channels, device):
    model, _ = create_shadow_context_jepa_model(
        latent_dim=latent_dim,
        base_channels=base_channels,
        device=device,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_decoder(checkpoint_path, latent_dim, base_channels, device):
    decoder, _ = create_shadow_latent_decoder(
        latent_dim=latent_dim,
        base_channels=base_channels,
        device=device,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    decoder.load_state_dict(checkpoint["model_state_dict"])
    decoder.eval()
    return decoder


def shadow_metrics(target, probability, threshold):
    prediction = probability >= threshold
    true_shadow = target.astype(bool)

    tp = np.logical_and(prediction, true_shadow).sum()
    fp = np.logical_and(prediction, ~true_shadow).sum()
    fn = np.logical_and(~prediction, true_shadow).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    return precision, recall, iou


def save_comparison(scene, start_shadow, target_shadow, target_recon,
                    pred_prob, threshold, output_path):
    pred_shadow = pred_prob >= threshold

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 6, figsize=(15, 3.4))

    panels = [
        (scene, "scene context", "gray", 0.0, 1.0),
        (start_shadow, "shadow t", "gray", 0.0, 1.0),
        (target_shadow, "target shadow", "gray", 0.0, 1.0),
        (target_recon, "target latent decode", "magma", 0.0, 1.0),
        (pred_prob, "predicted next prob", "magma", 0.0, 1.0),
        (pred_shadow, f"pred >= {threshold:.2f}", "gray", 0.0, 1.0),
    ]

    for axis, (image, title, cmap, vmin, vmax) in zip(axes, panels):
        axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Demo contextual shadow transition.")
    parser.add_argument("--context-jepa-checkpoint", required=True)
    parser.add_argument("--decoder-checkpoint", required=True)
    parser.add_argument("--output", default="outputs/shadow_context_transition_demo.png")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--jepa-base-channels", type=int, default=16)
    parser.add_argument("--decoder-base-channels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()

    context_jepa = load_context_jepa(
        args.context_jepa_checkpoint,
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

    scene, shadows, actions = generate_synthetic_shadow_mask_sequence(
        num_steps=2,
        random_projection=False,
        randomize_objects=True,
        img_width=224,
        img_height=224,
    )

    scene_tensor = torch.from_numpy(scene).unsqueeze(0).to(device)
    start_shadow_tensor = torch.from_numpy(shadows[0]).unsqueeze(0).to(device)
    target_shadow_tensor = torch.from_numpy(shadows[1]).unsqueeze(0).to(device)
    action_tensor = torch.from_numpy(actions[0]).unsqueeze(0).to(device)

    with torch.no_grad():
        z_scene = context_jepa.scene_encoder(scene_tensor)
        z_start = context_jepa.shadow_encoder(start_shadow_tensor)
        z_target = context_jepa.shadow_encoder(target_shadow_tensor)
        z_pred = context_jepa.predictor(z_start, z_scene, action_tensor)
        target_recon_tensor = decoder(z_target)
        pred_prob_tensor = decoder(z_pred)

    scene_image = scene.squeeze(0)
    start_shadow = shadows[0].squeeze(0)
    target_shadow = shadows[1].squeeze(0)
    target_recon = target_recon_tensor.squeeze(0).squeeze(0).cpu().numpy()
    pred_prob = pred_prob_tensor.squeeze(0).squeeze(0).cpu().numpy()

    recon_precision, recon_recall, recon_iou = shadow_metrics(
        target_shadow,
        target_recon,
        args.threshold,
    )
    pred_precision, pred_recall, pred_iou = shadow_metrics(
        target_shadow,
        pred_prob,
        args.threshold,
    )

    save_comparison(
        scene_image,
        start_shadow,
        target_shadow,
        target_recon,
        pred_prob,
        args.threshold,
        args.output,
    )

    print(f"Device: {device}")
    print(f"Scene tensor: {tuple(scene_tensor.shape)}")
    print(f"Start shadow tensor: {tuple(start_shadow_tensor.shape)}")
    print(f"Action tensor: {tuple(action_tensor.shape)}")
    print(f"Predicted latent tensor: {tuple(z_pred.shape)}")
    print(f"Threshold: {args.threshold:.2f}")
    print(f"Target-latent decode precision/recall/IoU: {recon_precision:.4f} / {recon_recall:.4f} / {recon_iou:.4f}")
    print(f"Predicted-transition precision/recall/IoU: {pred_precision:.4f} / {pred_recall:.4f} / {pred_iou:.4f}")
    print(f"Saved comparison: {args.output}")


if __name__ == "__main__":
    main()
