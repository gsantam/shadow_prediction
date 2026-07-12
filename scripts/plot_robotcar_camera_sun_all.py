#!/usr/bin/env python3
"""Plot sun-only vs all-runs learning curves split by RobotCar camera."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def load_losses(checkpoint_path: Path) -> tuple[list[float], list[float]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return (
        [float(value) for value in checkpoint["train_losses"]],
        [float(value) for value in checkpoint["val_losses"]],
    )


def plot_condition(ax, checkpoint_path: Path, label: str, color: str) -> tuple[int, float, float]:
    train_losses, val_losses = load_losses(checkpoint_path)
    epochs = list(range(1, len(val_losses) + 1))
    ax.plot(epochs, train_losses, color=color, linestyle=":", alpha=0.8, label=f"{label} train")
    ax.plot(epochs, val_losses, color=color, linewidth=2, label=f"{label} val")
    best_idx = min(range(len(val_losses)), key=val_losses.__getitem__)
    ax.scatter([best_idx + 1], [val_losses[best_idx]], color=color, s=34, zorder=3)
    return best_idx + 1, val_losses[best_idx], train_losses[best_idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera",
        action="append",
        nargs=3,
        metavar=("NAME", "SUN_CHECKPOINT", "ALL_CHECKPOINT"),
        required=True,
        help="Camera name, sun-only final checkpoint, all-runs final checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/robotcar_camera_sun_vs_all.png"),
    )
    args = parser.parse_args()

    camera_specs = args.camera
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    summaries = []

    for ax, (camera, sun_checkpoint, all_checkpoint) in zip(axes_flat, camera_specs):
        sun_summary = plot_condition(ax, Path(sun_checkpoint), "sun-only", "#2563eb")
        all_summary = plot_condition(ax, Path(all_checkpoint), "all-runs", "#dc2626")
        ax.set_title(camera)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss: 1 - cosine")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        summaries.append((camera, "sun-only", *sun_summary))
        summaries.append((camera, "all-runs", *all_summary))

    for ax in axes_flat[len(camera_specs) :]:
        ax.axis("off")

    fig.suptitle("RobotCar sun-only vs all-runs by camera, lr=3e-5")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"Saved camera split plot to {args.output}")

    for camera, condition, epoch, val_loss, train_loss in summaries:
        print(
            f"{camera:14s} {condition:8s} "
            f"best_epoch={epoch:2d}, best_val_loss={val_loss:.4f}, "
            f"train_loss_at_best={train_loss:.4f}"
        )


if __name__ == "__main__":
    main()
