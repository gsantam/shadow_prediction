#!/usr/bin/env python3
"""Plot RobotCar sun-predictor learning curves for multiple camera runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def load_losses(checkpoint_path: Path) -> tuple[list[float], list[float]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_losses = [float(value) for value in checkpoint["train_losses"]]
    val_losses = [float(value) for value in checkpoint["val_losses"]]
    return train_losses, val_losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "CHECKPOINT"),
        required=True,
        help="Camera label and checkpoint path. Can be repeated.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/robotcar_camera_learning_curves.png"),
    )
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    summary: list[tuple[str, int, float, float]] = []

    for idx, (label, checkpoint) in enumerate(args.run):
        train_losses, val_losses = load_losses(Path(checkpoint))
        epochs = list(range(1, len(val_losses) + 1))
        color = colors[idx % len(colors)]
        axes[0].plot(epochs, train_losses, color=color, label=label)
        axes[1].plot(epochs, val_losses, color=color, label=label)

        best_idx = min(range(len(val_losses)), key=val_losses.__getitem__)
        summary.append((label, best_idx + 1, val_losses[best_idx], train_losses[best_idx]))

    axes[0].set_title("Train")
    axes[1].set_title("Validation")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss: 1 - cosine")
        ax.grid(alpha=0.25)
        ax.legend()

    fig.suptitle("RobotCar Sun Direction Predictor by Camera")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"Saved camera curves to {args.output}")
    for label, epoch, val_loss, train_loss in summary:
        print(
            f"{label}: best_epoch={epoch}, "
            f"best_val_loss={val_loss:.4f}, train_loss_at_best={train_loss:.4f}"
        )


if __name__ == "__main__":
    main()
