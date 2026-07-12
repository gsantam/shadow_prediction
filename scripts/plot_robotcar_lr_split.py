#!/usr/bin/env python3
"""Plot RobotCar sun-predictor learning-rate curves split by dataset condition."""

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


def add_run(ax, checkpoint: Path, label: str, color: str) -> tuple[int, float, float]:
    train_losses, val_losses = load_losses(checkpoint)
    epochs = list(range(1, len(val_losses) + 1))
    ax.plot(epochs, train_losses, color=color, linestyle=":", alpha=0.9, label=f"{label} train")
    ax.plot(epochs, val_losses, color=color, linestyle="-", linewidth=2, label=f"{label} val")
    best_idx = min(range(len(val_losses)), key=val_losses.__getitem__)
    ax.scatter([best_idx + 1], [val_losses[best_idx]], color=color, s=36, zorder=3)
    return best_idx + 1, val_losses[best_idx], train_losses[best_idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sun-lr1e4", type=Path, required=True)
    parser.add_argument("--sun-lr3e5", type=Path, required=True)
    parser.add_argument("--all-lr1e4", type=Path, required=True)
    parser.add_argument("--all-lr3e5", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/robotcar_lr_split_sun_vs_all.png"),
    )
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    summaries = []

    summaries.append(
        ("sun-only lr=1e-4",)
        + add_run(axes[0], args.sun_lr1e4, "lr=1e-4", "#2563eb")
    )
    summaries.append(
        ("sun-only lr=3e-5",)
        + add_run(axes[0], args.sun_lr3e5, "lr=3e-5", "#0891b2")
    )
    summaries.append(
        ("all-runs lr=1e-4",)
        + add_run(axes[1], args.all_lr1e4, "lr=1e-4", "#dc2626")
    )
    summaries.append(
        ("all-runs lr=3e-5",)
        + add_run(axes[1], args.all_lr3e5, "lr=3e-5", "#f97316")
    )

    axes[0].set_title("Sun-tagged runs only")
    axes[1].set_title("All runs")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss: 1 - cosine")
        ax.grid(alpha=0.25)
        ax.legend()

    fig.suptitle("RobotCar stereo_centre learning-rate comparison")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)

    print(f"Saved LR split plot to {args.output}")
    for label, best_epoch, best_val_loss, train_loss in summaries:
        print(
            f"{label}: best_epoch={best_epoch}, "
            f"best_val_loss={best_val_loss:.4f}, train_loss_at_best={train_loss:.4f}"
        )


if __name__ == "__main__":
    main()
