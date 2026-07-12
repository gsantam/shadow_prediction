#!/usr/bin/env python3
"""Plot RobotCar sun-predictor learning curves from checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def load_curve(checkpoint_path: Path) -> dict[str, list[float]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    train_losses = [float(value) for value in checkpoint.get("train_losses", [])]
    val_losses = [float(value) for value in checkpoint.get("val_losses", [])]
    if not train_losses or not val_losses:
        raise ValueError(f"Checkpoint has no loss history: {checkpoint_path}")
    epochs = list(range(1, min(len(train_losses), len(val_losses)) + 1))
    return {
        "epochs": epochs,
        "train_losses": train_losses[: len(epochs)],
        "val_losses": val_losses[: len(epochs)],
    }


def add_curve(ax, curve: dict[str, list[float]], label: str, color: str) -> None:
    ax.plot(
        curve["epochs"],
        curve["train_losses"],
        marker="o",
        linestyle="-",
        color=color,
        label=f"{label} train",
    )
    ax.plot(
        curve["epochs"],
        curve["val_losses"],
        marker="s",
        linestyle="--",
        color=color,
        label=f"{label} val/test",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sun-checkpoint",
        type=Path,
        required=True,
        help="Checkpoint containing the sun-tagged training history.",
    )
    parser.add_argument(
        "--overall-checkpoint",
        type=Path,
        required=True,
        help="Checkpoint containing the all-runs training history.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/robotcar_learning_curves_sun_vs_overall.png"),
    )
    args = parser.parse_args()

    sun_curve = load_curve(args.sun_checkpoint)
    overall_curve = load_curve(args.overall_checkpoint)

    fig, ax = plt.subplots(figsize=(9, 5))
    add_curve(ax, sun_curve, "sun-only", "#2563eb")
    add_curve(ax, overall_curve, "overall", "#dc2626")
    ax.set_title("RobotCar Sun Direction Predictor")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss: 1 - cosine(pred, target)")
    ax.set_xticks(sorted(set(sun_curve["epochs"]) | set(overall_curve["epochs"])))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"Saved learning curves to {args.output}")
    print("sun-only")
    print("  train:", sun_curve["train_losses"])
    print("  val/test:", sun_curve["val_losses"])
    print("overall")
    print("  train:", overall_curve["train_losses"])
    print("  val/test:", overall_curve["val_losses"])


if __name__ == "__main__":
    main()
