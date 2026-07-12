#!/usr/bin/env python3
"""Evaluate joint all-camera RobotCar checkpoints with per-camera validation curves."""

from __future__ import annotations

import argparse
from collections import defaultdict
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from shadow_prediction.dataset_robotcar_sun import (
    DEFAULT_ARCHIVE,
    ROBOTCAR_CAMERAS,
    RobotCarArchiveSunDataset,
)
from shadow_prediction.model_robotcar_sun import create_robotcar_sun_model


def checkpoint_paths(run_dir: Path) -> list[Path]:
    paths = sorted(
        run_dir.glob("checkpoint_step*.pth"),
        key=lambda path: int(path.stem.replace("checkpoint_step", "")),
    )
    if not paths:
        raise FileNotFoundError(f"No checkpoint_step*.pth files found in {run_dir}")
    return paths


def make_val_loader(
    archive_path: Path,
    sun_runs_only: bool,
    image_size: int,
    max_train: int,
    max_val: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    train_dataset = RobotCarArchiveSunDataset(
        archive_path=archive_path,
        split="train",
        camera="all",
        image_size=image_size,
        include_heading=True,
        include_camera_ohe=True,
        max_samples=max_train,
        seed=7,
        sun_runs_only=sun_runs_only,
    )
    val_dataset = RobotCarArchiveSunDataset(
        archive_path=archive_path,
        split="val",
        camera="all",
        image_size=image_size,
        include_heading=True,
        include_camera_ohe=True,
        max_samples=max_val,
        seed=8,
        sun_runs_only=sun_runs_only,
        location_mean=train_dataset.location_mean,
        location_std=train_dataset.location_std,
    )
    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def evaluate_checkpoint(
    checkpoint_path: Path,
    model,
    loader: DataLoader,
    device: torch.device,
) -> tuple[int, dict[str, float]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loss_sums = defaultdict(float)
    counts = defaultdict(int)
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            pose = batch["pose"].to(device)
            target = batch["target"].to(device)
            camera_index = batch["camera_index"]

            output = model(image, pose)
            sample_loss = 1.0 - (output * target).sum(dim=1).clamp(-1.0, 1.0)
            sample_loss = sample_loss.detach().cpu()

            for idx, loss in zip(camera_index.tolist(), sample_loss.tolist()):
                camera = ROBOTCAR_CAMERAS[int(idx)]
                loss_sums[camera] += float(loss)
                counts[camera] += 1

    per_camera = {
        camera: loss_sums[camera] / max(counts[camera], 1)
        for camera in ROBOTCAR_CAMERAS
    }
    return int(checkpoint["epoch"]), per_camera


def evaluate_run(
    run_dir: Path,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], dict[str, list[float]]]:
    model = create_robotcar_sun_model(pose_dim=8, pretrained=False, device=device)
    epochs: list[int] = []
    curves = {camera: [] for camera in ROBOTCAR_CAMERAS}

    for path in checkpoint_paths(run_dir):
        epoch, per_camera = evaluate_checkpoint(path, model, loader, device)
        epochs.append(epoch)
        for camera in ROBOTCAR_CAMERAS:
            curves[camera].append(per_camera[camera])
        print(
            f"{run_dir.name} epoch={epoch:02d} "
            + " ".join(f"{camera}={per_camera[camera]:.4f}" for camera in ROBOTCAR_CAMERAS)
        )
    return epochs, curves


def plot_curves(
    output_path: Path,
    sun_epochs: list[int],
    sun_curves: dict[str, list[float]],
    all_epochs: list[int],
    all_curves: dict[str, list[float]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    colors = {
        "stereo_centre": "#2563eb",
        "mono_left": "#16a34a",
        "mono_right": "#f97316",
        "mono_rear": "#dc2626",
    }

    for camera in ROBOTCAR_CAMERAS:
        axes[0].plot(sun_epochs, sun_curves[camera], color=colors[camera], label=camera)
        axes[1].plot(all_epochs, all_curves[camera], color=colors[camera], label=camera)

    axes[0].set_title("Joint model: sun-only val")
    axes[1].set_title("Joint model: all-runs val")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Per-camera val loss: 1 - cosine")
        ax.grid(alpha=0.25)
        ax.legend()

    fig.suptitle("RobotCar joint all-camera model, camera-OHE, per-camera validation curves")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    print(f"Saved per-camera joint learning curves to {output_path}")


def print_best(label: str, epochs: list[int], curves: dict[str, list[float]]) -> None:
    print(label)
    for camera in ROBOTCAR_CAMERAS:
        values = curves[camera]
        best_idx = min(range(len(values)), key=values.__getitem__)
        print(
            f"  {camera:14s} best_epoch={epochs[best_idx]:2d} "
            f"best_val_loss={values[best_idx]:.4f} final_val_loss={values[-1]:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--sun-run-dir", type=Path, required=True)
    parser.add_argument("--all-run-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--max-train", type=int, default=8192)
    parser.add_argument("--max-val", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/robotcar_joint_per_camera_lc.png"),
    )
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    sun_loader = make_val_loader(
        archive_path=args.archive,
        sun_runs_only=True,
        image_size=args.image_size,
        max_train=args.max_train,
        max_val=args.max_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    all_loader = make_val_loader(
        archive_path=args.archive,
        sun_runs_only=False,
        image_size=args.image_size,
        max_train=args.max_train,
        max_val=args.max_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    sun_epochs, sun_curves = evaluate_run(args.sun_run_dir, sun_loader, device)
    all_epochs, all_curves = evaluate_run(args.all_run_dir, all_loader, device)
    plot_curves(args.output, sun_epochs, sun_curves, all_epochs, all_curves)
    print_best("sun-only", sun_epochs, sun_curves)
    print_best("all-runs", all_epochs, all_curves)


if __name__ == "__main__":
    main()
