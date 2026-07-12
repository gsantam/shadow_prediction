#!/usr/bin/env python3
"""Evaluate RobotCar sun prediction under image perturbations."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from shadow_prediction.dataset_robotcar_sun import (  # noqa: E402
    DEFAULT_ARCHIVE,
    ROBOTCAR_CAMERAS,
    RobotCarArchiveSunDataset,
)
from shadow_prediction.model_robotcar_sun import create_robotcar_sun_model  # noqa: E402


def luminance(image: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=image.dtype, device=image.device)
    return (image * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)


def box_blur(image: torch.Tensor, kernel_size: int = 15) -> torch.Tensor:
    radius = kernel_size // 2
    padded = F.pad(image, (radius, radius, radius, radius), mode="reflect")
    return F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)


def dark_mask(image: torch.Tensor, quantile: float = 0.35, max_threshold: float = 0.55) -> torch.Tensor:
    y = luminance(image)
    flat = y.flatten(start_dim=1)
    threshold = torch.quantile(flat, quantile, dim=1).view(-1, 1, 1, 1)
    threshold = torch.minimum(threshold, torch.full_like(threshold, max_threshold))
    return (y <= threshold).float()


def apply_ablation(image: torch.Tensor, name: str) -> torch.Tensor:
    if name == "original":
        return image
    if name == "grayscale":
        return luminance(image).repeat(1, 3, 1, 1)
    if name == "bw_threshold":
        y = luminance(image)
        threshold = torch.quantile(y.flatten(start_dim=1), 0.5, dim=1).view(-1, 1, 1, 1)
        return (y > threshold).float().repeat(1, 3, 1, 1)
    if name == "global_blur":
        return box_blur(image, kernel_size=17)
    if name == "dark_blur":
        mask = dark_mask(image)
        return image * (1.0 - mask) + box_blur(image, kernel_size=21) * mask
    if name == "dark_mean":
        mask = dark_mask(image)
        mean_color = image.mean(dim=(2, 3), keepdim=True)
        return image * (1.0 - mask) + mean_color * mask
    if name == "dark_lift":
        y = luminance(image)
        flat = y.flatten(start_dim=1)
        threshold = torch.quantile(flat, 0.35, dim=1).view(-1, 1, 1, 1)
        delta = (threshold - y).clamp(min=0.0)
        return (image + delta).clamp(0.0, 1.0)
    raise ValueError(f"Unknown ablation: {name}")


def make_dataset(
    archive_path: Path,
    sun_runs_only: bool,
    image_size: int,
    max_train: int,
    max_val: int,
) -> RobotCarArchiveSunDataset:
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
    return RobotCarArchiveSunDataset(
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


def evaluate_variant(
    model,
    dataset: RobotCarArchiveSunDataset,
    variant: str,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, float], list[tuple[float, int]]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    total_loss = 0.0
    total_angle = 0.0
    total_count = 0
    per_camera_loss = defaultdict(float)
    per_camera_angle = defaultdict(float)
    per_camera_count = defaultdict(int)
    original_errors: list[tuple[float, int]] = []

    seen = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            image = apply_ablation(batch["image"], variant).to(device)
            pose = batch["pose"].to(device)
            target = batch["target"].to(device)
            camera_index = batch["camera_index"]

            pred = model(image, pose)
            cosine = (pred * target).sum(dim=1).clamp(-1.0, 1.0)
            loss = 1.0 - cosine
            angle = torch.acos(cosine) * (180.0 / math.pi)

            loss_cpu = loss.detach().cpu()
            angle_cpu = angle.detach().cpu()
            total_loss += float(loss_cpu.sum())
            total_angle += float(angle_cpu.sum())
            total_count += int(loss_cpu.numel())

            for offset, (camera_idx, sample_loss, sample_angle) in enumerate(
                zip(camera_index.tolist(), loss_cpu.tolist(), angle_cpu.tolist())
            ):
                camera = ROBOTCAR_CAMERAS[int(camera_idx)]
                per_camera_loss[camera] += float(sample_loss)
                per_camera_angle[camera] += float(sample_angle)
                per_camera_count[camera] += 1
                if variant == "original":
                    original_errors.append((float(sample_angle), seen + offset))

            seen += int(loss_cpu.numel())

    metrics = {
        "loss": total_loss / max(total_count, 1),
        "angle_deg": total_angle / max(total_count, 1),
    }
    for camera in ROBOTCAR_CAMERAS:
        count = max(per_camera_count[camera], 1)
        metrics[f"{camera}_loss"] = per_camera_loss[camera] / count
        metrics[f"{camera}_angle_deg"] = per_camera_angle[camera] / count
    return metrics, original_errors


def plot_metrics(results: dict[str, dict[str, float]], output_path: Path) -> None:
    variants = list(results)
    losses = [results[name]["loss"] for name in variants]
    angles = [results[name]["angle_deg"] for name in variants]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(variants, losses, color="#2563eb")
    axes[0].set_ylabel("Validation loss: 1 - cosine")
    axes[0].set_title("Loss by image ablation")
    axes[1].bar(variants, angles, color="#dc2626")
    axes[1].set_ylabel("Mean angular error (deg)")
    axes[1].set_title("Angle by image ablation")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def tensor_to_image(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu().clamp(0.0, 1.0)
    return image.permute(1, 2, 0).numpy()


def plot_preview(
    dataset: RobotCarArchiveSunDataset,
    best_indices: list[int],
    variants: list[str],
    output_path: Path,
) -> None:
    rows = len(best_indices)
    cols = len(variants)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 2.5))
    axes_array = np.asarray(axes).reshape(rows, cols)
    for row_idx, sample_idx in enumerate(best_indices):
        sample = dataset[sample_idx]
        image = sample["image"].unsqueeze(0)
        camera = sample["camera"]
        for col_idx, variant in enumerate(variants):
            ax = axes_array[row_idx, col_idx]
            transformed = apply_ablation(image, variant)[0]
            ax.imshow(tensor_to_image(transformed))
            ax.axis("off")
            title = variant if row_idx == 0 else ""
            if col_idx == 0:
                title = f"{camera}\n{title}".strip()
            ax.set_title(title, fontsize=8)
    fig.suptitle("Image ablation preview on low-error validation samples", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/robotcar_image_ablations"))
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--max-train", type=int, default=8192)
    parser.add_argument("--max-val", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--all-runs", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "original",
            "grayscale",
            "global_blur",
            "dark_blur",
            "dark_mean",
            "dark_lift",
            "bw_threshold",
        ],
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

    dataset = make_dataset(
        archive_path=args.archive,
        sun_runs_only=not args.all_runs,
        image_size=args.image_size,
        max_train=args.max_train,
        max_val=args.max_val,
    )
    model = create_robotcar_sun_model(pose_dim=dataset.pose_dim, pretrained=False, device=device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    results: dict[str, dict[str, float]] = {}
    original_errors: list[tuple[float, int]] = []
    for variant in args.variants:
        metrics, errors = evaluate_variant(model, dataset, variant, args.batch_size, device)
        results[variant] = metrics
        if variant == "original":
            original_errors = errors
        print(
            f"{variant:12s} loss={metrics['loss']:.4f} "
            f"angle={metrics['angle_deg']:.2f} deg"
        )
        for camera in ROBOTCAR_CAMERAS:
            print(
                f"  {camera:14s} loss={metrics[f'{camera}_loss']:.4f} "
                f"angle={metrics[f'{camera}_angle_deg']:.2f}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_metrics(results, args.output_dir / "image_ablation_metrics.png")
    if original_errors:
        original_errors.sort()
        best_indices = [idx for _, idx in original_errors[:4]]
        plot_preview(dataset, best_indices, args.variants, args.output_dir / "image_ablation_preview.png")
    print(f"Saved ablation outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
