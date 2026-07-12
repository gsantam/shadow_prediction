#!/usr/bin/env python3
"""Render RobotCar validation images with the lowest sun-direction error."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from shadow_prediction.dataset_robotcar_sun import (  # noqa: E402
    DEFAULT_ARCHIVE,
    RobotCarArchiveSunDataset,
)
from shadow_prediction.model_robotcar_sun import create_robotcar_sun_model  # noqa: E402


def angle_error_deg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cosine = (pred * target).sum(dim=1).clamp(-1.0, 1.0)
    return torch.acos(cosine) * (180.0 / math.pi)


def vector_to_az_alt(vector: np.ndarray) -> tuple[float, float]:
    vector = vector / max(float(np.linalg.norm(vector)), 1e-8)
    azimuth = math.degrees(math.atan2(float(vector[0]), float(vector[1])))
    altitude = math.degrees(math.asin(max(-1.0, min(1.0, float(vector[2])))))
    return azimuth, altitude


def make_datasets(
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


def collect_best(
    checkpoint_path: Path,
    dataset: RobotCarArchiveSunDataset,
    batch_size: int,
    device: torch.device,
    top_k: int,
) -> list[dict[str, object]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = create_robotcar_sun_model(pose_dim=dataset.pose_dim, pretrained=False, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows: list[dict[str, object]] = []
    seen = 0
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            pose = batch["pose"].to(device)
            target = batch["target"].to(device)
            pred = model(image, pose)
            errors = angle_error_deg(pred, target).detach().cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()

            for offset, error in enumerate(errors):
                sample_idx = seen + offset
                sample_meta = dataset.samples[sample_idx]
                rows.append(
                    {
                        "sample_idx": sample_idx,
                        "error_deg": float(error),
                        "pred": pred_np[offset],
                        "target": target_np[offset],
                        "camera": sample_meta["camera"],
                        "track": sample_meta["track"],
                        "timestamp_us": sample_meta["timestamp_us"],
                        "image_path": sample_meta["image_path"],
                    }
                )
            seen += len(errors)

    rows.sort(key=lambda row: float(row["error_deg"]))
    return rows[:top_k]


def load_display_image(dataset: RobotCarArchiveSunDataset, image_path: str, size: int):
    image = dataset._load_image_member(image_path).convert("RGB")
    image.thumbnail((size, size), resample=2)
    return image


def render_individual(
    dataset: RobotCarArchiveSunDataset,
    row: dict[str, object],
    output_path: Path,
    image_px: int,
) -> None:
    image = load_display_image(dataset, str(row["image_path"]), image_px)
    pred_az, pred_alt = vector_to_az_alt(np.asarray(row["pred"], dtype=np.float32))
    target_az, target_alt = vector_to_az_alt(np.asarray(row["target"], dtype=np.float32))

    fig, ax = plt.subplots(figsize=(5, 4.8))
    ax.imshow(image)
    ax.axis("off")
    title = (
        f"{row['camera']} | err {float(row['error_deg']):.2f} deg\n"
        f"pred az/alt {pred_az:.1f}/{pred_alt:.1f} | "
        f"target {target_az:.1f}/{target_alt:.1f}"
    )
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_sheet(
    dataset: RobotCarArchiveSunDataset,
    rows: list[dict[str, object]],
    output_path: Path,
    image_px: int,
    columns: int,
) -> None:
    columns = max(1, columns)
    rows_count = math.ceil(len(rows) / columns)
    fig, axes = plt.subplots(rows_count, columns, figsize=(columns * 4.0, rows_count * 3.6))
    axes_array = np.asarray(axes).reshape(rows_count, columns)

    for ax in axes_array.flatten():
        ax.axis("off")

    for idx, row in enumerate(rows):
        ax = axes_array[idx // columns, idx % columns]
        image = load_display_image(dataset, str(row["image_path"]), image_px)
        pred_az, pred_alt = vector_to_az_alt(np.asarray(row["pred"], dtype=np.float32))
        target_az, target_alt = vector_to_az_alt(np.asarray(row["target"], dtype=np.float32))
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(
            f"#{idx + 1} {row['camera']} | {float(row['error_deg']):.2f} deg\n"
            f"pred {pred_az:.0f}/{pred_alt:.0f}  target {target_az:.0f}/{target_alt:.0f}",
            fontsize=8,
        )

    fig.suptitle("Best joint RobotCar sun-direction predictions on validation", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/robotcar_best_predictions"))
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--display-size", type=int, default=320)
    parser.add_argument("--max-train", type=int, default=8192)
    parser.add_argument("--max-val", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--all-runs", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    dataset = make_datasets(
        archive_path=args.archive,
        sun_runs_only=not args.all_runs,
        image_size=args.image_size,
        max_train=args.max_train,
        max_val=args.max_val,
    )
    best = collect_best(
        checkpoint_path=args.checkpoint,
        dataset=dataset,
        batch_size=args.batch_size,
        device=device,
        top_k=args.top_k,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = args.output_dir / "best_predictions_sheet.png"
    render_sheet(dataset, best, sheet_path, args.display_size, args.columns)
    for idx, row in enumerate(best, start=1):
        render_individual(
            dataset,
            row,
            args.output_dir / f"best_{idx:02d}_{row['camera']}_{float(row['error_deg']):.2f}deg.png",
            args.display_size,
        )

    print(f"Saved contact sheet to {sheet_path}")
    for idx, row in enumerate(best, start=1):
        pred_az, pred_alt = vector_to_az_alt(np.asarray(row["pred"], dtype=np.float32))
        target_az, target_alt = vector_to_az_alt(np.asarray(row["target"], dtype=np.float32))
        print(
            f"{idx:02d} camera={row['camera']:13s} "
            f"error={float(row['error_deg']):.3f}deg "
            f"pred={pred_az:.1f}/{pred_alt:.1f} "
            f"target={target_az:.1f}/{target_alt:.1f} "
            f"track={row['track']}"
        )


if __name__ == "__main__":
    main()
