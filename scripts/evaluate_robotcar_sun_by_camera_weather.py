#!/usr/bin/env python3
"""Evaluate RobotCar sun-direction checkpoints by camera and sun tag."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from shadow_prediction.dataset_robotcar_sun import (  # noqa: E402
    DEFAULT_ARCHIVE,
    ROBOTCAR_CAMERAS,
    RobotCarArchiveSunDataset,
    _load_robotcar_tags,
)
from shadow_prediction.model_robotcar_sun import BACKBONE_OPTIONS  # noqa: E402
from shadow_prediction.model_robotcar_sun import create_robotcar_sun_model  # noqa: E402


def make_eval_dataset(
    archive_path: Path,
    image_size: int,
    split: str,
    train_sun_runs_only: bool,
) -> RobotCarArchiveSunDataset:
    train_dataset = RobotCarArchiveSunDataset(
        archive_path=archive_path,
        split="train",
        camera="all",
        image_size=image_size,
        include_heading=True,
        include_camera_ohe=True,
        sun_runs_only=train_sun_runs_only,
    )
    return RobotCarArchiveSunDataset(
        archive_path=archive_path,
        split=split,
        camera="all",
        image_size=image_size,
        include_heading=True,
        include_camera_ohe=True,
        sun_runs_only=False,
        location_mean=train_dataset.location_mean,
        location_std=train_dataset.location_std,
    )


def empty_bucket() -> dict[str, float]:
    return {"count": 0, "loss_sum": 0.0, "angle_sum": 0.0}


def add_sample(bucket: dict[str, float], loss: float, angle_deg: float) -> None:
    bucket["count"] += 1
    bucket["loss_sum"] += float(loss)
    bucket["angle_sum"] += float(angle_deg)


def finalize(bucket: dict[str, float]) -> tuple[int, float, float]:
    count = int(bucket["count"])
    if count == 0:
        return 0, float("nan"), float("nan")
    return count, bucket["loss_sum"] / count, bucket["angle_sum"] / count


def evaluate(
    checkpoint_path: Path,
    archive_path: Path,
    backbone: str,
    image_size: int,
    batch_size: int,
    split: str,
    train_sun_runs_only: bool,
    device: torch.device,
    num_workers: int,
) -> list[dict[str, object]]:
    dataset = make_eval_dataset(
        archive_path=archive_path,
        image_size=image_size,
        split=split,
        train_sun_runs_only=train_sun_runs_only,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = create_robotcar_sun_model(
        pose_dim=dataset.pose_dim,
        pretrained=False,
        backbone=backbone,
        image_size=image_size,
        device=device,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tag_map = _load_robotcar_tags()
    buckets: dict[tuple[str, str], dict[str, float]] = {}

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            pose = batch["pose"].to(device)
            target = batch["target"].to(device)
            pred = model(image, pose)
            cosine = (pred * target).sum(dim=1).clamp(-1.0, 1.0)
            losses = (1.0 - cosine).detach().cpu()
            angles = (torch.acos(cosine) * (180.0 / math.pi)).detach().cpu()

            tracks = batch["track"]
            camera_indices = batch["camera_index"].detach().cpu().tolist()
            for track, camera_idx, loss, angle in zip(
                tracks,
                camera_indices,
                losses.tolist(),
                angles.tolist(),
            ):
                condition = "sun" if "sun" in tag_map.get(track, set()) else "not_sun"
                camera = ROBOTCAR_CAMERAS[int(camera_idx)]
                for key in (
                    (condition, camera),
                    (condition, "all"),
                    ("all", camera),
                    ("all", "all"),
                ):
                    buckets.setdefault(key, empty_bucket())
                    add_sample(buckets[key], loss, angle)

    rows: list[dict[str, object]] = []
    condition_order = ["sun", "not_sun", "all"]
    camera_order = [*ROBOTCAR_CAMERAS, "all"]
    for condition in condition_order:
        for camera in camera_order:
            bucket = buckets.get((condition, camera), empty_bucket())
            count, loss, angle = finalize(bucket)
            rows.append(
                {
                    "condition": condition,
                    "camera": camera,
                    "count": count,
                    "loss": loss,
                    "angle_deg": angle,
                }
            )
    return rows


def print_rows(rows: list[dict[str, object]]) -> None:
    print(f"{'condition':9s} {'camera':14s} {'count':>7s} {'loss':>9s} {'angle':>9s}")
    for row in rows:
        print(
            f"{str(row['condition']):9s} {str(row['camera']):14s} "
            f"{int(row['count']):7d} "
            f"{float(row['loss']):9.4f} "
            f"{float(row['angle_deg']):9.2f}"
        )


def save_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["condition", "camera", "count", "loss", "angle_deg"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved evaluation CSV to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--backbone", choices=BACKBONE_OPTIONS, default="vit_s_8_timm")
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument(
        "--train-normalization",
        choices=["sun", "all"],
        default="sun",
        help="Training subset used to compute location normalization.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/robotcar_sun_by_camera_weather.csv"),
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

    rows = evaluate(
        checkpoint_path=args.checkpoint,
        archive_path=args.archive,
        backbone=args.backbone,
        image_size=args.image_size,
        batch_size=args.batch_size,
        split=args.split,
        train_sun_runs_only=args.train_normalization == "sun",
        device=device,
        num_workers=args.num_workers,
    )
    print_rows(rows)
    save_csv(rows, args.output)


if __name__ == "__main__":
    main()
