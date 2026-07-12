#!/usr/bin/env python3
"""Plot RobotCar validation curves by sun tag and camera."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


CONDITIONS = ("sun", "not_sun", "all")
CAMERAS_WITH_ALL = (*ROBOTCAR_CAMERAS, "all")


def checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint_step(\d+)", path.stem)
    if match is None:
        return 0
    return int(match.group(1))


def checkpoint_paths(run_dir: Path) -> list[Path]:
    paths = sorted(run_dir.glob("checkpoint_step*.pth"), key=checkpoint_step)
    if not paths:
        raise FileNotFoundError(f"No checkpoint_step*.pth files found in {run_dir}")
    return paths


def select_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_eval_dataset(
    archive_path: Path,
    split: str,
    image_size: int,
    target_frame: str,
    train_sun_runs_only: bool,
) -> RobotCarArchiveSunDataset:
    train_dataset = RobotCarArchiveSunDataset(
        archive_path=archive_path,
        split="train",
        camera="all",
        image_size=image_size,
        target_frame=target_frame,
        include_heading=True,
        include_camera_ohe=True,
        sun_runs_only=train_sun_runs_only,
    )
    return RobotCarArchiveSunDataset(
        archive_path=archive_path,
        split=split,
        camera="all",
        image_size=image_size,
        target_frame=target_frame,
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


def load_checkpoint_models(
    paths: list[Path],
    pose_dim: int,
    backbone: str,
    image_size: int,
    device: torch.device,
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for checkpoint_path in paths:
        print(f"Loading {checkpoint_path.name}", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = create_robotcar_sun_model(
            pose_dim=pose_dim,
            pretrained=False,
            backbone=backbone,
            image_size=image_size,
            device=device,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        items.append(
            {
                "path": checkpoint_path,
                "epoch": int(checkpoint.get("epoch", 0)),
                "global_step": int(
                    checkpoint.get("global_step", checkpoint_step(checkpoint_path))
                ),
                "model": model,
            }
        )
    return items


def evaluate_checkpoints(
    checkpoint_items: list[dict[str, object]],
    loader: DataLoader,
    tag_map: dict[str, set[str]],
    device: torch.device,
    progress_every: int,
) -> list[dict[str, object]]:
    all_buckets: list[dict[tuple[str, str], dict[str, float]]] = [
        {} for _ in checkpoint_items
    ]
    total_batches = len(loader)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            image = batch["image"].to(device)
            pose = batch["pose"].to(device)
            target = batch["target"].to(device)

            tracks = batch["track"]
            camera_indices = batch["camera_index"].detach().cpu().tolist()
            sample_keys: list[tuple[tuple[str, str], ...]] = []
            for track, camera_idx in zip(tracks, camera_indices):
                condition = "sun" if "sun" in tag_map.get(track, set()) else "not_sun"
                camera = ROBOTCAR_CAMERAS[int(camera_idx)]
                sample_keys.append(
                    (
                    (condition, camera),
                    (condition, "all"),
                    ("all", camera),
                    ("all", "all"),
                    )
                )

            for item_idx, item in enumerate(checkpoint_items):
                model = item["model"]
                assert isinstance(model, torch.nn.Module)
                pred = model(image, pose)

                cosine = (pred * target).sum(dim=1).clamp(-1.0, 1.0)
                losses = (1.0 - cosine).detach().cpu().tolist()
                angles = (torch.acos(cosine) * (180.0 / math.pi)).detach().cpu().tolist()
                buckets = all_buckets[item_idx]
                for keys, loss, angle in zip(sample_keys, losses, angles):
                    for key in keys:
                        buckets.setdefault(key, empty_bucket())
                        add_sample(buckets[key], loss, angle)

            if progress_every > 0 and (
                batch_idx == 1
                or batch_idx == total_batches
                or batch_idx % progress_every == 0
            ):
                print(f"Evaluated batch {batch_idx}/{total_batches}", flush=True)

    rows: list[dict[str, object]] = []
    for item, buckets in zip(checkpoint_items, all_buckets):
        path = item["path"]
        assert isinstance(path, Path)
        for condition in CONDITIONS:
            for camera in CAMERAS_WITH_ALL:
                count, loss, angle = finalize(buckets.get((condition, camera), empty_bucket()))
                rows.append(
                    {
                        "epoch": int(item["epoch"]),
                        "global_step": int(item["global_step"]),
                        "checkpoint": path.name,
                        "condition": condition,
                        "camera": camera,
                        "count": count,
                        "loss": loss,
                        "angle_deg": angle,
                    }
                )
    return rows


def save_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "global_step",
                "checkpoint",
                "condition",
                "camera",
                "count",
                "loss",
                "angle_deg",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_breakdown(rows: list[dict[str, object]], output_path: Path) -> None:
    camera_colors = {
        "stereo_centre": "#2563eb",
        "mono_left": "#16a34a",
        "mono_right": "#f97316",
        "mono_rear": "#dc2626",
        "all": "#111827",
    }
    camera_styles = {
        "stereo_centre": "-",
        "mono_left": "-",
        "mono_right": "-",
        "mono_rear": "-",
        "all": "--",
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey="row")
    metrics = (("loss", "Loss: 1 - cosine"), ("angle_deg", "Angle error (deg)"))
    for col, condition in enumerate(CONDITIONS):
        for row_idx, (metric, ylabel) in enumerate(metrics):
            ax = axes[row_idx][col]
            for camera in CAMERAS_WITH_ALL:
                camera_rows = [
                    row
                    for row in rows
                    if row["condition"] == condition and row["camera"] == camera
                ]
                camera_rows.sort(key=lambda row: int(row["epoch"]))
                if not camera_rows:
                    continue
                epochs = [int(row["epoch"]) for row in camera_rows]
                values = [float(row[metric]) for row in camera_rows]
                ax.plot(
                    epochs,
                    values,
                    marker="o",
                    linewidth=2.0 if camera == "all" else 1.5,
                    color=camera_colors[camera],
                    linestyle=camera_styles[camera],
                    label=camera,
                )
            ax.set_title(condition)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)

    fig.suptitle("RobotCar ViT-S/8 validation curves by sun tag and camera")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)


def print_summary(rows: list[dict[str, object]]) -> None:
    latest_epoch = max(int(row["epoch"]) for row in rows)
    latest = [row for row in rows if int(row["epoch"]) == latest_epoch]
    print(f"Latest epoch: {latest_epoch}")
    print(f"{'condition':9s} {'camera':14s} {'count':>7s} {'loss':>9s} {'angle':>9s}")
    for condition in CONDITIONS:
        for camera in CAMERAS_WITH_ALL:
            matches = [
                row
                for row in latest
                if row["condition"] == condition and row["camera"] == camera
            ]
            if not matches:
                continue
            row = matches[0]
            print(
                f"{str(row['condition']):9s} {str(row['camera']):14s} "
                f"{int(row['count']):7d} {float(row['loss']):9.4f} "
                f"{float(row['angle_deg']):9.2f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--backbone", choices=BACKBONE_OPTIONS, default="vit_s_8_timm")
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--target-frame", choices=["car", "global"], default="car")
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
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--plot-output", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"Using device: {device}", flush=True)

    dataset = make_eval_dataset(
        archive_path=args.archive,
        split=args.split,
        image_size=args.image_size,
        target_frame=args.target_frame,
        train_sun_runs_only=args.train_normalization == "sun",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"Evaluation samples: {len(dataset):,}", flush=True)

    checkpoint_items = load_checkpoint_models(
        paths=checkpoint_paths(args.run_dir),
        pose_dim=dataset.pose_dim,
        backbone=args.backbone,
        image_size=args.image_size,
        device=device,
    )
    tag_map = _load_robotcar_tags()

    rows = evaluate_checkpoints(
        checkpoint_items=checkpoint_items,
        loader=loader,
        tag_map=tag_map,
        device=device,
        progress_every=args.progress_every,
    )

    csv_output = args.csv_output or args.run_dir / "val_breakdown_by_condition_camera.csv"
    plot_output = args.plot_output or args.run_dir / "val_breakdown_by_condition_camera.png"
    save_csv(rows, csv_output)
    plot_breakdown(rows, plot_output)
    print_summary(rows)
    print(f"Saved CSV to {csv_output}", flush=True)
    print(f"Saved plot to {plot_output}", flush=True)


if __name__ == "__main__":
    main()
