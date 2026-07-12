#!/usr/bin/env python3
"""Render same-location RobotCar validation pairs from different times."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from shadow_prediction.dataset_robotcar_sun import (  # noqa: E402
    DEFAULT_ARCHIVE,
    RobotCarArchiveSunDataset,
)
from shadow_prediction.model_robotcar_sun import create_robotcar_sun_model  # noqa: E402


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cosine = float(np.dot(a, b))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def vector_to_az_alt(vector: np.ndarray) -> tuple[float, float]:
    vector = vector / max(float(np.linalg.norm(vector)), 1e-8)
    azimuth = math.degrees(math.atan2(float(vector[0]), float(vector[1])))
    altitude = math.degrees(math.asin(max(-1.0, min(1.0, float(vector[2])))))
    return azimuth, altitude


def local_time(timestamp_us: int, timezone_name: str) -> str:
    dt = datetime.fromtimestamp(timestamp_us / 1_000_000, tz=timezone.utc).astimezone(
        ZoneInfo(timezone_name)
    )
    return dt.strftime("%Y-%m-%d %H:%M")


def local_hour(timestamp_us: int, timezone_name: str) -> float:
    dt = datetime.fromtimestamp(timestamp_us / 1_000_000, tz=timezone.utc).astimezone(
        ZoneInfo(timezone_name)
    )
    return dt.hour + dt.minute / 60.0 + dt.second / 3600.0


def make_dataset(
    archive_path: Path,
    camera: str,
    image_size: int,
    max_train: int,
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
        sun_runs_only=True,
    )
    return RobotCarArchiveSunDataset(
        archive_path=archive_path,
        split="val",
        camera=camera,
        image_size=image_size,
        include_heading=True,
        include_camera_ohe=True,
        max_samples=None,
        seed=8,
        sun_runs_only=True,
        location_mean=train_dataset.location_mean,
        location_std=train_dataset.location_std,
    )


def score_dataset(
    checkpoint_path: Path,
    dataset: RobotCarArchiveSunDataset,
    device: torch.device,
    batch_size: int,
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
            cosine = (pred * target).sum(dim=1).clamp(-1.0, 1.0)
            errors = torch.acos(cosine) * (180.0 / math.pi)
            pred_np = pred.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            errors_np = errors.detach().cpu().numpy()

            for offset, error in enumerate(errors_np):
                idx = seen + offset
                sample = dataset.samples[idx]
                rows.append(
                    {
                        "idx": idx,
                        "error_deg": float(error),
                        "pred": pred_np[offset],
                        "target": target_np[offset],
                        "easting": float(sample["easting"]),
                        "northing": float(sample["northing"]),
                        "timestamp_us": int(sample["timestamp_us"]),
                        "track": str(sample["track"]),
                        "camera": str(sample["camera"]),
                        "image_path": str(sample["image_path"]),
                    }
                )
            seen += len(errors_np)
    return rows


def find_pairs(
    rows: list[dict[str, object]],
    timezone_name: str,
    max_distance_m: float,
    min_hour_delta: float,
    min_target_delta_deg: float,
    max_mean_error_deg: float,
    top_k: int,
) -> list[dict[str, object]]:
    points = np.array(
        [(float(row["easting"]), float(row["northing"])) for row in rows],
        dtype=np.float32,
    )
    tree = cKDTree(points)
    candidates = []
    for i, j in tree.query_pairs(max_distance_m):
        a = rows[i]
        b = rows[j]
        if a["track"] == b["track"]:
            continue

        hour_a = local_hour(int(a["timestamp_us"]), timezone_name)
        hour_b = local_hour(int(b["timestamp_us"]), timezone_name)
        hour_delta = abs(hour_a - hour_b)
        if hour_delta < min_hour_delta:
            continue

        distance = float(np.linalg.norm(points[i] - points[j]))
        target_delta = angle_deg(
            np.asarray(a["target"], dtype=np.float32),
            np.asarray(b["target"], dtype=np.float32),
        )
        if target_delta < min_target_delta_deg:
            continue

        mean_error = 0.5 * (float(a["error_deg"]) + float(b["error_deg"]))
        if mean_error > max_mean_error_deg:
            continue

        # Put earlier local time first.
        if hour_a > hour_b:
            a, b = b, a
        candidates.append(
            {
                "a": a,
                "b": b,
                "distance_m": distance,
                "hour_delta": hour_delta,
                "target_delta_deg": target_delta,
                "mean_error_deg": mean_error,
            }
        )

    candidates.sort(
        key=lambda item: (
            float(item["mean_error_deg"]),
            -float(item["target_delta_deg"]),
            float(item["distance_m"]),
        )
    )

    selected = []
    used_indices: set[int] = set()
    used_location_keys: set[tuple[int, int]] = set()
    for candidate in candidates:
        a = candidate["a"]
        b = candidate["b"]
        location_key = (
            int(round((float(a["easting"]) + float(b["easting"])) * 0.5 / 10.0)),
            int(round((float(a["northing"]) + float(b["northing"])) * 0.5 / 10.0)),
        )
        if int(a["idx"]) in used_indices or int(b["idx"]) in used_indices:
            continue
        if location_key in used_location_keys:
            continue
        selected.append(candidate)
        used_indices.update([int(a["idx"]), int(b["idx"])])
        used_location_keys.add(location_key)
        if len(selected) >= top_k:
            break
    return selected


def load_display_image(dataset: RobotCarArchiveSunDataset, image_path: str, display_size: int):
    image = dataset._load_image_member(image_path).convert("RGB")
    image.thumbnail((display_size, display_size), resample=2)
    return image


def panel_title(row: dict[str, object], timezone_name: str) -> str:
    pred_az, pred_alt = vector_to_az_alt(np.asarray(row["pred"], dtype=np.float32))
    target_az, target_alt = vector_to_az_alt(np.asarray(row["target"], dtype=np.float32))
    return (
        f"{local_time(int(row['timestamp_us']), timezone_name)} | {row['track']}\n"
        f"err {float(row['error_deg']):.1f} deg | "
        f"pred {pred_az:.0f}/{pred_alt:.0f} target {target_az:.0f}/{target_alt:.0f}"
    )


def render_pair_sheet(
    dataset: RobotCarArchiveSunDataset,
    pairs: list[dict[str, object]],
    output_path: Path,
    timezone_name: str,
    display_size: int,
) -> None:
    fig, axes = plt.subplots(len(pairs), 2, figsize=(11, max(4, len(pairs) * 3.5)))
    axes_array = np.asarray(axes).reshape(len(pairs), 2)

    for row_idx, pair in enumerate(pairs):
        for col_idx, key in enumerate(["a", "b"]):
            row = pair[key]
            ax = axes_array[row_idx, col_idx]
            image = load_display_image(dataset, str(row["image_path"]), display_size)
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(panel_title(row, timezone_name), fontsize=8)

        axes_array[row_idx, 0].set_ylabel(
            f"dist {float(pair['distance_m']):.1f}m\n"
            f"sun delta {float(pair['target_delta_deg']):.0f} deg",
            fontsize=9,
            rotation=0,
            labelpad=48,
            va="center",
        )

    fig.suptitle(
        "Same/similar stereo_centre validation locations at different times",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def render_individual_pairs(
    dataset: RobotCarArchiveSunDataset,
    pairs: list[dict[str, object]],
    output_dir: Path,
    timezone_name: str,
    display_size: int,
) -> None:
    for idx, pair in enumerate(pairs, start=1):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, key in zip(axes, ["a", "b"]):
            row = pair[key]
            image = load_display_image(dataset, str(row["image_path"]), display_size)
            ax.imshow(image)
            ax.axis("off")
            ax.set_title(panel_title(row, timezone_name), fontsize=8)
        fig.suptitle(
            f"Pair {idx}: distance {float(pair['distance_m']):.1f}m, "
            f"target sun delta {float(pair['target_delta_deg']):.1f} deg, "
            f"mean error {float(pair['mean_error_deg']):.1f} deg",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"same_place_pair_{idx:02d}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--camera", default="stereo_centre")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/robotcar_same_place_times"))
    parser.add_argument("--timezone", default="Europe/London")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--max-distance-m", type=float, default=5.0)
    parser.add_argument("--min-hour-delta", type=float, default=3.0)
    parser.add_argument("--min-target-delta-deg", type=float, default=35.0)
    parser.add_argument("--max-mean-error-deg", type=float, default=12.0)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--display-size", type=int, default=360)
    parser.add_argument("--max-train", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=128)
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

    dataset = make_dataset(args.archive, args.camera, args.image_size, args.max_train)
    rows = score_dataset(args.checkpoint, dataset, device, args.batch_size)
    pairs = find_pairs(
        rows=rows,
        timezone_name=args.timezone,
        max_distance_m=args.max_distance_m,
        min_hour_delta=args.min_hour_delta,
        min_target_delta_deg=args.min_target_delta_deg,
        max_mean_error_deg=args.max_mean_error_deg,
        top_k=args.top_k,
    )
    if not pairs:
        raise RuntimeError("No same-location different-time pairs matched the filters")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = args.output_dir / "same_place_different_times_sheet.png"
    render_pair_sheet(dataset, pairs, sheet_path, args.timezone, args.display_size)
    render_individual_pairs(dataset, pairs, args.output_dir, args.timezone, args.display_size)

    print(f"Saved pair sheet to {sheet_path}")
    for idx, pair in enumerate(pairs, start=1):
        a = pair["a"]
        b = pair["b"]
        print(
            f"{idx:02d} dist={float(pair['distance_m']):.2f}m "
            f"sun_delta={float(pair['target_delta_deg']):.1f}deg "
            f"mean_error={float(pair['mean_error_deg']):.2f}deg"
        )
        for label, row in [("A", a), ("B", b)]:
            pred_az, pred_alt = vector_to_az_alt(np.asarray(row["pred"], dtype=np.float32))
            target_az, target_alt = vector_to_az_alt(np.asarray(row["target"], dtype=np.float32))
            print(
                f"  {label} {local_time(int(row['timestamp_us']), args.timezone)} "
                f"track={row['track']} error={float(row['error_deg']):.2f}deg "
                f"pred={pred_az:.1f}/{pred_alt:.1f} "
                f"target={target_az:.1f}/{target_alt:.1f}"
            )


if __name__ == "__main__":
    main()
