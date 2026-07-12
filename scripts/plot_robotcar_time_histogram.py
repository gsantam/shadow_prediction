#!/usr/bin/env python3
"""Plot RobotCar split time-of-day histograms from the Kaggle archive."""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from shadow_prediction.dataset_robotcar_sun import (  # noqa: E402
    ARCHIVE_ROOT,
    DEFAULT_ARCHIVE,
    _load_robotcar_tags,
)


def timestamp_to_hour(timestamp_us: int, tz: ZoneInfo) -> float:
    dt = datetime.fromtimestamp(timestamp_us / 1_000_000, tz=timezone.utc).astimezone(tz)
    return dt.hour + dt.minute / 60.0 + dt.second / 3600.0


def load_split_hours(
    archive_path: Path,
    split: str,
    camera: str,
    tz: ZoneInfo,
    sun_runs_only: bool,
) -> np.ndarray:
    tag_map = _load_robotcar_tags() if sun_runs_only else {}
    hours: list[float] = []

    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(f"{ARCHIVE_ROOT}/{split}.csv") as f:
            wrapper = io.TextIOWrapper(f, encoding="utf-8", newline="")
            reader = csv.DictReader(wrapper)
            for row in reader:
                if sun_runs_only and "sun" not in tag_map.get(row["track"], set()):
                    continue
                value = row.get(camera, "").strip()
                if not value or value.lower() in {"nan", "none"}:
                    continue
                hours.append(timestamp_to_hour(int(value), tz))

    return np.asarray(hours, dtype=np.float32)


def plot_histogram(
    hours_by_split: dict[str, np.ndarray],
    title: str,
    ax,
    bins: np.ndarray,
) -> None:
    colors = {
        "train": "#2563eb",
        "val": "#dc2626",
        "test": "#16a34a",
    }
    for split, hours in hours_by_split.items():
        weights = np.ones_like(hours) * (100.0 / max(len(hours), 1))
        ax.hist(
            hours,
            bins=bins,
            weights=weights,
            histtype="stepfilled",
            alpha=0.35,
            color=colors.get(split, None),
            edgecolor=colors.get(split, None),
            label=f"{split} (n={len(hours):,})",
        )

    ax.set_title(title)
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_ylabel("% of split")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--camera", default="stereo_centre")
    parser.add_argument("--timezone", default="Europe/London")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/robotcar_time_histogram_stereo_centre.png"),
    )
    args = parser.parse_args()

    tz = ZoneInfo(args.timezone)
    bins = np.arange(0, 25, 1)

    all_hours = {
        split: load_split_hours(args.archive, split, args.camera, tz, sun_runs_only=False)
        for split in args.splits
    }
    sun_hours = {
        split: load_split_hours(args.archive, split, args.camera, tz, sun_runs_only=True)
        for split in args.splits
    }

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    plot_histogram(
        all_hours,
        f"Oxford RobotCar {args.camera}: all runs, local time ({args.timezone})",
        axes[0],
        bins,
    )
    plot_histogram(
        sun_hours,
        f"Oxford RobotCar {args.camera}: sun-tagged runs only, local time ({args.timezone})",
        axes[1],
        bins,
    )
    axes[1].set_xlabel("Hour of day")
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"Saved histogram to {args.output}")

    for name, hours_by_split in [("all", all_hours), ("sun_only", sun_hours)]:
        print(name)
        for split, hours in hours_by_split.items():
            print(
                f"  {split}: n={len(hours):,}, "
                f"min={hours.min():.2f}, median={np.median(hours):.2f}, max={hours.max():.2f}"
            )


if __name__ == "__main__":
    main()
