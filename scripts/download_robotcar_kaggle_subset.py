#!/usr/bin/env python3
"""Download a small Kaggle RobotCar subset and join images to locations."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from bisect import bisect_left
from datetime import datetime, timezone
from pathlib import Path


DATASET = "creatorofuniverses/oxfordrobotcar-iprofi-hack-23"
ROOT_PREFIX = "pnvlad_oxford_robotcar"
LOCATION_NAME = "pointcloud_locations_20m_10overlap.csv"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT.parent / "data" / "robotcar" / "kaggle_subset"


def run_kaggle(args: list[str]) -> str:
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        local = REPO_ROOT / ".venv" / "bin" / "kaggle"
        if local.exists():
            kaggle = str(local)
    if kaggle is None:
        raise RuntimeError("Kaggle CLI not found. Install it with `pip install kaggle`.")
    return subprocess.check_output([kaggle, *args], text=True, stderr=subprocess.STDOUT)


def parse_kaggle_csv_listing(text: str) -> tuple[str | None, list[dict[str, str]]]:
    lines = text.splitlines()
    next_token = None
    if lines and lines[0].startswith("Next Page Token = "):
        next_token = lines[0].split(" = ", 1)[1].strip()
        lines = lines[1:]
    rows = list(csv.DictReader(lines))
    return next_token, rows


def list_dataset_files(run: str, camera: str, count: int) -> tuple[list[str], str]:
    image_prefix = f"{ROOT_PREFIX}/{run}/images_small/{camera}/"
    location_file = f"{ROOT_PREFIX}/{run}/{LOCATION_NAME}"
    token = None
    image_files: list[str] = []
    found_location = False

    while len(image_files) < count or not found_location:
        args = [
            "datasets",
            "files",
            DATASET,
            "--page-size",
            "200",
            "--csv",
        ]
        if token:
            args += ["--page-token", token]
        token, rows = parse_kaggle_csv_listing(run_kaggle(args))

        for row in rows:
            name = row["name"]
            if name == location_file:
                found_location = True
            if name.startswith(image_prefix) and name.endswith(".png"):
                image_files.append(name)
                if len(image_files) >= count and found_location:
                    break

        if not token:
            break

    if not found_location:
        raise RuntimeError(f"Could not find location file: {location_file}")
    if not image_files:
        raise RuntimeError(f"Could not find images under: {image_prefix}")

    return image_files[:count], location_file


def kaggle_download(file_name: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / Path(file_name).name
    if path.exists():
        return path
    run_kaggle(
        [
            "datasets",
            "download",
            "-d",
            DATASET,
            "-f",
            file_name,
            "-p",
            str(output_dir),
            "--quiet",
        ]
    )
    if not path.exists():
        raise RuntimeError(f"Expected downloaded file missing: {path}")
    return path


def read_locations(path: Path) -> tuple[list[int], list[dict[str, str]]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda row: int(row["timestamp"]))
    timestamps = [int(row["timestamp"]) for row in rows]
    return timestamps, rows


def nearest_location(
    image_timestamp: int, location_timestamps: list[int], locations: list[dict[str, str]]
) -> dict[str, str]:
    idx = bisect_left(location_timestamps, image_timestamp)
    candidates = []
    if idx < len(location_timestamps):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    best_idx = min(candidates, key=lambda i: abs(location_timestamps[i] - image_timestamp))
    return locations[best_idx]


def timestamp_to_utc(timestamp_us: int) -> str:
    dt = datetime.fromtimestamp(timestamp_us / 1_000_000, tz=timezone.utc)
    return dt.isoformat()


def write_manifest(
    manifest_path: Path,
    run: str,
    camera: str,
    image_paths: list[Path],
    location_path: Path,
) -> None:
    location_timestamps, locations = read_locations(location_path)
    rows: list[dict[str, str | int | float]] = []

    for image_path in sorted(image_paths):
        image_ts = int(image_path.stem)
        loc = nearest_location(image_ts, location_timestamps, locations)
        loc_ts = int(loc["timestamp"])
        rows.append(
            {
                "run": run,
                "camera": camera,
                "image_path": str(image_path),
                "timestamp_us": image_ts,
                "timestamp_utc": timestamp_to_utc(image_ts),
                "location_timestamp_us": loc_ts,
                "location_delta_us": abs(loc_ts - image_ts),
                "northing": loc["northing"],
                "easting": loc["easting"],
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="2014-05-19-13-20-57")
    parser.add_argument("--camera", default="stereo_centre")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    run_dir = output_dir / args.run
    camera_dir = run_dir / "images_small" / args.camera
    image_files, location_file = list_dataset_files(args.run, args.camera, args.count)

    location_path = kaggle_download(location_file, run_dir)
    image_paths = [kaggle_download(name, camera_dir) for name in image_files]

    manifest_path = run_dir / f"{args.camera}_manifest.csv"
    write_manifest(manifest_path, args.run, args.camera, image_paths, location_path)

    print(f"downloaded_images={len(image_paths)}")
    print(f"location_csv={location_path}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
