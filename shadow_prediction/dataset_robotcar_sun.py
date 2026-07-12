"""Dataset for RobotCar image/location to sun-direction prediction."""

from __future__ import annotations

import csv
import io
from pathlib import Path
import zipfile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from shadow_prediction.robotcar_sun import (
    enu_to_car_frame,
    estimate_heading_enu,
    load_location_rows,
    sun_vector_enu,
    utm_to_latlon,
)


DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "robotcar"
    / "kaggle_subset"
    / "2014-05-19-13-20-57"
    / "stereo_centre_manifest.csv"
)
DEFAULT_ARCHIVE = Path("/Users/donguille/Downloads/archive.zip")
ARCHIVE_ROOT = "pnvlad_oxford_robotcar"
ROBOTCAR_CAMERAS = ("stereo_centre", "mono_left", "mono_right", "mono_rear")
ROBOTCAR_CAMERA_TO_INDEX = {camera: idx for idx, camera in enumerate(ROBOTCAR_CAMERAS)}


def _read_csv_rows_from_zip(
    archive: zipfile.ZipFile,
    member: str,
) -> list[dict[str, str]]:
    with archive.open(member) as f:
        wrapper = io.TextIOWrapper(f, encoding="utf-8", newline="")
        return list(csv.DictReader(wrapper))


def _read_location_rows_from_zip(
    archive: zipfile.ZipFile,
    track: str,
) -> list[dict[str, float]]:
    candidates = [
        f"{ARCHIVE_ROOT}/{track}/pointcloud_locations_20m_10overlap.csv",
        f"{ARCHIVE_ROOT}/{track}/pointcloud_locations_20m.csv",
    ]
    for member in candidates:
        try:
            rows = _read_csv_rows_from_zip(archive, member)
            break
        except KeyError:
            continue
    else:
        raise FileNotFoundError(f"No location CSV found in archive for track {track}")

    location_rows = [
        {
            "timestamp": int(row["timestamp"]),
            "northing": float(row["northing"]),
            "easting": float(row["easting"]),
        }
        for row in rows
    ]
    location_rows.sort(key=lambda row: row["timestamp"])
    return location_rows


def _load_robotcar_tags() -> dict[str, set[str]]:
    tag_dir = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "robotcar"
        / "robotcar-dataset-sdk"
        / "tags"
    )
    if not tag_dir.exists():
        return {}

    tag_map: dict[str, set[str]] = {}
    for path in tag_dir.glob("*.yaml"):
        tags: set[str] = set()
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith("-"):
                tag = line[1:].strip()
                if tag:
                    tags.add(tag)
        tag_map[path.stem] = tags
    return tag_map


def _valid_archive_timestamp(value: str | None) -> bool:
    if value is None:
        return False
    value = value.strip()
    return bool(value) and value.lower() not in {"nan", "none"}


def _load_archive_samples(
    archive_path: str | Path,
    split: str,
    camera: str,
    max_samples: int | None = None,
    seed: int = 7,
    sun_runs_only: bool = False,
) -> list[dict[str, object]]:
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    tag_map = _load_robotcar_tags() if sun_runs_only else {}
    with zipfile.ZipFile(archive_path) as archive:
        rows = _read_csv_rows_from_zip(archive, f"{ARCHIVE_ROOT}/{split}.csv")

    if camera == "all":
        cameras = ROBOTCAR_CAMERAS
    else:
        cameras = (camera,)

    samples: list[dict[str, object]] = []
    for row in rows:
        track = row["track"]
        if sun_runs_only and "sun" not in tag_map.get(track, set()):
            continue
        pointcloud_timestamp_us = int(row["pointcloud"])
        for camera_name in cameras:
            if camera_name not in row:
                raise ValueError(f"Camera column {camera_name!r} not found in {split}.csv")
            if not _valid_archive_timestamp(row.get(camera_name)):
                continue

            image_timestamp_us = int(row[camera_name])
            samples.append(
                {
                    "track": track,
                    "camera": camera_name,
                    "camera_index": ROBOTCAR_CAMERA_TO_INDEX[camera_name],
                    "image_path": (
                        f"{ARCHIVE_ROOT}/{track}/images_small/{camera_name}/"
                        f"{image_timestamp_us}.png"
                    ),
                    "segmentation_path": (
                        f"{ARCHIVE_ROOT}/{track}/segmentation_masks_small/{camera_name}/"
                        f"{image_timestamp_us}.png"
                    ),
                    "timestamp_us": image_timestamp_us,
                    "pointcloud_timestamp_us": pointcloud_timestamp_us,
                    "northing": float(row["northing"]),
                    "easting": float(row["easting"]),
                }
            )

    if max_samples is not None and max_samples > 0 and len(samples) > max_samples:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(samples), generator=generator)[:max_samples].tolist()
        samples = [samples[idx] for idx in indices]

    if not samples:
        filter_text = " after sun-run filtering" if sun_runs_only else ""
        raise ValueError(f"No archive samples found for split {split!r}{filter_text}")
    return samples


def count_robotcar_archive_samples(
    archive_path: str | Path = DEFAULT_ARCHIVE,
    split: str = "train",
    camera: str = "stereo_centre",
    max_samples: int | None = None,
    sun_runs_only: bool = False,
) -> int:
    samples = _load_archive_samples(
        archive_path=archive_path,
        split=split,
        camera=camera,
        max_samples=max_samples,
        sun_runs_only=sun_runs_only,
    )
    return len(samples)


class RobotCarSunDataset(Dataset):
    """Returns image + pose features with global or car-relative sun direction."""

    def __init__(
        self,
        manifest_path: str | Path = DEFAULT_MANIFEST,
        image_size: int = 224,
        target_frame: str = "car",
        include_heading: bool = True,
    ):
        self.manifest_path = Path(manifest_path)
        self.image_size = image_size
        self.target_frame = target_frame
        self.include_heading = include_heading

        if target_frame not in {"car", "global"}:
            raise ValueError("target_frame must be 'car' or 'global'")
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with self.manifest_path.open(newline="") as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")

        location_path = self.manifest_path.parent / "pointcloud_locations_20m_10overlap.csv"
        if location_path.exists():
            self.location_rows = load_location_rows(location_path)
        else:
            self.location_rows = [
                {
                    "timestamp": int(row["location_timestamp_us"]),
                    "northing": float(row["northing"]),
                    "easting": float(row["easting"]),
                }
                for row in self.rows
            ]

        northings = np.array([float(row["northing"]) for row in self.rows], dtype=np.float32)
        eastings = np.array([float(row["easting"]) for row in self.rows], dtype=np.float32)
        self.location_mean = np.array([northings.mean(), eastings.mean()], dtype=np.float32)
        self.location_std = np.array([northings.std(), eastings.std()], dtype=np.float32)
        self.location_std = np.maximum(self.location_std, 1.0)

    @property
    def pose_dim(self) -> int:
        return 4 if self.include_heading else 2

    def __len__(self) -> int:
        return len(self.rows)

    def _load_image(self, image_path: Path) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        return torch.from_numpy(array)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image_path = Path(row["image_path"])
        timestamp_us = int(row["timestamp_us"])
        northing = float(row["northing"])
        easting = float(row["easting"])

        image = self._load_image(image_path)
        location = np.array([northing, easting], dtype=np.float32)
        location_norm = (location - self.location_mean) / self.location_std
        heading = estimate_heading_enu(timestamp_us, self.location_rows)

        lat, lon = utm_to_latlon(easting=easting, northing=northing)
        sun_global = sun_vector_enu(timestamp_us, lat, lon)
        if self.target_frame == "car":
            target = enu_to_car_frame(sun_global, heading)
        else:
            target = sun_global

        pose_parts = [location_norm]
        if self.include_heading:
            pose_parts.append(heading)
        pose = np.concatenate(pose_parts).astype(np.float32)

        sample = {
            "image": image,
            "pose": torch.from_numpy(pose),
            "target": torch.from_numpy(target.astype(np.float32)),
            "sun_global": torch.from_numpy(sun_global.astype(np.float32)),
            "heading": torch.from_numpy(heading.astype(np.float32)),
            "timestamp_us": torch.tensor(timestamp_us, dtype=torch.long),
            "location": torch.from_numpy(location),
        }
        return sample


class RobotCarArchiveSunDataset(Dataset):
    """RobotCar archive dataset for image/location to sun-direction prediction."""

    def __init__(
        self,
        archive_path: str | Path = DEFAULT_ARCHIVE,
        split: str = "train",
        camera: str = "stereo_centre",
        image_size: int = 224,
        target_frame: str = "car",
        include_heading: bool = True,
        include_camera_ohe: bool = False,
        include_segmentation: bool = False,
        max_samples: int | None = None,
        seed: int = 7,
        sun_runs_only: bool = False,
        location_mean: np.ndarray | None = None,
        location_std: np.ndarray | None = None,
    ):
        self.archive_path = Path(archive_path)
        self.split = split
        self.camera = camera
        self.image_size = image_size
        self.target_frame = target_frame
        self.include_heading = include_heading
        self.include_camera_ohe = include_camera_ohe
        self.include_segmentation = include_segmentation
        self.sun_runs_only = sun_runs_only
        self._zip: zipfile.ZipFile | None = None
        self._location_cache: dict[str, list[dict[str, float]]] = {}

        if target_frame not in {"car", "global"}:
            raise ValueError("target_frame must be 'car' or 'global'")

        self.samples = _load_archive_samples(
            archive_path=self.archive_path,
            split=split,
            camera=camera,
            max_samples=max_samples,
            seed=seed,
            sun_runs_only=sun_runs_only,
        )

        northings = np.array([float(row["northing"]) for row in self.samples], dtype=np.float32)
        eastings = np.array([float(row["easting"]) for row in self.samples], dtype=np.float32)
        if location_mean is None:
            location_mean = np.array([northings.mean(), eastings.mean()], dtype=np.float32)
        if location_std is None:
            location_std = np.array([northings.std(), eastings.std()], dtype=np.float32)
        self.location_mean = np.asarray(location_mean, dtype=np.float32)
        self.location_std = np.maximum(np.asarray(location_std, dtype=np.float32), 1.0)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_zip"] = None
        state["_location_cache"] = {}
        return state

    @property
    def pose_dim(self) -> int:
        pose_dim = 4 if self.include_heading else 2
        if self.include_camera_ohe:
            pose_dim += len(ROBOTCAR_CAMERAS)
        return pose_dim

    def __len__(self) -> int:
        return len(self.samples)

    def _get_zip(self) -> zipfile.ZipFile:
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.archive_path)
        return self._zip

    def _load_image_member(self, member: str) -> Image.Image:
        with self._get_zip().open(member) as f:
            data = f.read()
        return Image.open(io.BytesIO(data))

    def _load_image(self, member: str) -> torch.Tensor:
        image = self._load_image_member(member).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        return torch.from_numpy(array)

    def _load_segmentation(self, member: str) -> torch.Tensor:
        mask = self._load_image_member(member).convert("L")
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        return torch.from_numpy(np.asarray(mask, dtype=np.int64))

    def _location_rows(self, track: str) -> list[dict[str, float]]:
        if track not in self._location_cache:
            self._location_cache[track] = _read_location_rows_from_zip(self._get_zip(), track)
        return self._location_cache[track]

    def __getitem__(self, idx: int):
        row = self.samples[idx]
        timestamp_us = int(row["timestamp_us"])
        pointcloud_timestamp_us = int(row["pointcloud_timestamp_us"])
        northing = float(row["northing"])
        easting = float(row["easting"])
        track = str(row["track"])

        image = self._load_image(str(row["image_path"]))
        location = np.array([northing, easting], dtype=np.float32)
        location_norm = (location - self.location_mean) / self.location_std
        heading = estimate_heading_enu(pointcloud_timestamp_us, self._location_rows(track))

        lat, lon = utm_to_latlon(easting=easting, northing=northing)
        sun_global = sun_vector_enu(timestamp_us, lat, lon)
        if self.target_frame == "car":
            target = enu_to_car_frame(sun_global, heading)
        else:
            target = sun_global

        pose_parts = [location_norm]
        if self.include_heading:
            pose_parts.append(heading)
        if self.include_camera_ohe:
            camera_ohe = np.zeros(len(ROBOTCAR_CAMERAS), dtype=np.float32)
            camera_ohe[int(row["camera_index"])] = 1.0
            pose_parts.append(camera_ohe)
        pose = np.concatenate(pose_parts).astype(np.float32)

        sample = {
            "image": image,
            "pose": torch.from_numpy(pose),
            "target": torch.from_numpy(target.astype(np.float32)),
            "sun_global": torch.from_numpy(sun_global.astype(np.float32)),
            "heading": torch.from_numpy(heading.astype(np.float32)),
            "timestamp_us": torch.tensor(timestamp_us, dtype=torch.long),
            "pointcloud_timestamp_us": torch.tensor(pointcloud_timestamp_us, dtype=torch.long),
            "location": torch.from_numpy(location),
            "track": track,
            "camera_index": torch.tensor(int(row["camera_index"]), dtype=torch.long),
            "camera_ohe": torch.from_numpy(
                np.eye(len(ROBOTCAR_CAMERAS), dtype=np.float32)[int(row["camera_index"])]
            ),
        }
        sample["camera"] = str(row["camera"])
        if self.include_segmentation:
            sample["segmentation"] = self._load_segmentation(str(row["segmentation_path"]))
        return sample


def get_robotcar_sun_dataloaders(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    batch_size: int = 8,
    image_size: int = 224,
    val_fraction: float = 0.2,
    target_frame: str = "car",
    include_heading: bool = True,
    num_workers: int = 0,
    seed: int = 7,
):
    dataset = RobotCarSunDataset(
        manifest_path=manifest_path,
        image_size=image_size,
        target_frame=target_frame,
        include_heading=include_heading,
    )
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed)).tolist()
    val_size = max(1, int(round(len(indices) * val_fraction)))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    if not train_indices:
        raise ValueError("Need at least two samples to create train/validation splits")

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, dataset.pose_dim


def get_robotcar_archive_sun_dataloaders(
    archive_path: str | Path = DEFAULT_ARCHIVE,
    train_split: str = "train",
    val_split: str = "val",
    camera: str = "stereo_centre",
    batch_size: int = 8,
    image_size: int = 224,
    target_frame: str = "car",
    include_heading: bool = True,
    include_camera_ohe: bool = False,
    include_segmentation: bool = False,
    max_train: int | None = None,
    max_val: int | None = None,
    sun_runs_only: bool = False,
    num_workers: int = 0,
    seed: int = 7,
):
    train_dataset = RobotCarArchiveSunDataset(
        archive_path=archive_path,
        split=train_split,
        camera=camera,
        image_size=image_size,
        target_frame=target_frame,
        include_heading=include_heading,
        include_camera_ohe=include_camera_ohe,
        include_segmentation=include_segmentation,
        max_samples=max_train,
        seed=seed,
        sun_runs_only=sun_runs_only,
    )
    val_dataset = RobotCarArchiveSunDataset(
        archive_path=archive_path,
        split=val_split,
        camera=camera,
        image_size=image_size,
        target_frame=target_frame,
        include_heading=include_heading,
        include_camera_ohe=include_camera_ohe,
        include_segmentation=include_segmentation,
        max_samples=max_val,
        seed=seed + 1,
        sun_runs_only=sun_runs_only,
        location_mean=train_dataset.location_mean,
        location_std=train_dataset.location_std,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, train_dataset.pose_dim
