#!/usr/bin/env python3
"""Train RobotCar image/location -> sun-direction predictor."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from scripts.trainer_base import BaseTrainer
from shadow_prediction.dataset_robotcar_sun import (
    DEFAULT_ARCHIVE,
    DEFAULT_MANIFEST,
    count_robotcar_archive_samples,
    get_robotcar_archive_sun_dataloaders,
    get_robotcar_sun_dataloaders,
)
from shadow_prediction.model_robotcar_sun import create_robotcar_sun_model
from shadow_prediction.model_robotcar_sun import BACKBONE_OPTIONS


class DirectionCosineLoss(nn.Module):
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cosine = (output * target).sum(dim=1).clamp(-1.0, 1.0)
        return (1.0 - cosine).mean()


class RobotCarSunTrainer(BaseTrainer):
    def __init__(
        self,
        manifest_path: str | Path = DEFAULT_MANIFEST,
        archive_path: str | Path | None = None,
        image_size: int = 224,
        val_fraction: float = 0.2,
        train_split: str = "train",
        val_split: str = "val",
        camera: str = "stereo_centre",
        target_frame: str = "car",
        include_heading: bool = True,
        include_camera_ohe: bool | None = None,
        include_segmentation: bool = False,
        max_train: int | None = None,
        max_val: int | None = None,
        sun_runs_only: bool = False,
        num_workers: int = 0,
        pretrained: bool = False,
        backbone: str = "resnet18",
        **kwargs,
    ):
        self.manifest_path = Path(manifest_path)
        self.archive_path = Path(archive_path) if archive_path is not None else None
        self.image_size = image_size
        self.val_fraction = val_fraction
        self.train_split = train_split
        self.val_split = val_split
        self.camera = camera
        self.target_frame = target_frame
        self.include_heading = include_heading
        self.include_camera_ohe = (
            False
            if self.archive_path is None
            else camera == "all"
            if include_camera_ohe is None
            else include_camera_ohe
        )
        self.include_segmentation = include_segmentation
        self.max_train = max_train
        self.max_val = max_val
        self.sun_runs_only = sun_runs_only
        self.num_workers = num_workers
        self.pretrained = pretrained
        self.backbone = backbone
        self.pose_dim = 4 if include_heading else 2
        if self.include_camera_ohe:
            self.pose_dim += 4
        self.latest_batch_metrics = {}

        if self.archive_path is not None:
            train_size = count_robotcar_archive_samples(
                archive_path=self.archive_path,
                split=train_split,
                camera=camera,
                max_samples=max_train,
                sun_runs_only=sun_runs_only,
            )
            val_size = count_robotcar_archive_samples(
                archive_path=self.archive_path,
                split=val_split,
                camera=camera,
                max_samples=max_val,
                sun_runs_only=sun_runs_only,
            )
        else:
            with self.manifest_path.open() as f:
                sample_count = max(sum(1 for _ in f) - 1, 0)
            val_size = max(1, int(round(sample_count * val_fraction)))
            train_size = max(sample_count - val_size, 0)

        kwargs.setdefault("train_size", train_size)
        kwargs.setdefault("val_size", val_size)
        kwargs.setdefault("task_name", "RobotCar Sun Direction Training")
        kwargs.setdefault("loss_name", "1 - cosine")
        kwargs.setdefault("save_path", "robotcar_sun_predictor.pth")
        super().__init__(**kwargs)

    def create_model(self):
        return create_robotcar_sun_model(
            pose_dim=self.pose_dim,
            pretrained=self.pretrained,
            backbone=self.backbone,
            image_size=self.image_size,
            device=self.device,
        )

    def create_dataloaders(self):
        if self.archive_path is not None:
            train_loader, val_loader, pose_dim = get_robotcar_archive_sun_dataloaders(
                archive_path=self.archive_path,
                train_split=self.train_split,
                val_split=self.val_split,
                camera=self.camera,
                batch_size=self.batch_size,
                image_size=self.image_size,
                target_frame=self.target_frame,
                include_heading=self.include_heading,
                include_camera_ohe=self.include_camera_ohe,
                include_segmentation=self.include_segmentation,
                max_train=self.max_train,
                max_val=self.max_val,
                sun_runs_only=self.sun_runs_only,
                num_workers=self.num_workers,
            )
        else:
            train_loader, val_loader, pose_dim = get_robotcar_sun_dataloaders(
                manifest_path=self.manifest_path,
                batch_size=self.batch_size,
                image_size=self.image_size,
                val_fraction=self.val_fraction,
                target_frame=self.target_frame,
                include_heading=self.include_heading,
                num_workers=self.num_workers,
            )
        self.pose_dim = pose_dim
        return train_loader, val_loader

    def create_criterion(self):
        return DirectionCosineLoss()

    def forward_pass(self, batch):
        image = batch["image"].to(self.device)
        pose = batch["pose"].to(self.device)
        target = batch["target"].to(self.device)

        output = self.model(image, pose)
        loss = self.criterion(output, target)

        with torch.no_grad():
            cosine = (output * target).sum(dim=1).clamp(-1.0, 1.0)
            angle_deg = torch.acos(cosine).mean() * (180.0 / math.pi)
            self.latest_batch_metrics = {"angle_deg": float(angle_deg.detach().cpu())}

        return output, target, loss

    def get_progress_metrics(self):
        return self.latest_batch_metrics

    def get_extra_info_lines(self):
        source_lines = []
        if self.archive_path is not None:
            source_lines = [
                f"Archive: {self.archive_path}",
                f"Splits: train={self.train_split}, val={self.val_split}",
                f"Camera: {self.camera}",
                f"Include camera OHE: {self.include_camera_ohe}",
                f"Max train/val: {self.max_train}/{self.max_val}",
                f"Sun-tagged runs only: {self.sun_runs_only}",
                f"Include segmentation masks: {self.include_segmentation}",
            ]
        else:
            source_lines = [f"Manifest: {self.manifest_path}"]

        return source_lines + [
            f"Target frame: {self.target_frame}",
            f"Include heading: {self.include_heading}",
            f"Image size: {self.image_size}",
            f"Backbone: {self.backbone}",
            f"Pretrained backbone: {self.pretrained}",
        ]


def train(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    archive_path: str | Path | None = None,
    num_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    image_size: int = 224,
    val_fraction: float = 0.2,
    train_split: str = "train",
    val_split: str = "val",
    camera: str = "stereo_centre",
    target_frame: str = "car",
    include_heading: bool = True,
    include_camera_ohe: bool | None = None,
    include_segmentation: bool = False,
    max_train: int | None = None,
    max_val: int | None = None,
    sun_runs_only: bool = False,
    num_workers: int = 0,
    pretrained: bool = False,
    backbone: str = "resnet18",
    device=None,
    plot_curves: bool = False,
    resume_from_checkpoint: str | Path | None = None,
):
    trainer = RobotCarSunTrainer(
        manifest_path=manifest_path,
        archive_path=archive_path,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        image_size=image_size,
        val_fraction=val_fraction,
        train_split=train_split,
        val_split=val_split,
        camera=camera,
        target_frame=target_frame,
        include_heading=include_heading,
        include_camera_ohe=include_camera_ohe,
        include_segmentation=include_segmentation,
        max_train=max_train,
        max_val=max_val,
        sun_runs_only=sun_runs_only,
        num_workers=num_workers,
        pretrained=pretrained,
        backbone=backbone,
        device=device,
        plot_curves=plot_curves,
        resume_from_checkpoint=(
            str(resume_from_checkpoint) if resume_from_checkpoint is not None else None
        ),
    )
    return trainer.train()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--archive", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--camera", default="stereo_centre")
    camera_ohe_group = parser.add_mutually_exclusive_group()
    camera_ohe_group.add_argument("--camera-ohe", dest="camera_ohe", action="store_true")
    camera_ohe_group.add_argument("--no-camera-ohe", dest="camera_ohe", action="store_false")
    parser.set_defaults(camera_ohe=None)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--sun-runs-only", action="store_true")
    parser.add_argument("--include-segmentation", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--target-frame", choices=["car", "global"], default="car")
    parser.add_argument("--no-heading", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--backbone", choices=BACKBONE_OPTIONS, default="resnet18")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--plot-curves", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None
    train(
        manifest_path=args.manifest,
        archive_path=args.archive,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.image_size,
        val_fraction=args.val_fraction,
        train_split=args.train_split,
        val_split=args.val_split,
        camera=args.camera,
        target_frame=args.target_frame,
        include_heading=not args.no_heading,
        include_camera_ohe=args.camera_ohe,
        include_segmentation=args.include_segmentation,
        max_train=args.max_train,
        max_val=args.max_val,
        sun_runs_only=args.sun_runs_only,
        num_workers=args.num_workers,
        pretrained=args.pretrained,
        backbone=args.backbone,
        device=device,
        plot_curves=args.plot_curves,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )


if __name__ == "__main__":
    main()
