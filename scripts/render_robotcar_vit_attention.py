#!/usr/bin/env python3
"""Render ViT attention heatmaps for RobotCar sun-direction predictions."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
from types import MethodType

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from shadow_prediction.dataset_robotcar_sun import (  # noqa: E402
    DEFAULT_ARCHIVE,
    RobotCarArchiveSunDataset,
)
from shadow_prediction.model_robotcar_sun import BACKBONE_OPTIONS  # noqa: E402
from shadow_prediction.model_robotcar_sun import create_robotcar_sun_model  # noqa: E402


def angle_error_deg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cosine = (pred * target).sum(dim=1).clamp(-1.0, 1.0)
    return torch.acos(cosine) * (180.0 / math.pi)


def vector_to_az_alt(vector: np.ndarray) -> tuple[float, float]:
    vector = vector / max(float(np.linalg.norm(vector)), 1e-8)
    azimuth = math.degrees(math.atan2(float(vector[0]), float(vector[1])))
    altitude = math.degrees(math.asin(max(-1.0, min(1.0, float(vector[2])))))
    return azimuth, altitude


def patch_torchvision_encoder_blocks(model) -> None:
    """Patch torchvision ViT encoder blocks to retain per-head attention weights."""

    def forward_with_attention(self, input: torch.Tensor):
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        x = self.ln_1(input)
        x, attn = self.self_attention(
            x,
            x,
            x,
            need_weights=True,
            average_attn_weights=False,
        )
        self._last_attention = attn.detach()
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y

    for block in model.image_encoder.encoder.layers:
        block.forward = MethodType(forward_with_attention, block)


def patch_timm_encoder_blocks(model) -> None:
    """Patch timm ViT attention blocks to retain per-head attention weights."""

    def forward_with_attention(self, x: torch.Tensor, attn_mask=None, is_causal: bool = False):
        if attn_mask is not None or is_causal:
            raise NotImplementedError("Attention rendering expects plain self-attention.")

        batch_size, tokens, _ = x.shape
        gate = self.gate(x).sigmoid() if self.gate is not None else None
        qkv = self.qkv(x).reshape(
            batch_size,
            tokens,
            3,
            self.num_heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        self._last_attention = attn.detach()
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(batch_size, tokens, self.attn_dim)
        x = self.norm(x)
        if gate is not None:
            x = x * gate
        x = self.proj(x)
        return self.proj_drop(x)

    for block in model.image_encoder.blocks:
        block.attn.forward = MethodType(forward_with_attention, block.attn)


def patch_encoder_blocks(model) -> None:
    if hasattr(model.image_encoder, "encoder") and hasattr(model.image_encoder.encoder, "layers"):
        patch_torchvision_encoder_blocks(model)
    elif hasattr(model.image_encoder, "blocks"):
        patch_timm_encoder_blocks(model)
    else:
        raise ValueError("Attention rendering only supports ViT-style backbones.")


def get_attention_tensors(model) -> list[torch.Tensor]:
    if hasattr(model.image_encoder, "encoder") and hasattr(model.image_encoder.encoder, "layers"):
        blocks = model.image_encoder.encoder.layers
        tensors = [getattr(block, "_last_attention", None) for block in blocks]
    elif hasattr(model.image_encoder, "blocks"):
        blocks = model.image_encoder.blocks
        tensors = [getattr(block.attn, "_last_attention", None) for block in blocks]
    else:
        raise ValueError("Attention rendering only supports ViT-style backbones.")

    if any(attn is None for attn in tensors):
        raise RuntimeError("Attention tensors were not captured")
    return tensors


def attention_grid_size(attn: torch.Tensor) -> int:
    num_patches = attn.shape[-1] - 1
    grid_size = int(round(math.sqrt(num_patches)))
    if grid_size * grid_size != num_patches:
        raise ValueError(f"Cannot reshape {num_patches} patches into a square grid")
    return grid_size


def attention_maps(attentions: list[torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    grid_size = attention_grid_size(attentions[-1])

    last = attentions[-1][0].mean(dim=0)
    last_cls = last[0, 1:].reshape(grid_size, grid_size)
    last_cls = normalize_map(last_cls)

    rollout = torch.eye(attentions[0].shape[-1], device=attentions[0].device)
    for attn in attentions:
        avg = attn[0].mean(dim=0)
        avg = avg + torch.eye(avg.shape[0], device=avg.device)
        avg = avg / avg.sum(dim=-1, keepdim=True)
        rollout = avg @ rollout
    rollout_cls = rollout[0, 1:].reshape(grid_size, grid_size)
    rollout_cls = normalize_map(rollout_cls)

    return last_cls.cpu().numpy(), rollout_cls.cpu().numpy()


def normalize_map(attn_map: torch.Tensor) -> torch.Tensor:
    attn_map = attn_map - attn_map.min()
    return attn_map / attn_map.max().clamp_min(1e-8)


def upsample_map(attn_map: np.ndarray, image_size: int) -> np.ndarray:
    tensor = torch.from_numpy(attn_map).float().view(1, 1, *attn_map.shape)
    tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return tensor[0, 0].numpy()


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


def collect_best_indices(
    model,
    dataset: RobotCarArchiveSunDataset,
    device: torch.device,
    batch_size: int,
    top_k: int,
) -> list[dict[str, object]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    rows = []
    seen = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            pose = batch["pose"].to(device)
            target = batch["target"].to(device)
            pred = model(image, pose)
            error = angle_error_deg(pred, target).detach().cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            for offset, sample_error in enumerate(error):
                idx = seen + offset
                rows.append(
                    {
                        "idx": idx,
                        "error_deg": float(sample_error),
                        "pred": pred_np[offset],
                        "target": target_np[offset],
                    }
                )
            seen += len(error)

    rows.sort(key=lambda row: float(row["error_deg"]))
    return rows[:top_k]


def tensor_to_image(image: torch.Tensor) -> np.ndarray:
    return image.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def render_overlay(ax, image: np.ndarray, attn_map: np.ndarray, title: str) -> None:
    ax.imshow(image)
    ax.imshow(attn_map, cmap="magma", alpha=0.52, vmin=0.0, vmax=1.0)
    ax.axis("off")
    ax.set_title(title, fontsize=8)


def render_attention_for_sample(
    model,
    dataset: RobotCarArchiveSunDataset,
    row: dict[str, object],
    device: torch.device,
    image_size: int,
) -> dict[str, object]:
    sample = dataset[int(row["idx"])]
    image = sample["image"].unsqueeze(0).to(device)
    pose = sample["pose"].unsqueeze(0).to(device)
    target = sample["target"].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(image, pose)
        error = float(angle_error_deg(pred, target).detach().cpu()[0])
        attentions = get_attention_tensors(model)
        last_map, rollout_map = attention_maps(attentions)

    return {
        "image": tensor_to_image(sample["image"]),
        "last_patch_map": last_map,
        "rollout_patch_map": rollout_map,
        "last_map": upsample_map(last_map, image_size),
        "rollout_map": upsample_map(rollout_map, image_size),
        "camera": sample["camera"],
        "track": dataset.samples[int(row["idx"])]["track"],
        "error_deg": error,
        "pred": pred.detach().cpu().numpy()[0],
        "target": target.detach().cpu().numpy()[0],
    }


def render_sheet(items: list[dict[str, object]], output_path: Path, backbone: str) -> None:
    fig, axes = plt.subplots(len(items), 3, figsize=(10.5, max(3.2, len(items) * 3.0)))
    axes_array = np.asarray(axes).reshape(len(items), 3)
    for row_idx, item in enumerate(items):
        image = item["image"]
        pred_az, pred_alt = vector_to_az_alt(np.asarray(item["pred"], dtype=np.float32))
        target_az, target_alt = vector_to_az_alt(np.asarray(item["target"], dtype=np.float32))
        header = (
            f"{item['camera']} | err {float(item['error_deg']):.2f} deg\n"
            f"pred {pred_az:.0f}/{pred_alt:.0f} target {target_az:.0f}/{target_alt:.0f}"
        )

        axes_array[row_idx, 0].imshow(image)
        axes_array[row_idx, 0].axis("off")
        axes_array[row_idx, 0].set_title(header, fontsize=8)
        render_overlay(axes_array[row_idx, 1], image, item["last_map"], "last-layer CLS attention")
        render_overlay(axes_array[row_idx, 2], image, item["rollout_map"], "attention rollout")

    fig.suptitle(
        f"{backbone} attention maps for RobotCar sun-direction predictions",
        fontsize=14,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def render_patch_grid(ax, patch_map: np.ndarray, title: str) -> None:
    ax.imshow(patch_map, cmap="magma", interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(-0.5, patch_map.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, patch_map.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.25, alpha=0.45)
    ax.tick_params(which="both", left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_title(title, fontsize=8)


def render_patch_score_sheet(
    items: list[dict[str, object]],
    output_path: Path,
    backbone: str,
) -> None:
    fig, axes = plt.subplots(len(items), 3, figsize=(10.5, max(3.2, len(items) * 3.0)))
    axes_array = np.asarray(axes).reshape(len(items), 3)
    for row_idx, item in enumerate(items):
        image = item["image"]
        pred_az, pred_alt = vector_to_az_alt(np.asarray(item["pred"], dtype=np.float32))
        target_az, target_alt = vector_to_az_alt(np.asarray(item["target"], dtype=np.float32))
        header = (
            f"{item['camera']} | err {float(item['error_deg']):.2f} deg\n"
            f"pred {pred_az:.0f}/{pred_alt:.0f} target {target_az:.0f}/{target_alt:.0f}"
        )

        axes_array[row_idx, 0].imshow(image)
        axes_array[row_idx, 0].axis("off")
        axes_array[row_idx, 0].set_title(header, fontsize=8)
        render_patch_grid(
            axes_array[row_idx, 1],
            np.asarray(item["last_patch_map"], dtype=np.float32),
            "last-layer CLS patch scores",
        )
        render_patch_grid(
            axes_array[row_idx, 2],
            np.asarray(item["rollout_patch_map"], dtype=np.float32),
            "rollout patch scores",
        )

    fig.suptitle(
        f"{backbone} per-patch attention scores for RobotCar sun-direction predictions",
        fontsize=14,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def render_individuals(items: list[dict[str, object]], output_dir: Path) -> None:
    for idx, item in enumerate(items, start=1):
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5))
        image = item["image"]
        axes[0].imshow(image)
        axes[0].axis("off")
        axes[0].set_title(f"{item['camera']} | err {float(item['error_deg']):.2f} deg", fontsize=8)
        render_overlay(axes[1], image, item["last_map"], "last layer")
        render_overlay(axes[2], image, item["rollout_map"], "rollout")
        fig.tight_layout()
        fig.savefig(
            output_dir / f"vit_attention_{idx:02d}_{item['camera']}_{float(item['error_deg']):.2f}deg.png",
            dpi=160,
        )
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/robotcar_vit_attention"))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--max-train", type=int, default=8192)
    parser.add_argument("--max-val", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--backbone", choices=BACKBONE_OPTIONS, default="vit_b_16")
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

    dataset = make_dataset(
        archive_path=args.archive,
        sun_runs_only=not args.all_runs,
        image_size=args.image_size,
        max_train=args.max_train,
        max_val=args.max_val,
    )
    model = create_robotcar_sun_model(
        pose_dim=dataset.pose_dim,
        backbone=args.backbone,
        image_size=args.image_size,
        pretrained=False,
        device=device,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    best_rows = collect_best_indices(model, dataset, device, args.batch_size, args.top_k)
    patch_encoder_blocks(model)
    items = [
        render_attention_for_sample(model, dataset, row, device, args.image_size)
        for row in best_rows
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = args.output_dir / "vit_attention_sheet.png"
    patch_sheet_path = args.output_dir / "vit_attention_patch_scores_sheet.png"
    render_sheet(items, sheet_path, args.backbone)
    render_patch_score_sheet(items, patch_sheet_path, args.backbone)
    render_individuals(items, args.output_dir)
    print(f"Saved ViT attention sheet to {sheet_path}")
    print(f"Saved ViT patch-score sheet to {patch_sheet_path}")
    for idx, item in enumerate(items, start=1):
        print(
            f"{idx:02d} camera={item['camera']:13s} "
            f"error={float(item['error_deg']):.3f}deg track={item['track']}"
        )


if __name__ == "__main__":
    main()
