"""Image backbone model for RobotCar sun-direction prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

try:
    import timm
except ImportError:  # pragma: no cover - exercised only when optional dependency is absent
    timm = None


TIMM_BACKBONES = {
    "vit_s_8_timm": "vit_small_patch8_224",
    "vit_b_8_timm": "vit_base_patch8_224",
}
BACKBONE_OPTIONS = ("resnet18", "vit_b_16", *TIMM_BACKBONES)


def _resnet18(pretrained: bool):
    if pretrained:
        try:
            return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except AttributeError:
            return models.resnet18(pretrained=True)
    try:
        return models.resnet18(weights=None)
    except TypeError:
        return models.resnet18(pretrained=False)


def _vit_b_16(pretrained: bool, image_size: int):
    if pretrained:
        try:
            return models.vit_b_16(
                weights=models.ViT_B_16_Weights.DEFAULT,
                image_size=image_size,
            )
        except AttributeError:
            return models.vit_b_16(pretrained=True, image_size=image_size)
    return models.vit_b_16(weights=None, image_size=image_size)


def _timm_backbone(backbone: str, pretrained: bool, image_size: int):
    if timm is None:
        raise ImportError(
            "timm is required for patch-8 ViT backbones. Install it with "
            "`python -m pip install timm`."
        )

    model_name = TIMM_BACKBONES[backbone]
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,
        img_size=image_size,
    )
    return model, int(model.num_features)


class RobotCarSunPredictor(nn.Module):
    """Predict a unit sun-direction vector from RGB image and pose features."""

    def __init__(
        self,
        pose_dim: int = 4,
        pretrained: bool = False,
        backbone: str = "resnet18",
        image_size: int = 224,
    ):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "resnet18":
            self.image_encoder = _resnet18(pretrained=pretrained)
            image_dim = self.image_encoder.fc.in_features
            self.image_encoder.fc = nn.Identity()
        elif backbone == "vit_b_16":
            self.image_encoder = _vit_b_16(pretrained=pretrained, image_size=image_size)
            image_dim = self.image_encoder.heads.head.in_features
            self.image_encoder.heads = nn.Identity()
        elif backbone in TIMM_BACKBONES:
            self.image_encoder, image_dim = _timm_backbone(
                backbone=backbone,
                pretrained=pretrained,
                image_size=image_size,
            )
        else:
            options = "', '".join(BACKBONE_OPTIONS)
            raise ValueError(f"backbone must be one of '{options}'")

        self.pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(image_dim + 64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )

    def forward(self, image: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(image)
        pose_features = self.pose_mlp(pose)
        output = self.head(torch.cat([image_features, pose_features], dim=1))
        return F.normalize(output, dim=1)


def create_robotcar_sun_model(
    pose_dim: int = 4,
    pretrained: bool = False,
    backbone: str = "resnet18",
    image_size: int = 224,
    device=None,
) -> RobotCarSunPredictor:
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model = RobotCarSunPredictor(
        pose_dim=pose_dim,
        pretrained=pretrained,
        backbone=backbone,
        image_size=image_size,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Created RobotCarSunPredictor with backbone={backbone}, "
        f"pose_dim={pose_dim}, image_size={image_size}"
    )
    print(f"Total parameters: {total_params:,}")
    return model
