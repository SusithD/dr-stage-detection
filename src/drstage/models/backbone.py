"""Create timm backbones and expose their input mean/std and the Grad-CAM target layer."""

import logging

import timm
import torch.nn as nn

logger = logging.getLogger(__name__)


def create_backbone(name: str, pretrained: bool, drop_path_rate: float) -> tuple[nn.Module, int]:
    """Build a headless timm model and return it with its feature width.

    num_classes=0 makes timm return pooled features, so the head is bolted on
    separately.
    """
    backbone = timm.create_model(
        name, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path_rate
    )
    num_features = backbone.num_features
    return backbone, num_features


def normalization_stats(backbone: nn.Module) -> tuple[list[float], list[float]]:
    """Read the input mean/std from the timm pretrained config, falling back to ImageNet's."""
    if hasattr(backbone, "pretrained_cfg") and backbone.pretrained_cfg:
        cfg = backbone.pretrained_cfg
        if "mean" in cfg and "std" in cfg:
            mean = list(cfg["mean"])
            std = list(cfg["std"])
            if len(mean) == 3 and len(std) == 3:
                return mean, std

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    return imagenet_mean, imagenet_std


def cam_target_layer(backbone: nn.Module, name: str) -> nn.Module:
    """Pick the layer Grad-CAM should hook for each supported family. Unknown
    families raise so a new backbone gets an explicit mapping instead of a
    guessed layer.
    """
    if "convnext" in name.lower():
        return backbone.stages[-1]
    elif "efficientnet" in name.lower():
        return backbone.conv_head
    elif "resnet" in name.lower():
        return backbone.layer4[-1]
    else:
        raise ValueError(
            f"Unsupported architecture for CAM: {name}. "
            f"Supported: convnext*, efficientnet*, resnet*"
        )
