"""Backbone creation and utilities for timm models."""

import logging

import timm
import torch.nn as nn

logger = logging.getLogger(__name__)


def create_backbone(name: str, pretrained: bool, drop_path_rate: float) -> tuple[nn.Module, int]:
    """Create timm backbone with num_classes=0 and return feature dimension.

    Args:
        name: Model name (e.g., 'convnext_base.fb_in22k_ft_in1k_384').
        pretrained: Whether to load pretrained weights.
        drop_path_rate: Stochastic depth rate.

    Returns:
        Tuple of (backbone module, feature dimension).
    """
    backbone = timm.create_model(
        name, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path_rate
    )
    num_features = backbone.num_features
    return backbone, num_features


def normalization_stats(backbone: nn.Module) -> tuple[list[float], list[float]]:
    """Get mean and std from backbone.pretrained_cfg or ImageNet defaults.

    Args:
        backbone: Backbone module.

    Returns:
        Tuple of (mean, std) as lists of 3 floats.
    """
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
    """Get target layer for Grad-CAM based on architecture.

    Args:
        backbone: Backbone module.
        name: Model name (used for architecture detection).

    Returns:
        Target layer module.

    Raises:
        ValueError: If architecture is not supported.
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
