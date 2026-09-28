"""Assemble backbone and head into DRModel, plus backbone freezing and optimizer param groups."""

import logging

import torch.nn as nn
from torch import Tensor

from drstage.models.backbone import create_backbone
from drstage.models.heads import Head, build_head

logger = logging.getLogger(__name__)


class DRModel(nn.Module):
    """Backbone that produces pooled features, followed by a CORN or CE head."""

    def __init__(self, backbone: nn.Module, head: Head, num_features: int):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.num_features = num_features

    def forward(self, x: Tensor) -> Tensor:
        """[N, 3, H, W] images to [N, out_dim] logits."""
        features = self.backbone(x)
        logits = self.head(features)
        return logits

    def predict(self, logits: Tensor) -> tuple:
        """Decode logits with the head; see Head.predict for the tuple layout."""
        return self.head.predict(logits)


def build_model(model_cfg: dict, pretrained: bool = True) -> DRModel:
    """Build a DRModel from the model section of a training config.

    Reads backbone, head and num_classes, plus optional dropout and drop_path_rate.
    """
    backbone, num_features = create_backbone(
        model_cfg["backbone"],
        pretrained=pretrained,
        drop_path_rate=model_cfg.get("drop_path_rate", 0.0),
    )

    head = build_head(
        model_cfg["head"],
        in_features=num_features,
        num_classes=model_cfg["num_classes"],
        dropout=model_cfg.get("dropout", 0.0),
    )

    return DRModel(backbone, head, num_features)


def set_backbone_trainable(model: DRModel, trainable: bool, freeze_bn_stats: bool = False) -> None:
    """Toggle requires_grad on the backbone.

    With freeze_bn_stats the frozen backbone is also put in eval() so any BatchNorm
    layers keep their pretrained running stats during the head-only warm-up.
    """
    for param in model.backbone.parameters():
        param.requires_grad = trainable

    if not trainable and freeze_bn_stats:
        model.backbone.eval()


def param_groups(
    model: DRModel, backbone_lr: float, head_lr: float, weight_decay: float
) -> list[dict]:
    """Split trainable params into four AdamW groups: backbone/head x decay/no-decay.

    Only tensors with ndim > 1 that are not biases get weight decay, so norm scales
    and biases are left alone. Frozen params are skipped. Group order matters:
    Trainer._get_lrs reads 0-1 as backbone and 2-3 as head.
    """

    def needs_decay(param_name: str, param: nn.Parameter) -> bool:
        """Weight decay only for matrices and conv kernels, never biases or norm params."""
        return param.ndim > 1 and not param_name.endswith("bias")

    backbone_decay = []
    backbone_no_decay = []
    head_decay = []
    head_no_decay = []

    for name, param in model.backbone.named_parameters():
        if not param.requires_grad:
            continue
        if needs_decay(name, param):
            backbone_decay.append(param)
        else:
            backbone_no_decay.append(param)

    for name, param in model.head.named_parameters():
        if not param.requires_grad:
            continue
        if needs_decay(name, param):
            head_decay.append(param)
        else:
            head_no_decay.append(param)

    groups = [
        {
            "params": backbone_decay,
            "lr": backbone_lr,
            "weight_decay": weight_decay,
            "name": "backbone_decay",
        },
        {
            "params": backbone_no_decay,
            "lr": backbone_lr,
            "weight_decay": 0.0,
            "name": "backbone_no_decay",
        },
        {
            "params": head_decay,
            "lr": head_lr,
            "weight_decay": weight_decay,
            "name": "head_decay",
        },
        {
            "params": head_no_decay,
            "lr": head_lr,
            "weight_decay": 0.0,
            "name": "head_no_decay",
        },
    ]

    return groups
