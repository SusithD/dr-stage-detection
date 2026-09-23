"""Model factory and utilities."""

import logging

import torch.nn as nn
from torch import Tensor

from drstage.models.backbone import create_backbone
from drstage.models.heads import Head, build_head

logger = logging.getLogger(__name__)


class DRModel(nn.Module):
    """DR stage detection model combining backbone and head."""

    def __init__(self, backbone: nn.Module, head: Head, num_features: int):
        """Initialize DRModel.

        Args:
            backbone: Feature extraction backbone.
            head: Classification head.
            num_features: Number of features from backbone.
        """
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.num_features = num_features

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: [N, 3, H, W] image tensor.

        Returns:
            [N, out_dim] logits.
        """
        features = self.backbone(x)
        logits = self.head(features)
        return logits

    def predict(self, logits: Tensor) -> tuple:
        """Predict grades and probabilities.

        Args:
            logits: [N, out_dim] logits.

        Returns:
            Tuple of (grade, prob_dr, class_probs) from head.predict().
        """
        return self.head.predict(logits)


def build_model(model_cfg: dict, pretrained: bool = True) -> DRModel:
    """Build a complete DR stage detection model.

    Args:
        model_cfg: Model config dict with keys:
            - backbone: Model name
            - head: 'corn' or 'ce'
            - num_classes: Number of classes
            - dropout: Dropout rate
            - drop_path_rate: Stochastic depth rate
        pretrained: Whether to use pretrained backbone weights.

    Returns:
        DRModel instance.
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
    """Set backbone trainability and optionally freeze BN stats.

    Args:
        model: DRModel instance.
        trainable: Whether to enable gradient computation.
        freeze_bn_stats: If True and trainable=False, keep BN in eval() mode.
    """
    for param in model.backbone.parameters():
        param.requires_grad = trainable

    if not trainable and freeze_bn_stats:
        model.backbone.eval()


def param_groups(
    model: DRModel, backbone_lr: float, head_lr: float, weight_decay: float
) -> list[dict]:
    """Create parameter groups for optimizer with differentiated learning rates.

    Creates 4 groups:
    - backbone decay (requires_grad=True, ndim>1)
    - backbone no-decay (requires_grad=True, ndim<=1 or bias)
    - head decay (ndim>1)
    - head no-decay (ndim<=1 or bias)

    Args:
        model: DRModel instance.
        backbone_lr: Learning rate for backbone.
        head_lr: Learning rate for head.
        weight_decay: Weight decay coefficient.

    Returns:
        List of 4 parameter group dicts with 'params', 'lr', 'weight_decay', 'name'.
    """

    def needs_decay(param_name: str, param: nn.Parameter) -> bool:
        """Check if parameter should have weight decay."""
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
