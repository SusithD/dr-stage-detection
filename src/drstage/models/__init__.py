"""Models module: backbone, heads, losses, and factory."""

from drstage.models.backbone import cam_target_layer, create_backbone, normalization_stats
from drstage.models.factory import DRModel, build_model, param_groups, set_backbone_trainable
from drstage.models.heads import CeHead, CornHead, Head, build_head
from drstage.models.losses import (
    CornLoss,
    WeightedCrossEntropy,
    build_criterion,
    class_weights_from_counts,
    corn_loss,
)

__all__ = [
    "create_backbone",
    "normalization_stats",
    "cam_target_layer",
    "Head",
    "CornHead",
    "CeHead",
    "build_head",
    "class_weights_from_counts",
    "corn_loss",
    "CornLoss",
    "WeightedCrossEntropy",
    "build_criterion",
    "DRModel",
    "build_model",
    "set_backbone_trainable",
    "param_groups",
]
