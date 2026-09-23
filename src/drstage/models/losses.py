"""Loss functions and class weighting."""

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor, Tensor


def class_weights_from_counts(counts: Sequence[int], power: float) -> Tensor:
    """Compute class weights from label counts.

    w_c = (N / (K * n_c)) ** power, then rescaled to mean 1.
    power=0 gives all ones.

    Args:
        counts: Sequence of per-class sample counts.
        power: Power for weighting (0 disables).

    Returns:
        [K] weight tensor.
    """
    counts = torch.tensor(counts, dtype=torch.float32)
    num_classes = len(counts)
    total = counts.sum()

    if power == 0:
        return torch.ones(num_classes, dtype=torch.float32)

    weights = (total / (num_classes * counts)) ** power
    weights = weights / weights.mean()

    return weights


def corn_loss(
    logits: Tensor,
    targets: LongTensor,
    num_classes: int,
    sample_weights: Tensor | None = None,
) -> Tensor:
    """CORN loss with vectorised masking.

    For each binary task k in 0..K-2:
    - S_k = {i : y_i >= k} (conditional subset)
    - binary_label = 1[y_i > k] for i in S_k
    - loss_k = mean BCEWithLogits(logits[i,k], label) weighted by w_i

    Logits outside S_k do not influence the loss.

    Args:
        logits: [N, K-1] model outputs.
        targets: [N] class labels in 0..K-1.
        num_classes: Number of classes (K).
        sample_weights: [N] sample weights or None (treated as 1.0).

    Returns:
        Scalar loss.
    """
    logits = logits.float()
    targets = targets.long()
    device = logits.device
    batch_size = logits.size(0)

    if sample_weights is None:
        sample_weights = torch.ones(batch_size, device=device, dtype=torch.float32)
    else:
        sample_weights = sample_weights.float()

    k = torch.arange(num_classes - 1, device=device)
    in_subset = (targets[:, None] >= k).float() * sample_weights[:, None]
    labels = (targets[:, None] > k).float()
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (bce * in_subset).sum() / in_subset.sum().clamp_min(1e-12)


class CornLoss(nn.Module):
    """CORN loss module with class weighting."""

    def __init__(self, num_classes: int, class_weights: Tensor | None = None):
        """Initialize CORN loss.

        Args:
            num_classes: Number of classes (K).
            class_weights: [K] class weights or None.
        """
        super().__init__()
        self.num_classes = num_classes
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: Tensor, targets: LongTensor) -> Tensor:
        """Compute CORN loss.

        Args:
            logits: [N, K-1] logits.
            targets: [N] labels.

        Returns:
            Scalar loss.
        """
        sample_weights = None
        if self.class_weights is not None:
            sample_weights = self.class_weights[targets]

        return corn_loss(logits, targets, self.num_classes, sample_weights)


class WeightedCrossEntropy(nn.Module):
    """Weighted cross-entropy loss module."""

    def __init__(self, class_weights: Tensor | None = None, label_smoothing: float = 0.0):
        """Initialize weighted CE loss.

        Args:
            class_weights: [K] class weights or None.
            label_smoothing: Label smoothing coefficient.
        """
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    def forward(self, logits: Tensor, targets: LongTensor) -> Tensor:
        """Compute weighted CE loss.

        Args:
            logits: [N, K] logits.
            targets: [N] labels.

        Returns:
            Scalar loss.
        """
        return self.ce_loss(logits, targets)


def build_criterion(
    head: str,
    class_weights: Tensor | None,
    num_classes: int,
    label_smoothing: float,
) -> nn.Module:
    """Build loss criterion.

    Args:
        head: Head kind ('corn' or 'ce').
        class_weights: [K] class weights or None.
        num_classes: Number of classes.
        label_smoothing: Label smoothing (CE only).

    Returns:
        Loss module.

    Raises:
        ValueError: If head is not 'corn' or 'ce'.
    """
    if head == "corn":
        return CornLoss(num_classes, class_weights)
    elif head == "ce":
        return WeightedCrossEntropy(class_weights, label_smoothing)
    else:
        raise ValueError(f"Unknown head: {head}. Must be 'corn' or 'ce'.")
