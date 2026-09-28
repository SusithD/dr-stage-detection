"""CORN ordinal loss, weighted cross-entropy and the class weighting they share."""

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import LongTensor, Tensor


def class_weights_from_counts(counts: Sequence[int], power: float) -> Tensor:
    """Inverse-frequency class weights, (N / (K * n_c)) ** power, rescaled to mean 1.

    power=0.5 is the square-root weighting used by default; power=0 turns weighting off.
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
    """CORN loss: K-1 binary cross-entropies over conditional subsets, vectorised with a mask.

    Task k only sees samples with y >= k and asks whether y > k, so logits outside that
    subset never touch the loss. Optional per-sample weights scale each term and the
    normaliser. Runs in float32 regardless of autocast.

    Args:
        logits: [N, K-1] raw head outputs.
        targets: [N] integer grades in 0..K-1.
        sample_weights: [N] per-sample weights, or None for uniform.
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
    """Module wrapper around corn_loss that looks up a per-sample weight from the class weights.

    Weights are registered as a buffer so they follow the module across devices.
    """

    def __init__(self, num_classes: int, class_weights: Tensor | None = None):
        super().__init__()
        self.num_classes = num_classes
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: Tensor, targets: LongTensor) -> Tensor:
        """Index the class weights by target and hand everything to corn_loss."""
        sample_weights = None
        if self.class_weights is not None:
            sample_weights = self.class_weights[targets]

        return corn_loss(logits, targets, self.num_classes, sample_weights)


class WeightedCrossEntropy(nn.Module):
    """nn.CrossEntropyLoss with optional class weights and label smoothing."""

    def __init__(self, class_weights: Tensor | None = None, label_smoothing: float = 0.0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    def forward(self, logits: Tensor, targets: LongTensor) -> Tensor:
        """Mean weighted cross-entropy over the batch."""
        return self.ce_loss(logits, targets)


def build_criterion(
    head: str,
    class_weights: Tensor | None,
    num_classes: int,
    label_smoothing: float,
) -> nn.Module:
    """Pick the loss matching the head kind. label_smoothing only applies to the CE path."""
    if head == "corn":
        return CornLoss(num_classes, class_weights)
    elif head == "ce":
        return WeightedCrossEntropy(class_weights, label_smoothing)
    else:
        raise ValueError(f"Unknown head: {head}. Must be 'corn' or 'ce'.")
