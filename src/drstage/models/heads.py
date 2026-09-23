"""Classification heads for CORN and CE loss."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import LongTensor, Tensor


class Head(nn.Module, ABC):
    """Abstract base class for classification heads."""

    num_classes: int
    out_dim: int

    @abstractmethod
    def forward(self, features: Tensor) -> Tensor:
        """Compute logits from features.

        Args:
            features: [N, in_features] feature tensor.

        Returns:
            Logits tensor [N, out_dim].
        """

    @abstractmethod
    def predict(self, logits: Tensor) -> tuple[LongTensor, Tensor, Tensor]:
        """Decode logits to grade, prob_dr, and class probabilities.

        Args:
            logits: [N, out_dim] logits tensor.

        Returns:
            Tuple of (grade [N], prob_dr [N], class_probs [N, num_classes]).
        """


class CornHead(Head):
    """Cumulative Output Regression Network (CORN) head.

    out_dim = num_classes - 1 (K-1 binary classifiers).
    """

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.0):
        """Initialize CORN head.

        Args:
            in_features: Input feature dimension.
            num_classes: Number of classes (K).
            dropout: Dropout rate.
        """
        super().__init__()
        self.num_classes = num_classes
        self.out_dim = num_classes - 1

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_features, self.out_dim)

    def forward(self, features: Tensor) -> Tensor:
        """Compute CORN logits.

        Args:
            features: [N, in_features] feature tensor.

        Returns:
            [N, num_classes-1] logits.
        """
        return self.linear(self.dropout(features))

    def predict(self, logits: Tensor) -> tuple[LongTensor, Tensor, Tensor]:
        """Decode CORN logits.

        Computes P(y>k) = cumprod(sigmoid(logits)), then derives class probs.

        Args:
            logits: [N, K-1] logits.

        Returns:
            - grade: [N] predicted grades
            - prob_dr: [N] probability of any DR (referable)
            - class_probs: [N, K] class probabilities
        """
        logits = logits.float()
        cum = torch.cumprod(torch.sigmoid(logits), dim=1)

        grade = (cum > 0.5).sum(dim=1).long()

        prob_dr = cum[:, 0]

        class_probs = torch.zeros(
            logits.size(0), self.num_classes, device=logits.device, dtype=logits.dtype
        )
        class_probs[:, 0] = 1.0 - cum[:, 0]

        for k in range(1, self.num_classes - 1):
            class_probs[:, k] = cum[:, k - 1] - cum[:, k]

        class_probs[:, self.num_classes - 1] = cum[:, self.num_classes - 2]

        return grade, prob_dr, class_probs


class CeHead(Head):
    """Cross-entropy head.

    out_dim = num_classes (K).
    """

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.0):
        """Initialize CE head.

        Args:
            in_features: Input feature dimension.
            num_classes: Number of classes (K).
            dropout: Dropout rate.
        """
        super().__init__()
        self.num_classes = num_classes
        self.out_dim = num_classes

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, features: Tensor) -> Tensor:
        """Compute CE logits.

        Args:
            features: [N, in_features] feature tensor.

        Returns:
            [N, num_classes] logits.
        """
        return self.linear(self.dropout(features))

    def predict(self, logits: Tensor) -> tuple[LongTensor, Tensor, Tensor]:
        """Decode CE logits via softmax.

        Args:
            logits: [N, K] logits.

        Returns:
            - grade: [N] predicted grades (argmax)
            - prob_dr: [N] probability of any DR (1 - softmax[:,0])
            - class_probs: [N, K] softmax probabilities
        """
        class_probs = torch.softmax(logits, dim=1)
        grade = torch.argmax(class_probs, dim=1).long()
        prob_dr = 1.0 - class_probs[:, 0]

        return grade, prob_dr, class_probs


def build_head(kind: str, in_features: int, num_classes: int, dropout: float = 0.0) -> Head:
    """Build a head of the specified kind.

    Args:
        kind: Head kind ('corn' or 'ce').
        in_features: Input feature dimension.
        num_classes: Number of classes.
        dropout: Dropout rate.

    Returns:
        Head instance.

    Raises:
        ValueError: If kind is not 'corn' or 'ce'.
    """
    if kind == "corn":
        return CornHead(in_features, num_classes, dropout)
    elif kind == "ce":
        return CeHead(in_features, num_classes, dropout)
    else:
        raise ValueError(f"Unknown head kind: {kind}. Must be 'corn' or 'ce'.")
