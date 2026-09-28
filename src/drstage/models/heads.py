"""Classification heads: CORN ordinal (K-1 binary tasks) and plain cross-entropy."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import LongTensor, Tensor


class Head(nn.Module, ABC):
    """Interface every head implements: a forward to logits and a predict that decodes them."""

    num_classes: int
    out_dim: int

    @abstractmethod
    def forward(self, features: Tensor) -> Tensor:
        """Map [N, in_features] features to [N, out_dim] logits."""

    @abstractmethod
    def predict(self, logits: Tensor) -> tuple[LongTensor, Tensor, Tensor]:
        """Decode [N, out_dim] logits into (grade [N], prob_dr [N], class_probs [N, K]).

        prob_dr is the probability of any DR, i.e. grade >= 1, and feeds the
        binary metrics.
        """


class CornHead(Head):
    """CORN ordinal head: K-1 binary logits, one per 'is the grade above k' question."""

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.out_dim = num_classes - 1

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_features, self.out_dim)

    def forward(self, features: Tensor) -> Tensor:
        """Dropout then a single linear layer to [N, K-1] logits."""
        return self.linear(self.dropout(features))

    def predict(self, logits: Tensor) -> tuple[LongTensor, Tensor, Tensor]:
        """Turn CORN logits into a grade and class probabilities.

        P(y > k) is the running product of the sigmoids, so it can only fall with
        k; the grade is how many of those exceed 0.5 and class k gets P(y > k-1) -
        P(y > k). Cast to float32 first so the cumulative product is not done in
        half precision.
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
    """Plain softmax head with K logits."""

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.out_dim = num_classes

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, features: Tensor) -> Tensor:
        """Dropout then a single linear layer to [N, K] logits."""
        return self.linear(self.dropout(features))

    def predict(self, logits: Tensor) -> tuple[LongTensor, Tensor, Tensor]:
        """Softmax, argmax for the grade, and 1 - P(class 0) as the probability of any DR."""
        class_probs = torch.softmax(logits, dim=1)
        grade = torch.argmax(class_probs, dim=1).long()
        prob_dr = 1.0 - class_probs[:, 0]

        return grade, prob_dr, class_probs


def build_head(kind: str, in_features: int, num_classes: int, dropout: float = 0.0) -> Head:
    """Pick a head by name, 'corn' or 'ce'."""
    if kind == "corn":
        return CornHead(in_features, num_classes, dropout)
    elif kind == "ce":
        return CeHead(in_features, num_classes, dropout)
    else:
        raise ValueError(f"Unknown head kind: {kind}. Must be 'corn' or 'ce'.")
