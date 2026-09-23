"""Optimizer and learning rate scheduler for two-stage training."""

import logging
import math

import torch
import torch.optim
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


def build_optimizer(groups: list[dict], cfg_optim: dict) -> torch.optim.AdamW:
    """Build AdamW optimizer with parameter groups.

    Args:
        groups: List of parameter group dicts with keys 'params', 'lr', 'weight_decay', 'name'.
        cfg_optim: Optimizer config with keys 'betas', 'eps'.

    Returns:
        AdamW optimizer instance.
    """
    betas = tuple(cfg_optim.get("betas", [0.9, 0.999]))
    eps = cfg_optim.get("eps", 1.0e-8)

    return torch.optim.AdamW(groups, betas=betas, eps=eps)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> LambdaLR:
    """Build per-step learning rate scheduler with linear warmup + cosine annealing.

    lr(s) = base * s / warmup for s < warmup
    lr(s) = base * (min + (1-min)*0.5*(1+cos(pi*t))) for s >= warmup,
    where t = (s - warmup) / (total - warmup).

    Args:
        optimizer: Optimizer instance.
        total_steps: Total number of optimizer steps.
        warmup_steps: Number of warmup steps.
        min_lr_ratio: Minimum LR as fraction of base LR (typically 0.01).

    Returns:
        LambdaLR scheduler.
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        if total_steps <= warmup_steps:
            return min_lr_ratio
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)
