"""AdamW for both stages and the linear warm-up + cosine LR schedule used in stage 2."""

import logging
import math

import torch
import torch.optim
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


def build_optimizer(groups: list[dict], cfg_optim: dict) -> torch.optim.AdamW:
    """AdamW over the given groups, with betas/eps from the optim config or torch defaults."""
    betas = tuple(cfg_optim.get("betas", [0.9, 0.999]))
    eps = cfg_optim.get("eps", 1.0e-8)

    return torch.optim.AdamW(groups, betas=betas, eps=eps)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> LambdaLR:
    """Per-step LambdaLR: linear warm-up to the base LR, then cosine down to min_lr_ratio of it.

    Steps here are optimiser steps, not micro-batches, so the caller must divide by
    accum_steps. If total_steps <= warmup_steps the post-warm-up LR just sits at the floor.
    """

    def lr_lambda(step: int) -> float:
        """Learning-rate multiplier for an optimiser step: linear warm-up then cosine decay."""
        if step < warmup_steps:
            return step / warmup_steps
        if total_steps <= warmup_steps:
            return min_lr_ratio
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)
