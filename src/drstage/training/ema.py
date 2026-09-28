"""Exponential moving average of model weights, validated and checkpointed in stage 2."""

import torch
from timm.utils.model_ema import ModelEmaV3


def build_ema(model: torch.nn.Module, decay: float, device: torch.device) -> ModelEmaV3:
    """Create an EMA copy of the model whose decay warms up from 0 towards ``decay``.

    Warm-up keeps the decay low at first so the average follows the live weights instead of
    clinging to the copy it started from. The averaged model lives at ``ema.module``.
    """
    return ModelEmaV3(model, decay=decay, use_warmup=True, device=device, exclude_buffers=False)


def reset_ema(ema: ModelEmaV3, model: torch.nn.Module) -> None:
    """Copy the live weights into the EMA so stage-1 weights do not dominate stage 2."""
    ema.module.load_state_dict(model.state_dict())
