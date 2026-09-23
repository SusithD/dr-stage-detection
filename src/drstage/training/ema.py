"""Exponential moving average of model weights, validated and checkpointed in stage 2."""

import torch
from timm.utils.model_ema import ModelEmaV3


def build_ema(model: torch.nn.Module, decay: float, device: torch.device) -> ModelEmaV3:
    """Create an EMA copy of the model whose decay warms up from 0 towards ``decay``.

    Args:
        model: Model to average.
        decay: Maximum decay rate.
        device: Device holding the EMA weights.

    Returns:
        ModelEmaV3 wrapper; the averaged model is ``ema.module``.
    """
    return ModelEmaV3(model, decay=decay, use_warmup=True, device=device, exclude_buffers=False)


def reset_ema(ema: ModelEmaV3, model: torch.nn.Module) -> None:
    """Copy the live weights into the EMA so stage-1 weights do not dominate stage 2."""
    ema.module.load_state_dict(model.state_dict())
