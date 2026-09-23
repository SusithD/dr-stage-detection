"""Automatic mixed precision handling with device-aware policy."""

import logging
from contextlib import contextmanager
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class AmpPolicy:
    """AMP policy configuration."""

    enabled: bool
    dtype: torch.dtype | None
    use_scaler: bool
    device_type: str = "cpu"


def resolve_amp(device: torch.device, cfg_amp: dict) -> AmpPolicy:
    """Resolve AMP policy based on device and config.

    According to spec amendment:
    - MPS: float16 without GradScaler
    - CUDA: float16 (or configured) with GradScaler
    - CPU: disabled

    Args:
        device: torch.device object.
        cfg_amp: Config dict with key 'dtype' (auto, float16, bfloat16, or off).

    Returns:
        AmpPolicy instance.
    """
    dtype_cfg = cfg_amp.get("dtype", "auto")

    if dtype_cfg == "off":
        return AmpPolicy(enabled=False, dtype=None, use_scaler=False)

    if device.type == "cpu":
        return AmpPolicy(enabled=False, dtype=None, use_scaler=False)

    if device.type == "mps":
        if dtype_cfg == "auto":
            dtype = torch.float16
        elif dtype_cfg == "float16":
            dtype = torch.float16
        elif dtype_cfg == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        return AmpPolicy(enabled=True, dtype=dtype, use_scaler=False, device_type="mps")

    if device.type == "cuda":
        if dtype_cfg == "auto":
            dtype = torch.float16
        elif dtype_cfg == "float16":
            dtype = torch.float16
        elif dtype_cfg == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        return AmpPolicy(enabled=True, dtype=dtype, use_scaler=True, device_type="cuda")

    return AmpPolicy(enabled=False, dtype=None, use_scaler=False)


class AmpContext:
    """Context manager for AMP training with device-aware GradScaler."""

    def __init__(self, policy: AmpPolicy):
        """Initialize AMP context.

        Args:
            policy: AmpPolicy instance.
        """
        self.policy = policy
        self.scaler = torch.amp.GradScaler("cuda") if policy.use_scaler else None

    @contextmanager
    def autocast(self):
        """Context manager for autocast during forward pass."""
        if self.policy.enabled:
            with torch.autocast(device_type=self.policy.device_type, dtype=self.policy.dtype):
                yield
        else:
            yield

    def backward(self, loss: torch.Tensor) -> bool:
        """Backpropagate one micro-batch loss; returns False (no backward) if it is non-finite."""
        if not torch.isfinite(loss):
            logger.warning("Non-finite loss detected; discarding this accumulation window")
            return False
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        return True

    def step(self, optimizer: torch.optim.Optimizer, params, clip_norm: float, ok: bool) -> bool:
        """Clip and apply accumulated gradients, or discard them when the window was not finite.

        Args:
            optimizer: Optimizer to step.
            params: Parameters whose gradients are clipped.
            clip_norm: Maximum gradient norm.
            ok: False if any micro-batch in the window had a non-finite loss.

        Returns:
            True if the optimizer stepped, False if the update was skipped.
        """
        if ok:
            if self.scaler:
                self.scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, clip_norm)
            ok = bool(torch.isfinite(grad_norm))
            if not ok:
                logger.warning("Non-finite gradient norm detected; skipping step")
        if ok:
            if self.scaler:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
        elif self.scaler:
            self.scaler.update()
        optimizer.zero_grad(set_to_none=True)
        return ok
