"""Autocast policy per device: MPS runs without a GradScaler, CUDA with one, CPU stays fp32."""

import logging
from contextlib import contextmanager
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class AmpPolicy:
    """What autocast should do on this device, and whether a GradScaler is involved."""

    enabled: bool
    dtype: torch.dtype | None
    use_scaler: bool
    device_type: str = "cpu"


def resolve_amp(device: torch.device, cfg_amp: dict) -> AmpPolicy:
    """Decide the autocast dtype and scaler use from the device and the amp config.

    MPS runs autocast without a GradScaler and leans on the non-finite guard in
    AmpContext instead. 'auto' and anything unrecognised resolve to float16; 'off' or a
    CPU device disables autocast entirely.
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
    """Autocast, backward and step under one AMP policy, with the non-finite guard."""

    def __init__(self, policy: AmpPolicy):
        """Only builds a GradScaler when the policy asks for one (CUDA)."""
        self.policy = policy
        self.scaler = torch.amp.GradScaler("cuda") if policy.use_scaler else None

    @contextmanager
    def autocast(self):
        """Autocast for the forward pass; a no-op when AMP is disabled."""
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
        """Clip and apply the accumulated gradients, or throw them away if the window went bad.

        ok is False when any micro-batch in the window had a non-finite loss; a non-finite
        clipped norm also skips the update. Gradients are zeroed either way, and the scaler
        still updates so it can back off its scale. Returns True only if the step happened.
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
