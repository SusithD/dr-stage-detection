"""Atomic checkpoint writes, loading with a format check, and the inference bundle layout."""

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

CHECKPOINT_FORMAT = 1


def save_checkpoint(path: Path, state: dict) -> None:
    """Save state atomically: write to a .tmp beside the target, then rename over it. A crash
    mid-write leaves the old checkpoint intact.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.parent / f"{path.name}.tmp"
    torch.save(state, tmp_path)
    tmp_path.replace(path)

    logger.debug(f"Saved checkpoint to {path}")


def load_checkpoint(path: Path, map_location: str = "cpu") -> dict:
    """Load a checkpoint and refuse it unless format_version matches CHECKPOINT_FORMAT."""
    state = torch.load(path, map_location=map_location)

    assert state.get("format_version") == CHECKPOINT_FORMAT, (
        f"Checkpoint format {state.get('format_version')} "
        f"does not match expected {CHECKPOINT_FORMAT}"
    )

    logger.debug(f"Loaded checkpoint from {path}")
    return state


def resolve_checkpoint(spec: str | Path) -> Path:
    """Turn a file, run dir or experiment dir into a concrete best.pt path.

    A run dir is anything holding checkpoints/best.pt; an experiment dir (runs/<exp>) picks
    the newest timestamped run, which works because run folders sort by name.
    """
    spec = Path(spec)

    if spec.is_file():
        return spec

    checkpoints_dir = spec / "checkpoints"
    if checkpoints_dir.is_dir():
        best_pt = checkpoints_dir / "best.pt"
        if best_pt.exists():
            return best_pt

    run_dirs = sorted(spec.glob("*/checkpoints/best.pt"), reverse=True)
    if run_dirs:
        return run_dirs[0]

    raise FileNotFoundError(
        f"Could not resolve checkpoint from {spec}. "
        f"Expected file, run dir with checkpoints/, or exp dir runs/<exp>"
    )


def build_inference_bundle(
    cfg: dict,
    model: object,
    mean: list[float],
    std: list[float],
    preprocess_cfg: dict,
    git_commit: str,
) -> dict:
    """Everything the predictor needs to rebuild and run the model, minus the weights.

    The preprocess block drops the 'methods' catalogue and pins the method and cache size
    actually used, so inference can reproduce the training input pipeline from the
    checkpoint alone.
    """
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    bundle = {
        "model": {
            "backbone": model_cfg["backbone"],
            "head": model_cfg["head"],
            "num_classes": model_cfg["num_classes"],
            "dropout": model_cfg.get("dropout", 0.0),
            "drop_path_rate": model_cfg.get("drop_path_rate", 0.0),
            "img_size": data_cfg["img_size"],
        },
        "preprocess": {
            **{k: v for k, v in preprocess_cfg.items() if k != "methods"},
            "method": data_cfg["method"],
            "size": data_cfg["cache_size"],
        },
        "normalize": {
            "mean": mean,
            "std": std,
        },
        "class_names": ["No DR", "Mild", "Moderate", "Severe", "Proliferative"],
        "tta": {"flips": True},
        "config": cfg,
        "git_commit": git_commit,
    }

    return bundle
