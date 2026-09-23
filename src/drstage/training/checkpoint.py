"""Checkpoint saving and loading utilities."""

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

CHECKPOINT_FORMAT = 1


def save_checkpoint(path: Path, state: dict) -> None:
    """Save checkpoint to file atomically (write to tmp then rename).

    Args:
        path: Path to save checkpoint.
        state: Checkpoint state dict.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.parent / f"{path.name}.tmp"
    torch.save(state, tmp_path)
    tmp_path.replace(path)

    logger.debug(f"Saved checkpoint to {path}")


def load_checkpoint(path: Path, map_location: str = "cpu") -> dict:
    """Load checkpoint from file.

    Args:
        path: Path to checkpoint file.
        map_location: Device mapping (default cpu).

    Returns:
        Checkpoint state dict.

    Raises:
        AssertionError: If checkpoint format version does not match.
    """
    state = torch.load(path, map_location=map_location)

    assert state.get("format_version") == CHECKPOINT_FORMAT, (
        f"Checkpoint format {state.get('format_version')} "
        f"does not match expected {CHECKPOINT_FORMAT}"
    )

    logger.debug(f"Loaded checkpoint from {path}")
    return state


def resolve_checkpoint(spec: str | Path) -> Path:
    """Resolve checkpoint path from various specs.

    - file path -> itself
    - run dir (contains checkpoints/) -> checkpoints/best.pt
    - exp dir (runs/<exp>) -> newest timestamp's best.pt

    Args:
        spec: Checkpoint specification (file, run dir, or exp dir).

    Returns:
        Path to checkpoint file.

    Raises:
        FileNotFoundError: If checkpoint cannot be resolved.
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
    """Build inference bundle for checkpoint (everything except weights).

    Args:
        cfg: Training config.
        model: DRModel instance.
        mean: Normalization mean (3 floats).
        std: Normalization std (3 floats).
        preprocess_cfg: Preprocessing config from data.yaml.
        git_commit: Git commit hash.

    Returns:
        Dict with bundle metadata for checkpoint.
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
