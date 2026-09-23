"""Training config loading with base inheritance and override support."""

import logging
from collections.abc import Sequence
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override dict into base dict, overriding matching keys.

    Args:
        base: Base dictionary.
        override: Override dictionary.

    Returns:
        Merged dictionary with override values taking precedence.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_train_config(path: str | Path, overrides: Sequence[str] = ()) -> dict:
    """Load training config with base inheritance and override support.

    If cfg has 'base' key, loads it recursively (relative to yaml's directory or repo cwd),
    deep-merges child over base, then drops 'base' key. Applies overrides with strict
    checking (KeyError on unknown keys).

    Args:
        path: Path to training YAML config.
        overrides: List of "key.subkey=value" overrides (strict; raises KeyError if unknown).

    Returns:
        Resolved configuration dictionary without 'base' key.

    Raises:
        KeyError: If an override key does not exist in the resolved config.
    """
    path = Path(path)

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    if "base" in cfg:
        base_path = cfg["base"]

        if not Path(base_path).is_absolute():
            base_dir = path.parent
            base_path = base_dir / base_path
            if not base_path.exists():
                base_path = Path.cwd() / cfg["base"]

        base_cfg = load_train_config(base_path)
        cfg.pop("base")
        cfg = _deep_merge(base_cfg, cfg)

    for override in overrides:
        key_path, value_str = override.split("=", 1)
        keys = key_path.split(".")

        obj = cfg
        for key in keys[:-1]:
            if key not in obj:
                raise KeyError(f"Unknown config key: {key_path}")
            obj = obj[key]

        final_key = keys[-1]
        if final_key not in obj:
            raise KeyError(f"Unknown config key: {key_path}")

        obj[final_key] = yaml.safe_load(value_str)

    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict) -> None:
    """Validate required config keys and values.

    Args:
        cfg: Configuration dictionary.

    Raises:
        ValueError: If validation fails.
    """
    if cfg.get("model", {}).get("head") not in ("corn", "ce"):
        raise ValueError(
            f"model.head must be 'corn' or 'ce', got {cfg.get('model', {}).get('head')}"
        )

    img_size = cfg.get("data", {}).get("img_size")
    cache_size = cfg.get("data", {}).get("cache_size")
    if img_size and cache_size and img_size > cache_size:
        logger.warning(f"data.img_size ({img_size}) > cache_size ({cache_size}); will upscale")

    accum_steps = cfg.get("train", {}).get("accum_steps", 1)
    if accum_steps < 1:
        raise ValueError(f"train.accum_steps must be >= 1, got {accum_steps}")
