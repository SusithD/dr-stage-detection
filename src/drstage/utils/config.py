"""Configuration loading and saving with override support."""

from collections.abc import Sequence
from pathlib import Path

import yaml


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> dict:
    """Load YAML config with command-line overrides.

    Args:
        path: Path to YAML config file.
        overrides: List of "key.subkey=value" overrides, coerced via yaml.safe_load.

    Returns:
        Loaded configuration dict with overrides applied.

    Raises:
        KeyError: If an override path does not exist in config.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

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

    return cfg


def save_config(cfg: dict, path: Path) -> None:
    """Write config to YAML, creating parent directories if needed.

    Args:
        cfg: Configuration dict.
        path: Path to write YAML.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
