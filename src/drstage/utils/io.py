"""File I/O utilities for JSON, CSV, and directory management."""

import json
import tempfile
from pathlib import Path

import pandas as pd


def ensure_dir(p: Path) -> Path:
    """Ensure directory exists and return the path."""
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(obj: object, path: Path, indent: int = 2) -> None:
    """Write JSON to file, creating parent directories."""
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)


def read_json(path: Path) -> object:
    """Read JSON from file."""
    with open(path) as f:
        return json.load(f)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write CSV atomically via temporary file and rename."""
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, suffix=".tmp") as tmp:
        tmp_path = Path(tmp.name)
    try:
        df.to_csv(tmp_path, index=False)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
