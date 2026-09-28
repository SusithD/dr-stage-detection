"""CLI for training: parse args, pick the run dir and device, then hand off to Trainer."""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

from drstage.training.config import load_train_config
from drstage.training.trainer import Trainer
from drstage.utils.config import save_config
from drstage.utils.logging import get_logger

logger = None


def _get_device(device_str: str) -> torch.device:
    """Resolve 'auto' to CUDA, then MPS, then CPU; other strings go straight to torch.device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    return torch.device(device_str)


def _create_run_dir(cfg: dict) -> Path:
    """Make runs_root/exp_name/<timestamp> and return it."""
    runs_root = Path(cfg.get("runs_root", "runs"))
    exp_name = cfg.get("exp_name", "base")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = runs_root / exp_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_metadata(run_dir: Path, cfg: dict) -> None:
    """Write meta.json with git state, library versions and the launch command.

    git is run from three levels above the run dir, which is the repo root for the default
    runs/<exp>/<timestamp> layout.
    """
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=run_dir.parent.parent.parent,
        ).stdout.strip()
    except Exception:
        git_commit = "unknown"

    try:
        git_status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=run_dir.parent.parent.parent,
        ).stdout.strip()
        git_dirty = len(git_status) > 0
    except Exception:
        git_dirty = False

    metadata = {
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "torch": torch.__version__,
        "timm": None,
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "amp_policy": {
            "enabled": cfg.get("amp", {}).get("dtype") != "off",
            "dtype": cfg.get("amp", {}).get("dtype", "auto"),
        },
        "argv": " ".join(__import__("sys").argv),
        "started_at": datetime.now().isoformat(),
        "hostname": __import__("socket").gethostname(),
    }

    try:
        import timm

        metadata["timm"] = timm.__version__
    except Exception:
        pass

    with open(run_dir / "meta.json", "w") as f:
        json.dump(metadata, f, indent=2)


def main(argv=None) -> int:
    """Parse arguments, set up the run directory and logger, and run training.

    --resume takes either a run dir or a last.pt path and reuses that run's directory unless
    --run-dir overrides it. The run dir is printed as the last line of stdout so shell
    scripts can pick it up.
    """
    parser = argparse.ArgumentParser(description="Train DR stage detection model")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument(
        "-o", "--override", action="append", default=[], help="Config overrides (key.subkey=value)"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint (run dir or last.pt path)"
    )
    parser.add_argument("--run-dir", type=str, default=None, help="Custom run directory path")

    args = parser.parse_args(argv)

    cfg = load_train_config(args.config, args.override)

    device = _get_device(cfg.get("device", "auto"))

    resume_path = Path(args.resume) if args.resume else None
    if resume_path is not None and resume_path.is_dir():
        resume_path = resume_path / "checkpoints" / "last.pt"
    if resume_path is not None and not resume_path.is_file():
        parser.error(f"no resumable checkpoint at {resume_path}")

    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif resume_path is not None:
        run_dir = resume_path.parent.parent
    else:
        run_dir = _create_run_dir(cfg)

    run_dir.mkdir(parents=True, exist_ok=True)

    global logger
    logger = get_logger("drstage", log_file=run_dir / "train.log")

    logger.info(f"Training config loaded from {args.config}")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Device: {device}")

    save_config(cfg, run_dir / "config.yaml")
    _save_metadata(run_dir, cfg)

    trainer = Trainer(cfg, run_dir, device)

    if resume_path is not None:
        logger.info(f"Resuming from {resume_path}")

    summary = trainer.fit(resume_from=resume_path)

    logger.info(f"Training completed. Summary: {json.dumps(summary, indent=2)}")
    print(str(run_dir))

    return 0


if __name__ == "__main__":
    code = main()
    logging.shutdown()
    sys.stdout.flush()
    # persistent spawn-mode DataLoader workers can block interpreter shutdown on macOS
    os._exit(code)
