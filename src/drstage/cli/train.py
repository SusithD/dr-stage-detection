"""Training CLI entry point."""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import torch

from drstage.training.config import load_train_config
from drstage.training.trainer import Trainer
from drstage.utils.config import save_config
from drstage.utils.logging import get_logger

logger = None


def _get_device(device_str: str) -> torch.device:
    """Resolve device string to torch.device.

    Args:
        device_str: Device spec ('auto', 'cuda', 'mps', 'cpu').

    Returns:
        torch.device instance.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    return torch.device(device_str)


def _create_run_dir(cfg: dict) -> Path:
    """Create run directory with timestamp.

    Args:
        cfg: Training config.

    Returns:
        Path to run directory.
    """
    runs_root = Path(cfg.get("runs_root", "runs"))
    exp_name = cfg.get("exp_name", "base")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = runs_root / exp_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_metadata(run_dir: Path, cfg: dict) -> None:
    """Save metadata about the training run.

    Args:
        run_dir: Run directory path.
        cfg: Training config.
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
    """Main training CLI entry point.

    Args:
        argv: Command-line arguments (default sys.argv[1:]).

    Returns:
        Exit code.
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

    if args.run_dir:
        run_dir = Path(args.run_dir)
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

    resume_path = None
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_dir():
            from drstage.training.checkpoint import resolve_checkpoint

            resume_path = resolve_checkpoint(resume_path)
        logger.info(f"Resuming from {resume_path}")

    summary = trainer.fit(resume_from=resume_path)

    logger.info(f"Training completed. Summary: {json.dumps(summary, indent=2)}")
    print(str(run_dir))

    return 0


if __name__ == "__main__":
    exit(main())
