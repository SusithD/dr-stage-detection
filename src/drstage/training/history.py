"""Training history tracking and early stopping."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EpochRecord:
    """Record of metrics for one training epoch."""

    epoch: int
    stage: int
    train_loss: float
    val_loss: float
    val_acc: float
    val_qwk: float
    val_macro_f1: float
    val_auc: float
    lr_backbone: float
    lr_head: float
    epoch_time_s: float
    skipped_steps: int
    is_best: bool


class HistoryWriter:
    """Write training history to CSV and JSON each epoch."""

    def __init__(self, run_dir: Path):
        """Initialize history writer.

        Args:
            run_dir: Run directory path.
        """
        self.run_dir = Path(run_dir)
        self.csv_path = self.run_dir / "history.csv"
        self.json_path = self.run_dir / "history.json"
        self.records = []

    def append(self, record: EpochRecord) -> None:
        """Append epoch record and write to CSV/JSON.

        Args:
            record: EpochRecord instance.
        """
        self.records.append(record)

        df = pd.DataFrame([asdict(r) for r in self.records])
        df.to_csv(self.csv_path, index=False)

        with open(self.json_path, "w") as f:
            json.dump([asdict(r) for r in self.records], f, indent=2)

        logger.debug(f"Wrote history: {len(self.records)} epochs")

    def load_from_checkpoint(self, history_data: list[dict]) -> None:
        """Load history from checkpoint (for resume).

        Args:
            history_data: List of epoch dicts from checkpoint.
        """
        self.records = [
            EpochRecord(
                epoch=r["epoch"],
                stage=r["stage"],
                train_loss=r["train_loss"],
                val_loss=r["val_loss"],
                val_acc=r["val_acc"],
                val_qwk=r["val_qwk"],
                val_macro_f1=r["val_macro_f1"],
                val_auc=r["val_auc"],
                lr_backbone=r["lr_backbone"],
                lr_head=r["lr_head"],
                epoch_time_s=r["epoch_time_s"],
                skipped_steps=r["skipped_steps"],
                is_best=r["is_best"],
            )
            for r in history_data
        ]


class EarlyStopping:
    """Early stopping monitor with patience and minimum delta."""

    def __init__(self, patience: int, min_delta: float = 0.0, mode: str = "max"):
        """Initialize early stopping.

        Args:
            patience: Number of epochs with no improvement before stopping.
            min_delta: Minimum change to qualify as improvement.
            mode: 'max' for maximization (e.g., accuracy), 'min' for minimization (e.g., loss).
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best = float("-inf") if mode == "max" else float("inf")
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Check if there was an improvement.

        Args:
            value: Current metric value.

        Returns:
            True if improved, False if no improvement (counter incremented).
        """
        if self.mode == "max":
            improved = value > self.best + self.min_delta
        else:
            improved = value < self.best - self.min_delta

        if improved:
            self.best = value
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False

    def state_dict(self) -> dict:
        """Get state for checkpointing.

        Returns:
            State dict with best, counter.
        """
        return {"best": self.best, "counter": self.counter}

    def load_state_dict(self, state: dict) -> None:
        """Load state from checkpoint.

        Args:
            state: State dict from checkpoint.
        """
        self.best = state["best"]
        self.counter = state["counter"]
        self.should_stop = self.counter >= self.patience
