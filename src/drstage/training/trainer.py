"""Two-stage training loop, from head-only warm-up to full fine-tuning with EMA."""

import json
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from drstage.data.loaders import build_dataloaders
from drstage.data.registry import CLASS_NAMES
from drstage.evaluation.metrics import compute_metrics
from drstage.models.backbone import normalization_stats
from drstage.models.factory import (
    build_model,
    param_groups,
    set_backbone_trainable,
)
from drstage.models.losses import build_criterion, class_weights_from_counts
from drstage.training.amp import AmpContext, resolve_amp
from drstage.training.checkpoint import (
    CHECKPOINT_FORMAT,
    build_inference_bundle,
    save_checkpoint,
)
from drstage.training.ema import build_ema, reset_ema
from drstage.training.history import EarlyStopping, EpochRecord, HistoryWriter
from drstage.training.schedule import build_optimizer, build_scheduler
from drstage.utils.seed import seed_everything

logger = logging.getLogger(__name__)


class Trainer:
    """Run the two-stage schedule and own everything it touches: model, loaders, AMP, EMA.

    Stage 1 trains the head on a frozen backbone for warmup_epochs. Stage 2 unfreezes it
    with a lower backbone LR, starts the cosine schedule, resets the EMA and turns on
    early stopping on validation QWK. Only stage 2 can produce a best checkpoint, and when
    EMA is enabled the EMA weights are what get validated and saved.
    """

    def __init__(self, cfg: dict, run_dir: Path, device: torch.device):
        """Build model, loaders, loss, AMP and EMA up front; optimizers are created per stage."""
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.device = device

        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        seed_everything(cfg["seed"], deterministic=False)

        self.model = build_model(cfg["model"], pretrained=cfg["model"].get("pretrained", True))
        self.model = self.model.to(device)

        if cfg["model"].get("channels_last", False):
            self.model = self.model.to(memory_format=torch.channels_last)

        self.train_loader, self.val_loader, self.data_info = build_dataloaders(
            cfg, *normalization_stats(self.model.backbone), device
        )

        class_counts = self.data_info["class_counts"]
        class_weight_power = cfg.get("imbalance", {}).get("class_weight_power", 0.5)
        sampler_enabled = cfg.get("imbalance", {}).get("sampler", {}).get("enabled", False)
        weights_with_sampler = cfg.get("imbalance", {}).get("weights_with_sampler", False)

        class_weights = None
        # class weights are skipped when the sampler already balances, unless explicitly asked for
        if class_weight_power > 0 and (not sampler_enabled or weights_with_sampler):
            class_weights = class_weights_from_counts(class_counts, class_weight_power)
            class_weights = class_weights.to(device)

        self.criterion = build_criterion(
            cfg["model"]["head"],
            class_weights,
            cfg["model"]["num_classes"],
            cfg.get("imbalance", {}).get("label_smoothing", 0.0),
        )

        self.amp_policy = resolve_amp(device, cfg.get("amp", {}))
        self.amp = AmpContext(self.amp_policy)

        self.ema = None
        if cfg.get("ema", {}).get("enabled", False):
            self.ema = build_ema(self.model, cfg["ema"]["decay"], device)

        self.optimizer = None
        self.scheduler = None
        self.early_stopping = None

        self.history_writer = HistoryWriter(run_dir)

        from drstage.utils.logging import get_logger

        self._logger = get_logger("drstage", log_file=run_dir / "train.log")

        self.epochs = cfg["train"]["epochs"]
        self.warmup_epochs = cfg["train"]["warmup_epochs"]
        self.accum_steps = cfg["train"]["accum_steps"]
        self.clip_grad_norm = cfg["train"]["clip_grad_norm"]
        self.freeze_bn_stats_stage1 = cfg["train"].get("freeze_bn_stats_stage1", True)
        self.limit_train_batches = cfg["train"].get("limit_train_batches", 0)
        self.limit_val_batches = cfg["train"].get("limit_val_batches", 0)
        self.log_every_steps = cfg["train"].get("log_every_steps", 50)

        self.current_epoch = 0
        self.current_stage = 1
        self.global_step = 0
        self.total_skipped_steps = 0
        self.best_val_qwk = -1.0
        self.best_epoch: int | None = None
        self.best_metrics = {}
        self.start_time = None
        self.started_at = datetime.now().isoformat()

    def fit(self, resume_from: Path | None = None) -> dict:
        """Run the epoch loop, moving to stage 2 after warmup_epochs, and return the summary.

        Resume expects last.pt. The returned dict also lands in summary.json; keys are
        best_epoch, best_val_qwk, best_metrics, epochs_run, stopped_early, total_time_s,
        skipped_steps.
        """
        self.start_time = time.time()

        if not resume_from:
            self._save_config_and_metadata()

        if resume_from:
            self._load_checkpoint(resume_from)
            start_epoch = self.current_epoch + 1
        else:
            start_epoch = 1
            self._setup_stage1()

        logger.info(f"Training from epoch {start_epoch} to {self.epochs}")

        for epoch in range(start_epoch, self.epochs + 1):
            epoch_start = time.time()

            self.current_epoch = epoch

            if epoch == self.warmup_epochs + 1 and not resume_from:
                self._setup_stage2()
            elif resume_from and epoch == self.warmup_epochs + 1:
                self._setup_stage2()
                resume_from = None

            train_loss, skipped = self._train_epoch(self.train_loader, epoch)

            use_ema = self.ema is not None and self.current_stage == 2
            model_for_eval = self.ema.module if use_ema else self.model
            val_loss, metrics, val_df = self._validate(model_for_eval, self.val_loader)

            epoch_time = time.time() - epoch_start

            lr_backbone, lr_head = self._get_lrs()

            record = EpochRecord(
                epoch=epoch,
                stage=self.current_stage,
                train_loss=train_loss,
                val_loss=val_loss,
                val_acc=metrics["multiclass"]["accuracy"],
                val_qwk=metrics["multiclass"]["qwk"],
                val_macro_f1=metrics["multiclass"]["macro_f1"],
                val_auc=metrics["binary"].get("roc_auc", 0.0),
                lr_backbone=lr_backbone,
                lr_head=lr_head,
                epoch_time_s=epoch_time,
                skipped_steps=skipped,
                is_best=False,
            )

            improved = False
            if self.current_stage == 2:
                improved = self.early_stopping.step(metrics["multiclass"]["qwk"])
                record.is_best = improved

                if improved:
                    self.best_epoch = epoch
                    self.best_val_qwk = metrics["multiclass"]["qwk"]
                    self.best_metrics = metrics
                    self._save_best_checkpoint(model_for_eval, metrics)
                    val_df.to_csv(self.run_dir / "val_predictions.csv", index=False)

            self.history_writer.append(record)

            logger.info(
                f"Epoch {epoch}/{self.epochs} | "
                f"stage {self.current_stage} | "
                f"train_loss {train_loss:.4f} | "
                f"val_loss {val_loss:.4f} | "
                f"val_qwk {metrics['multiclass']['qwk']:.4f} | "
                f"time {epoch_time:.1f}s"
            )

            self._save_last_checkpoint(epoch)

            if self.current_stage == 2 and self.early_stopping.should_stop:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        total_time = time.time() - self.start_time
        summary = {
            "best_epoch": None,
            "best_val_qwk": None,
            "best_metrics": None,
            "epochs_run": self.current_epoch,
            "stopped_early": self.current_stage == 2 and self.early_stopping.should_stop,
            "total_time_s": total_time,
            "skipped_steps": self.total_skipped_steps,
        }

        if self.best_metrics:
            summary["best_epoch"] = self.best_epoch
            summary["best_val_qwk"] = self.best_val_qwk
            summary["best_metrics"] = self.best_metrics

        self._save_summary(summary)

        return summary

    def _setup_stage1(self) -> None:
        """Stage 1: freeze the backbone and optimize only the head, with no weight decay.

        The head LR is backbone_lr x head_lr_mult, the same as it will be in stage 2.
        """
        logger.info("Setting up stage 1: frozen backbone, head-only training")

        self.current_stage = 1
        set_backbone_trainable(
            self.model, trainable=False, freeze_bn_stats=self.freeze_bn_stats_stage1
        )

        head_lr = self.cfg["optim"]["backbone_lr"] * self.cfg["optim"].get("head_lr_mult", 10)
        head_groups = [
            {
                "params": self.model.head.parameters(),
                "lr": head_lr,
                "weight_decay": 0.0,
                "name": "head",
            }
        ]

        self.optimizer = build_optimizer(head_groups, self.cfg["optim"])

    def _setup_stage2(self) -> None:
        """Stage 2: unfreeze the backbone, discriminative LRs, warm-up + cosine schedule.

        Schedule length is counted in optimizer steps (micro-batches / accum_steps, rounded up)
        over the remaining epochs. The EMA is reset to the current weights and early stopping
        starts fresh here.
        """
        logger.info("Setting up stage 2: unfrozen backbone, discriminative LRs")

        self.current_stage = 2

        set_backbone_trainable(self.model, trainable=True, freeze_bn_stats=False)

        backbone_lr = self.cfg["optim"]["backbone_lr"]
        head_lr = backbone_lr * self.cfg["optim"].get("head_lr_mult", 10)

        groups = param_groups(self.model, backbone_lr, head_lr, self.cfg["optim"]["weight_decay"])
        self.optimizer = build_optimizer(groups, self.cfg["optim"])

        epochs_stage2 = self.epochs - self.warmup_epochs
        steps_per_epoch = len(self.train_loader)
        if self.limit_train_batches > 0:
            steps_per_epoch = min(self.limit_train_batches, steps_per_epoch)

        total_steps = epochs_stage2 * ((steps_per_epoch + self.accum_steps - 1) // self.accum_steps)
        warmup_steps = self.cfg["optim"].get("lr_warmup_epochs", 1) * (
            (steps_per_epoch + self.accum_steps - 1) // self.accum_steps
        )

        self.scheduler = build_scheduler(
            self.optimizer,
            total_steps,
            warmup_steps,
            self.cfg["optim"].get("min_lr_ratio", 0.01),
        )

        if self.ema:
            reset_ema(self.ema, self.model)

        self.early_stopping = EarlyStopping(
            patience=self.cfg.get("early_stopping", {}).get("patience", 6),
            min_delta=self.cfg.get("early_stopping", {}).get("min_delta", 0.002),
            mode="max",
        )

    def _train_epoch(self, loader: DataLoader, epoch: int) -> tuple[float, int]:
        """Train one epoch, accumulating gradients over ``accum_steps`` micro-batches.

        backward() runs on every micro-batch with the loss divided by accum_steps; the
        optimizer steps at the end of each window and the whole window is dropped if any loss
        in it was non-finite. The scheduler and EMA advance at every window end, even when
        the step was skipped. Returns (mean loss over finite batches, skipped optimizer steps).
        """
        self.model.train()
        # model.train() flips BN back to train mode, so re-freeze it every epoch in stage 1
        if self.freeze_bn_stats_stage1 and self.current_stage == 1:
            set_backbone_trainable(self.model, trainable=False, freeze_bn_stats=True)

        n_batches = len(loader)
        if self.limit_train_batches > 0:
            n_batches = min(n_batches, self.limit_train_batches)

        total_loss = 0.0
        n_seen = 0
        skipped_steps = 0
        window_ok = True
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(loader):
            if batch_idx >= n_batches:
                break
            images = batch["image"].to(self.device, non_blocking=True)
            if self.cfg["model"].get("channels_last", False):
                images = images.contiguous(memory_format=torch.channels_last)
            labels = batch["label"].to(self.device, non_blocking=True)

            with self.amp.autocast():
                logits = self.model(images)
            loss = self.criterion(logits.float(), labels)
            window_ok = self.amp.backward(loss / self.accum_steps) and window_ok

            loss_value = loss.item()
            if np.isfinite(loss_value):
                total_loss += loss_value
                n_seen += 1

            end_of_window = (batch_idx + 1) % self.accum_steps == 0 or batch_idx + 1 == n_batches
            if end_of_window:
                if not self.amp.step(self.optimizer, params, self.clip_grad_norm, window_ok):
                    skipped_steps += 1
                    self.total_skipped_steps += 1
                window_ok = True
                if self.scheduler:
                    self.scheduler.step()
                self.global_step += 1
                if self.ema:
                    self.ema.update(self.model, step=self.global_step)

            if batch_idx % self.log_every_steps == 0 and batch_idx > 0:
                logger.info(
                    f"epoch {epoch} batch {batch_idx}/{n_batches} "
                    f"loss {total_loss / max(n_seen, 1):.4f}"
                )

        return total_loss / max(n_seen, 1), skipped_steps

    def _validate(self, model: nn.Module, loader: DataLoader) -> tuple[float, dict, object]:
        """Evaluate on the loader under autocast and gather per-image predictions.

        Returns (mean val loss, metrics dict from compute_metrics, DataFrame of per-image
        label/pred/prob_dr/p0..pK-1) - the frame becomes val_predictions.csv on a new best.
        """
        model.eval()

        total_loss = 0.0
        num_batches = 0
        all_labels = []
        all_preds = []
        all_class_probs = []
        all_prob_dr = []

        all_ids: list[str] = []
        with torch.inference_mode():
            for batch_idx, batch in enumerate(loader):
                if self.limit_val_batches > 0 and batch_idx >= self.limit_val_batches:
                    break

                images = batch["image"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)

                with self.amp.autocast():
                    logits = model(images)
                logits = logits.float()
                total_loss += self.criterion(logits, labels).item()
                num_batches += 1

                preds, prob_dr, class_probs = model.predict(logits)
                all_ids.extend(batch["image_id"])
                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_prob_dr.append(prob_dr.cpu().numpy())
                all_class_probs.append(class_probs.cpu().numpy())

        val_loss = total_loss / num_batches if num_batches > 0 else 0.0

        labels = np.concatenate(all_labels)
        preds = np.concatenate(all_preds)
        prob_dr = np.concatenate(all_prob_dr)
        class_probs = np.concatenate(all_class_probs)

        metrics = compute_metrics(labels, preds, class_probs, prob_dr, list(CLASS_NAMES))
        pred_df = pd.DataFrame(
            {
                "image_id": all_ids,
                "label": labels,
                "pred": preds,
                "prob_dr": prob_dr,
                **{f"p{i}": class_probs[:, i] for i in range(class_probs.shape[1])},
            }
        )
        return val_loss, metrics, pred_df

    def _get_lrs(self) -> tuple[float, float]:
        """Current (backbone LR, head LR), read off the param group order set by param_groups.

        Stage 1 has only the head group, so the backbone LR is reported as 0.
        """
        lrs = [group["lr"] for group in self.optimizer.param_groups]

        if self.current_stage == 1:
            return 0.0, lrs[0] if lrs else 0.0

        backbone_lrs = lrs[:2]
        head_lrs = lrs[2:]
        return (
            backbone_lrs[0] if backbone_lrs else 0.0,
            head_lrs[0] if head_lrs else 0.0,
        )

    def _save_best_checkpoint(self, model: nn.Module, metrics: dict) -> None:
        """Write best.pt: the evaluated weights (EMA when enabled) plus the inference bundle.

        Optimizer and resume state are left out on purpose; best.pt is for inference only.
        """
        mean, std = normalization_stats(self.model.backbone)

        preprocess_cfg_path = Path(self.cfg["data"]["config"])
        if preprocess_cfg_path.exists():
            import yaml

            with open(preprocess_cfg_path) as f:
                preprocess_cfg = yaml.safe_load(f).get("preprocess", {})
        else:
            preprocess_cfg = {}

        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        if not git_commit:
            git_commit = "unknown"

        bundle = build_inference_bundle(self.cfg, model, mean, std, preprocess_cfg, git_commit)
        bundle["created_at"] = datetime.now().isoformat()
        bundle["best_metric"] = metrics["multiclass"]["qwk"]

        state = {
            "format_version": CHECKPOINT_FORMAT,
            "kind": "best",
            "model_state": model.state_dict(),
            "model": bundle["model"],
            "preprocess": bundle["preprocess"],
            "normalize": bundle["normalize"],
            "class_names": bundle["class_names"],
            "tta": bundle["tta"],
            "config": bundle["config"],
            "git_commit": bundle["git_commit"],
            "created_at": bundle["created_at"],
            "best_metric": bundle["best_metric"],
            "ema_state": None,
            "optimizer_state": None,
            "scheduler_state": None,
            "scaler_state": None,
            "epoch": None,
            "global_step": None,
            "stage": None,
            "early_stopping": None,
            "history": None,
        }

        save_checkpoint(self.checkpoint_dir / "best.pt", state)

    def _save_last_checkpoint(self, epoch: int) -> None:
        """Write last.pt with the full resume state, history included."""
        state = {
            "format_version": CHECKPOINT_FORMAT,
            "kind": "last",
            "model_state": self.model.state_dict(),
            "model": {
                "backbone": self.cfg["model"]["backbone"],
                "head": self.cfg["model"]["head"],
                "num_classes": self.cfg["model"]["num_classes"],
                "dropout": self.cfg["model"].get("dropout", 0.0),
                "drop_path_rate": self.cfg["model"].get("drop_path_rate", 0.0),
                "img_size": self.cfg["data"]["img_size"],
            },
            "ema_state": self.ema.module.state_dict() if self.ema else None,
            "optimizer_state": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state": self.amp.scaler.state_dict() if self.amp.scaler else None,
            "epoch": epoch,
            "global_step": self.global_step,
            "stage": self.current_stage,
            "early_stopping": self.early_stopping.state_dict() if self.early_stopping else None,
            "history": [
                {
                    "epoch": r.epoch,
                    "stage": r.stage,
                    "train_loss": r.train_loss,
                    "val_loss": r.val_loss,
                    "val_acc": r.val_acc,
                    "val_qwk": r.val_qwk,
                    "val_macro_f1": r.val_macro_f1,
                    "val_auc": r.val_auc,
                    "lr_backbone": r.lr_backbone,
                    "lr_head": r.lr_head,
                    "epoch_time_s": r.epoch_time_s,
                    "skipped_steps": r.skipped_steps,
                    "is_best": r.is_best,
                }
                for r in self.history_writer.records
            ],
        }

        save_checkpoint(self.checkpoint_dir / "last.pt", state)

    def _load_checkpoint(self, path: Path) -> None:
        """Restore from last.pt. The stage setup is re-run first so the optimizer and scheduler
        exist before their state dicts are loaded into them.
        """
        from drstage.training.checkpoint import load_checkpoint

        state = load_checkpoint(path, map_location=str(self.device))

        self.model.load_state_dict(state["model_state"])
        self.current_epoch = state["epoch"]
        self.global_step = state["global_step"]
        self.current_stage = state["stage"]

        self.history_writer.load_from_checkpoint(state.get("history", []))

        if self.current_stage == 1:
            self._setup_stage1()
        else:
            self._setup_stage2()

        if self.optimizer and state.get("optimizer_state"):
            self.optimizer.load_state_dict(state["optimizer_state"])

        if self.scheduler and state.get("scheduler_state"):
            self.scheduler.load_state_dict(state["scheduler_state"])

        if self.ema and state.get("ema_state"):
            self.ema.module.load_state_dict(state["ema_state"])

        if self.amp.scaler and state.get("scaler_state"):
            self.amp.scaler.load_state_dict(state["scaler_state"])

        if self.early_stopping and state.get("early_stopping"):
            self.early_stopping.load_state_dict(state["early_stopping"])

        logger.info(
            f"Resumed from checkpoint at epoch {self.current_epoch}, stage {self.current_stage}"
        )

    def _save_summary(self, summary: dict) -> None:
        """Dump the fit() summary to summary.json."""
        with open(self.run_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    def _save_config_and_metadata(self) -> None:
        """Write config.yaml and meta.json (git state, library versions, device, argv)."""
        import socket

        from drstage.utils.config import save_config

        save_config(self.cfg, self.run_dir / "config.yaml")

        try:
            git_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
            ).stdout.strip()
        except Exception:
            git_commit = "unknown"

        try:
            git_status = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
            ).stdout.strip()
            git_dirty = len(git_status) > 0
        except Exception:
            git_dirty = False

        metadata = {
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "torch": torch.__version__,
            "timm": None,
            "device": str(self.device),
            "amp_policy": {
                "enabled": self.cfg.get("amp", {}).get("dtype") != "off",
                "dtype": self.cfg.get("amp", {}).get("dtype", "auto"),
            },
            "argv": " ".join(__import__("sys").argv),
            "started_at": self.started_at,
            "hostname": socket.gethostname(),
        }

        try:
            import timm

            metadata["timm"] = timm.__version__
        except Exception:
            pass

        with open(self.run_dir / "meta.json", "w") as f:
            json.dump(metadata, f, indent=2)
