"""Smoke tests for training end-to-end."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import torch
import yaml

from drstage.models.factory import build_model
from drstage.training.checkpoint import load_checkpoint
from drstage.training.config import load_train_config
from drstage.training.trainer import Trainer


class TestTrainSmoke:
    """End-to-end training smoke tests."""

    @pytest.mark.slow
    def test_train_end_to_end(self, tiny_split, tmp_path):
        """Test complete training pipeline on tiny synthetic dataset."""
        split_csv = tiny_split["split_csv"]

        config_path = tmp_path / "config.yaml"
        config = {
            "exp_name": "smoke_test",
            "seed": 42,
            "device": "cpu",
            "runs_root": str(tmp_path / "runs"),
            "data": {
                "config": str(tmp_path / "data.yaml"),
                "processed_root": str(tmp_path / "processed"),
                "dataset": "synth",
                "split_csv": str(split_csv),
                "method": "none",
                "cache_size": 64,
                "img_size": 64,
                "cache_format": "png",
                "batch_size": 4,
                "eval_batch_size": 4,
                "num_workers": 0,
                "aug": {
                    "rotate_limit": 45,
                    "rotate_p": 0.5,
                    "rrc_scale": [0.9, 1.0],
                    "brightness": 0.1,
                    "contrast": 0.1,
                    "hue": 5,
                    "sat": 5,
                    "color_p": 0.5,
                },
            },
            "model": {
                "backbone": "resnet18",
                "pretrained": False,
                "head": "corn",
                "num_classes": 5,
                "dropout": 0.1,
                "drop_path_rate": 0.0,
                "channels_last": False,
            },
            "imbalance": {
                "class_weight_power": 0.0,
                "sampler": {"enabled": False, "power": 0.5},
                "weights_with_sampler": False,
                "label_smoothing": 0.0,
            },
            "train": {
                "epochs": 2,
                "warmup_epochs": 1,
                "accum_steps": 1,
                "clip_grad_norm": 1.0,
                "freeze_bn_stats_stage1": True,
                "limit_train_batches": 0,
                "limit_val_batches": 0,
                "log_every_steps": 10,
            },
            "optim": {
                "backbone_lr": 1e-4,
                "head_lr_mult": 10,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "lr_warmup_epochs": 1,
                "min_lr_ratio": 0.01,
            },
            "amp": {"dtype": "off"},
            "ema": {"enabled": True, "decay": 0.9995},
            "early_stopping": {"metric": "val_qwk", "patience": 3, "min_delta": 0.0},
        }

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        preprocess_config = {"circle_crop": {}, "ben_graham": {}, "clahe_l": {}}
        with open(tmp_path / "data.yaml", "w") as f:
            yaml.dump({"preprocess": preprocess_config}, f)

        device = torch.device("cpu")
        trainer = Trainer(config, tmp_path / "runs" / "smoke_test" / "000000-000000", device)

        summary = trainer.fit()

        assert summary["epochs_run"] == 2
        assert summary["best_val_qwk"] is not None
        assert summary["total_time_s"] > 0

        run_dir = tmp_path / "runs" / "smoke_test"
        run_dirs = sorted(run_dir.glob("*"))
        assert len(run_dirs) > 0
        actual_run_dir = run_dirs[-1]

        assert (actual_run_dir / "config.yaml").exists()
        assert (actual_run_dir / "meta.json").exists()
        assert (actual_run_dir / "train.log").exists()
        assert (actual_run_dir / "history.csv").exists()
        assert (actual_run_dir / "history.json").exists()
        assert (actual_run_dir / "summary.json").exists()
        assert (actual_run_dir / "checkpoints" / "best.pt").exists()
        assert (actual_run_dir / "checkpoints" / "last.pt").exists()
        assert (actual_run_dir / "val_predictions.csv").exists()

        history_df = pd.read_csv(actual_run_dir / "history.csv")
        assert len(history_df) == 2
        assert list(history_df["stage"].values) == [1, 2]

        best_ckpt = load_checkpoint(actual_run_dir / "checkpoints" / "best.pt")
        assert best_ckpt["format_version"] == 1
        assert best_ckpt["kind"] == "best"
        assert best_ckpt["optimizer_state"] is None
        assert "model_state" in best_ckpt

        model = build_model(best_ckpt["model"], pretrained=False)
        model.load_state_dict(best_ckpt["model_state"])
        assert model is not None

    def test_resume_from_checkpoint(self, tiny_split, tmp_path):
        """Test resuming training from last.pt checkpoint."""
        split_csv = tiny_split["split_csv"]

        config_path = tmp_path / "config.yaml"
        config = {
            "exp_name": "smoke_resume",
            "seed": 42,
            "device": "cpu",
            "runs_root": str(tmp_path / "runs"),
            "data": {
                "config": str(tmp_path / "data.yaml"),
                "processed_root": str(tmp_path / "processed"),
                "dataset": "synth",
                "split_csv": str(split_csv),
                "method": "none",
                "cache_size": 64,
                "img_size": 64,
                "cache_format": "png",
                "batch_size": 4,
                "eval_batch_size": 4,
                "num_workers": 0,
                "aug": {
                    "rotate_limit": 45,
                    "rotate_p": 0.5,
                    "rrc_scale": [0.9, 1.0],
                    "brightness": 0.1,
                    "contrast": 0.1,
                    "hue": 5,
                    "sat": 5,
                    "color_p": 0.5,
                },
            },
            "model": {
                "backbone": "resnet18",
                "pretrained": False,
                "head": "corn",
                "num_classes": 5,
                "dropout": 0.1,
                "drop_path_rate": 0.0,
                "channels_last": False,
            },
            "imbalance": {
                "class_weight_power": 0.0,
                "sampler": {"enabled": False, "power": 0.5},
                "weights_with_sampler": False,
                "label_smoothing": 0.0,
            },
            "train": {
                "epochs": 2,
                "warmup_epochs": 1,
                "accum_steps": 1,
                "clip_grad_norm": 1.0,
                "freeze_bn_stats_stage1": True,
                "limit_train_batches": 0,
                "limit_val_batches": 0,
                "log_every_steps": 10,
            },
            "optim": {
                "backbone_lr": 1e-4,
                "head_lr_mult": 10,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "lr_warmup_epochs": 1,
                "min_lr_ratio": 0.01,
            },
            "amp": {"dtype": "off"},
            "ema": {"enabled": True, "decay": 0.9995},
            "early_stopping": {"metric": "val_qwk", "patience": 3, "min_delta": 0.0},
        }

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        preprocess_config = {"circle_crop": {}, "ben_graham": {}, "clahe_l": {}}
        with open(tmp_path / "data.yaml", "w") as f:
            yaml.dump({"preprocess": preprocess_config}, f)

        device = torch.device("cpu")
        run_dir = tmp_path / "runs" / "smoke_resume" / "000000-000000"
        trainer1 = Trainer(config, run_dir, device)
        trainer1.fit()

        history1 = pd.read_csv(run_dir / "history.csv")
        assert len(history1) == 2

        config["train"]["epochs"] = 3
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        trainer2 = Trainer(config, run_dir, device)
        trainer2.fit(resume_from=run_dir / "checkpoints" / "last.pt")

        history2 = pd.read_csv(run_dir / "history.csv")
        assert len(history2) == 3

    def test_nan_loss_guard(self, tiny_split, tmp_path):
        """Test that NaN loss causes step to be skipped."""
        split_csv = tiny_split["split_csv"]

        config = {
            "exp_name": "smoke_nan",
            "seed": 42,
            "device": "cpu",
            "runs_root": str(tmp_path / "runs"),
            "data": {
                "config": str(tmp_path / "data.yaml"),
                "processed_root": str(tmp_path / "processed"),
                "dataset": "synth",
                "split_csv": str(split_csv),
                "method": "none",
                "cache_size": 64,
                "img_size": 64,
                "cache_format": "png",
                "batch_size": 4,
                "eval_batch_size": 4,
                "num_workers": 0,
                "aug": {
                    "rotate_limit": 45,
                    "rotate_p": 0.5,
                    "rrc_scale": [0.9, 1.0],
                    "brightness": 0.1,
                    "contrast": 0.1,
                    "hue": 5,
                    "sat": 5,
                    "color_p": 0.5,
                },
            },
            "model": {
                "backbone": "resnet18",
                "pretrained": False,
                "head": "corn",
                "num_classes": 5,
                "dropout": 0.1,
                "drop_path_rate": 0.0,
                "channels_last": False,
            },
            "imbalance": {
                "class_weight_power": 0.0,
                "sampler": {"enabled": False, "power": 0.5},
                "weights_with_sampler": False,
                "label_smoothing": 0.0,
            },
            "train": {
                "epochs": 1,
                "warmup_epochs": 0,
                "accum_steps": 1,
                "clip_grad_norm": 1.0,
                "freeze_bn_stats_stage1": True,
                "limit_train_batches": 2,
                "limit_val_batches": 0,
                "log_every_steps": 1,
            },
            "optim": {
                "backbone_lr": 1e-4,
                "head_lr_mult": 10,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "lr_warmup_epochs": 0,
                "min_lr_ratio": 0.01,
            },
            "amp": {"dtype": "off"},
            "ema": {"enabled": False},
            "early_stopping": {"metric": "val_qwk", "patience": 1, "min_delta": 0.0},
        }

        preprocess_config = {"circle_crop": {}, "ben_graham": {}, "clahe_l": {}}
        with open(tmp_path / "data.yaml", "w") as f:
            yaml.dump({"preprocess": preprocess_config}, f)

        device = torch.device("cpu")
        run_dir = tmp_path / "runs" / "smoke_nan" / "000000-000000"
        trainer = Trainer(config, run_dir, device)

        original_criterion = trainer.criterion
        call_count = [0]

        def mock_criterion(logits, labels):
            call_count[0] += 1
            if call_count[0] == 2:
                return torch.tensor(float("nan"), requires_grad=True)
            return original_criterion(logits, labels)

        with patch.object(trainer, "criterion", side_effect=mock_criterion):
            trainer.fit()

        history = pd.read_csv(run_dir / "history.csv")
        assert history["skipped_steps"].iloc[0] > 0

    def test_all_configs_load(self):
        """Test that all experiment configs load successfully."""
        configs_dir = Path(__file__).parent.parent / "configs" / "train"

        for config_file in sorted(configs_dir.glob("*.yaml")):
            cfg = load_train_config(config_file)
            assert isinstance(cfg, dict)
            assert "model" in cfg
            assert "train" in cfg
            assert "optim" in cfg

    def test_config_override_unknown_key(self, tiny_split, tmp_path):
        """Test that unknown config overrides raise KeyError."""
        cache_root = tiny_split["cache_root"]

        config_path = tmp_path / "config.yaml"
        config = {
            "exp_name": "test",
            "seed": 42,
            "device": "cpu",
            "data": {
                "processed_root": str(cache_root.parent.parent),
                "dataset": "synth",
                "img_size": 64,
            },
            "model": {"head": "corn", "num_classes": 5},
            "train": {"epochs": 1, "warmup_epochs": 1},
            "optim": {"backbone_lr": 1e-4},
            "amp": {},
            "ema": {},
        }

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with pytest.raises(KeyError):
            load_train_config(config_path, ["unknown_key=value"])

        with pytest.raises(KeyError):
            load_train_config(config_path, ["model.unknown_key=value"])
