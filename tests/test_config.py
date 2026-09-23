"""Tests for config, seed, and logging utilities."""

import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from drstage.utils.config import load_config, save_config
from drstage.utils.logging import get_logger
from drstage.utils.seed import seed_everything


class TestConfig:
    """Tests for load_config and save_config."""

    def test_load_basic_yaml(self, tmp_path: Path) -> None:
        """Load a basic YAML file."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("key: value\nnumber: 42\n")

        cfg = load_config(yaml_file)

        assert cfg["key"] == "value"
        assert cfg["number"] == 42

    def test_load_nested_yaml(self, tmp_path: Path) -> None:
        """Load YAML with nested structures."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("outer:\n  inner: value\n  number: 100\n")

        cfg = load_config(yaml_file)

        assert cfg["outer"]["inner"] == "value"
        assert cfg["outer"]["number"] == 100

    def test_override_int_value(self, tmp_path: Path) -> None:
        """Override int value via command line."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("preprocess:\n  size: 512\n")

        cfg = load_config(yaml_file, overrides=["preprocess.size=384"])

        assert cfg["preprocess"]["size"] == 384
        assert isinstance(cfg["preprocess"]["size"], int)

    def test_override_bool_true(self, tmp_path: Path) -> None:
        """Override to bool True."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("flag: false\n")

        cfg = load_config(yaml_file, overrides=["flag=true"])

        assert cfg["flag"] is True

    def test_override_bool_false(self, tmp_path: Path) -> None:
        """Override to bool False."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("flag: true\n")

        cfg = load_config(yaml_file, overrides=["flag=false"])

        assert cfg["flag"] is False

    def test_override_float(self, tmp_path: Path) -> None:
        """Override to float."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("threshold: 0.5\n")

        cfg = load_config(yaml_file, overrides=["threshold=0.75"])

        assert cfg["threshold"] == 0.75
        assert isinstance(cfg["threshold"], float)

    def test_override_string(self, tmp_path: Path) -> None:
        """Override to string."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("name: old\n")

        cfg = load_config(yaml_file, overrides=["name=new"])

        assert cfg["name"] == "new"

    def test_override_nested_value(self, tmp_path: Path) -> None:
        """Override nested value."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("outer:\n  inner: 10\n")

        cfg = load_config(yaml_file, overrides=["outer.inner=20"])

        assert cfg["outer"]["inner"] == 20

    def test_override_unknown_key_raises(self, tmp_path: Path) -> None:
        """Unknown key path raises KeyError."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("existing: 1\n")

        with pytest.raises(KeyError):
            load_config(yaml_file, overrides=["nonexistent=1"])

    def test_override_unknown_nested_key_raises(self, tmp_path: Path) -> None:
        """Unknown nested key path raises KeyError."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("outer:\n  inner: 1\n")

        with pytest.raises(KeyError):
            load_config(yaml_file, overrides=["outer.nonexistent=1"])

    def test_save_config(self, tmp_path: Path) -> None:
        """Save config to YAML."""
        cfg = {"key": "value", "number": 42, "nested": {"inner": True}}
        out_path = tmp_path / "out.yaml"

        save_config(cfg, out_path)

        assert out_path.exists()
        reloaded = load_config(out_path)
        assert reloaded == cfg

    def test_multiple_overrides(self, tmp_path: Path) -> None:
        """Apply multiple overrides."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("a: 1\nb: 2\nc:\n  d: 3\n")

        cfg = load_config(yaml_file, overrides=["a=10", "c.d=30"])

        assert cfg["a"] == 10
        assert cfg["b"] == 2
        assert cfg["c"]["d"] == 30


class TestSeed:
    """Tests for seed_everything."""

    def test_numpy_reproducibility(self) -> None:
        """Same seed produces same numpy arrays."""
        seed_everything(42)
        arr1 = np.random.randn(5)

        seed_everything(42)
        arr2 = np.random.randn(5)

        np.testing.assert_array_equal(arr1, arr2)

    def test_torch_reproducibility(self) -> None:
        """Same seed produces same torch tensors."""
        seed_everything(42)
        t1 = torch.randn(5)

        seed_everything(42)
        t2 = torch.randn(5)

        assert torch.allclose(t1, t2)

    def test_different_seeds_different_values(self) -> None:
        """Different seeds produce different values."""
        seed_everything(42)
        arr1 = np.random.randn(5)

        seed_everything(43)
        arr2 = np.random.randn(5)

        assert not np.allclose(arr1, arr2)

    def test_deterministic_flag(self) -> None:
        """Deterministic flag doesn't raise errors."""
        seed_everything(42, deterministic=True)
        seed_everything(42, deterministic=False)


class TestLogging:
    """Tests for get_logger."""

    def test_get_logger_basic(self) -> None:
        """Create logger without file."""
        logger = get_logger("test_logger")

        assert logger.name == "test_logger"
        assert len(logger.handlers) > 0

    def test_get_logger_idempotent(self) -> None:
        """Calling get_logger twice doesn't duplicate handlers."""
        logger_name = "idempotent_test"
        logger1 = get_logger(logger_name)
        n_handlers_1 = len(logger1.handlers)

        logger2 = get_logger(logger_name)
        n_handlers_2 = len(logger2.handlers)

        assert logger1 is logger2
        assert n_handlers_1 == n_handlers_2

    def test_get_logger_with_file(self, tmp_path: Path) -> None:
        """Create logger with file output."""
        log_file = tmp_path / "test.log"
        logger = get_logger("file_test", log_file=log_file)

        logger.info("Test message")

        assert log_file.exists()
        assert "Test message" in log_file.read_text()

    def test_get_logger_format(self, tmp_path: Path) -> None:
        """Logger format includes asctime, levelname, name."""
        log_file = tmp_path / "format_test.log"
        logger = get_logger("format_test", log_file=log_file)

        logger.info("Format test")

        content = log_file.read_text()
        assert "format_test" in content
        assert "INFO" in content

    def test_get_logger_custom_level(self, tmp_path: Path) -> None:
        """Logger respects custom level."""
        log_file = tmp_path / "level_test.log"
        logger = get_logger("level_test", log_file=log_file, level=logging.WARNING)

        logger.info("Should not appear")
        logger.warning("Should appear")

        content = log_file.read_text()
        assert "Should not appear" not in content
        assert "Should appear" in content
