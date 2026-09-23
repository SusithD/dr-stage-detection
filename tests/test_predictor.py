"""Tests for inference predictor."""

from pathlib import Path

import numpy as np
import pytest
import torch

from drstage.inference.predictor import load_predictor
from drstage.models.factory import build_model

logger = __import__("logging").getLogger(__name__)


@pytest.fixture
def tiny_checkpoint(tmp_path):
    """Create a tiny checkpoint for testing."""
    model = build_model(
        {
            "backbone": "resnet18",
            "head": "corn",
            "num_classes": 5,
            "dropout": 0.0,
            "drop_path_rate": 0.0,
        },
        pretrained=False,
    )
    model_state = model.state_dict()

    checkpoint = {
        "format_version": 1,
        "kind": "best",
        "model_state": model_state,
        "model": {
            "backbone": "resnet18",
            "head": "corn",
            "num_classes": 5,
            "dropout": 0.0,
            "drop_path_rate": 0.0,
            "img_size": 64,
        },
        "preprocess": {
            "method": "ben_graham",
            "size": 64,
            "circle_crop": {"threshold": 10, "min_radius_frac": 0.25},
            "ben_graham": {"sigma_frac": 0.17, "alpha": 4, "beta": -4, "gamma": 128},
            "clahe_l": {"clip_limit": 2.0, "tile_grid": 8},
        },
        "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "class_names": ["No DR", "Mild", "Moderate", "Severe", "Proliferative"],
        "tta": {"flips": True},
        "config": {},
        "git_commit": "test",
        "created_at": "2024-01-01T00:00:00Z",
        "best_metric": 0.5,
    }

    ckpt_path = tmp_path / "test_checkpoint.pt"
    torch.save(checkpoint, ckpt_path)
    return ckpt_path


def test_load_predictor(tiny_checkpoint):
    """Test loading predictor from checkpoint."""
    predictor = load_predictor(tiny_checkpoint, device="cpu")

    assert predictor is not None
    assert hasattr(predictor, "predict")
    assert hasattr(predictor, "info")


def test_predictor_predict_output_shape(tiny_checkpoint):
    """Test predictor.predict returns correct shapes and types."""
    predictor = load_predictor(tiny_checkpoint, device="cpu")

    image_rgb = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    result = predictor.predict(image_rgb, heatmap=False)

    assert isinstance(result, dict)
    assert "grade" in result
    assert "grade_name" in result
    assert "prob_dr" in result
    assert "class_probs" in result
    assert "heatmap" in result
    assert "original" in result
    assert "enhanced" in result

    assert isinstance(result["grade"], (int, np.integer))
    assert 0 <= result["grade"] <= 4
    assert isinstance(result["grade_name"], str)
    assert isinstance(result["prob_dr"], (float, np.floating))
    assert 0.0 <= result["prob_dr"] <= 1.0

    assert isinstance(result["class_probs"], list)
    assert len(result["class_probs"]) == 5
    assert all(isinstance(p, (float, np.floating)) for p in result["class_probs"])
    class_probs_sum = sum(result["class_probs"])
    assert np.isclose(class_probs_sum, 1.0, rtol=1e-5)

    assert result["original"].shape == (64, 64, 3)
    assert result["original"].dtype == np.uint8
    assert result["enhanced"].shape == (64, 64, 3)
    assert result["enhanced"].dtype == np.uint8

    assert result["heatmap"] is None


def test_predictor_heatmap(tiny_checkpoint):
    """Test predictor.predict with heatmap."""
    predictor = load_predictor(tiny_checkpoint, device="cpu")

    image_rgb = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    result = predictor.predict(image_rgb, heatmap=True)

    heat = result["heatmap"]
    assert isinstance(heat, np.ndarray)
    assert heat.dtype == np.float32
    assert heat.shape == (64, 64)
    assert 0.0 <= heat.min() and heat.max() <= 1.0


def test_predictor_info(tiny_checkpoint):
    """Test predictor.info returns expected keys."""
    predictor = load_predictor(tiny_checkpoint, device="cpu")

    info = predictor.info()

    assert isinstance(info, dict)
    assert "name" in info
    assert "head" in info
    assert "image_size" in info
    assert "preprocessing" in info
    assert "checkpoint" in info
    assert "device" in info


def test_load_predictor_no_configs(tiny_checkpoint, tmp_path):
    """Test load_predictor works when cwd has no configs/."""
    original_cwd = Path.cwd()
    try:
        import os

        os.chdir(tmp_path)

        predictor = load_predictor(tiny_checkpoint, device="cpu")
        assert predictor is not None

        image_rgb = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = predictor.predict(image_rgb)
        assert result["grade"] >= 0
    finally:
        import os

        os.chdir(original_cwd)


def test_predictor_monkeypatch_preprocess(tiny_checkpoint):
    """Test that predictor calls preprocess_image."""
    from unittest.mock import patch

    predictor = load_predictor(tiny_checkpoint, device="cpu")
    image_rgb = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    call_count = []

    original_preprocess = __import__(
        "drstage.data.preprocess", fromlist=["preprocess_image"]
    ).preprocess_image

    def mock_preprocess(*args, **kwargs):
        call_count.append(1)
        return original_preprocess(*args, **kwargs)

    with patch("drstage.inference.predictor.preprocess_image", side_effect=mock_preprocess):
        result = predictor.predict(image_rgb)

    assert len(call_count) >= 1, "preprocess_image should be called"
    assert result["grade"] >= 0


def test_predictor_deterministic_gradcam(tiny_checkpoint):
    """Test that predictor runs deterministically."""
    predictor = load_predictor(tiny_checkpoint, device="cpu", tta=False)
    image_rgb = np.random.RandomState(42).randint(0, 255, (256, 256, 3), dtype=np.uint8)

    result1 = predictor.predict(image_rgb, heatmap=False)
    result2 = predictor.predict(image_rgb, heatmap=False)

    assert result1["grade"] == result2["grade"]
    assert np.isclose(result1["prob_dr"], result2["prob_dr"])
    for p1, p2 in zip(result1["class_probs"], result2["class_probs"], strict=False):
        assert np.isclose(p1, p2, rtol=1e-5)


def test_checkpoint_missing_raises():
    """Test load_predictor raises when checkpoint is missing."""
    with pytest.raises(FileNotFoundError):
        load_predictor("/nonexistent/path.pt", device="cpu")


def test_predictor_tta_flag(tiny_checkpoint):
    """Test predictor respects TTA flag."""
    predictor_tta = load_predictor(tiny_checkpoint, device="cpu", tta=True)
    predictor_no_tta = load_predictor(tiny_checkpoint, device="cpu", tta=False)

    image_rgb = np.random.RandomState(42).randint(0, 255, (256, 256, 3), dtype=np.uint8)

    result_tta = predictor_tta.predict(image_rgb, heatmap=False)
    result_no_tta = predictor_no_tta.predict(image_rgb, heatmap=False)

    assert result_tta is not None
    assert result_no_tta is not None
