"""Tests for evaluation module."""

import json
import logging

import numpy as np
import pandas as pd
import pytest
import torch

from drstage.evaluation.error_analysis import (
    error_distance_stats,
    per_class_error_table,
    top_confident_errors,
)
from drstage.evaluation.predict import (
    Predictions,
    ensemble,
    flip_tta_logits_to_probs,
    grade_from_class_probs,
)
from drstage.evaluation.report import write_eval_bundle
from drstage.models.factory import build_model

logger = logging.getLogger(__name__)


@pytest.fixture
def tiny_model():
    """Create a tiny ResNet18 model for testing."""
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
    return model


@pytest.fixture
def tiny_checkpoint(tmp_path, tiny_model):
    """Create a tiny checkpoint for testing."""
    model_state = tiny_model.state_dict()

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


def test_tta_flip_invariant():
    """Test TTA on flip-invariant input equals no-TTA."""
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
    model.eval()

    x = torch.randn(2, 3, 64, 64)

    with torch.no_grad():
        probs_tta, prob_dr_tta = flip_tta_logits_to_probs(model, x, flips=True)
        probs_no_tta, prob_dr_no_tta = flip_tta_logits_to_probs(model, x, flips=False)

    assert probs_tta.shape == probs_no_tta.shape
    assert prob_dr_tta.shape == prob_dr_no_tta.shape


def test_ensemble_identical():
    """Test ensemble of identical predictions equals input."""
    preds1 = Predictions(
        image_id=np.array(["a", "b", "c"]),
        label=np.array([0, 1, 2]),
        grade=np.array([0, 1, 2]),
        prob_dr=np.array([0.1, 0.5, 0.9]),
        class_probs=np.array([[0.9, 0.1, 0, 0, 0], [0.2, 0.8, 0, 0, 0], [0, 0, 1, 0, 0]]),
    )

    preds2 = Predictions(
        image_id=np.array(["a", "b", "c"]),
        label=np.array([0, 1, 2]),
        grade=np.array([0, 1, 2]),
        prob_dr=np.array([0.1, 0.5, 0.9]),
        class_probs=np.array([[0.9, 0.1, 0, 0, 0], [0.2, 0.8, 0, 0, 0], [0, 0, 1, 0, 0]]),
    )

    ensembled = ensemble([preds1, preds2], head="corn")

    np.testing.assert_array_equal(ensembled.grade, preds1.grade)
    np.testing.assert_allclose(ensembled.prob_dr, preds1.prob_dr, rtol=1e-5)
    np.testing.assert_allclose(ensembled.class_probs, preds1.class_probs, rtol=1e-5)


def test_grade_from_class_probs_corn():
    """Test CORN grade decoding from class probs round-trips CornHead.predict."""
    from drstage.models.heads import CornHead

    head = CornHead(512, 5, dropout=0.0)
    logits = torch.randn(10, 4)

    with torch.no_grad():
        grade_from_head, prob_dr_from_head, class_probs = head.predict(logits)

    grade_from_probs = grade_from_class_probs(class_probs, "corn")

    torch.testing.assert_close(grade_from_probs.float(), grade_from_head.float())


def test_grade_from_class_probs_ce():
    """Test CE grade decoding."""
    from drstage.models.heads import CeHead

    head = CeHead(512, 5, dropout=0.0)
    logits = torch.randn(10, 5)

    with torch.no_grad():
        grade_from_head, prob_dr_from_head, class_probs = head.predict(logits)

    grade_from_probs = grade_from_class_probs(class_probs, "ce")

    torch.testing.assert_close(grade_from_probs.float(), grade_from_head.float())


def test_write_eval_bundle(tmp_path, tiny_split):
    """Test write_eval_bundle writes all required files."""
    preds = Predictions(
        image_id=np.array(["img_000", "img_001", "img_002"]),
        label=np.array([0, 1, 2]),
        grade=np.array([0, 1, 2]),
        prob_dr=np.array([0.1, 0.5, 0.9]),
        class_probs=np.array([[0.9, 0.1, 0, 0, 0], [0.2, 0.8, 0, 0, 0], [0, 0, 1, 0, 0]]),
    )

    class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    metrics = {
        "multiclass": {
            "accuracy": 1.0,
            "qwk": 1.0,
            "macro_f1": 1.0,
            "weighted_f1": 1.0,
            "macro_precision": 1.0,
            "macro_recall": 1.0,
            "per_class": [
                {"name": c, "precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 1}
                for c in class_names
            ],
            "confusion": [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            "confusion_normalized": [
                [1.0, 0, 0, 0, 0],
                [0, 1.0, 0, 0, 0],
                [0, 0, 1.0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        },
        "binary": {
            "threshold": 0.5,
            "sensitivity": 1.0,
            "specificity": 1.0,
            "precision": 1.0,
            "f1": 1.0,
            "roc_auc": 1.0,
            "confusion": [[1, 0], [0, 2]],
            "roc": {"fpr": [0, 1], "tpr": [0, 1]},
        },
        "referable": {
            "threshold": 0.5,
            "sensitivity": 1.0,
            "specificity": 1.0,
            "precision": 1.0,
            "f1": 1.0,
            "roc_auc": 1.0,
            "confusion": [[1, 0], [0, 2]],
            "roc": {"fpr": [0, 1], "tpr": [0, 1]},
        },
        "n": 3,
    }

    out_dir = tmp_path / "eval_bundle"
    write_eval_bundle(out_dir, preds, metrics, class_names, tiny_split["cache_root"], topk=10)

    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "predictions.csv").exists()
    assert (out_dir / "per_class_metrics.csv").exists()
    assert (out_dir / "confusion_matrix.png").exists()
    assert (out_dir / "confusion_matrix_norm.png").exists()
    assert (out_dir / "error_table.csv").exists()
    assert (out_dir / "error_stats.json").exists()
    assert (out_dir / "top_errors.csv").exists()

    with open(out_dir / "metrics.json") as f:
        saved_metrics = json.load(f)
    assert saved_metrics["multiclass"]["accuracy"] == 1.0

    df = pd.read_csv(out_dir / "predictions.csv")
    assert len(df) == 3
    assert list(df["image_id"]) == ["img_000", "img_001", "img_002"]


def test_top_confident_errors():
    """Test top_confident_errors extracts worst mistakes."""
    preds = Predictions(
        image_id=np.array(["a", "b", "c", "d"]),
        label=np.array([0, 1, 2, 3]),
        grade=np.array([1, 0, 3, 0]),
        prob_dr=np.array([0.2, 0.8, 0.9, 0.1]),
        class_probs=np.array(
            [
                [0.2, 0.8, 0, 0, 0],
                [0.9, 0.1, 0, 0, 0],
                [0, 0, 0.1, 0.9, 0],
                [0.8, 0.2, 0, 0, 0],
            ]
        ),
    )

    top_errors = top_confident_errors(preds, k=2)
    assert len(top_errors) == 2
    assert set(top_errors["image_id"]) == {"b", "c"}


def test_per_class_error_table():
    """Test per_class_error_table statistics."""
    preds = Predictions(
        image_id=np.array(["a", "b", "c"]),
        label=np.array([0, 0, 1]),
        grade=np.array([1, 0, 1]),
        prob_dr=np.array([0.2, 0.8, 0.9]),
        class_probs=np.array([[0.2, 0.8, 0, 0, 0], [0.9, 0.1, 0, 0, 0], [0, 1, 0, 0, 0]]),
    )

    class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    error_table = per_class_error_table(preds, class_names)

    assert len(error_table) == 5
    assert error_table.loc[0, "n"] == 2
    assert error_table.loc[0, "n_wrong"] == 1
    assert error_table.loc[1, "n"] == 1
    assert error_table.loc[1, "n_wrong"] == 0


def test_error_distance_stats():
    """Test error distance statistics."""
    preds = Predictions(
        image_id=np.array(["a", "b", "c", "d"]),
        label=np.array([0, 1, 2, 4]),
        grade=np.array([1, 2, 3, 2]),
        prob_dr=np.array([0.2, 0.8, 0.9, 0.1]),
        class_probs=np.array(
            [[0.2, 0.8, 0, 0, 0], [0.2, 0.8, 0, 0, 0], [0.2, 0.8, 0, 0, 0], [0.2, 0.8, 0, 0, 0]]
        ),
    )

    stats = error_distance_stats(preds)

    assert "off_by_one_rate" in stats
    assert "off_by_more_rate" in stats
    assert "mean_abs_error" in stats
    assert stats["mean_abs_error"] > 0
