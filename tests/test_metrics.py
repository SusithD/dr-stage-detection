"""Tests for metrics."""

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from drstage.data.registry import CLASS_NAMES
from drstage.evaluation.metrics import compute_metrics, quadratic_weighted_kappa


class TestQuadraticWeightedKappa:
    """Tests for QWK computation."""

    def test_qwk_perfect_agreement(self):
        """QWK = 1.0 for perfect predictions."""
        y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])

        qwk = quadratic_weighted_kappa(y_true, y_pred, num_classes=5)

        assert abs(qwk - 1.0) < 1e-6

    def test_qwk_matches_sklearn(self):
        """QWK matches sklearn implementation on random data."""
        np.random.seed(42)
        y_true = np.random.randint(0, 5, 100)
        y_pred = np.random.randint(0, 5, 100)

        qwk = quadratic_weighted_kappa(y_true, y_pred, num_classes=5)
        sklearn_qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=list(range(5)))

        assert abs(qwk - sklearn_qwk) < 1e-10

    def test_qwk_constant_predictions(self):
        """QWK = 0.0 when all predictions are constant."""
        y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        qwk = quadratic_weighted_kappa(y_true, y_pred, num_classes=5)

        assert qwk == 0.0

    def test_qwk_binary_labels(self):
        """QWK works with binary task (labels 0,1)."""
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])

        qwk = quadratic_weighted_kappa(y_true, y_pred, num_classes=2)
        sklearn_qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=[0, 1])

        assert abs(qwk - sklearn_qwk) < 1e-10


class TestComputeMetrics:
    """Tests for comprehensive metrics computation."""

    def test_compute_metrics_perfect_predictions(self):
        """Metrics for perfect predictions."""
        y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        class_probs = np.zeros((10, 5))
        for i in range(10):
            class_probs[i, y_pred[i]] = 1.0
        prob_dr = 1.0 - class_probs[:, 0]

        metrics = compute_metrics(y_true, y_pred, class_probs, prob_dr, CLASS_NAMES)

        assert metrics["multiclass"]["accuracy"] == 1.0
        assert metrics["multiclass"]["qwk"] == 1.0
        assert metrics["multiclass"]["macro_f1"] == 1.0
        assert metrics["binary"]["sensitivity"] == 1.0
        assert metrics["binary"]["specificity"] == 1.0
        assert metrics["referable"]["sensitivity"] == 1.0
        assert metrics["n"] == 10

    def test_compute_metrics_return_structure(self):
        """compute_metrics returns required keys."""
        y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4, 1, 2, 3, 4, 0])
        class_probs = np.random.dirichlet([1, 1, 1, 1, 1], 10)
        prob_dr = 1.0 - class_probs[:, 0]

        metrics = compute_metrics(y_true, y_pred, class_probs, prob_dr, CLASS_NAMES)

        assert "multiclass" in metrics
        assert "binary" in metrics
        assert "referable" in metrics
        assert "n" in metrics

        multiclass = metrics["multiclass"]
        assert "accuracy" in multiclass
        assert "qwk" in multiclass
        assert "macro_f1" in multiclass
        assert "weighted_f1" in multiclass
        assert "macro_precision" in multiclass
        assert "macro_recall" in multiclass
        assert "per_class" in multiclass
        assert "confusion" in multiclass
        assert "confusion_normalized" in multiclass

        assert len(multiclass["per_class"]) == 5
        assert "name" in multiclass["per_class"][0]
        assert "precision" in multiclass["per_class"][0]
        assert "recall" in multiclass["per_class"][0]
        assert "f1" in multiclass["per_class"][0]
        assert "support" in multiclass["per_class"][0]

    def test_compute_metrics_binary_section(self):
        """Binary metrics section computed correctly."""
        y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        y_pred = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        class_probs = np.zeros((10, 5))
        for i in range(10):
            class_probs[i, y_pred[i]] = 1.0
        prob_dr = 1.0 - class_probs[:, 0]

        metrics = compute_metrics(y_true, y_pred, class_probs, prob_dr, CLASS_NAMES)

        binary = metrics["binary"]
        assert binary["threshold"] == 0.5
        assert binary["sensitivity"] == 1.0
        assert binary["specificity"] == 1.0
        assert binary["precision"] == 1.0
        assert "roc_auc" in binary
        assert "confusion" in binary
        assert "roc" in binary

    def test_compute_metrics_binary_sensitivity_specificity(self):
        """Binary sensitivity and specificity computed correctly."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 1, 1, 0, 1])
        class_probs = np.zeros((8, 5))
        for i in range(8):
            class_probs[i, y_pred[i]] = 1.0
        prob_dr = 1.0 - class_probs[:, 0]

        metrics = compute_metrics(y_true, y_pred, class_probs, prob_dr, CLASS_NAMES)

        binary = metrics["binary"]
        tp = 3
        tn = 3
        fp = 1
        fn = 1
        expected_sensitivity = tp / (tp + fn)
        expected_specificity = tn / (tn + fp)

        assert abs(binary["sensitivity"] - expected_sensitivity) < 1e-6
        assert abs(binary["specificity"] - expected_specificity) < 1e-6

    def test_compute_metrics_referable_section(self):
        """Referable metrics (y>=2) computed correctly."""
        y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        y_pred = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        class_probs = np.zeros((10, 5))
        for i in range(10):
            class_probs[i, y_pred[i]] = 1.0
        prob_dr = 1.0 - class_probs[:, 0]

        metrics = compute_metrics(y_true, y_pred, class_probs, prob_dr, CLASS_NAMES)

        referable = metrics["referable"]
        assert "sensitivity" in referable
        assert "specificity" in referable
        assert "roc_auc" in referable
        assert referable["threshold"] == 0.5

    def test_compute_metrics_json_serializable(self):
        """All metrics values are JSON-serializable (plain Python types)."""
        import json

        y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        class_probs = np.zeros((10, 5))
        for i in range(10):
            class_probs[i, y_pred[i]] = 1.0
        prob_dr = 1.0 - class_probs[:, 0]

        metrics = compute_metrics(y_true, y_pred, class_probs, prob_dr, CLASS_NAMES)

        try:
            json_str = json.dumps(metrics)
            assert json_str
        except (TypeError, ValueError) as e:
            pytest.fail(f"Metrics not JSON-serializable: {e}")

    def test_compute_metrics_confusion_matrix_shape(self):
        """Confusion matrix has shape [5, 5]."""
        y_true = np.random.randint(0, 5, 50)
        y_pred = np.random.randint(0, 5, 50)
        class_probs = np.random.dirichlet([1, 1, 1, 1, 1], 50)
        prob_dr = 1.0 - class_probs[:, 0]

        metrics = compute_metrics(y_true, y_pred, class_probs, prob_dr, CLASS_NAMES)

        cm = metrics["multiclass"]["confusion"]
        assert len(cm) == 5
        assert all(len(row) == 5 for row in cm)

    def test_compute_metrics_per_class_counts_match(self):
        """Per-class support values match confusion matrix row sums."""
        y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        y_pred = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 0])
        class_probs = np.random.dirichlet([1, 1, 1, 1, 1], 10)
        prob_dr = 1.0 - class_probs[:, 0]

        metrics = compute_metrics(y_true, y_pred, class_probs, prob_dr, CLASS_NAMES)

        cm = np.array(metrics["multiclass"]["confusion"])
        support = [p["support"] for p in metrics["multiclass"]["per_class"]]
        cm_row_sums = cm.sum(axis=1).tolist()

        assert support == cm_row_sums

    def test_compute_metrics_with_single_class_in_true(self):
        """Metrics handle data with only one true class."""
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([0, 0, 1, 0, 0])
        class_probs = np.zeros((5, 5))
        class_probs[:, y_pred[0]] = 1.0
        prob_dr = 1.0 - class_probs[:, 0]

        metrics = compute_metrics(y_true, y_pred, class_probs, prob_dr, CLASS_NAMES)

        assert metrics["n"] == 5
        assert "accuracy" in metrics["multiclass"]
