"""Evaluation metrics for DR stage detection."""

import logging
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5) -> float:
    """Compute quadratic weighted kappa (Cohen's kappa with quadratic weights).

    Returns 0.0 if predictions are constant.

    Args:
        y_true: [N] true labels.
        y_pred: [N] predicted labels.
        num_classes: Number of classes.

    Returns:
        QWK score in [-1, 1].
    """
    if len(np.unique(y_pred)) == 1:
        logger.warning("Predictions are constant; returning QWK=0.0")
        return 0.0

    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=list(range(num_classes)))
    return float(kappa)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_probs: np.ndarray,
    prob_dr: np.ndarray,
    class_names: Sequence[str],
) -> dict:
    """Compute comprehensive multiclass, binary, and referable metrics.

    Args:
        y_true: [N] true class labels (0..4).
        y_pred: [N] predicted class labels.
        class_probs: [N, 5] per-class probabilities (must sum to 1).
        prob_dr: [N] probability of any DR (referable; score for binary).
        class_names: Sequence of 5 class names.

    Returns:
        Dict with keys 'multiclass', 'binary', 'referable', 'n'.
    """
    num_classes = 5
    n = len(y_true)

    multiclass_metrics = {}

    accuracy = float((y_pred == y_true).mean())
    multiclass_metrics["accuracy"] = accuracy

    qwk = quadratic_weighted_kappa(y_true, y_pred, num_classes)
    multiclass_metrics["qwk"] = qwk

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    multiclass_metrics["macro_f1"] = macro_f1

    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    multiclass_metrics["weighted_f1"] = weighted_f1

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    multiclass_metrics["macro_precision"] = float(precision)
    multiclass_metrics["macro_recall"] = float(recall)

    per_class_list = []
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    for i in range(num_classes):
        per_class_list.append(
            {
                "name": class_names[i],
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
        )
    multiclass_metrics["per_class"] = per_class_list

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    multiclass_metrics["confusion"] = cm.tolist()

    cm_norm = cm.astype(np.float32)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(row_sums > 0, cm_norm / row_sums, 0)
    multiclass_metrics["confusion_normalized"] = cm_norm.tolist()

    binary_metrics = {}
    binary_metrics["threshold"] = 0.5

    y_binary = (y_true >= 1).astype(int)
    y_pred_binary = (y_pred >= 1).astype(int)

    cm_binary = confusion_matrix(y_binary, y_pred_binary, labels=[0, 1])
    tn, fp, fn, tp = cm_binary.ravel()

    sensitivity = float(tp / (tp + fn) if (tp + fn) > 0 else 0)
    specificity = float(tn / (tn + fp) if (tn + fp) > 0 else 0)
    precision_binary = float(tp / (tp + fp) if (tp + fp) > 0 else 0)
    f1_binary = float(
        2 * (precision_binary * sensitivity) / (precision_binary + sensitivity)
        if (precision_binary + sensitivity) > 0
        else 0
    )

    binary_metrics["sensitivity"] = sensitivity
    binary_metrics["specificity"] = specificity
    binary_metrics["precision"] = precision_binary
    binary_metrics["f1"] = f1_binary

    if len(np.unique(y_binary)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_binary, prob_dr))
        except Exception:
            roc_auc = 0.0
    else:
        roc_auc = 0.0

    binary_metrics["roc_auc"] = roc_auc

    binary_metrics["confusion"] = [[int(tn), int(fp)], [int(fn), int(tp)]]

    if len(np.unique(y_binary)) > 1:
        try:
            fpr, tpr, _ = roc_curve(y_binary, prob_dr)
            binary_metrics["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        except Exception:
            binary_metrics["roc"] = {"fpr": [], "tpr": []}
    else:
        binary_metrics["roc"] = {"fpr": [], "tpr": []}

    referable_metrics = {}
    y_referable = (y_true >= 2).astype(int)
    y_pred_referable = (y_pred >= 2).astype(int)
    referable_score = class_probs[:, 2:].sum(axis=1)

    referable_metrics["threshold"] = 0.5

    cm_ref = confusion_matrix(y_referable, y_pred_referable, labels=[0, 1])
    tn_ref, fp_ref, fn_ref, tp_ref = cm_ref.ravel()

    sensitivity_ref = float(tp_ref / (tp_ref + fn_ref) if (tp_ref + fn_ref) > 0 else 0)
    specificity_ref = float(tn_ref / (tn_ref + fp_ref) if (tn_ref + fp_ref) > 0 else 0)
    precision_ref = float(tp_ref / (tp_ref + fp_ref) if (tp_ref + fp_ref) > 0 else 0)
    f1_ref = float(
        2 * (precision_ref * sensitivity_ref) / (precision_ref + sensitivity_ref)
        if (precision_ref + sensitivity_ref) > 0
        else 0
    )

    referable_metrics["sensitivity"] = sensitivity_ref
    referable_metrics["specificity"] = specificity_ref
    referable_metrics["precision"] = precision_ref
    referable_metrics["f1"] = f1_ref

    if len(np.unique(y_referable)) > 1:
        try:
            roc_auc_ref = float(roc_auc_score(y_referable, referable_score))
        except Exception:
            roc_auc_ref = 0.0
    else:
        roc_auc_ref = 0.0

    referable_metrics["roc_auc"] = roc_auc_ref

    referable_metrics["confusion"] = [[int(tn_ref), int(fp_ref)], [int(fn_ref), int(tp_ref)]]

    if len(np.unique(y_referable)) > 1:
        try:
            fpr_ref, tpr_ref, _ = roc_curve(y_referable, referable_score)
            referable_metrics["roc"] = {"fpr": fpr_ref.tolist(), "tpr": tpr_ref.tolist()}
        except Exception:
            referable_metrics["roc"] = {"fpr": [], "tpr": []}
    else:
        referable_metrics["roc"] = {"fpr": [], "tpr": []}

    return {
        "multiclass": multiclass_metrics,
        "binary": binary_metrics,
        "referable": referable_metrics,
        "n": int(n),
    }
