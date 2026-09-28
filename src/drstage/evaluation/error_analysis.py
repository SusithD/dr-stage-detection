"""Diagnostic breakdowns of prediction errors by type and class."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def top_confident_errors(pred, k: int) -> pd.DataFrame:
    """Find the k incorrect predictions with the highest predicted confidence."""
    mask = pred.grade != pred.label
    wrong_indices = np.where(mask)[0]

    if len(wrong_indices) == 0:
        return pd.DataFrame(columns=["image_id", "label", "pred", "confidence", "prob_dr"])

    confidences = pred.class_probs[wrong_indices].max(axis=1)
    top_k_indices = wrong_indices[np.argsort(-confidences)[:k]]

    return pd.DataFrame(
        {
            "image_id": pred.image_id[top_k_indices],
            "label": pred.label[top_k_indices],
            "pred": pred.grade[top_k_indices],
            "confidence": pred.class_probs[top_k_indices].max(axis=1),
            "prob_dr": pred.prob_dr[top_k_indices],
        }
    )


def per_class_error_table(pred, class_names) -> pd.DataFrame:
    """Compute error rates, off-by-one rates and directional errors for each class."""
    rows = []

    for class_idx in range(5):
        mask = pred.label == class_idx
        if mask.sum() == 0:
            rows.append(
                {
                    "class": class_names[class_idx],
                    "n": 0,
                    "n_wrong": 0,
                    "error_rate": 0.0,
                    "under_graded": 0,
                    "over_graded": 0,
                    "off_by_one": 0,
                    "off_by_more": 0,
                }
            )
            continue

        class_indices = np.where(mask)[0]
        n_total = len(class_indices)

        wrong_mask = pred.grade[class_indices] != pred.label[class_indices]
        n_wrong = wrong_mask.sum()

        under_graded = (pred.grade[class_indices] < pred.label[class_indices]).sum()
        over_graded = (pred.grade[class_indices] > pred.label[class_indices]).sum()

        abs_diff = np.abs(pred.grade[class_indices] - pred.label[class_indices])
        off_by_one = (abs_diff == 1).sum()
        off_by_more = (abs_diff > 1).sum()

        rows.append(
            {
                "class": class_names[class_idx],
                "n": n_total,
                "n_wrong": int(n_wrong),
                "error_rate": float(n_wrong / n_total) if n_total > 0 else 0.0,
                "under_graded": int(under_graded),
                "over_graded": int(over_graded),
                "off_by_one": int(off_by_one),
                "off_by_more": int(off_by_more),
            }
        )

    return pd.DataFrame(rows)


def error_distance_stats(pred) -> dict:
    """Compute the fraction of errors that are off-by-one vs larger gaps."""
    mask = pred.grade != pred.label
    if mask.sum() == 0:
        return {
            "off_by_one_rate": 0.0,
            "off_by_more_rate": 0.0,
            "mean_abs_error": 0.0,
        }

    error_indices = np.where(mask)[0]
    abs_diff = np.abs(pred.grade[error_indices] - pred.label[error_indices])

    off_by_one = (abs_diff == 1).sum()
    off_by_more = (abs_diff > 1).sum()
    n_errors = len(error_indices)

    return {
        "off_by_one_rate": float(off_by_one / n_errors) if n_errors > 0 else 0.0,
        "off_by_more_rate": float(off_by_more / n_errors) if n_errors > 0 else 0.0,
        "mean_abs_error": float(abs_diff.mean()),
    }
