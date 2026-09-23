"""Write evaluation bundles and comparison files."""

import json
import logging
from pathlib import Path

import pandas as pd

from drstage.evaluation.error_analysis import (
    error_distance_stats,
    per_class_error_table,
    top_confident_errors,
)
from drstage.evaluation.plots import (
    plot_confusion_matrix,
    plot_error_gallery,
    plot_roc,
)

logger = logging.getLogger(__name__)


def write_eval_bundle(
    out_dir: Path,
    pred,
    metrics: dict,
    class_names,
    cache_root: Path,
    topk: int = 20,
) -> None:
    """Write comprehensive evaluation bundle.

    Args:
        out_dir: Output directory.
        pred: Predictions object.
        metrics: Dict from compute_metrics.
        class_names: List of class names.
        cache_root: Root of cached images (for error gallery).
        topk: Number of top errors to show.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Wrote metrics.json")

    pred_df = pred.to_frame()
    pred_df = pred_df.sort_values("image_id").reset_index(drop=True)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    logger.info(f"Wrote predictions.csv ({len(pred_df)} samples)")

    per_class = per_class_error_table(pred, class_names)
    per_class.to_csv(out_dir / "per_class_metrics.csv", index=False)
    logger.info("Wrote per_class_metrics.csv")

    cm = metrics["multiclass"]["confusion"]
    cm = __import__("numpy").array(cm)
    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png", normalized=False)
    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix_norm.png", normalized=True)
    logger.info("Wrote confusion matrices")

    binary_metrics = metrics["binary"]
    if binary_metrics.get("roc", {}).get("fpr"):
        plot_roc(
            binary_metrics["roc"],
            binary_metrics["roc_auc"],
            "ROC (Binary)",
            out_dir / "roc_binary.png",
        )

    referable_metrics = metrics["referable"]
    if referable_metrics.get("roc", {}).get("fpr"):
        plot_roc(
            referable_metrics["roc"],
            referable_metrics["roc_auc"],
            "ROC (Referable)",
            out_dir / "roc_referable.png",
        )
    logger.info("Wrote ROC curves")

    error_table = per_class_error_table(pred, class_names)
    error_table.to_csv(out_dir / "error_table.csv", index=False)

    error_stats = error_distance_stats(pred)
    with open(out_dir / "error_stats.json", "w") as f:
        json.dump(error_stats, f, indent=2)
    logger.info("Wrote error_table.csv and error_stats.json")

    top_errors = top_confident_errors(pred, k=topk)
    top_errors.to_csv(out_dir / "top_errors.csv", index=False)

    if len(top_errors) > 0 and cache_root.exists():
        try:
            plot_error_gallery(top_errors, cache_root, out_dir / "top_errors_grid.png", ncols=5)
            logger.info("Wrote top_errors_grid.png")
        except Exception as e:
            logger.warning(f"Could not create error gallery: {e}")
    logger.info("Wrote top_errors.csv")


def write_comparison(rows: list[dict], out: Path) -> None:
    """Write model comparison CSV.

    Args:
        rows: List of dicts with model, dataset, tta, accuracy, qwk, macro_f1,
            auc, sensitivity, specificity, referable_auc.
        out: Output file path.
    """
    df = pd.DataFrame(rows)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"Wrote comparison.csv ({len(df)} rows)")
