"""Visualization plots for evaluation results."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def _row_normalise(cm: np.ndarray) -> np.ndarray:
    """Divide each confusion-matrix row by its total, leaving empty rows as zeros."""
    totals = cm.sum(axis=1, keepdims=True).astype(np.float32)
    return np.divide(cm, totals, out=np.zeros_like(totals * cm, dtype=np.float32), where=totals > 0)


def plot_confusion_matrix(cm, class_names, out: Path, normalized: bool = False) -> None:
    """Plot confusion matrix.

    Args:
        cm: [K, K] confusion matrix.
        class_names: List of K class names.
        out: Output file path.
        normalized: Whether to normalize by row.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    if normalized:
        cm_plot = _row_normalise(cm)
        im = ax.imshow(cm_plot, cmap="Blues", vmin=0, vmax=1)
    else:
        im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    title = "Confusion Matrix (normalized)" if normalized else "Confusion Matrix"
    ax.set_title(title)

    if normalized:
        cm_norm = _row_normalise(cm)
    else:
        cm_norm = cm.astype(np.float32) / cm.max()

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if normalized:
                text = f"{cm_plot[i, j]:.2f}"
            else:
                text = f"{cm[i, j]}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
            )

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc(roc: dict, auc: float, title: str, out: Path) -> None:
    """Plot ROC curve.

    Args:
        roc: Dict with 'fpr' and 'tpr' lists.
        auc: AUC score.
        title: Plot title.
        out: Output file path.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    fpr = roc.get("fpr", [])
    tpr = roc.get("tpr", [])

    if len(fpr) > 0 and len(tpr) > 0:
        ax.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curves(history, out: Path) -> None:
    """Plot training curves from history DataFrame.

    Args:
        history: DataFrame with train_loss, val_loss, val_acc, val_qwk,
            val_macro_f1, lr_backbone columns.
        out: Output file path.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    if "epoch" not in history.columns:
        epochs = np.arange(len(history))
    else:
        epochs = history["epoch"].values

    ax = axes[0, 0]
    if "train_loss" in history.columns:
        ax.plot(epochs, history["train_loss"], label="Train", linewidth=2)
    if "val_loss" in history.columns:
        ax.plot(epochs, history["val_loss"], label="Val", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if "val_acc" in history.columns:
        ax.plot(epochs, history["val_acc"], label="Accuracy", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if "val_qwk" in history.columns:
        ax.plot(epochs, history["val_qwk"], label="QWK", linewidth=2)
    if "val_macro_f1" in history.columns:
        ax.plot(epochs, history["val_macro_f1"], label="Macro F1", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("QWK and Macro F1")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if "lr_backbone" in history.columns:
        ax.plot(epochs, history["lr_backbone"], label="Backbone LR", linewidth=2)
    if "lr_head" in history.columns:
        ax.plot(epochs, history["lr_head"], label="Head LR", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def plot_error_gallery(rows, cache_root: Path, out: Path, ncols: int = 5) -> None:
    """Plot gallery of error examples.

    Args:
        rows: DataFrame with image_id, label, pred, confidence columns.
        cache_root: Root path to cached images.
        out: Output file path.
        ncols: Number of columns in grid.
    """
    from drstage.data.preprocess import load_rgb

    n_rows = int(np.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(n_rows, ncols, figsize=(ncols * 2, n_rows * 2))

    if n_rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.reshape(n_rows, ncols)

    class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

    for ax, (_, row) in zip(axes.flat, rows.iterrows(), strict=False):
        image_id = row["image_id"]

        try:
            img_path = cache_root / f"{image_id}.png"
            img = load_rgb(img_path)
            ax.imshow(img)
        except Exception as e:
            logger.warning(f"Failed to load {image_id}: {e}")
            ax.text(0.5, 0.5, "Not found", ha="center", va="center")

        label = int(row["label"])
        pred = int(row["pred"])
        conf = row.get("confidence", 0.0)

        title = f"True: {class_names[label]}\nPred: {class_names[pred]} ({conf:.2f})"
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    for idx in range(len(rows), len(axes.flat)):
        axes.flat[idx].axis("off")

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cam_grid(items: list[dict], out: Path, ncols: int = 5) -> None:
    """Plot grid of Grad-CAM overlays.

    Args:
        items: List of dicts with 'image', 'heatmap', 'title'.
        out: Output file path.
        ncols: Number of columns in grid.
    """
    n_items = len(items)
    n_rows = int(np.ceil(n_items / ncols))

    fig, axes = plt.subplots(n_rows, ncols, figsize=(ncols * 2.5, n_rows * 2.5))

    if n_rows == 1:
        axes = axes.reshape(1, -1)
    else:
        axes = axes.reshape(n_rows, ncols)

    for ax, item in zip(axes.flat, items, strict=False):
        img = item["image"]
        heatmap = item["heatmap"]
        title = item["title"]

        ax.imshow(img)
        if heatmap is not None:
            ax.imshow(heatmap, cmap="jet", alpha=0.4)
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    for idx in range(n_items, len(axes.flat)):
        axes.flat[idx].axis("off")

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
