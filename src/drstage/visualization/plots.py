"""Visualization functions for data analysis and preprocessing."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from drstage.data.preprocess import circle_crop, load_rgb, preprocess_image
from drstage.data.registry import CLASS_NAMES


def plot_class_distribution(frames: dict[str, pd.DataFrame], out: Path) -> None:
    """Show the DR grade distribution across datasets as grouped bar chart.

    Args:
        frames: Dict of dataset_name -> DataFrame with 'label' column.
        out: Output path for PNG.
    """
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    x_pos = np.arange(len(CLASS_NAMES))
    width = 0.35

    for i, (dataset, df) in enumerate(frames.items()):
        counts = df["label"].value_counts().reindex(range(5), fill_value=0)
        percentages = 100 * counts / counts.sum()
        ax.bar(x_pos + i * width, percentages, width, label=dataset)

    ax.set_xlabel("Grade")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Class Distribution by Dataset")
    ax.set_xticks(x_pos + width / 2)
    ax.set_xticklabels(CLASS_NAMES)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_split_distribution(split_df: pd.DataFrame, out: Path) -> None:
    """Show how each DR grade is distributed across train/val/test splits as grouped
    bar chart.

    Args:
        split_df: DataFrame with 'split' and 'label' columns.
        out: Output path for PNG.
    """
    out.parent.mkdir(parents=True, exist_ok=True)

    split_order = ["train", "val", "test"]
    fig, ax = plt.subplots(figsize=(10, 5))

    splits_present = [s for s in split_order if s in split_df["split"].values]
    x_pos = np.arange(len(CLASS_NAMES))
    width = 0.25

    for i, split in enumerate(splits_present):
        split_data = split_df[split_df["split"] == split]
        counts = split_data["label"].value_counts().reindex(range(5), fill_value=0)
        percentages = 100 * counts / len(split_data) if len(split_data) > 0 else counts
        ax.bar(x_pos + i * width, percentages, width, label=split)

    ax.set_xlabel("Grade")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Split Distribution across Classes")
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(CLASS_NAMES)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dup_examples(pairs: list[tuple[str, str, int]], out: Path, max_pairs: int) -> None:
    """Visualize the most similar duplicate pairs side by side.

    Args:
        pairs: List of (path_a, path_b, hamming_distance) tuples, sorted by distance.
        out: Output path for PNG.
        max_pairs: Maximum pairs to display.
    """
    out.parent.mkdir(parents=True, exist_ok=True)

    pairs_to_show = pairs[:max_pairs]
    n_pairs = len(pairs_to_show)

    fig, axes = plt.subplots(n_pairs, 2, figsize=(8, 3 * n_pairs))
    if n_pairs == 1:
        axes = axes.reshape(1, -1)

    for idx, (path_a, path_b, distance) in enumerate(pairs_to_show):
        try:
            img_a = load_rgb(Path(path_a))
            img_b = load_rgb(Path(path_b))
            axes[idx, 0].imshow(img_a)
            axes[idx, 0].set_title("Image A")
            axes[idx, 0].axis("off")
            axes[idx, 1].imshow(img_b)
            axes[idx, 1].set_title(f"Image B (distance={distance})")
            axes[idx, 1].axis("off")
        except Exception:
            axes[idx, 0].text(0.5, 0.5, "Failed to load", ha="center", va="center")
            axes[idx, 1].text(0.5, 0.5, "Failed to load", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_image_size_hist(manifest: pd.DataFrame, out: Path) -> None:
    """Show width and height distributions of raw images.

    Args:
        manifest: DataFrame with 'width' and 'height' columns.
        out: Output path for PNG.
    """
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.hist(manifest["width"], bins=30, edgecolor="black", alpha=0.7)
    ax1.set_xlabel("Width (pixels)")
    ax1.set_ylabel("Count")
    ax1.set_title("Image Width Distribution")
    ax1.grid(axis="y", alpha=0.3)

    ax2.hist(manifest["height"], bins=30, edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Height (pixels)")
    ax2.set_ylabel("Count")
    ax2.set_title("Image Height Distribution")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_preprocess_examples(rows: pd.DataFrame, cfg: dict, out: Path) -> None:
    """Show each image through the full preprocessing pipeline side by side: raw,
    circle crop, Ben Graham, and CLAHE.

    Args:
        rows: DataFrame with 'image_id', 'path', 'label' columns.
        cfg: Preprocess config dict.
        out: Output path for PNG.
    """
    out.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 4, figsize=(12, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    methods = ["Raw", "Circle crop", "Ben Graham", "CLAHE (L)"]

    for row_idx, (_, row) in enumerate(rows.iterrows()):
        path = Path(row["path"])
        label = row["label"]
        grade_name = CLASS_NAMES[label]

        try:
            img = load_rgb(path)

            axes[row_idx, 0].imshow(img)
            axes[row_idx, 0].set_title(f"{methods[0]} ({grade_name})")
            axes[row_idx, 0].axis("off")

            cropped_result = circle_crop(img, **cfg.get("circle_crop", {}))
            axes[row_idx, 1].imshow(cropped_result.image)
            axes[row_idx, 1].set_title(methods[1])
            axes[row_idx, 1].axis("off")

            ben_graham_img, _ = preprocess_image(img, "ben_graham", cfg.get("size", 512), cfg)
            axes[row_idx, 2].imshow(ben_graham_img)
            axes[row_idx, 2].set_title(methods[2])
            axes[row_idx, 2].axis("off")

            clahe_img, _ = preprocess_image(img, "clahe_l", cfg.get("size", 512), cfg)
            axes[row_idx, 3].imshow(clahe_img)
            axes[row_idx, 3].set_title(methods[3])
            axes[row_idx, 3].axis("off")

        except Exception:
            for col in range(4):
                axes[row_idx, col].text(0.5, 0.5, "Failed to load", ha="center", va="center")
                axes[row_idx, col].axis("off")

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
