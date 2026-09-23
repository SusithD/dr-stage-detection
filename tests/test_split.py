"""Tests for split module."""

import numpy as np
import pandas as pd
import pytest

from drstage.data.split import (
    check_split,
    split_summary,
    stratified_group_split,
)


def test_stratified_group_split_ratios_invalid():
    """Test stratified_group_split raises on invalid ratios."""
    df = pd.DataFrame(
        {
            "image_id": ["img1"],
            "label": [0],
            "dup_group": [0],
        }
    )

    with pytest.raises(ValueError, match="must have keys"):
        stratified_group_split(df, {"train": 0.5, "val": 0.3}, seed=42)


def test_stratified_group_split_ratios_dont_sum():
    """Test stratified_group_split raises when ratios don't sum to ~1.0."""
    df = pd.DataFrame(
        {
            "image_id": ["img1"],
            "label": [0],
            "dup_group": [0],
        }
    )

    with pytest.raises(ValueError, match="must sum"):
        stratified_group_split(
            df,
            {"train": 0.3, "val": 0.3, "test": 0.3},
            seed=42,
        )


def test_stratified_group_split_deterministic_seed():
    """Test stratified_group_split produces identical results for same seed."""
    np.random.seed(42)
    rng = np.random.default_rng(42)

    n_rows = 500
    labels_dist = [6266, 630, 4477, 236, 913]
    labels_norm = [x / sum(labels_dist) for x in labels_dist]

    labels = rng.choice(5, n_rows, p=labels_norm)
    groups = np.arange(n_rows)

    df = pd.DataFrame(
        {
            "image_id": [f"img{i:04d}" for i in range(n_rows)],
            "label": labels,
            "dup_group": groups,
        }
    )

    split1 = stratified_group_split(
        df.copy(),
        {"train": 0.5455, "val": 0.1818, "test": 0.2727},
        seed=42,
    )

    split2 = stratified_group_split(
        df.copy(),
        {"train": 0.5455, "val": 0.1818, "test": 0.2727},
        seed=42,
    )

    pd.testing.assert_frame_equal(split1, split2)


def test_stratified_group_split_different_seed():
    """Test stratified_group_split produces different results for different seeds."""
    np.random.seed(42)
    rng = np.random.default_rng(42)

    n_rows = 500
    labels_dist = [6266, 630, 4477, 236, 913]
    labels_norm = [x / sum(labels_dist) for x in labels_dist]

    labels = rng.choice(5, n_rows, p=labels_norm)
    groups = np.arange(n_rows)

    df = pd.DataFrame(
        {
            "image_id": [f"img{i:04d}" for i in range(n_rows)],
            "label": labels,
            "dup_group": groups,
        }
    )

    split42 = stratified_group_split(
        df.copy(),
        {"train": 0.5455, "val": 0.1818, "test": 0.2727},
        seed=42,
    )

    split43 = stratified_group_split(
        df.copy(),
        {"train": 0.5455, "val": 0.1818, "test": 0.2727},
        seed=43,
    )

    assert not split42.equals(split43)


def test_stratified_group_split_sizes_within_tolerance():
    """Test stratified_group_split produces splits within ±2% of target ratio."""
    np.random.seed(42)
    rng = np.random.default_rng(42)

    n_rows = 500
    labels_dist = [6266, 630, 4477, 236, 913]
    labels_norm = [x / sum(labels_dist) for x in labels_dist]

    labels = rng.choice(5, n_rows, p=labels_norm)
    groups = np.arange(n_rows)

    df = pd.DataFrame(
        {
            "image_id": [f"img{i:04d}" for i in range(n_rows)],
            "label": labels,
            "dup_group": groups,
        }
    )

    ratios = {"train": 0.5455, "val": 0.1818, "test": 0.2727}
    split = stratified_group_split(df, ratios, seed=42)

    for split_name, target_ratio in ratios.items():
        split_df = split[split["split"] == split_name]
        actual_ratio = len(split_df) / len(split)
        tolerance = 0.02
        assert abs(actual_ratio - target_ratio) <= tolerance, (
            f"{split_name}: expected {target_ratio:.4f}, got {actual_ratio:.4f}"
        )


def test_stratified_group_split_all_classes_in_all_splits():
    """Test all 5 classes appear in every split."""
    np.random.seed(42)
    rng = np.random.default_rng(42)

    n_rows = 500
    labels_dist = [6266, 630, 4477, 236, 913]
    labels_norm = [x / sum(labels_dist) for x in labels_dist]

    labels = rng.choice(5, n_rows, p=labels_norm)
    groups = np.arange(n_rows)

    df = pd.DataFrame(
        {
            "image_id": [f"img{i:04d}" for i in range(n_rows)],
            "label": labels,
            "dup_group": groups,
        }
    )

    split = stratified_group_split(
        df,
        {"train": 0.5455, "val": 0.1818, "test": 0.2727},
        seed=42,
    )

    for split_name in ["train", "val", "test"]:
        split_df = split[split["split"] == split_name]
        unique_labels = split_df["label"].unique()
        assert len(unique_labels) == 5, (
            f"{split_name} missing class(es); has {sorted(unique_labels)}"
        )


def test_stratified_group_split_sorted_by_image_id():
    """Test stratified_group_split result is sorted by image_id."""
    df = pd.DataFrame(
        {
            "image_id": [f"img{i:02d}" for i in range(30)],
            "label": [i % 5 for i in range(30)],
            "dup_group": list(range(30)),
        }
    )

    split = stratified_group_split(
        df,
        {"train": 0.5, "val": 0.3, "test": 0.2},
        seed=42,
    )

    assert split["image_id"].is_monotonic_increasing


def test_stratified_group_split_group_not_split():
    """Test images in same group always go to same split."""
    n_groups = 30
    images_per_group = 3
    df = pd.DataFrame(
        {
            "image_id": [f"img_{i}_{j}" for i in range(n_groups) for j in range(images_per_group)],
            "label": [i % 5 for i in range(n_groups) for _ in range(images_per_group)],
            "dup_group": [i for i in range(n_groups) for _ in range(images_per_group)],
        }
    )

    split = stratified_group_split(
        df,
        {"train": 0.5, "val": 0.3, "test": 0.2},
        seed=42,
    )

    for group_id in split["dup_group"].unique():
        group_df = split[split["dup_group"] == group_id]
        assert len(group_df["split"].unique()) == 1, (
            f"Group {group_id} split across multiple splits"
        )


def test_check_split_valid():
    """Test check_split passes for valid split."""
    df = pd.DataFrame(
        {
            "image_id": [f"img{i}" for i in range(30)],
            "split": ["train"] * 15 + ["val"] * 9 + ["test"] * 6,
            "label": [i % 5 for i in range(30)],
        }
    )

    check_split(df)


def test_check_split_missing_class():
    """Test check_split raises if a split is missing a class."""
    df = pd.DataFrame(
        {
            "image_id": [f"img{i}" for i in range(20)],
            "split": ["train"] * 10 + ["val"] * 5 + ["test"] * 5,
            "label": [0, 1, 2, 3, 4] * 4,
        }
    )

    df.loc[df["split"] == "val", "label"] = 0

    with pytest.raises(ValueError, match="Missing classes|missing class"):
        check_split(df)


def test_check_split_duplicate_image_ids():
    """Test check_split raises on duplicate image_ids."""
    base_labels = [0, 1, 2, 3, 4]
    labels = (base_labels * 6)[:30]
    df = pd.DataFrame(
        {
            "image_id": ["img1", "img1", "img2"] + [f"img{i}" for i in range(27)],
            "split": ["train"] * 15 + ["val"] * 9 + ["test"] * 6,
            "label": labels,
        }
    )

    with pytest.raises(ValueError, match="Duplicate"):
        check_split(df)


def test_check_split_unknown_split():
    """Test check_split raises on unknown split value."""
    df = pd.DataFrame(
        {
            "image_id": [f"img{i}" for i in range(30)],
            "split": ["train"] * 10 + ["val"] * 10 + ["unknown"] * 10,
            "label": [i % 5 for i in range(30)],
        }
    )

    with pytest.raises(ValueError, match="Unknown split"):
        check_split(df)


def test_split_summary():
    """Test split_summary returns correct statistics."""
    df = pd.DataFrame(
        {
            "split": ["train"] * 15 + ["val"] * 9 + ["test"] * 6,
            "label": [i % 5 for i in range(30)],
        }
    )

    summary = split_summary(df)

    assert summary["total_samples"] == 30
    assert summary["split_counts"]["train"] == 15
    assert summary["split_counts"]["val"] == 9
    assert summary["split_counts"]["test"] == 6
    assert abs(summary["split_ratios"]["train"] - 0.5) < 0.01
    assert abs(summary["split_ratios"]["val"] - 0.3) < 0.01
    assert abs(summary["split_ratios"]["test"] - 0.2) < 0.01


def test_split_summary_per_class():
    """Test split_summary includes per-class statistics."""
    df = pd.DataFrame(
        {
            "split": ["train"] * 10 + ["val"] * 5 + ["test"] * 5,
            "label": [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
        }
    )

    summary = split_summary(df)

    assert "train_class_counts" in summary
    assert "val_class_counts" in summary
    assert "test_class_counts" in summary
    assert "train_class_ratios" in summary
    assert "val_class_ratios" in summary
    assert "test_class_ratios" in summary
