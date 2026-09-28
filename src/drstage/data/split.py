"""Train/val/test splitting with stratification by group."""

import logging

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)


def stratified_group_split(
    df: pd.DataFrame,
    ratios: dict[str, float],
    seed: int,
    label_col: str = "label",
    group_col: str = "dup_group",
) -> pd.DataFrame:
    """Stratified split at group level, stratified by label.

    Groups (duplicate sets) always land in a single split. Group label is the
    majority class; ties go to lower grade (0 < 1 < 2 < 3 < 4).

    Args:
        df: DataFrame with image_id, label, dup_group columns
        ratios: Dict with keys {train, val, test}; must sum to ~1.0
        seed: Random seed
        label_col: Column name for labels
        group_col: Column name for group IDs

    Returns:
        DataFrame with original columns plus 'split' column, sorted by image_id

    Raises:
        ValueError: if ratios invalid
    """
    if set(ratios.keys()) != {"train", "val", "test"}:
        raise ValueError("ratios must have keys {train, val, test}")

    ratio_sum = sum(ratios.values())
    if abs(ratio_sum - 1.0) > 0.01:
        raise ValueError(f"ratios must sum to ~1.0, got {ratio_sum}")

    df = df.copy()

    group_labels = {}
    for _, group_df in df.groupby(group_col):
        group_id = group_df[group_col].iloc[0]
        labels = group_df[label_col].values
        counts = pd.Series(labels).value_counts()
        if len(counts) == 1:
            group_label = counts.index[0]
        else:
            max_count = counts.max()
            tied_labels = counts[counts == max_count].index.tolist()
            group_label = min(tied_labels)
        group_labels[group_id] = group_label

    df["group_label"] = df[group_col].map(group_labels)

    group_df = df.groupby(group_col).first().reset_index()[[group_col, "group_label"]]
    group_df = group_df.sort_values(group_col)

    train_ratio = ratios["train"]
    val_ratio = ratios["val"]
    test_ratio = ratios["test"]

    splitter_test = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_ratio,
        random_state=seed,
    )
    train_val_idx, test_idx = next(splitter_test.split(group_df, group_df["group_label"]))

    train_val_groups = group_df.iloc[train_val_idx].reset_index(drop=True)

    val_size_of_remainder = val_ratio / (train_ratio + val_ratio)
    splitter_val = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_size_of_remainder,
        random_state=seed + 1,
    )
    train_idx, val_idx = next(splitter_val.split(train_val_groups, train_val_groups["group_label"]))

    train_groups = set(train_val_groups.iloc[train_idx][group_col].values)
    val_groups = set(train_val_groups.iloc[val_idx][group_col].values)
    test_groups = set(group_df.iloc[test_idx][group_col].values)

    def assign_split(group_id):
        """Map a group ID to its assigned split (train, val, or test)."""
        if group_id in train_groups:
            return "train"
        elif group_id in val_groups:
            return "val"
        elif group_id in test_groups:
            return "test"
        else:
            return None

    df["split"] = df[group_col].map(assign_split)

    df = df.drop(columns=["group_label"])

    return df.sort_values("image_id").reset_index(drop=True)


def split_summary(df: pd.DataFrame) -> dict:
    """Summarize split distribution by split and class.

    Args:
        df: DataFrame with split, label columns

    Returns:
        Dictionary with per-split per-class counts and achieved ratios
    """
    total = len(df)

    split_counts = df["split"].value_counts().to_dict()
    split_ratios = {k: v / total for k, v in split_counts.items()}

    summary = {
        "split_counts": split_counts,
        "split_ratios": split_ratios,
        "total_samples": total,
    }

    for split in ["train", "val", "test"]:
        if split in split_counts:
            split_df = df[df["split"] == split]
            class_counts = split_df["label"].value_counts().sort_index().to_dict()
            class_ratios = {k: v / len(split_df) for k, v in class_counts.items()}
            summary[f"{split}_class_counts"] = class_counts
            summary[f"{split}_class_ratios"] = class_ratios

    return summary


def check_split(df: pd.DataFrame) -> None:
    """Validate split integrity.

    Args:
        df: DataFrame with image_id, split, label columns

    Raises:
        ValueError: if split is invalid
    """
    if df["image_id"].duplicated().any():
        raise ValueError("Duplicate image_ids in split")

    required_splits = {"train", "val", "test"}
    valid_split_values = required_splits
    invalid = df[~df["split"].isin(valid_split_values)]
    if len(invalid) > 0:
        raise ValueError(f"Unknown split values: {invalid['split'].unique()}")

    actual_splits = set(df["split"].unique())
    if not required_splits.issubset(actual_splits):
        missing = required_splits - actual_splits
        raise ValueError(f"Missing splits: {missing}")

    for split in required_splits:
        split_df = df[df["split"] == split]
        if split_df["label"].nunique() < 5:
            raise ValueError(
                f"Split '{split}' missing class(es); has: {sorted(split_df['label'].unique())}"
            )
