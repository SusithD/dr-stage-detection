"""Cached fundus dataset for training and evaluation."""

import logging
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from drstage.data.cache import cached_path, verify_cache
from drstage.data.preprocess import load_rgb, resize_square

logger = logging.getLogger(__name__)


class CachedFundusDataset(Dataset):
    """PyTorch Dataset for cached fundus images with on-the-fly resizing."""

    def __init__(
        self,
        frame: pd.DataFrame,
        cache_root: Path,
        img_size: int,
        transform=None,
        cache_fmt: str = "png",
    ):
        """Initialize dataset from DataFrame and cache directory.

        Args:
            frame: DataFrame with columns image_id, label (at minimum).
            cache_root: Path to cache directory containing cached images.
            img_size: Target image size for resizing.
            transform: albumentations Compose object or None.
            cache_fmt: Format of cached images (png, jpg, jpeg).

        Raises:
            RuntimeError: If verify_cache finds missing cached files.
        """
        self.frame = frame.reset_index(drop=True)
        self.cache_root = Path(cache_root)
        self.img_size = img_size
        self.transform = transform
        self.cache_fmt = cache_fmt

        missing = verify_cache(frame, self.cache_root, cache_fmt)
        if missing:
            raise RuntimeError(
                f"Missing {len(missing)} cached files: {missing[:5]} "
                f"(use prepare_data --stage cache)"
            )

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict:
        """Load, resize, transform, and return image and label.

        Returns:
            dict with keys: image (FloatTensor [3,S,S]), label (int64 tensor),
            image_id (str), index (int).
        """
        row = self.frame.iloc[idx]
        image_id = row["image_id"]
        label = row["label"]

        cache_path = cached_path(self.cache_root, image_id, self.cache_fmt)
        img = load_rgb(cache_path)

        cached_size = img.shape[0]
        if cached_size != self.img_size:
            img = resize_square(img, self.img_size)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        return {
            "image": img,
            "label": torch.tensor(label, dtype=torch.int64),
            "image_id": image_id,
            "index": idx,
        }


def load_split(csv: Path, split: str | None) -> pd.DataFrame:
    """Load split from split CSV file.

    Args:
        csv: Path to split CSV (columns: image_id, path, label, split, ...).
        split: Split name (train/val/test) or None for all rows.

    Returns:
        Filtered DataFrame with columns: image_id, label.
    """
    df = pd.read_csv(csv)

    if split is not None:
        df = df[df["split"] == split]

    return df[["image_id", "label"]].reset_index(drop=True)
