"""DataLoader construction with spawn context and weighted sampling."""

import logging
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler

from drstage.data.cache import cache_dir
from drstage.data.dataset import CachedFundusDataset, load_split
from drstage.data.transforms import build_eval_transforms, build_train_transforms
from drstage.utils.seed import worker_init_fn

logger = logging.getLogger(__name__)


def sampler_weights(labels: np.ndarray, power: float) -> torch.Tensor:
    """Compute per-sample weights for balanced sampling.

    Args:
        labels: Array of class labels.
        power: Exponent for weighting (0 = uniform, 1 = exact inverse frequency).

    Returns:
        Tensor of weights, one per sample, normalized to sum to batch_size.
    """
    unique, counts = np.unique(labels, return_counts=True)
    n_classes = len(unique)
    n_total = len(labels)

    if power == 0:
        return torch.ones(n_total)

    class_weights = (n_total / (n_classes * counts)) ** power
    weights = class_weights[labels]

    return torch.from_numpy(weights).float()


def build_dataloader(
    dataset: CachedFundusDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    sampler: Sampler | None = None,
    drop_last: bool = False,
    seed: int = 0,
) -> DataLoader:
    """Build DataLoader with platform-aware multiprocessing settings.

    Args:
        dataset: PyTorch Dataset.
        batch_size: Batch size.
        shuffle: Shuffle data (ignored when sampler is provided).
        num_workers: Number of worker processes.
        device: torch.device object.
        sampler: Sampler instance or None.
        drop_last: Drop incomplete batches.
        seed: Random seed for DataLoader generator.

    Returns:
        DataLoader instance.
    """
    ctx = None
    if num_workers > 0:
        import sys

        if sys.platform == "darwin":
            ctx = mp.get_context("spawn")

    pin_memory = device.type == "cuda"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        multiprocessing_context=ctx,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=torch.Generator().manual_seed(seed),
    )


def build_dataloaders(
    cfg: dict,
    mean: list[float],
    std: list[float],
    device: torch.device,
) -> tuple[DataLoader, DataLoader, dict]:
    """Build train and val dataloaders from config, with optional weighted sampling for
    class imbalance.

    Args:
        cfg: Config dict with keys: data.*, model.channels_last, imbalance.*
        mean: Normalization mean.
        std: Normalization std.
        device: torch.device object.

    Returns:
        (train_loader, val_loader, info_dict) where info_dict contains:
        class_counts (list[int]), n_train, n_val.
    """
    data_cfg = cfg["data"]
    imbalance_cfg = cfg.get("imbalance", {})

    processed_root = Path(data_cfg["processed_root"])
    split_csv = Path(data_cfg["split_csv"])
    dataset_name = data_cfg["dataset"]
    method = data_cfg["method"]
    cache_size = data_cfg["cache_size"]
    img_size = data_cfg["img_size"]
    batch_size = data_cfg["batch_size"]
    eval_batch_size = data_cfg.get("eval_batch_size", batch_size)
    num_workers = data_cfg.get("num_workers", 0)
    cache_fmt = data_cfg.get("cache_format", "png")

    cache_root = cache_dir(processed_root, dataset_name, method, cache_size)

    train_frame = load_split(split_csv, "train")
    val_frame = load_split(split_csv, "val")

    train_transforms = build_train_transforms(img_size, mean, std, data_cfg["aug"])
    eval_transforms = build_eval_transforms(img_size, mean, std)

    train_dataset = CachedFundusDataset(
        train_frame, cache_root, img_size, train_transforms, cache_fmt
    )
    val_dataset = CachedFundusDataset(val_frame, cache_root, img_size, eval_transforms, cache_fmt)

    train_sampler = None
    if imbalance_cfg.get("sampler", {}).get("enabled", False):
        power = imbalance_cfg["sampler"].get("power", 0.5)
        weights = sampler_weights(train_frame["label"].values, power)
        train_sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )

    train_loader = build_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        device=device,
        sampler=train_sampler,
        seed=cfg["seed"],
    )

    val_loader = build_dataloader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        device=device,
        seed=cfg["seed"],
    )

    unique, counts = np.unique(train_frame["label"].values, return_counts=True)
    class_counts = [0] * 5
    for label, count in zip(unique, counts, strict=False):
        class_counts[label] = int(count)

    info = {
        "class_counts": class_counts,
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
    }

    return train_loader, val_loader, info
