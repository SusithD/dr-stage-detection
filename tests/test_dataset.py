"""Tests for dataset and dataloader implementations."""

import numpy as np
import torch

from drstage.data.dataset import CachedFundusDataset, load_split
from drstage.data.loaders import build_dataloader, build_dataloaders, sampler_weights
from drstage.data.transforms import build_eval_transforms, build_train_transforms


def test_load_split(tiny_split):
    """Test load_split filters correctly."""
    split_csv = tiny_split["split_csv"]

    train = load_split(split_csv, "train")
    assert len(train) == 10

    val = load_split(split_csv, "val")
    assert len(val) == 10

    test = load_split(split_csv, "test")
    assert len(test) == 10

    all_rows = load_split(split_csv, None)
    assert len(all_rows) == 30


def test_cached_fundus_dataset_shapes(tiny_split):
    """Test dataset item shapes and dtypes."""
    split_csv = tiny_split["split_csv"]
    cache_root = tiny_split["cache_root"]

    frame = load_split(split_csv, "train")
    transform = build_eval_transforms(64, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    dataset = CachedFundusDataset(frame, cache_root, 64, transform)
    assert len(dataset) == 10

    item = dataset[0]
    assert "image" in item
    assert "label" in item
    assert "image_id" in item
    assert "index" in item

    image = item["image"]
    assert isinstance(image, torch.Tensor)
    assert image.dtype == torch.float32
    assert image.shape == (3, 64, 64)

    label = item["label"]
    assert isinstance(label, torch.Tensor)
    assert label.dtype == torch.int64

    assert isinstance(item["image_id"], str)
    assert isinstance(item["index"], int)


def test_dataset_on_the_fly_resize(tiny_split):
    """Test dataset resizes cached images when size differs."""
    split_csv = tiny_split["split_csv"]
    cache_root = tiny_split["cache_root"]

    frame = load_split(split_csv, "train")
    transform = build_eval_transforms(128, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    dataset = CachedFundusDataset(frame, cache_root, 128, transform)
    item = dataset[0]

    assert item["image"].shape == (3, 128, 128)


def test_eval_transform_determinism(tiny_split):
    """Test eval transforms are deterministic."""
    split_csv = tiny_split["split_csv"]
    cache_root = tiny_split["cache_root"]

    frame = load_split(split_csv, "train")
    transform = build_eval_transforms(64, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    dataset = CachedFundusDataset(frame, cache_root, 64, transform)
    item1 = dataset[0]
    item2 = dataset[0]

    torch.testing.assert_close(item1["image"], item2["image"])


def test_train_transform_construction(tiny_split):
    """Test train transforms construct correctly."""
    split_csv = tiny_split["split_csv"]
    cache_root = tiny_split["cache_root"]

    frame = load_split(split_csv, "train")

    aug = {
        "rotate_limit": 180,
        "rotate_p": 0.7,
        "rrc_scale": [0.85, 1.0],
        "brightness": 0.2,
        "contrast": 0.2,
        "hue": 5,
        "sat": 10,
        "color_p": 0.5,
    }
    transform = build_train_transforms(64, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], aug)

    dataset = CachedFundusDataset(frame, cache_root, 64, transform)
    item = dataset[0]

    assert item["image"].shape == (3, 64, 64)
    assert item["image"].dtype == torch.float32


def test_sampler_weights():
    """Test sampler_weights computes correct per-sample inverse-frequency weights."""
    labels = np.array([0, 0, 0, 1, 1, 2])

    weights = sampler_weights(labels, 0.0)
    assert torch.allclose(weights, torch.ones(6))

    weights = sampler_weights(labels, 1.0)
    assert weights[0] < weights[3]
    assert weights[3] < weights[5]

    total = len(labels)
    n_classes = 3
    class_weights = (total / (n_classes * np.array([3, 2, 1]))) ** 1.0
    expected = torch.tensor([class_weights[i] for i in labels], dtype=torch.float32)

    torch.testing.assert_close(weights, expected, rtol=1e-5, atol=1e-7)


def test_build_dataloader_spawn_context(tiny_split):
    """Test dataloader with num_workers>0 on macOS uses spawn context."""
    split_csv = tiny_split["split_csv"]
    cache_root = tiny_split["cache_root"]

    frame = load_split(split_csv, "train")
    transform = build_eval_transforms(64, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    dataset = CachedFundusDataset(frame, cache_root, 64, transform)
    device = torch.device("cpu")

    loader = build_dataloader(dataset, batch_size=4, shuffle=False, num_workers=2, device=device)

    batch = next(iter(loader))
    assert "image" in batch
    assert "label" in batch
    assert batch["image"].shape[0] == 4


def test_build_dataloaders(tiny_split):
    """Test build_dataloaders integration."""
    split_csv = tiny_split["split_csv"]
    cache_root = tiny_split["cache_root"]

    processed_root = cache_root.parents[3]

    cfg = {
        "seed": 42,
        "device": "cpu",
        "data": {
            "processed_root": str(processed_root),
            "split_csv": str(split_csv),
            "dataset": "synth",
            "method": "none",
            "cache_size": 64,
            "img_size": 64,
            "cache_format": "png",
            "batch_size": 4,
            "eval_batch_size": 8,
            "num_workers": 0,
            "aug": {
                "rotate_limit": 180,
                "rotate_p": 0.7,
                "rrc_scale": [0.85, 1.0],
                "brightness": 0.2,
                "contrast": 0.2,
                "hue": 5,
                "sat": 10,
                "color_p": 0.5,
            },
        },
        "imbalance": {
            "class_weight_power": 0.5,
            "sampler": {"enabled": False, "power": 0.5},
            "weights_with_sampler": False,
        },
    }

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    device = torch.device("cpu")

    train_loader, val_loader, info = build_dataloaders(cfg, mean, std, device)

    assert info["n_train"] == 10
    assert info["n_val"] == 10
    assert len(info["class_counts"]) == 5

    batch = next(iter(train_loader))
    assert batch["image"].shape[0] <= 4
    assert batch["image"].shape == (batch["image"].shape[0], 3, 64, 64)

    batch = next(iter(val_loader))
    assert batch["image"].shape[0] <= 8
