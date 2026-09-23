"""Shared pytest fixtures: synthetic fundus-like images and a tiny labelled manifest."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest


def make_fundus(
    width: int = 640,
    height: int = 480,
    radius: int | None = None,
    dark: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """Draw a textured bright disc on a black background, mimicking a fundus photograph."""
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width, 3), np.uint8)
    r = radius or int(min(width, height) * 0.45)
    colour = (3, 2, 1) if dark else (200, 90, 40)
    cv2.circle(img, (width // 2, height // 2), r, colour, -1)
    if not dark:
        noise = rng.integers(0, 40, (height, width, 3), dtype=np.uint8)
        mask = img.any(axis=2, keepdims=True)
        img = np.where(mask, cv2.add(img, noise), img)
        for _ in range(6):
            cx, cy = (
                rng.integers(width // 3, 2 * width // 3),
                rng.integers(height // 3, 2 * height // 3),
            )
            cv2.circle(img, (int(cx), int(cy)), int(rng.integers(3, 12)), (120, 20, 20), -1)
    return img


@pytest.fixture
def synthetic_fundus():
    return make_fundus


@pytest.fixture
def tiny_manifest(tmp_path: Path) -> dict:
    """Write 12 labelled synthetic images (2 exact duplicates, 1 near duplicate) plus a CSV.

    Also writes one unlabelled image (must be ignored) and lists one CSV row whose file is missing.
    """
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rows = []
    for i in range(10):
        img = make_fundus(640 + 16 * i, 480, seed=i)
        cv2.imwrite(str(image_dir / f"img{i:02d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        rows.append((f"img{i:02d}.jpg", i % 5))

    exact = cv2.imread(str(image_dir / "img00.jpg"))
    cv2.imwrite(str(image_dir / "dup_exact.jpg"), exact)
    rows.append(("dup_exact.jpg", rows[0][1]))

    near = cv2.imread(str(image_dir / "img01.jpg"))
    near = cv2.convertScaleAbs(near, alpha=1.03, beta=2)
    cv2.imwrite(str(image_dir / "dup_near.jpg"), near)
    rows.append(("dup_near.jpg", rows[1][1]))

    cv2.imwrite(str(image_dir / "unlabelled.jpg"), exact)
    rows.append(("missing_file.jpg", 2))

    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(rows, columns=["id_code", "diagnosis"]).to_csv(csv_path, index=False)
    return {
        "csv": csv_path,
        "image_dir": image_dir,
        "n_labelled_present": 12,
        "exact_pair": ("img00", "dup_exact"),
        "near_pair": ("img01", "dup_near"),
    }


@pytest.fixture
def tiny_split(tmp_path: Path) -> dict:
    """Create 30 synthetic 64px cached images covering all 5 labels.

    Writes cache in <tmp>/processed/images/synth/none/64/ and split CSV.
    Returns dict with cache_root, split_csv, preprocess_cfg.
    """
    cache_root = tmp_path / "processed" / "images" / "synth" / "none" / "64"
    cache_root.mkdir(parents=True, exist_ok=True)

    split_rows = []
    for i in range(30):
        label = i % 5
        image_id = f"synth_{i:03d}"

        img = make_fundus(64, 64, seed=i)
        img_path = cache_root / f"{image_id}.png"
        cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        split_num = (i // 5) % 3
        if split_num == 0:
            split = "train"
        elif split_num == 1:
            split = "val"
        else:
            split = "test"

        split_rows.append({"image_id": image_id, "label": label, "split": split})

    split_csv = tmp_path / "split.csv"
    pd.DataFrame(split_rows).to_csv(split_csv, index=False)

    preprocess_cfg = {
        "circle_crop": {"threshold": 10, "min_radius_frac": 0.25},
        "ben_graham": {"sigma_frac": 0.17, "alpha": 4, "beta": -4, "gamma": 128},
        "clahe_l": {"clip_limit": 2.0, "tile_grid": 8},
    }

    return {
        "cache_root": cache_root,
        "split_csv": split_csv,
        "preprocess_cfg": preprocess_cfg,
        "img_size": 64,
    }
