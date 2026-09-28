"""Image caching and preprocessing pipeline."""

import logging
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from drstage.data.preprocess import load_rgb, preprocess_image, save_rgb
from drstage.utils.io import ensure_dir, write_json

logger = logging.getLogger(__name__)


def cache_dir(processed_root: Path, dataset: str, method: str, size: int) -> Path:
    """Return cache directory for a given dataset, method, and size."""
    return processed_root / "images" / dataset / method / str(size)


def cached_path(cache_root: Path, image_id: str, fmt: str = "png") -> Path:
    """Return path for cached image."""
    return cache_root / f"{image_id}.{fmt}"


def _process_one(args: tuple) -> tuple:
    """Process and cache a single image; must be module-level for multiprocessing spawn."""
    image_id, path, dataset, method, cfg, cache_root, fmt, overwrite = args
    cache_output = cached_path(cache_root, image_id, fmt)

    if cache_output.exists() and not overwrite:
        return (image_id, str(cache_output), False, "")

    try:
        cv2.setNumThreads(1)
        img = load_rgb(Path(path))
        processed, fallback_used = preprocess_image(img, method, cfg["size"], cfg)
        ensure_dir(cache_output.parent)
        save_rgb(processed, cache_output, fmt=fmt, jpeg_quality=cfg.get("cache_jpeg_quality", 95))
        return (image_id, str(cache_output), fallback_used, "")
    except Exception as e:
        return (image_id, "", False, str(e))


def build_cache(
    manifest: pd.DataFrame,
    dataset: str,
    method: str,
    cfg: dict,
    processed_root: Path,
    num_workers: int,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Preprocess and cache all images in parallel. Writes a manifest JSON with
    processing stats and error summaries."""
    cache_root = cache_dir(processed_root, dataset, method, cfg["size"])
    fmt = cfg.get("cache_format", "png")

    manifest = manifest.sort_values("image_id").reset_index(drop=True)

    args_list = [
        (
            row["image_id"],
            row["path"],
            dataset,
            method,
            cfg,
            cache_root,
            fmt,
            overwrite,
        )
        for _, row in manifest.iterrows()
    ]

    results = []
    start_time = time.time()

    if num_workers == 0:
        for args in tqdm(args_list, desc=f"Caching {dataset}/{method}"):
            results.append(_process_one(args))
    else:
        with Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_process_one, args_list, chunksize=16),
                    total=len(args_list),
                    desc=f"Caching {dataset}/{method}",
                )
            )

    elapsed_s = time.time() - start_time

    result_df = pd.DataFrame(results, columns=["image_id", "cache_path", "crop_fallback", "error"])
    result_df = result_df.sort_values("image_id").reset_index(drop=True)

    n_images = len(result_df)
    n_errors = (result_df["error"] != "").sum()
    n_fallback = result_df["crop_fallback"].sum()

    manifest_json = {
        "n_images": int(n_images),
        "n_errors": int(n_errors),
        "n_fallback": int(n_fallback),
        "size": int(cfg["size"]),
        "method": method,
        "params": cfg.get(method, {}),
        "created_at": datetime.now().isoformat(),
        "elapsed_s": float(elapsed_s),
    }

    manifest_path = cache_root / "_manifest.json"
    write_json(manifest_json, manifest_path)

    for _, row in result_df.iterrows():
        if row["error"]:
            logger.warning(f"Failed to process {row['image_id']}: {row['error']}")

    return result_df


def verify_cache(manifest: pd.DataFrame, cache_root: Path, fmt: str = "png") -> list[str]:
    """Return list of image_ids whose cached files are missing."""
    missing = []
    for image_id in manifest["image_id"]:
        path = cached_path(cache_root, image_id, fmt)
        if not path.exists():
            missing.append(image_id)
    return missing
