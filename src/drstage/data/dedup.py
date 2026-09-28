"""Duplicate detection and deduplication."""

import logging
from multiprocessing import Pool
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_phash(img: np.ndarray, hash_size: int = 16) -> str:
    """Compute perceptual hash of RGB image.

    Args:
        img: RGB uint8 image (H, W, 3)
        hash_size: hash grid size (default 16)

    Returns:
        Hex string hash
    """
    pil_img = Image.fromarray(img, mode="RGB")
    return str(imagehash.phash(pil_img, hash_size=hash_size))


def hamming(a: str, b: str) -> int:
    """Compute Hamming distance between two hex hash strings."""
    a_int = int(a, 16)
    b_int = int(b, 16)
    xor = a_int ^ b_int
    return bin(xor).count("1")


def _compute_hash_worker(args: tuple) -> tuple:
    """Hash a single image; must be module-level for multiprocessing spawn."""
    image_id, path, cfg = args

    try:
        from drstage.data.preprocess import circle_crop, load_rgb, resize_square

        img = load_rgb(Path(path))
        cropped = circle_crop(
            img,
            threshold=cfg["preprocess"]["circle_crop"]["threshold"],
            min_radius_frac=cfg["preprocess"]["circle_crop"]["min_radius_frac"],
        )
        resized = resize_square(cropped.image, cfg["dedup"]["hash_image_size"])
        phash = compute_phash(resized, hash_size=cfg["dedup"]["hash_size"])
        return (image_id, phash, None)
    except Exception as e:
        logger.warning(f"Failed to hash {image_id}: {e}")
        return (image_id, None, str(e))


def compute_hashes(manifest: pd.DataFrame, cfg: dict, num_workers: int) -> pd.DataFrame:
    """Compute perceptual hashes for all images in manifest.

    Args:
        manifest: DataFrame with image_id, path, dataset columns
        cfg: full config dict with preprocess and dedup sections
        num_workers: number of worker processes (0 for sequential)

    Returns:
        DataFrame with image_id, phash, dataset columns, sorted by image_id
    """
    tasks = [(row["image_id"], row["path"], cfg) for _, row in manifest.iterrows()]

    results = []
    if num_workers == 0:
        for task in tqdm(tasks, desc="Computing hashes"):
            results.append(_compute_hash_worker(task))
    else:
        with Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_compute_hash_worker, tasks, chunksize=16),
                    total=len(tasks),
                    desc="Computing hashes",
                )
            )

    ok = [(image_id, phash) for image_id, phash, _ in results if phash is not None]
    n_failed = len(results) - len(ok)
    if n_failed:
        logger.warning(f"Hash computation failed for {n_failed} images")
    df = pd.DataFrame(ok, columns=["image_id", "phash"])
    if "dataset" in manifest.columns:
        df = df.merge(manifest[["image_id", "dataset"]], on="image_id", how="left")
    return df.sort_values("image_id").reset_index(drop=True)


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _hash_bytes(hex_hashes: pd.Series) -> np.ndarray:
    """Convert hex hash strings to byte arrays for fast distance computation."""
    return np.stack([np.frombuffer(bytes.fromhex(h), dtype=np.uint8) for h in hex_hashes])


def _block_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise Hamming distances; result shape is (len(a), len(b))."""
    return _POPCOUNT[np.bitwise_xor(a[:, None, :], b[None, :, :])].sum(axis=2, dtype=np.uint16)


def _close_pairs(
    a: np.ndarray, b: np.ndarray, threshold: int, same: bool, block: int = 512
) -> list[tuple[int, int, int]]:
    """Return (i, j, distance) pairs within threshold; i < j when a set is compared to itself."""
    pairs = []
    for start in range(0, len(a), block):
        dist = _block_distances(a[start : start + block], b)
        ii, jj = np.nonzero(dist <= threshold)
        for i, j in zip(ii.tolist(), jj.tolist(), strict=True):
            gi = start + i
            if not same or gi < j:
                pairs.append((gi, j, int(dist[i, j])))
    return pairs


class UnionFind:
    """Merge duplicate pairs into groups using a disjoint-set forest."""

    def __init__(self, n: int):
        """Initialize with n elements, each in its own set."""
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        """Find the root of x, with path compression."""
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        """Merge the sets containing x and y by their roots."""
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[max(rx, ry)] = min(rx, ry)


def find_duplicate_groups(hashes: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Group images whose perceptual hashes are within a Hamming distance threshold.

    Args:
        hashes: DataFrame with image_id, phash columns.
        threshold: Maximum Hamming distance for two images to count as duplicates.

    Returns:
        DataFrame with image_id, dup_group (numbered by first sorted image_id), sorted by image_id.
    """
    hashes = hashes.sort_values("image_id").reset_index(drop=True)
    if len(hashes) == 0:
        return pd.DataFrame(columns=["image_id", "dup_group"])

    bits = _hash_bytes(hashes["phash"])
    uf = UnionFind(len(hashes))
    for i, j, _ in _close_pairs(bits, bits, threshold, same=True):
        uf.union(i, j)

    roots = [uf.find(i) for i in range(len(hashes))]
    group_ids = {root: gid for gid, root in enumerate(dict.fromkeys(roots))}
    return pd.DataFrame(
        {"image_id": hashes["image_id"], "dup_group": [group_ids[r] for r in roots]}
    )


def summarise_groups(groups: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Summarize duplicate groups.

    Args:
        groups: DataFrame with image_id, dup_group columns
        labels: DataFrame with image_id, label columns

    Returns:
        Dictionary with group statistics
    """
    merged = groups.merge(labels[["image_id", "label"]], on="image_id", how="left")
    n_images = len(merged)
    n_groups_multi = (merged.groupby("dup_group").size() > 1).sum()
    n_images_in_multi = merged[
        merged.groupby("dup_group")["dup_group"].transform("size") > 1
    ].shape[0]

    conflicting_groups = 0
    for _, group_df in merged.groupby("dup_group"):
        if group_df["label"].nunique() > 1:
            conflicting_groups += 1

    group_sizes = merged.groupby("dup_group").size()
    group_size_hist = group_sizes.value_counts().sort_index().to_dict()

    return {
        "n_images": int(n_images),
        "n_groups_multi": int(n_groups_multi),
        "n_images_in_multi": int(n_images_in_multi),
        "n_conflicting_groups": int(conflicting_groups),
        "group_size_hist": {int(k): int(v) for k, v in group_size_hist.items()},
        "max_group_size": int(group_sizes.max()) if len(group_sizes) > 0 else 0,
    }


def resolve_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve duplicates: keep first per group, drop conflicting groups entirely.

    Args:
        df: DataFrame with image_id, label, dup_group columns

    Returns:
        Tuple of (kept, dropped) DataFrames. dropped has 'reason' column.
    """
    df = df.sort_values("image_id").reset_index(drop=True)

    kept_rows = []
    dropped_rows = []

    for _, group_df in df.groupby("dup_group"):
        group_labels = group_df["label"].unique()

        if len(group_labels) > 1:
            for _, row in group_df.iterrows():
                dropped_rows.append({**row.to_dict(), "reason": "conflicting_labels"})
        else:
            sorted_group = group_df.sort_values("image_id")
            kept_rows.append(sorted_group.iloc[0].to_dict())

            for _, row in sorted_group.iloc[1:].iterrows():
                dropped_rows.append({**row.to_dict(), "reason": "duplicate"})

    kept = pd.DataFrame(kept_rows) if kept_rows else pd.DataFrame()
    dropped = pd.DataFrame(dropped_rows) if dropped_rows else pd.DataFrame()

    if len(kept) > 0:
        kept = kept.sort_values("image_id").reset_index(drop=True)
    if len(dropped) > 0:
        dropped = dropped.sort_values("image_id").reset_index(drop=True)

    return kept, dropped


def cross_dataset_overlap(
    a_hashes: pd.DataFrame, b_hashes: pd.DataFrame, threshold: int
) -> pd.DataFrame:
    """Find image pairs across two datasets whose hashes are within the threshold.

    Args:
        a_hashes: DataFrame with image_id, phash columns (dataset A).
        b_hashes: DataFrame with image_id, phash columns (dataset B).
        threshold: Maximum Hamming distance for a match.

    Returns:
        DataFrame with a_id, b_id, distance columns.
    """
    a = a_hashes.sort_values("image_id").reset_index(drop=True)
    b = b_hashes.sort_values("image_id").reset_index(drop=True)
    if len(a) == 0 or len(b) == 0:
        return pd.DataFrame(columns=["a_id", "b_id", "distance"])

    pairs = _close_pairs(_hash_bytes(a["phash"]), _hash_bytes(b["phash"]), threshold, same=False)
    return pd.DataFrame(
        [(a.at[i, "image_id"], b.at[j, "image_id"], d) for i, j, d in pairs],
        columns=["a_id", "b_id", "distance"],
    )


def cross_split_leakage(split_df: pd.DataFrame) -> int:
    """Count duplicate groups that span multiple splits.

    Args:
        split_df: DataFrame with split, dup_group columns

    Returns:
        Number of groups spanning multiple splits
    """
    return int((split_df.groupby("dup_group")["split"].nunique() > 1).sum())


def duplicate_example_pairs(
    groups: pd.DataFrame,
    hashes: pd.DataFrame,
    manifest: pd.DataFrame,
    max_pairs: int,
) -> list[tuple[str, str, int]]:
    """Extract example duplicate pairs sorted by Hamming distance (highest first).

    Args:
        groups: DataFrame with image_id, dup_group
        hashes: DataFrame with image_id, phash
        manifest: DataFrame with image_id, path
        max_pairs: max number of pairs to return

    Returns:
        List of (path_a, path_b, distance) tuples, sorted by distance descending
    """
    merged = groups.merge(hashes, on="image_id").merge(
        manifest[["image_id", "path"]], on="image_id"
    )

    pairs = []
    for _, group_df in merged.groupby("dup_group"):
        if len(group_df) > 1:
            sorted_ids = group_df.sort_values("image_id")["image_id"].values
            for i in range(len(sorted_ids) - 1):
                id_a = sorted_ids[i]
                id_b = sorted_ids[i + 1]

                hash_a = merged[merged["image_id"] == id_a]["phash"].values[0]
                hash_b = merged[merged["image_id"] == id_b]["phash"].values[0]
                dist = hamming(hash_a, hash_b)

                path_a = merged[merged["image_id"] == id_a]["path"].values[0]
                path_b = merged[merged["image_id"] == id_b]["path"].values[0]

                pairs.append((path_a, path_b, dist))

    pairs.sort(key=lambda x: -x[2])
    return pairs[:max_pairs]
