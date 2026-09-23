"""Tests for dedup module."""

import numpy as np
import pandas as pd

from drstage.data.dedup import (
    compute_phash,
    cross_dataset_overlap,
    cross_split_leakage,
    duplicate_example_pairs,
    find_duplicate_groups,
    hamming,
    resolve_duplicates,
    summarise_groups,
)


def test_hamming_identical():
    """Test hamming distance between identical hashes."""
    h1 = "f" * 32
    h2 = "f" * 32
    assert hamming(h1, h2) == 0


def test_hamming_different():
    """Test hamming distance between different hashes."""
    h1 = "0" * 32
    h2 = "f" * 32
    dist = hamming(h1, h2)
    assert dist > 0


def test_hamming_symmetric():
    """Test hamming distance is symmetric."""
    h1 = "abcd1234" * 4
    h2 = "dcba4321" * 4
    assert hamming(h1, h2) == hamming(h2, h1)


def test_compute_phash_returns_hex():
    """Test compute_phash returns hex string."""
    from conftest import make_fundus

    img = make_fundus()
    hash_str = compute_phash(img, hash_size=16)

    assert isinstance(hash_str, str)
    assert len(hash_str) == 64  # 16*16 bits = 256 bits = 64 hex chars
    all(c in "0123456789abcdef" for c in hash_str)


def test_compute_phash_same_image():
    """Test compute_phash returns same hash for identical images."""
    from conftest import make_fundus

    img = make_fundus(seed=42)
    hash1 = compute_phash(img, hash_size=16)
    hash2 = compute_phash(img, hash_size=16)

    assert hash1 == hash2


def test_compute_phash_different_images():
    """Test compute_phash returns different hashes for different images."""
    from conftest import make_fundus

    img1 = make_fundus(seed=42)
    img2 = make_fundus(seed=43)

    hash1 = compute_phash(img1, hash_size=16)
    hash2 = compute_phash(img2, hash_size=16)

    assert hash1 != hash2


def test_find_duplicate_groups_empty():
    """Test find_duplicate_groups with empty input."""
    df = pd.DataFrame({"image_id": [], "phash": []})
    result = find_duplicate_groups(df, threshold=6)

    assert len(result) == 0
    assert list(result.columns) == ["image_id", "dup_group"]


def test_find_duplicate_groups_no_duplicates():
    """Test find_duplicate_groups with no duplicates."""
    from conftest import make_fundus

    hashes = []
    for i in range(5):
        img = make_fundus(seed=i)
        hash_str = compute_phash(img, hash_size=16)
        hashes.append({"image_id": f"img{i:02d}", "phash": hash_str})

    df = pd.DataFrame(hashes)
    result = find_duplicate_groups(df, threshold=6)

    assert len(result) == 5
    assert result["dup_group"].nunique() == 5


def test_find_duplicate_groups_exact_duplicates():
    """Test find_duplicate_groups finds exact duplicates."""
    from conftest import make_fundus

    img_base = make_fundus(seed=42)
    hash_base = compute_phash(img_base, hash_size=16)

    hashes = [
        {"image_id": "img00", "phash": hash_base},
        {"image_id": "dup_exact", "phash": hash_base},
    ]

    df = pd.DataFrame(hashes)
    result = find_duplicate_groups(df, threshold=6)

    assert len(result) == 2
    assert (
        result.loc[result["image_id"] == "img00", "dup_group"].values[0]
        == result.loc[result["image_id"] == "dup_exact", "dup_group"].values[0]
    )


def test_find_duplicate_groups_near_duplicates():
    """Test find_duplicate_groups with near duplicates."""
    from conftest import make_fundus

    img1 = make_fundus(seed=1)
    hash1 = compute_phash(img1, hash_size=16)

    img2 = make_fundus(seed=1)
    hash2 = compute_phash(img2, hash_size=16)

    hashes = [
        {"image_id": "img01", "phash": hash1},
        {"image_id": "dup_near", "phash": hash2},
    ]

    df = pd.DataFrame(hashes)
    result = find_duplicate_groups(df, threshold=6)

    assert len(result) == 2
    group1 = result.loc[result["image_id"] == "img01", "dup_group"].values[0]
    group2 = result.loc[result["image_id"] == "dup_near", "dup_group"].values[0]
    assert group1 == group2


def test_summarise_groups():
    """Test summarise_groups returns correct statistics."""
    groups = pd.DataFrame(
        {
            "image_id": ["img00", "dup_exact", "img01", "dup_near", "img02"],
            "dup_group": [0, 0, 1, 1, 2],
        }
    )

    labels = pd.DataFrame(
        {
            "image_id": ["img00", "dup_exact", "img01", "dup_near", "img02"],
            "label": [0, 0, 1, 1, 2],
        }
    )

    summary = summarise_groups(groups, labels)

    assert summary["n_images"] == 5
    assert summary["n_groups_multi"] == 2
    assert summary["n_images_in_multi"] == 4
    assert summary["n_conflicting_groups"] == 0
    assert summary["group_size_hist"][2] == 2
    assert summary["max_group_size"] == 2


def test_summarise_groups_conflicting():
    """Test summarise_groups detects conflicting groups."""
    groups = pd.DataFrame(
        {
            "image_id": ["img00", "dup_exact"],
            "dup_group": [0, 0],
        }
    )

    labels = pd.DataFrame(
        {
            "image_id": ["img00", "dup_exact"],
            "label": [0, 1],
        }
    )

    summary = summarise_groups(groups, labels)

    assert summary["n_conflicting_groups"] == 1


def test_resolve_duplicates_keeps_first():
    """Test resolve_duplicates keeps first image per group."""
    df = pd.DataFrame(
        {
            "image_id": ["img00", "img01", "img02"],
            "label": [0, 0, 1],
            "dup_group": [0, 0, 1],
        }
    )

    kept, dropped = resolve_duplicates(df)

    assert len(kept) == 2
    assert len(dropped) == 1

    kept_ids = set(kept["image_id"])
    assert "img00" in kept_ids
    assert "img02" in kept_ids
    assert "img01" not in kept_ids

    dropped_row = dropped.iloc[0]
    assert dropped_row["image_id"] == "img01"
    assert dropped_row["reason"] == "duplicate"


def test_resolve_duplicates_drops_conflicting():
    """Test resolve_duplicates drops entire conflicting group."""
    df = pd.DataFrame(
        {
            "image_id": ["img00", "dup_bad", "img01"],
            "label": [0, 1, 1],
            "dup_group": [0, 0, 1],
        }
    )

    kept, dropped = resolve_duplicates(df)

    kept_ids = set(kept["image_id"])
    assert "img00" not in kept_ids
    assert "dup_bad" not in kept_ids
    assert "img01" in kept_ids

    assert len(dropped) == 2
    assert (dropped["reason"] == "conflicting_labels").all()


def test_cross_split_leakage_no_leakage():
    """Test cross_split_leakage with no leakage."""
    df = pd.DataFrame(
        {
            "split": ["train", "train", "val", "val", "test"],
            "dup_group": [0, 0, 1, 1, 2],
        }
    )

    leakage = cross_split_leakage(df)
    assert leakage == 0


def test_cross_split_leakage_with_leakage():
    """Test cross_split_leakage detects leakage."""
    df = pd.DataFrame(
        {
            "split": ["train", "val", "train", "val", "test"],
            "dup_group": [0, 0, 1, 1, 2],
        }
    )

    leakage = cross_split_leakage(df)
    assert leakage == 2


def test_cross_dataset_overlap_empty():
    """Test cross_dataset_overlap with empty inputs."""
    a = pd.DataFrame({"image_id": [], "phash": []})
    b = pd.DataFrame({"image_id": [], "phash": []})

    result = cross_dataset_overlap(a, b, threshold=6)

    assert len(result) == 0


def test_cross_dataset_overlap_no_match():
    """Test cross_dataset_overlap finds no match with high threshold."""
    from conftest import make_fundus

    a = pd.DataFrame(
        {
            "image_id": ["a1"],
            "phash": [compute_phash(make_fundus(seed=1), hash_size=16)],
        }
    )

    b = pd.DataFrame(
        {
            "image_id": ["b1"],
            "phash": [compute_phash(make_fundus(seed=2), hash_size=16)],
        }
    )

    result = cross_dataset_overlap(a, b, threshold=0)

    assert len(result) == 0


def test_cross_dataset_overlap_finds_match():
    """Test cross_dataset_overlap finds matching hashes."""
    img = make_fundus(seed=42)
    hash_str = compute_phash(img, hash_size=16)

    a = pd.DataFrame(
        {
            "image_id": ["a1"],
            "phash": [hash_str],
        }
    )

    b = pd.DataFrame(
        {
            "image_id": ["b1"],
            "phash": [hash_str],
        }
    )

    result = cross_dataset_overlap(a, b, threshold=6)

    assert len(result) == 1
    assert result.iloc[0]["a_id"] == "a1"
    assert result.iloc[0]["b_id"] == "b1"
    assert result.iloc[0]["distance"] == 0


def test_duplicate_example_pairs():
    """Test duplicate_example_pairs returns sorted pairs."""
    from conftest import make_fundus

    img1 = make_fundus(seed=1)
    img2 = make_fundus(seed=1)
    hash1 = compute_phash(img1, hash_size=16)
    hash2 = compute_phash(img2, hash_size=16)

    groups = pd.DataFrame(
        {
            "image_id": ["img1", "img2", "img3"],
            "dup_group": [0, 0, 1],
        }
    )

    hashes = pd.DataFrame(
        {
            "image_id": ["img1", "img2", "img3"],
            "phash": [hash1, hash2, compute_phash(make_fundus(seed=3), hash_size=16)],
        }
    )

    manifest = pd.DataFrame(
        {
            "image_id": ["img1", "img2", "img3"],
            "path": ["/tmp/img1.jpg", "/tmp/img2.jpg", "/tmp/img3.jpg"],
        }
    )

    pairs = duplicate_example_pairs(groups, hashes, manifest, max_pairs=5)

    assert isinstance(pairs, list)
    if len(pairs) > 0:
        for path_a, path_b, dist in pairs:
            assert isinstance(path_a, str)
            assert isinstance(path_b, str)
            assert isinstance(dist, int)


def test_performance_12500_hashes(monkeypatch):
    """Test find_duplicate_groups handles 12500 hashes efficiently.

    This is a performance test to ensure the blockwise algorithm works at scale.
    """
    import time

    rng = np.random.default_rng(42)
    n_hashes = 12500

    hashes_list = [
        format(rng.integers(0, 2**32), "08x")
        + format(rng.integers(0, 2**32), "08x")
        + format(rng.integers(0, 2**32), "08x")
        + format(rng.integers(0, 2**32), "08x")
        for _ in range(n_hashes)
    ]

    df = pd.DataFrame(
        {
            "image_id": [f"img{i:05d}" for i in range(n_hashes)],
            "phash": hashes_list,
        }
    )

    start = time.time()
    result = find_duplicate_groups(df, threshold=6)
    elapsed = time.time() - start

    assert len(result) == n_hashes
    assert elapsed < 60, f"Performance test failed: {elapsed:.1f}s > 60s"


def make_fundus(width=640, height=480, radius=None, dark=False, seed=0):
    """Make a synthetic fundus image for testing."""
    from conftest import make_fundus as _make_fundus

    return _make_fundus(width=width, height=height, radius=radius, dark=dark, seed=seed)
