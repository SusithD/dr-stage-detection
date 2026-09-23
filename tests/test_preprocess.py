"""Tests for preprocessing, caching, and I/O utilities."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drstage.data.cache import build_cache, cache_dir
from drstage.data.preprocess import (
    ben_graham,
    circle_crop,
    clahe_l,
    preprocess_image,
)
from drstage.utils.io import write_csv


class TestCircleCrop:
    """Tests for circle_crop function."""

    def test_circle_crop_1000x700_square_output(self, synthetic_fundus):
        """Test circle_crop on 1000x700 synthetic disc image -> square output."""
        img = synthetic_fundus(width=1000, height=700, radius=200)
        result = circle_crop(img, threshold=10, min_radius_frac=0.25)

        assert result.image.shape[0] == result.image.shape[1], "Output should be square"
        assert result.image.ndim == 3, "Output should be 3D (HxWx3)"

        non_black = np.any(result.image != 0, axis=2)
        non_black_width = np.max(np.where(non_black.any(axis=0))[0]) - np.min(
            np.where(non_black.any(axis=0))[0]
        )
        coverage = non_black_width / result.image.shape[1]
        assert coverage >= 0.95, f"Disc should span >=95% of output width, got {coverage:.2%}"

    def test_circle_crop_512x512_disc_touches_edges(self, synthetic_fundus):
        """Test circle_crop on 512x512 image where disc touches edges."""
        img = synthetic_fundus(width=512, height=512, radius=256)
        result = circle_crop(img, threshold=10, min_radius_frac=0.25)

        h, w = result.image.shape[:2]
        assert abs(h - 512) <= 4, f"Height should be ~512, got {h}"
        assert abs(w - 512) <= 4, f"Width should be ~512, got {w}"
        assert not result.fallback_used, "Fallback should not be used for valid disc"

    def test_circle_crop_very_dark_image(self, synthetic_fundus):
        """Test circle_crop on very dark image triggers fallback."""
        img = synthetic_fundus(width=640, height=480, dark=True)
        result = circle_crop(img, threshold=10, min_radius_frac=0.25)

        assert result.fallback_used, "Fallback should be used for dark image"
        h, w = result.image.shape[:2]
        assert h == w, "Fallback result should be square"


class TestBenGraham:
    """Tests for Ben Graham preprocessing."""

    def test_ben_graham_output_properties(self, synthetic_fundus):
        """Test ben_graham output dtype, shape, and mask properties."""
        img = synthetic_fundus(width=512, height=512, radius=200)
        sigma = 10.0
        result = ben_graham(img, sigma=sigma, mask_radius_frac=0.92)

        assert result.dtype == np.uint8, f"Output dtype should be uint8, got {result.dtype}"
        assert result.shape == img.shape, f"Shape should be preserved: {img.shape}"

        h, w = result.shape[:2]
        y, x = np.ogrid[:h, :w]
        cx, cy = w // 2, h // 2
        radius = int(w * 0.92 / 2)
        outside_mask = (x - cx) ** 2 + (y - cy) ** 2 > radius**2

        corners = result[outside_mask]
        assert np.all(corners == 0), "Pixels outside mask should be zero"


class TestCLAHE:
    """Tests for CLAHE preprocessing."""

    def test_clahe_l_preserves_shape_dtype(self, synthetic_fundus):
        """Test clahe_l preserves shape and dtype."""
        img = synthetic_fundus(width=512, height=512, radius=200)
        result = clahe_l(img, clip_limit=2.0, tile_grid=8)

        assert result.shape == img.shape, f"Shape should be preserved: {img.shape}"
        assert result.dtype == np.uint8, f"Dtype should be uint8, got {result.dtype}"


class TestPreprocessImage:
    """Tests for preprocess_image function."""

    @pytest.fixture
    def preprocess_cfg(self):
        """Minimal preprocessing config."""
        return {
            "size": 256,
            "circle_crop": {"threshold": 10, "min_radius_frac": 0.25},
            "ben_graham": {
                "sigma_frac": 0.0195,
                "alpha": 4,
                "beta": -4,
                "gamma": 128,
                "mask_radius_frac": 0.92,
            },
            "clahe_l": {"clip_limit": 2.0, "tile_grid": 8},
        }

    def test_preprocess_image_all_methods(self, synthetic_fundus, preprocess_cfg):
        """Test preprocess_image for each method returns correct shape/dtype."""
        img = synthetic_fundus(width=512, height=512, radius=200)
        size = 256

        for method in ["ben_graham", "clahe_l", "none"]:
            result, fallback_used = preprocess_image(img, method, size, preprocess_cfg)
            assert result.shape == (size, size, 3), (
                f"Method {method}: expected shape ({size}, {size}, 3), got {result.shape}"
            )
            assert result.dtype == np.uint8, (
                f"Method {method}: expected dtype uint8, got {result.dtype}"
            )

    def test_preprocess_image_unknown_method_raises(self, synthetic_fundus, preprocess_cfg):
        """Test preprocess_image raises ValueError for unknown method."""
        img = synthetic_fundus(width=512, height=512, radius=200)
        with pytest.raises(ValueError, match="Unknown preprocessing method"):
            preprocess_image(img, "unknown_method", 256, preprocess_cfg)


class TestBuildCache:
    """Tests for build_cache function."""

    @pytest.fixture
    def cache_cfg(self):
        """Minimal cache config."""
        return {
            "size": 256,
            "cache_format": "png",
            "cache_jpeg_quality": 95,
            "circle_crop": {"threshold": 10, "min_radius_frac": 0.25},
            "ben_graham": {
                "sigma_frac": 0.0195,
                "alpha": 4,
                "beta": -4,
                "gamma": 128,
                "mask_radius_frac": 0.92,
            },
            "clahe_l": {"clip_limit": 2.0, "tile_grid": 8},
        }

    def test_build_cache_creates_files_and_manifest(self, tiny_manifest, tmp_path, cache_cfg):
        """Test build_cache writes PNG files and _manifest.json."""
        csv_path = tiny_manifest["csv"]
        image_dir = tiny_manifest["image_dir"]
        processed_root = tmp_path / "processed"

        df = pd.read_csv(csv_path)
        manifest = pd.DataFrame(
            {
                "image_id": [Path(row[0]).stem for row in df.values],
                "path": [str(image_dir / row[0]) for row in df.values],
            }
        )

        result = build_cache(
            manifest,
            dataset="test",
            method="none",
            cfg=cache_cfg,
            processed_root=processed_root,
            num_workers=2,
            overwrite=False,
        )

        cache_root = cache_dir(processed_root, "test", "none", cache_cfg["size"])

        assert result is not None, "build_cache should return DataFrame"
        assert "image_id" in result.columns, "Result should have image_id column"
        assert "cache_path" in result.columns, "Result should have cache_path column"
        assert "error" in result.columns, "Result should have error column"
        assert "crop_fallback" in result.columns, "Result should have crop_fallback column"

        manifest_json = cache_root / "_manifest.json"
        assert manifest_json.exists(), "_manifest.json should be created"

        with open(manifest_json) as f:
            manifest_data = json.load(f)
        assert "n_images" in manifest_data, "_manifest should have n_images"
        assert "n_errors" in manifest_data, "_manifest should have n_errors"
        assert "created_at" in manifest_data, "_manifest should have created_at"

    def test_build_cache_skip_existing_without_overwrite(self, tiny_manifest, tmp_path, cache_cfg):
        """Test build_cache skips existing files when overwrite=False."""
        csv_path = tiny_manifest["csv"]
        image_dir = tiny_manifest["image_dir"]
        processed_root = tmp_path / "processed"

        df = pd.read_csv(csv_path)
        manifest = pd.DataFrame(
            {
                "image_id": [Path(row[0]).stem for row in df.values],
                "path": [str(image_dir / row[0]) for row in df.values],
            }
        )

        build_cache(
            manifest,
            dataset="test",
            method="none",
            cfg=cache_cfg,
            processed_root=processed_root,
            num_workers=2,
            overwrite=False,
        )

        cache_root = cache_dir(processed_root, "test", "none", cache_cfg["size"])
        first_file = list(cache_root.glob("*.png"))[0]
        mtime_first = first_file.stat().st_mtime

        time.sleep(0.1)

        build_cache(
            manifest,
            dataset="test",
            method="none",
            cfg=cache_cfg,
            processed_root=processed_root,
            num_workers=2,
            overwrite=False,
        )

        mtime_second = first_file.stat().st_mtime
        assert mtime_first == mtime_second, "File mtime should not change when skipping"

    def test_build_cache_corrupt_file_no_exception(self, tiny_manifest, tmp_path, cache_cfg):
        """Test build_cache handles corrupt files gracefully."""
        csv_path = tiny_manifest["csv"]
        image_dir = tiny_manifest["image_dir"]
        processed_root = tmp_path / "processed"

        corrupt_file = image_dir / "corrupt.jpg"
        corrupt_file.write_bytes(b"not an image")

        df = pd.read_csv(csv_path)
        df = pd.concat(
            [
                df,
                pd.DataFrame({"id_code": ["corrupt.jpg"], "diagnosis": [0]}),
            ],
            ignore_index=True,
        )

        manifest = pd.DataFrame(
            {
                "image_id": [Path(row[0]).stem for row in df.values],
                "path": [str(image_dir / row[0]) for row in df.values],
            }
        )

        result = build_cache(
            manifest,
            dataset="test",
            method="none",
            cfg=cache_cfg,
            processed_root=processed_root,
            num_workers=2,
            overwrite=False,
        )

        corrupt_row = result[result["image_id"] == "corrupt"]
        assert not corrupt_row.empty, "Result should contain corrupt image row"
        assert corrupt_row.iloc[0]["error"] != "", "Error should not be empty for corrupt file"

        result_valid = result[
            (result["image_id"] != "corrupt") & (result["image_id"] != "missing_file")
        ]
        assert (result_valid["error"] == "").all(), "Valid images should have no error"


class TestWriteCSV:
    """Tests for write_csv function."""

    def test_write_csv_atomic(self, tmp_path):
        """Test write_csv is atomic and content roundtrips."""
        csv_path = tmp_path / "subdir" / "test.csv"
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        write_csv(df, csv_path)

        assert csv_path.exists(), "File should exist after write"

        df_read = pd.read_csv(csv_path)
        pd.testing.assert_frame_equal(df, df_read, "Content should roundtrip exactly")

    def test_write_csv_creates_parent_dirs(self, tmp_path):
        """Test write_csv creates parent directories."""
        csv_path = tmp_path / "a" / "b" / "c" / "test.csv"
        df = pd.DataFrame({"col": [1, 2]})

        write_csv(df, csv_path)

        assert csv_path.parent.exists(), "Parent directories should be created"
        assert csv_path.exists(), "File should exist"
