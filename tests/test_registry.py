"""Tests for registry module."""

import pandas as pd
import pytest

from drstage.data.registry import CLASS_NAMES, DatasetSpec, load_manifest, spec_from_config


def test_class_names():
    """Test CLASS_NAMES tuple."""
    assert CLASS_NAMES == ("No DR", "Mild", "Moderate", "Severe", "Proliferative")
    assert len(CLASS_NAMES) == 5


def test_dataset_spec_frozen():
    """Test DatasetSpec is immutable."""
    from pathlib import Path

    spec = DatasetSpec(
        name="test",
        csv=Path("test.csv"),
        image_dir=Path("imgs"),
        id_col="id",
        label_col="label",
        id_has_extension=True,
        image_ext="",
    )

    with pytest.raises(AttributeError):
        spec.name = "other"


def test_spec_from_config():
    """Test spec_from_config parses config dict correctly."""
    cfg = {
        "datasets": {
            "ddr": {
                "csv": "data/raw/ddr/DR_grading.csv",
                "image_dir": "data/raw/ddr/DR_grading/DR_grading",
                "id_col": "id_code",
                "label_col": "diagnosis",
                "id_has_extension": True,
                "image_ext": "",
            }
        }
    }

    spec = spec_from_config("ddr", cfg)

    assert spec.name == "ddr"
    assert spec.id_col == "id_code"
    assert spec.label_col == "diagnosis"
    assert spec.id_has_extension is True
    assert spec.image_ext == ""


def test_load_manifest_tiny_manifest(tiny_manifest):
    """Test load_manifest with tiny_manifest fixture.

    Should drop missing_file row and exclude unlabelled.jpg.
    Result should have 12 rows.
    """
    spec = DatasetSpec(
        name="test",
        csv=tiny_manifest["csv"],
        image_dir=tiny_manifest["image_dir"],
        id_col="id_code",
        label_col="diagnosis",
        id_has_extension=True,
        image_ext="",
    )

    manifest = load_manifest(spec, read_dims=True)

    assert len(manifest) == tiny_manifest["n_labelled_present"], (
        f"Expected {tiny_manifest['n_labelled_present']} rows, got {len(manifest)}"
    )


def test_load_manifest_columns(tiny_manifest):
    """Test load_manifest returns correct columns."""
    spec = DatasetSpec(
        name="test",
        csv=tiny_manifest["csv"],
        image_dir=tiny_manifest["image_dir"],
        id_col="id_code",
        label_col="diagnosis",
        id_has_extension=True,
        image_ext="",
    )

    manifest = load_manifest(spec, read_dims=True)

    expected_columns = {"image_id", "path", "label", "dataset", "width", "height", "file_bytes"}
    assert set(manifest.columns) == expected_columns


def test_load_manifest_dimensions(tiny_manifest):
    """Test load_manifest reads image dimensions correctly."""
    spec = DatasetSpec(
        name="test",
        csv=tiny_manifest["csv"],
        image_dir=tiny_manifest["image_dir"],
        id_col="id_code",
        label_col="diagnosis",
        id_has_extension=True,
        image_ext="",
    )

    manifest = load_manifest(spec, read_dims=True)

    assert (manifest["width"] > 0).all(), "All widths should be > 0"
    assert (manifest["height"] > 0).all(), "All heights should be > 0"
    assert (manifest["file_bytes"] > 0).all(), "All file_bytes should be > 0"


def test_load_manifest_labels_are_int(tiny_manifest):
    """Test load_manifest labels are integers."""
    spec = DatasetSpec(
        name="test",
        csv=tiny_manifest["csv"],
        image_dir=tiny_manifest["image_dir"],
        id_col="id_code",
        label_col="diagnosis",
        id_has_extension=True,
        image_ext="",
    )

    manifest = load_manifest(spec, read_dims=True)

    assert manifest["label"].dtype in [int, "int64", "int32"], (
        f"Labels should be int, got {manifest['label'].dtype}"
    )


def test_load_manifest_sorted_by_image_id(tiny_manifest):
    """Test load_manifest returns sorted by image_id."""
    spec = DatasetSpec(
        name="test",
        csv=tiny_manifest["csv"],
        image_dir=tiny_manifest["image_dir"],
        id_col="id_code",
        label_col="diagnosis",
        id_has_extension=True,
        image_ext="",
    )

    manifest = load_manifest(spec, read_dims=True)

    assert manifest["image_id"].is_monotonic_increasing, "Should be sorted by image_id"


def test_load_manifest_invalid_labels(tmp_path):
    """Test load_manifest raises ValueError for labels outside 0..4."""
    import cv2
    from conftest import make_fundus

    image_dir = tmp_path / "images"
    image_dir.mkdir()

    img = make_fundus()
    cv2.imwrite(str(image_dir / "img.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(
        [("img.jpg", 5)],
        columns=["id_code", "diagnosis"],
    ).to_csv(csv_path, index=False)

    spec = DatasetSpec(
        name="test",
        csv=csv_path,
        image_dir=image_dir,
        id_col="id_code",
        label_col="diagnosis",
        id_has_extension=True,
        image_ext="",
    )

    with pytest.raises(ValueError, match="outside.*0.*4"):
        load_manifest(spec, read_dims=True)


def test_load_manifest_no_dims(tiny_manifest):
    """Test load_manifest with read_dims=False."""
    spec = DatasetSpec(
        name="test",
        csv=tiny_manifest["csv"],
        image_dir=tiny_manifest["image_dir"],
        id_col="id_code",
        label_col="diagnosis",
        id_has_extension=True,
        image_ext="",
    )

    manifest = load_manifest(spec, read_dims=False)

    assert len(manifest) > 0
    assert (manifest["width"] == 0).all()
    assert (manifest["height"] == 0).all()
    assert (manifest["file_bytes"] == 0).all()
