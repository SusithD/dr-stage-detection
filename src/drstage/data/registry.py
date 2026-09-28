"""Dataset registry and manifest loading."""

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

CLASS_NAMES = ("No DR", "Mild", "Moderate", "Severe", "Proliferative")


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset specification for manifest loading."""

    name: str
    csv: Path
    image_dir: Path
    id_col: str
    label_col: str
    id_has_extension: bool
    image_ext: str


def spec_from_config(name: str, cfg: dict) -> DatasetSpec:
    """Build a DatasetSpec from the config's dataset section."""
    ds = cfg["datasets"][name]
    return DatasetSpec(
        name=name,
        csv=Path(ds["csv"]),
        image_dir=Path(ds["image_dir"]),
        id_col=ds["id_col"],
        label_col=ds["label_col"],
        id_has_extension=ds["id_has_extension"],
        image_ext=ds["image_ext"],
    )


def load_manifest(spec: DatasetSpec, read_dims: bool = True) -> pd.DataFrame:
    """Load the CSV and build a manifest, dropping rows with missing or unlabelled files.
    Reads image dimensions via PIL when requested; skips files with invalid labels.

    Args:
        spec: DatasetSpec defining the dataset
        read_dims: whether to read image dimensions (via PIL lazy header)

    Returns:
        Manifest DataFrame with columns: image_id, path, label, dataset, width,
        height, file_bytes. Labels are validated to be in [0, 4].
    """
    csv_df = pd.read_csv(spec.csv)

    rows = []
    missing_count = 0
    unlabelled_count = 0

    for _, row in csv_df.iterrows():
        file_id = row[spec.id_col]
        if spec.id_has_extension:
            image_id = Path(file_id).stem
            image_path = spec.image_dir / file_id
        else:
            image_id = file_id
            image_path = spec.image_dir / (file_id + spec.image_ext)

        if not image_path.exists():
            missing_count += 1
            continue

        label = row[spec.label_col]
        if not isinstance(label, int):
            try:
                label = int(label)
            except (ValueError, TypeError):
                unlabelled_count += 1
                continue

        if label < 0 or label > 4:
            raise ValueError(f"Label {label} for image {image_id} outside [0, 4] range")

        width, height, file_bytes = 0, 0, 0
        if read_dims:
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                file_bytes = image_path.stat().st_size
            except Exception as e:
                logger.warning(f"Failed to read dimensions for {image_id}: {e}")
                continue

        rows.append(
            {
                "image_id": image_id,
                "path": str(image_path.absolute()),
                "label": label,
                "dataset": spec.name,
                "width": width,
                "height": height,
                "file_bytes": file_bytes,
            }
        )

    if missing_count > 0:
        logger.info(f"{spec.name}: dropped {missing_count} rows with missing files")
    if unlabelled_count > 0:
        logger.info(f"{spec.name}: skipped {unlabelled_count} rows without valid labels")

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    return df.sort_values("image_id").reset_index(drop=True)
