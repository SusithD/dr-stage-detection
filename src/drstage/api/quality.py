"""Heuristic fundus image quality check: exposure, contrast, sharpness and field of view.

Thresholds are the 1st (fail) and 5th (warn) percentiles of each measure over 1,200 DDR training
images, so a "fail" means the photograph is less usable than 99% of the images the model learned
from. This is a transparent screen, not a trained gradability model.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from drstage.data.preprocess import circle_crop, estimate_fundus_mask, resize_square

ANALYSIS_SIZE = 512
MASK_THRESHOLD = 10

# (fail below, warn below) unless noted; derived from the DDR training distribution
BRIGHTNESS_LOW = (36.0, 52.0)
BRIGHTNESS_HIGH_WARN = 198.0
CONTRAST_LOW = (11.0, 14.0)
SHARPNESS_LOW = (0.75, 1.8)
FIELD_LOW = (0.60, 0.69)


@dataclass
class QualityCheck:
    """Outcome of one quality measure."""

    key: str
    label: str
    status: str
    value: float
    message: str


def _grade(value: float, fail_below: float, warn_below: float) -> str:
    """Map a measured value to ok, warn or fail using the two thresholds."""
    if value < fail_below:
        return "fail"
    if value < warn_below:
        return "warn"
    return "ok"


def assess_quality(image_rgb: np.ndarray) -> list[QualityCheck]:
    """Measure exposure, contrast, sharpness and retinal field coverage of a fundus photograph.

    Args:
        image_rgb: RGB uint8 photograph.

    Returns:
        One QualityCheck per measure, each with status "ok", "warn" or "fail".
    """
    crop = circle_crop(image_rgb, threshold=MASK_THRESHOLD, min_radius_frac=0.25)
    img = resize_square(crop.image, ANALYSIS_SIZE)
    mask, _ = estimate_fundus_mask(img, MASK_THRESHOLD)
    if not mask.any():
        mask = np.ones(img.shape[:2], dtype=bool)

    lightness = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 0][mask].astype(np.float32)
    brightness = float(lightness.mean())
    contrast = float(lightness.std())
    green = cv2.GaussianBlur(img[..., 1].astype(np.float32), (0, 0), 1.0)
    sharpness = float(cv2.Laplacian(green, cv2.CV_32F)[mask].var())
    field = 0.0 if crop.fallback_used else float(mask.mean())

    if brightness > BRIGHTNESS_HIGH_WARN:
        exposure = QualityCheck(
            "exposure",
            "Exposure",
            "warn",
            brightness,
            "Overexposed; bright lesions may be washed out.",
        )
    else:
        status = _grade(brightness, *BRIGHTNESS_LOW)
        exposure = QualityCheck(
            "exposure",
            "Exposure",
            status,
            brightness,
            {
                "ok": "Well exposed.",
                "warn": "Darker than most screening photographs.",
                "fail": "Too dark to assess reliably. Retake with more illumination.",
            }[status],
        )

    contrast_status = _grade(contrast, *CONTRAST_LOW)
    sharp_status = _grade(sharpness, *SHARPNESS_LOW)
    field_status = _grade(field, *FIELD_LOW)
    return [
        exposure,
        QualityCheck(
            "contrast",
            "Contrast",
            contrast_status,
            contrast,
            {
                "ok": "Vessels and background are well separated.",
                "warn": "Low contrast; small lesions may be missed.",
                "fail": "Very low contrast, often from cataract or haze. Consider retaking.",
            }[contrast_status],
        ),
        QualityCheck(
            "sharpness",
            "Focus",
            sharp_status,
            sharpness,
            {
                "ok": "In focus.",
                "warn": "Slightly blurred; microaneurysms may be hard to see.",
                "fail": "Out of focus. Retake the photograph.",
            }[sharp_status],
        ),
        QualityCheck(
            "field",
            "Field of view",
            field_status,
            field,
            {
                "ok": "The retina fills the frame.",
                "warn": "The retina covers less of the frame than usual.",
                "fail": "The retinal disc could not be located. Check this is a fundus photograph.",
            }[field_status],
        ),
    ]


def is_gradable(checks: list[QualityCheck]) -> bool:
    """A photograph is gradable when no check fails."""
    return all(c.status != "fail" for c in checks)
