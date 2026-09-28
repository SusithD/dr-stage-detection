"""Image preprocessing: loading, cropping, resizing, enhancement."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from drstage.utils.io import ensure_dir


@dataclass
class CropResult:
    """Cropped image with bounding box and fallback flag."""

    image: np.ndarray
    bbox: tuple[int, int, int, int]
    fallback_used: bool


def load_rgb(path: Path) -> np.ndarray:
    """Load image as RGB, converting from BGR (OpenCV format)."""
    if not Path(path).exists():
        raise FileNotFoundError(path)
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to decode image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_rgb(img: np.ndarray, path: Path, fmt: str = "png", jpeg_quality: int = 95) -> None:
    """Save image in the specified format."""
    ensure_dir(path.parent)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if fmt.lower() == "jpg" or fmt.lower() == "jpeg":
        cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    else:
        cv2.imwrite(str(path), img_bgr)


def estimate_fundus_mask(
    img: np.ndarray, threshold: int
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Find the fundus disc by thresholding the brightest channel and extracting the
    largest connected component."""
    binary = (img.max(axis=2) > threshold).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        return np.zeros(binary.shape, dtype=bool), (0, 0, 0)

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = (int(v) for v in stats[largest, :4])
    cx, cy, r = x + w // 2, y + h // 2, max(w, h) // 2
    return labels == largest, (cx, cy, r)


def circle_crop(img: np.ndarray, threshold: int = 10, min_radius_frac: float = 0.25) -> CropResult:
    """Crop the retina out of the black border and pad it to a square. Falls back to
    center crop if the detected disc is too small."""
    h, w = img.shape[:2]
    mask, (cx, cy, r) = estimate_fundus_mask(img, threshold)

    min_h_w = min(h, w)
    min_radius = int(min_radius_frac * min_h_w)

    if r < min_radius or r == 0:
        bbox = (0, 0, w, h)
        size = min_h_w
        x_start = (w - size) // 2
        y_start = (h - size) // 2
        cropped = img[y_start : y_start + size, x_start : x_start + size]
        return CropResult(cropped, bbox, True)

    x_min = max(0, cx - r)
    x_max = min(w, cx + r)
    y_min = max(0, cy - r)
    y_max = min(h, cy + r)

    cropped = img[y_min:y_max, x_min:x_max]

    bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

    crop_h, crop_w = cropped.shape[:2]
    size = max(crop_h, crop_w)

    pad_y_top = (size - crop_h) // 2
    pad_y_bottom = size - crop_h - pad_y_top
    pad_x_left = (size - crop_w) // 2
    pad_x_right = size - crop_w - pad_x_left

    padded = cv2.copyMakeBorder(
        cropped, pad_y_top, pad_y_bottom, pad_x_left, pad_x_right, cv2.BORDER_CONSTANT, value=0
    )

    return CropResult(padded, bbox, False)


def resize_square(img: np.ndarray, size: int) -> np.ndarray:
    """Resize to a square. Uses INTER_AREA for downsampling and INTER_CUBIC for
    upsampling to maintain quality."""
    h, w = img.shape[:2]
    if (h, w) == (size, size):
        return img
    if h > size or w > size:
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)


def circular_mask(size: int, radius_frac: float) -> np.ndarray:
    """Create a circular mask to isolate the fundus region."""
    y, x = np.ogrid[:size, :size]
    cx, cy = size // 2, size // 2
    radius = int(size * radius_frac / 2)
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
    return mask


def ben_graham(
    img: np.ndarray,
    sigma: float,
    alpha: float = 4,
    beta: float = -4,
    gamma: float = 128,
    mask_radius_frac: float = 0.92,
) -> np.ndarray:
    """Enhance the fundus with weighted Gaussian blur and apply a circular mask. Helps
    bring out vessel structure and reduces the effect of vignetting."""
    blur = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma)
    result = cv2.addWeighted(img.astype(np.float32), alpha, blur, beta, gamma)
    result = np.clip(result, 0, 255).astype(np.uint8)

    mask = circular_mask(img.shape[0], mask_radius_frac)
    result[~mask] = 0

    return result


def clahe_l(img: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Boost local contrast via CLAHE on the L channel in LAB space. Highlights vessel
    networks and microaneurysms."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_channel = lab[..., 0]

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l_equalized = clahe.apply(l_channel)

    lab[..., 0] = l_equalized
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return result


def preprocess_image(img: np.ndarray, method: str, size: int, cfg: dict) -> tuple[np.ndarray, bool]:
    """Apply the full preprocessing pipeline: circle crop, resize, and enhancement.
    Returns the processed image and a flag indicating whether the fallback crop was used."""
    circle_crop_result = circle_crop(img, **cfg["circle_crop"])
    cropped_img = circle_crop_result.image
    fallback_used = circle_crop_result.fallback_used

    resized = resize_square(cropped_img, size)

    if method == "ben_graham":
        sigma = cfg["ben_graham"]["sigma_frac"] * size
        processed = ben_graham(
            resized,
            sigma=sigma,
            **{k: v for k, v in cfg["ben_graham"].items() if k != "sigma_frac"},
        )
    elif method == "clahe_l":
        processed = clahe_l(resized, **cfg["clahe_l"])
    elif method == "none":
        processed = resized
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")

    return processed, fallback_used
