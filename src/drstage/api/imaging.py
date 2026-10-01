"""Image decoding, encoding to data URLs, and thumbnail generation."""

import base64
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def decode_upload(data: bytes) -> np.ndarray:
    """Load uploaded image bytes and convert to RGB uint8."""
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError("The file isn't a readable image. Upload a JPG or PNG fundus photograph.")

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("The file isn't a readable image. Upload a JPG or PNG fundus photograph.")

    if img.dtype != np.uint8:
        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def to_png_data_url(img: np.ndarray) -> str:
    """Encode image to PNG and wrap as a data URL for embedding in JSON."""
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if len(img.shape) == 3 and img.shape[2] == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        img_bgra = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        img_bgr = img_bgra
    else:
        raise ValueError("Image must be RGB or RGBA uint8")

    _, buffer = cv2.imencode(".png", img_bgr)
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def heatmap_to_rgba(cam: np.ndarray, colormap: str = "inferno") -> np.ndarray:
    """Map heatmap values to RGBA using a colourmap, with alpha from the heatmap itself."""
    if cam.min() < 0 or cam.max() > 1:
        cam = np.clip(cam, 0, 1)

    cmap = plt.get_cmap(colormap)
    rgba = cmap(cam)

    rgba_uint8 = (rgba * 255).astype(np.uint8)

    alpha = (cam * 255).astype(np.uint8)
    rgba_uint8[..., 3] = alpha

    return rgba_uint8


def thumbnail_jpeg(path: Path, size: int = 256) -> bytes:
    """Load, circle-crop, resize and encode to JPEG."""
    from drstage.data.preprocess import circle_crop, load_rgb, resize_square

    img = load_rgb(path)
    circle_crop_result = circle_crop(img)
    cropped = circle_crop_result.image
    resized = resize_square(cropped, size)

    img_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buffer.tobytes()


def summarise_attention(cam: np.ndarray) -> tuple[str, float]:
    """Summarise attention map location and spread on a 3x3 grid."""
    cam = np.clip(cam, 0, 1)

    weights = cam**2
    weight_sum = weights.sum()

    if weight_sum > 0:
        h_indices, w_indices = np.mgrid[0 : cam.shape[0], 0 : cam.shape[1]]
        centre_h = (weights * h_indices).sum() / weight_sum
        centre_w = (weights * w_indices).sum() / weight_sum
    else:
        centre_h = cam.shape[0] / 2
        centre_w = cam.shape[1] / 2

    h_norm = centre_h / cam.shape[0]
    w_norm = centre_w / cam.shape[1]

    row = "upper" if h_norm < 1 / 3 else "lower" if h_norm > 2 / 3 else ""
    col = "left" if w_norm < 1 / 3 else "right" if w_norm > 2 / 3 else ""

    parts = [row, col]
    region_str = " ".join(p for p in parts if p) or "centre"

    spread = float(np.round((cam >= 0.5).sum() / cam.size, 3))

    return region_str, spread
