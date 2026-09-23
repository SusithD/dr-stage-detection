"""Image encoding, decoding, and transformations for the API."""

import base64
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def decode_upload(data: bytes) -> np.ndarray:
    """Decode uploaded image bytes to RGB uint8 array.

    Args:
        data: Image file bytes (JPG, PNG, TIFF).

    Returns:
        RGB uint8 ndarray.

    Raises:
        ValueError: If data is not a valid, decodable image.
    """
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
    """Convert RGB or RGBA uint8 image to PNG data URL.

    Args:
        img: RGB or RGBA uint8 ndarray.

    Returns:
        Data URL string (data:image/png;base64,...).
    """
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
    """Convert CAM (float32, [0, 1]) to RGBA using a perceptual colormap.

    Args:
        cam: Float32 HxW array in [0, 1].
        colormap: Matplotlib colormap name.

    Returns:
        RGBA uint8 HxW4 array.
    """
    if cam.min() < 0 or cam.max() > 1:
        cam = np.clip(cam, 0, 1)

    cmap = plt.get_cmap(colormap)
    rgba = cmap(cam)

    rgba_uint8 = (rgba * 255).astype(np.uint8)

    alpha = (cam * 255).astype(np.uint8)
    rgba_uint8[..., 3] = alpha

    return rgba_uint8


def thumbnail_jpeg(path: Path, size: int = 256) -> bytes:
    """Load image, circle crop, resize, and encode to JPEG.

    Args:
        path: Path to image file.
        size: Target size in pixels (square).

    Returns:
        JPEG bytes.

    Raises:
        FileNotFoundError: If image file does not exist.
        ValueError: If image cannot be decoded.
    """
    from drstage.data.preprocess import circle_crop, load_rgb, resize_square

    img = load_rgb(path)
    circle_crop_result = circle_crop(img)
    cropped = circle_crop_result.image
    resized = resize_square(cropped, size)

    img_bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buffer.tobytes()
