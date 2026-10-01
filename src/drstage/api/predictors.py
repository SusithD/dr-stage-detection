"""Predictor interface and a lightweight demo that doesn't require a checkpoint."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from drstage.data.preprocess import circle_crop, preprocess_image, resize_square
from drstage.utils.config import load_config


@dataclass
class Prediction:
    """Single prediction: grade, probabilities and preprocessed images."""

    grade: int
    grade_name: str
    prob_dr: float
    class_probs: list[float]
    original: np.ndarray
    enhanced: np.ndarray
    heatmap: np.ndarray


class Predictor(Protocol):
    """Any object that can predict grades and return a Prediction dataclass."""

    demo: bool

    def info(self) -> dict | None:
        """Return metadata dict, or None if running in demo mode."""
        ...

    def predict(self, image_rgb: np.ndarray) -> Prediction:
        """Grade one image and return the prediction with visualisations."""
        ...


class DemoPredictor:
    """Placeholder: real preprocessing but synthetic predictions for development."""

    demo = True

    def __init__(self):
        """Load the preprocessing config."""
        cfg = load_config(Path("configs/data.yaml"))
        self._preprocess_cfg = cfg["preprocess"]
        self._grade_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

    def info(self) -> None:
        """Demo mode returns no model info."""
        return None

    def predict(self, image_rgb: np.ndarray) -> Prediction:
        """Preprocess realistically but generate deterministic grades from the image hash."""
        size = self._preprocess_cfg["size"]

        enhanced, _ = preprocess_image(
            image_rgb, method="ben_graham", size=size, cfg=self._preprocess_cfg
        )

        circle_crop_result = circle_crop(image_rgb, **self._preprocess_cfg["circle_crop"])
        original = resize_square(circle_crop_result.image, size)

        img_hash = int(hashlib.sha256(image_rgb.tobytes()).hexdigest(), 16)
        rng = np.random.default_rng(img_hash % (2**32))
        class_probs = rng.dirichlet([1.0] * 5)
        class_probs = class_probs / class_probs.sum()

        prob_dr = float(1.0 - class_probs[0])
        # decode as for CORN to stay consistent with the main model
        grade = int((1.0 - np.cumsum(class_probs)[:-1] > 0.5).sum())
        grade_name = self._grade_names[grade]

        abs_dev = np.abs(enhanced.astype(float) - enhanced.mean())
        max_idx = np.unravel_index(np.argmax(abs_dev), abs_dev.shape)
        heatmap = np.zeros((size, size), dtype=np.float32)
        y, x = np.ogrid[:size, :size]
        heatmap = np.exp(-((x - max_idx[1]) ** 2 + (y - max_idx[0]) ** 2) / (2 * (size / 8) ** 2))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        return Prediction(
            grade=grade,
            grade_name=grade_name,
            prob_dr=prob_dr,
            class_probs=class_probs.tolist(),
            original=original,
            enhanced=enhanced,
            heatmap=heatmap,
        )


def load_predictor(checkpoint: str | None, device: str = "auto", tta: bool = True) -> Predictor:
    """Load a trained model from a checkpoint, or return the demo predictor if none is set.

    tta=False skips the four-flip averaging, which is worth it on a slow CPU host.
    """
    if checkpoint is None:
        return DemoPredictor()

    try:
        from drstage.inference.predictor import load_predictor as load_real
    except ImportError as e:
        raise RuntimeError(
            "Could not import drstage.inference.predictor. "
            "The model training module may not be available."
        ) from e

    try:
        real_predictor = load_real(checkpoint, device, tta=tta)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model from checkpoint: {checkpoint}. "
            f"Ensure the file exists and is a valid checkpoint. Error: {e}"
        ) from e

    class RealPredictorAdapter:
        """Adapt a dict-returning predictor to the Prediction dataclass interface."""

        demo = False

        def __init__(self, inner):
            """Wrap the real predictor from the training module."""
            self.inner = inner

        def info(self) -> dict:
            """Return the model metadata."""
            return self.inner.info()

        def predict(self, image_rgb: np.ndarray) -> Prediction:
            """Convert the dict output to a Prediction object."""
            result = self.inner.predict(image_rgb)
            return Prediction(
                grade=result["grade"],
                grade_name=result["grade_name"],
                prob_dr=result["prob_dr"],
                class_probs=result["class_probs"],
                original=result["original"],
                enhanced=result["enhanced"],
                heatmap=result["heatmap"],
            )

    return RealPredictorAdapter(real_predictor)
