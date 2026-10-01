"""Single-model predictor wrapping a checkpoint, with preprocessing and optional CAM."""

import logging
from pathlib import Path

import numpy as np
import torch

from drstage.data.preprocess import circle_crop, preprocess_image, resize_square
from drstage.data.transforms import build_eval_transforms
from drstage.evaluation.gradcam import ArgmaxTarget, CornExpectedGradeTarget, build_cam
from drstage.evaluation.predict import flip_tta_logits_to_probs, grade_from_class_probs
from drstage.models.backbone import cam_target_layer
from drstage.models.factory import build_model
from drstage.training.checkpoint import resolve_checkpoint

logger = logging.getLogger(__name__)


class Predictor:
    """Load a checkpoint and grade fundus photographs with optional TTA and Grad-CAM."""

    def __init__(
        self,
        model,
        bundle: dict,
        device: torch.device,
        tta: bool = True,
        cam_method: str = "gradcam",
    ):
        """Store the model, preprocessing config, and optionally build a CAM generator."""
        self.model = model.to(device).eval()
        self.device = device
        self.tta = tta
        self.cam_method = cam_method

        self.bundle = bundle
        self.grade_names = tuple(
            bundle.get("class_names", ["No DR", "Mild", "Moderate", "Severe", "Proliferative"])
        )
        self.img_size = bundle["model"]["img_size"]
        self.head = bundle["model"]["head"]
        self.preprocess_cfg = bundle.get("preprocess", {})
        self.normalize_mean = bundle.get("normalize", {}).get("mean", [0.5, 0.5, 0.5])
        self.normalize_std = bundle.get("normalize", {}).get("std", [0.5, 0.5, 0.5])
        self.method = self.preprocess_cfg.get("method", "ben_graham")
        self.preprocess_size = self.preprocess_cfg.get("size", 512)
        self.checkpoint_path = bundle.get("checkpoint", "")

        self.eval_transform = build_eval_transforms(
            self.img_size, self.normalize_mean, self.normalize_std
        )

        try:
            self.cam = build_cam(self.model, bundle["model"]["backbone"], method=cam_method)
            # Only the CAM layer needs gradients, so backprop stops there instead of running
            # back to the input; on a CPU this makes the heatmap about 3x faster.
            self.model.requires_grad_(False)
            cam_target_layer(self.model.backbone, bundle["model"]["backbone"]).requires_grad_(True)
        except Exception as e:
            logger.warning(f"Could not build CAM: {e}")
            self.cam = None

    def preprocess(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
        """Crop, enhance and normalise the image, matching training-time preprocessing exactly.

        Returns the original and preprocessed images alongside the normalised tensor.
        """
        crop = circle_crop(image_rgb, **self.preprocess_cfg["circle_crop"]).image
        original = resize_square(crop, self.img_size)
        enhanced, _ = preprocess_image(
            image_rgb, self.method, self.preprocess_size, self.preprocess_cfg
        )
        enhanced = resize_square(enhanced, self.img_size)
        tensor = self.eval_transform(image=enhanced)["image"].unsqueeze(0).to(self.device)
        return original, enhanced, tensor

    def predict(self, image_rgb: np.ndarray, heatmap: bool = True) -> dict:
        """Preprocess, infer and optionally generate a Grad-CAM heatmap for one image."""
        original, enhanced, x = self.preprocess(image_rgb)

        heatmap_array, cam_logits = None, None
        if heatmap and self.cam is not None:
            try:
                with torch.enable_grad():
                    target = (
                        CornExpectedGradeTarget() if self.head == "corn" else ArgmaxTarget("ce")
                    )
                    cam = self.cam(input_tensor=x.clone(), targets=[target])
                heatmap_array = np.clip(cam[0], 0.0, 1.0).astype(np.float32)
                cam_logits = self.cam.outputs.detach()
            except Exception:
                logger.exception("Grad-CAM failed; returning the prediction without a heatmap")

        if cam_logits is not None and not self.tta:
            # without TTA the CAM forward pass already holds the logits we need
            _, prob_dr, class_probs = self.model.predict(cam_logits)
        else:
            with torch.inference_mode():
                class_probs, prob_dr = flip_tta_logits_to_probs(self.model, x, self.tta)
        grade = int(grade_from_class_probs(class_probs, self.head).item())

        return {
            "grade": grade,
            "grade_name": self.grade_names[grade],
            "prob_dr": float(prob_dr[0]),
            "class_probs": [float(p) for p in class_probs[0].cpu()],
            "heatmap": heatmap_array,
            "original": original,
            "enhanced": enhanced,
        }

    def info(self) -> dict:
        """Return metadata: backbone, head, image size, preprocessing method and device."""
        return {
            "name": self.bundle["model"]["backbone"],
            "head": self.head,
            "image_size": self.img_size,
            "preprocessing": self.method,
            "checkpoint": str(self.checkpoint_path),
            "device": str(self.device),
        }


def load_predictor(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    tta: bool = True,
) -> Predictor:
    """Load model weights and configuration from a checkpoint file or directory."""
    checkpoint_path = resolve_checkpoint(checkpoint_path)

    if isinstance(device, str):
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        device = torch.device(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if checkpoint.get("format_version") != 1:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint.get('format_version')}")

    bundle = {k: v for k, v in checkpoint.items() if k != "model_state"}

    model = build_model(checkpoint["model"], pretrained=False)
    model.load_state_dict(checkpoint["model_state"])

    bundle["checkpoint"] = str(checkpoint_path)

    predictor = Predictor(model, bundle, device, tta=tta)
    return predictor
