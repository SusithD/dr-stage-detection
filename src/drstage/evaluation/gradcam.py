"""Grad-CAM targets and heatmap generation for explaining model predictions."""

import logging

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from torch import Tensor

from drstage.models.factory import DRModel

logger = logging.getLogger(__name__)


class CornExpectedGradeTarget:
    """Grad-CAM target using the expected ordinal grade for a CORN head."""

    def __call__(self, model_output: Tensor) -> Tensor:
        """Return E[grade] = sum of sigmoid(logits), the expected grade."""
        logits_f32 = model_output.float()
        cum = torch.sigmoid(logits_f32).cumprod(dim=-1)
        expected_grade = cum.sum(dim=-1)
        return expected_grade


class ProbDrTarget:
    """Grad-CAM target using the probability of any retinopathy."""

    def __init__(self, head: str):
        """Store whether this is a CORN or cross-entropy head."""
        self.head = head

    def __call__(self, model_output: Tensor) -> Tensor:
        """Return P(DR>0): sigmoid of logits[0] for CORN, or 1 - P(grade 0) for CE."""
        if self.head == "corn":
            return torch.sigmoid(model_output[..., 0])
        elif self.head == "ce":
            probs = torch.softmax(model_output, dim=-1)
            return 1.0 - probs[..., 0]
        else:
            raise ValueError(f"Unknown head: {self.head}")


class ArgmaxTarget:
    """Grad-CAM target using the highest predicted class probability."""

    def __init__(self, head: str, predictions: np.ndarray | None = None):
        """Store head type; predictions arg is unused but kept for compatibility."""
        self.head = head
        self.predictions = predictions

    def __call__(self, model_output: Tensor) -> Tensor:
        """Return expected grade (CORN) or max probability (CE)."""
        if self.head == "corn":
            target = CornExpectedGradeTarget()
            return target(model_output)
        elif self.head == "ce":
            probs = torch.softmax(model_output, dim=-1)
            max_probs = torch.max(probs, dim=-1).values
            return max_probs
        else:
            raise ValueError(f"Unknown head: {self.head}")


def build_cam(model: DRModel, backbone_name: str, method: str = "gradcam") -> GradCAM:
    """Initialise GradCAM by selecting the target layer from the backbone."""
    from drstage.models.backbone import cam_target_layer

    target_layer = cam_target_layer(model.backbone, backbone_name)
    cam = GradCAM(model=model, target_layers=[target_layer])
    return cam


def cam_for_batch(
    cam: GradCAM, x: Tensor, target: str, head: str, preds: np.ndarray | None = None
) -> np.ndarray:
    """Compute Grad-CAM heatmaps by backpropagating the target through the model."""
    if target == "expected_grade":
        if head != "corn":
            raise ValueError("expected_grade target only for CORN head")
        targets = [CornExpectedGradeTarget()]
    elif target == "prob_dr":
        targets = [ProbDrTarget(head)]
    elif target == "argmax":
        targets = [ArgmaxTarget(head, preds)]
    else:
        raise ValueError(f"Unknown target: {target}")

    grayscale_cam = cam(input_tensor=x.float(), targets=targets)
    return grayscale_cam.astype(np.float32)


def select_examples(pred, per_grade: int, seed: int = 0):
    """Choose representative examples: the most confident correct and incorrect per grade."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    selected = []

    for grade in range(5):
        mask_correct = (pred.grade == grade) & (pred.grade == pred.label)
        mask_incorrect = (pred.grade == grade) & (pred.grade != pred.label)

        correct_indices = np.where(mask_correct)[0]
        incorrect_indices = np.where(mask_incorrect)[0]

        if len(correct_indices) > 0:
            sampled = rng.choice(
                correct_indices, size=min(per_grade, len(correct_indices)), replace=False
            )
            selected.extend(sampled)

        if len(incorrect_indices) > 0:
            conf = pred.class_probs[incorrect_indices].max(axis=1)
            top_indices = incorrect_indices[np.argsort(-conf)[:per_grade]]
            selected.extend(top_indices)

    selected_array = np.array(selected)
    return pd.DataFrame(
        {
            "index": selected_array,
            "image_id": pred.image_id[selected_array],
            "label": pred.label[selected_array],
            "grade": pred.grade[selected_array],
        }
    )
