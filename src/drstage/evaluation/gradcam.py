"""Grad-CAM visualization for DR stage predictions."""

import logging

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from torch import Tensor

from drstage.models.factory import DRModel

logger = logging.getLogger(__name__)


class CornExpectedGradeTarget:
    """Grad-CAM target: expected grade for CORN head.

    E[grade] = sum_k P(y>k) = sum of sigmoid(logits).
    """

    def __call__(self, model_output: Tensor) -> Tensor:
        """Compute expected grade as target.

        Args:
            model_output: CORN logits, [K-1] for one sample or [N, K-1].

        Returns:
            [N] scalar target (one per sample).
        """
        logits_f32 = model_output.float()
        cum = torch.sigmoid(logits_f32).cumprod(dim=-1)
        expected_grade = cum.sum(dim=-1)
        return expected_grade


class ProbDrTarget:
    """Grad-CAM target: probability of any DR.

    For CORN: logits[0] (P(y>0)).
    For CE: 1 - softmax[0].
    """

    def __init__(self, head: str):
        """Initialize target.

        Args:
            head: 'corn' or 'ce'.
        """
        self.head = head

    def __call__(self, model_output: Tensor) -> Tensor:
        """Compute prob_dr as target.

        Args:
            model_output: [N, out_dim] logits.

        Returns:
            [N] scalar target.
        """
        if self.head == "corn":
            return torch.sigmoid(model_output[..., 0])
        elif self.head == "ce":
            probs = torch.softmax(model_output, dim=-1)
            return 1.0 - probs[..., 0]
        else:
            raise ValueError(f"Unknown head: {self.head}")


class ArgmaxTarget:
    """Grad-CAM target: argmax class probability."""

    def __init__(self, head: str, predictions: np.ndarray | None = None):
        """Initialize target.

        Args:
            head: 'corn' or 'ce'.
            predictions: [N] predicted grades (optional, used for alignment).
        """
        self.head = head
        self.predictions = predictions

    def __call__(self, model_output: Tensor) -> Tensor:
        """Compute argmax class probability as target.

        Args:
            model_output: [N, out_dim] logits.

        Returns:
            [N] scalar target.
        """
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
    """Build GradCAM for the model.

    Args:
        model: DRModel instance.
        backbone_name: Name of the backbone (used to find target layer).
        method: CAM method ('gradcam' or 'gradcam++').

    Returns:
        GradCAM instance.
    """
    from drstage.models.backbone import cam_target_layer

    target_layer = cam_target_layer(model.backbone, backbone_name)
    cam = GradCAM(model=model, target_layers=[target_layer])
    return cam


def cam_for_batch(
    cam: GradCAM, x: Tensor, target: str, head: str, preds: np.ndarray | None = None
) -> np.ndarray:
    """Generate Grad-CAM heatmaps for a batch.

    Args:
        cam: GradCAM instance.
        x: [N, 3, H, W] images.
        target: 'expected_grade', 'prob_dr', or 'argmax'.
        head: 'corn' or 'ce'.
        preds: [N] predicted grades (for argmax alignment).

    Returns:
        [N, H, W] heatmaps in [0, 1].
    """
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
    """Select examples for visualization: per_grade correct + incorrect.

    Args:
        pred: Predictions object.
        per_grade: Number of correct and incorrect per grade.
        seed: Random seed.

    Returns:
        DataFrame with selected indices.
    """
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
