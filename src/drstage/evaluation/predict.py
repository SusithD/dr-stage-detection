"""Inference utilities: TTA, ensemble, and prediction aggregation."""

import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from drstage.models.factory import DRModel

logger = logging.getLogger(__name__)


@dataclass
class Predictions:
    """Aggregated predictions from a dataset."""

    image_id: np.ndarray
    label: np.ndarray
    grade: np.ndarray
    prob_dr: np.ndarray
    class_probs: np.ndarray

    def to_frame(self):
        """Convert to pandas DataFrame with one row per image."""
        import pandas as pd

        data = {
            "image_id": self.image_id,
            "label": self.label,
            "pred": self.grade,
            "prob_dr": self.prob_dr,
        }
        for i in range(self.class_probs.shape[1]):
            data[f"p{i}"] = self.class_probs[:, i]
        return pd.DataFrame(data)


def flip_tta_logits_to_probs(model: DRModel, x: Tensor, flips: bool) -> tuple[Tensor, Tensor]:
    """Apply Test-Time Augmentation via flips and average probabilities.

    For CORN: average sigmoid(logits) across {id, hflip, vflip, hvflip}.
    For CE: average softmax(logits) across same flips.

    Args:
        model: DRModel with forward() and predict().
        x: [N, 3, H, W] image tensor.
        flips: Whether to apply flips.

    Returns:
        (class_probs, prob_dr) both averaged over all TTA variants.
    """
    if not flips:
        with torch.no_grad():
            logits = model(x)
            grade, prob_dr, class_probs = model.predict(logits)
        return class_probs, prob_dr

    variants = [x]
    variants.append(torch.flip(x, [-1]))
    variants.append(torch.flip(x, [-2]))
    variants.append(torch.flip(x, [-1, -2]))

    all_class_probs = []
    all_prob_dr = []

    with torch.no_grad():
        for variant in variants:
            logits = model(variant)
            grade, prob_dr, class_probs = model.predict(logits)
            all_class_probs.append(class_probs)
            all_prob_dr.append(prob_dr)

    class_probs_avg = torch.stack(all_class_probs).mean(dim=0)
    prob_dr_avg = torch.stack(all_prob_dr).mean(dim=0)

    return class_probs_avg, prob_dr_avg


def grade_from_class_probs(class_probs: Tensor, head: str) -> torch.LongTensor:
    """Decode grade from class probabilities.

    For CORN: p0 + p1 + ... + p_k should sum through cumulative rule.
    For CE: argmax(class_probs).

    Args:
        class_probs: [N, K] class probabilities.
        head: 'corn' or 'ce'.

    Returns:
        [N] predicted grades.
    """
    if head == "ce":
        return torch.argmax(class_probs, dim=1).long()

    if head != "corn":
        raise ValueError(f"Unknown head: {head}. Must be 'corn' or 'ce'.")

    class_probs_f32 = class_probs.float()
    cumsum_p = torch.cumsum(class_probs_f32, dim=1)
    cum = 1.0 - cumsum_p

    grade = (cum > 0.5).sum(dim=1).long()
    return grade


def run_inference(
    model: DRModel,
    loader: DataLoader,
    device: torch.device,
    amp,
    tta: bool,
    head: str,
    limit_batches: int = 0,
) -> Predictions:
    """Run inference on a dataset.

    Args:
        model: DRModel instance.
        loader: DataLoader yielding batches with 'image', 'label', 'image_id'.
        device: Device to run on.
        amp: AmpPolicy object.
        tta: Whether to apply TTA (flip averaging).
        head: 'corn' or 'ce'.
        limit_batches: Max batches to process (0 = all).

    Returns:
        Predictions aggregated over the dataset.
    """
    model.eval()
    model = model.to(device)

    all_image_ids = []
    all_labels = []
    all_grades = []
    all_prob_dr = []
    all_class_probs = []

    num_batches = 0
    for batch in loader:
        if limit_batches > 0 and num_batches >= limit_batches:
            break

        x = batch["image"].to(device)
        labels = batch["label"].cpu().numpy()
        image_ids = batch["image_id"]

        with torch.no_grad():
            if amp.enabled:
                with torch.autocast(device_type=device.type, dtype=amp.dtype):
                    class_probs, prob_dr = flip_tta_logits_to_probs(model, x, tta)
            else:
                class_probs, prob_dr = flip_tta_logits_to_probs(model, x, tta)

        grade = grade_from_class_probs(class_probs, head)

        all_image_ids.extend(image_ids)
        all_labels.append(labels)
        all_grades.append(grade.cpu().numpy())
        all_prob_dr.append(prob_dr.cpu().numpy())
        all_class_probs.append(class_probs.cpu().numpy())

        num_batches += 1

    labels_array = np.concatenate(all_labels, axis=0)
    grades_array = np.concatenate(all_grades, axis=0)
    prob_dr_array = np.concatenate(all_prob_dr, axis=0)
    class_probs_array = np.concatenate(all_class_probs, axis=0)

    return Predictions(
        image_id=np.array(all_image_ids),
        label=labels_array,
        grade=grades_array,
        prob_dr=prob_dr_array,
        class_probs=class_probs_array,
    )


def ensemble(preds: list[Predictions], head: str) -> Predictions:
    """Ensemble predictions from multiple models.

    Aligns on image_id, averages class_probs and prob_dr, re-derives grade.

    Args:
        preds: List of Predictions objects.
        head: 'corn' or 'ce'.

    Returns:
        Ensembled Predictions.
    """
    if not preds:
        raise ValueError("No predictions to ensemble")

    if len(preds) == 1:
        return preds[0]

    import pandas as pd

    all_dfs = []
    for pred in preds:
        df = pd.DataFrame(
            {
                "image_id": pred.image_id,
                "label": pred.label,
                "grade_individual": pred.grade,
                "prob_dr_individual": pred.prob_dr,
            }
        )
        for i in range(pred.class_probs.shape[1]):
            df[f"p{i}"] = pred.class_probs[:, i]
        all_dfs.append(df)

    merged = all_dfs[0].copy()
    for df in all_dfs[1:]:
        merged = pd.merge(merged, df, on=["image_id", "label"], suffixes=("", "_x"))

    class_prob_cols = [c for c in merged.columns if c.startswith("p") and "_" not in c]
    for i, col in enumerate(class_prob_cols):
        related_cols = [merged[col]] + [merged[c] for c in merged.columns if c.startswith(f"p{i}_")]
        merged[col] = pd.concat(related_cols, axis=1).mean(axis=1)

    prob_dr_cols = [c for c in merged.columns if c.startswith("prob_dr")]
    merged["prob_dr"] = merged[prob_dr_cols].mean(axis=1)

    class_probs_array = merged[class_prob_cols].values.astype(np.float32)
    grade_array = grade_from_class_probs(torch.from_numpy(class_probs_array), head).numpy()

    return Predictions(
        image_id=merged["image_id"].values,
        label=merged["label"].values.astype(np.int64),
        grade=grade_array.astype(np.int64),
        prob_dr=merged["prob_dr"].values.astype(np.float32),
        class_probs=class_probs_array,
    )
