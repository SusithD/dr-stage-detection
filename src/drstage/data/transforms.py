"""Albumentations-based image transforms for training and evaluation."""

import logging

import albumentations as A
import cv2

logger = logging.getLogger(__name__)


def build_train_transforms(
    img_size: int, mean: list[float], std: list[float], aug: dict
) -> A.Compose:
    """Build training transforms with augmentations.

    Args:
        img_size: Target image size.
        mean: Normalization mean (RGB).
        std: Normalization std (RGB).
        aug: Augmentation config dict with keys: rotate_limit, rotate_p, rrc_scale,
            brightness, contrast, hue, sat, color_p.

    Returns:
        albumentations Compose object.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(
                limit=aug["rotate_limit"],
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
                p=aug["rotate_p"],
            ),
            A.RandomResizedCrop(
                size=(img_size, img_size),
                scale=tuple(aug["rrc_scale"]),
                ratio=(1, 1),
                p=1.0,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=aug["brightness"],
                contrast_limit=aug["contrast"],
                p=aug["color_p"],
            ),
            A.HueSaturationValue(
                hue_shift_limit=aug["hue"],
                sat_shift_limit=aug["sat"],
                val_shift_limit=0,
                p=aug["color_p"],
            ),
            A.Normalize(mean=mean, std=std),
            A.ToTensorV2(),
        ],
    )


def build_eval_transforms(img_size: int, mean: list[float], std: list[float]) -> A.Compose:
    """Build evaluation transforms (no augmentation).

    Args:
        img_size: Target image size.
        mean: Normalization mean (RGB).
        std: Normalization std (RGB).

    Returns:
        albumentations Compose object.
    """
    return A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=mean, std=std),
            A.ToTensorV2(),
        ],
    )
