"""CLI for Grad-CAM visualization."""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from drstage.data.dataset import CachedFundusDataset, load_split
from drstage.data.loaders import build_dataloader
from drstage.data.preprocess import load_rgb
from drstage.data.transforms import build_eval_transforms
from drstage.evaluation.gradcam import build_cam, cam_for_batch, select_examples
from drstage.evaluation.predict import run_inference
from drstage.inference.predictor import load_predictor
from drstage.utils.seed import seed_everything

logger = logging.getLogger(__name__)


def resolve_device(device: str) -> torch.device:
    """Resolve device string to torch.device.

    Args:
        device: 'auto', 'mps', 'cuda', or 'cpu'.

    Returns:
        torch.device object.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device)


def main(argv=None):
    """Main Grad-CAM CLI.

    Args:
        argv: Command-line arguments.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path or run dir")
    parser.add_argument("--dataset", default="ddr", help="Dataset name")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--csv", help="Custom split CSV path")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--per-grade", type=int, default=3, help="Examples per grade")
    parser.add_argument(
        "--target", default="expected_grade", help="CAM target: expected_grade, prob_dr, argmax"
    )
    parser.add_argument("--method", default="gradcam", help="CAM method: gradcam, gradcam++")
    parser.add_argument("--device", default="auto", help="Device: auto, mps, cuda, cpu")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    seed_everything(42, deterministic=False)

    try:
        predictor = load_predictor(args.checkpoint, device=device)
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return 1

    model = predictor.model
    model_info = predictor.info()
    backbone_name = model_info["name"]
    head = model_info["head"]
    img_size = model_info["image_size"]
    method = model_info["preprocessing"]

    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    eval_transform = build_eval_transforms(img_size, mean, std)

    processed_root = Path("data/processed")
    if args.csv:
        split_df = pd.read_csv(args.csv)
    else:
        csv_path = processed_root / "splits" / f"{args.dataset}_split.csv"
        split_df = load_split(csv_path, args.split)

    cache_root = processed_root / "images" / args.dataset / method / str(img_size)

    try:
        dataset = CachedFundusDataset(split_df, cache_root, img_size, eval_transform)
        loader = build_dataloader(
            dataset, batch_size=1, shuffle=False, num_workers=0, device=device
        )
    except RuntimeError as e:
        logger.error(f"Failed to load dataset: {e}")
        return 1

    logger.info("Running inference...")
    predictions = run_inference(
        model,
        loader,
        device,
        amp=None,
        tta=False,
        head=head,
        limit_batches=0,
    )

    logger.info("Selecting examples...")
    examples = select_examples(predictions, per_grade=args.per_grade)

    logger.info("Building Grad-CAM...")
    cam = build_cam(model, backbone_name, method=args.method)

    overlays_dir = out_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    correct_items = []
    incorrect_items = []

    for _, row in examples.iterrows():
        image_id = row["image_id"]
        label = int(row["label"])
        pred = int(row["grade"])

        try:
            img_path = cache_root / f"{image_id}.png"
            img_rgb = load_rgb(img_path)

            x = eval_transform(image=img_rgb)["image"].unsqueeze(0).to(device)

            model.eval()
            for param in model.parameters():
                param.requires_grad = True

            heatmap = cam_for_batch(cam, x, args.target, head, np.array([pred]))[0]

            heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

            overlay_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            heatmap_color = cv2.applyColorMap(
                (heatmap_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            overlay = cv2.addWeighted(overlay_bgr, 0.6, heatmap_color, 0.4, 0)

            overlay_path = overlays_dir / f"{image_id}.png"
            cv2.imwrite(str(overlay_path), overlay)

            class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
            title = f"True: {class_names[label]}, Pred: {class_names[pred]}"

            item = {
                "image": img_rgb,
                "heatmap": heatmap_normalized,
                "title": title,
            }

            if pred == label:
                correct_items.append(item)
            else:
                incorrect_items.append(item)

        except Exception as e:
            logger.warning(f"Failed to process {image_id}: {e}")
            continue

    if correct_items:
        from drstage.evaluation.plots import plot_cam_grid

        plot_cam_grid(correct_items[:10], out_dir / "cam_grid_correct.png", ncols=5)

    if incorrect_items:
        from drstage.evaluation.plots import plot_cam_grid

        plot_cam_grid(incorrect_items[:10], out_dir / "cam_grid_incorrect.png", ncols=5)

    examples.to_csv(out_dir / "cam_examples.csv", index=False)

    logger.info(f"Grad-CAM complete. Results in {out_dir}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
