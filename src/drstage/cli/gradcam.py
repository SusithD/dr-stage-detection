"""Visualise Grad-CAM heatmaps for the most confident correct and incorrect predictions per grade.

Heatmaps come from the same Predictor as the web app, so they show what a clinician would see.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from drstage.data.preprocess import load_rgb  # noqa: E402
from drstage.data.registry import CLASS_NAMES  # noqa: E402
from drstage.inference.predictor import load_predictor  # noqa: E402
from drstage.utils.io import write_csv  # noqa: E402
from drstage.utils.logging import get_logger  # noqa: E402

logger = logging.getLogger("drstage.gradcam")


def select_examples(preds: pd.DataFrame, per_grade: int) -> pd.DataFrame:
    """Pick the most confident correct and incorrect predictions for each true grade."""
    probs = preds[[f"p{i}" for i in range(len(CLASS_NAMES))]].to_numpy()
    preds = preds.assign(confidence=probs.max(axis=1), correct=preds["label"] == preds["pred"])
    picked = (
        preds.sort_values("confidence", ascending=False)
        .groupby(["label", "correct"], group_keys=False)
        .head(per_grade)
    )
    return picked.sort_values(["correct", "label", "confidence"], ascending=[False, True, False])


def plot_pairs(items: list[dict], out: Path, title: str) -> None:
    """Display side-by-side image and Grad-CAM pairs in a grid."""
    pairs_per_row = 2
    rows = int(np.ceil(len(items) / pairs_per_row))
    fig, axes = plt.subplots(rows, pairs_per_row * 2, figsize=(pairs_per_row * 5, rows * 2.6))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for i, item in enumerate(items):
        r, c = divmod(i, pairs_per_row)
        left, right = axes[r, 2 * c], axes[r, 2 * c + 1]
        left.imshow(item["image"])
        left.set_title(item["caption"], fontsize=8, loc="left")
        right.imshow(item["image"])
        right.imshow(item["heatmap"], cmap="inferno", alpha=item["heatmap"] * 0.75)
        right.set_title("Grad-CAM", fontsize=8, loc="left")
    fig.suptitle(title, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """Generate Grad-CAM visualisations from a predictions CSV and save to disk."""
    parser = argparse.ArgumentParser(description="Grad-CAM figures for correct and incorrect cases")
    parser.add_argument("--checkpoint", required=True, help="checkpoint file or run directory")
    parser.add_argument("--predictions", required=True, help="predictions.csv from evaluate")
    parser.add_argument("--split-csv", required=True, help="split CSV with image_id and path")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--per-grade", type=int, default=2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    out = Path(args.out)
    get_logger("drstage", log_file=out / "gradcam.log")
    predictor = load_predictor(args.checkpoint, device=args.device, tta=False)
    paths = pd.read_csv(args.split_csv).set_index("image_id")["path"]
    examples = select_examples(pd.read_csv(args.predictions), args.per_grade)

    groups: dict[bool, list[dict]] = {True: [], False: []}
    for row in examples.itertuples():
        result = predictor.predict(load_rgb(Path(paths[row.image_id])), heatmap=True)
        overlay_dir = out / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        plt.imsave(overlay_dir / f"{row.image_id}_heatmap.png", result["heatmap"], cmap="inferno")
        groups[bool(row.correct)].append(
            {
                "image": result["original"],
                "heatmap": result["heatmap"],
                "caption": (
                    f"True: {CLASS_NAMES[row.label]}\n"
                    f"Predicted: {CLASS_NAMES[row.pred]} ({row.confidence:.0%})"
                ),
            }
        )

    if groups[True]:
        plot_pairs(
            groups[True], out / "cam_correct.png", "Correct predictions, most confident per grade"
        )
    if groups[False]:
        plot_pairs(
            groups[False],
            out / "cam_incorrect.png",
            "Incorrect predictions, most confident per grade",
        )
    write_csv(examples, out / "cam_examples.csv")
    logger.info(f"Grad-CAM figures written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
