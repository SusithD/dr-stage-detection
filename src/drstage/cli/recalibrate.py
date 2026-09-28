"""Site recalibration of decision thresholds on an external dataset.

Simulates deploying at a new clinic: a small stratified calibration subset of the external set
is used to choose DR and referable-DR thresholds that restore a target sensitivity, and the
effect is measured on the remaining, untouched images.
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
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from drstage.utils.io import write_json  # noqa: E402
from drstage.utils.logging import get_logger  # noqa: E402

logger = logging.getLogger("drstage.recalibrate")

TASKS = {
    "any_dr": {"label": lambda y: y >= 1, "score": ["p1", "p2", "p3", "p4"]},
    "referable_dr": {"label": lambda y: y >= 2, "score": ["p2", "p3", "p4"]},
}


def operating_point(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    """Sensitivity and specificity of ``score >= threshold``."""
    pred = score >= threshold
    pos, neg = y.sum(), (~y).sum()
    return {
        "threshold": float(threshold),
        "sensitivity": float((pred & y).sum() / pos) if pos else float("nan"),
        "specificity": float((~pred & ~y).sum() / neg) if neg else float("nan"),
    }


def threshold_for_sensitivity(y: np.ndarray, score: np.ndarray, target: float) -> float:
    """Highest threshold whose sensitivity on (y, score) is at least ``target``."""
    positives = np.sort(score[y])[::-1]
    k = int(np.ceil(target * len(positives))) - 1
    return float(positives[max(k, 0)])


def main(argv: list[str] | None = None) -> int:
    """Recalibrate thresholds on part of an external set and report the effect on the rest."""
    parser = argparse.ArgumentParser(description="Recalibrate thresholds on an external set")
    parser.add_argument("--predictions", required=True, help="external predictions.csv")
    parser.add_argument("--reference", required=True, help="in-domain test predictions.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    out = Path(args.out)
    get_logger("drstage", log_file=out / "recalibrate.log")
    ext = pd.read_csv(args.predictions)
    ref = pd.read_csv(args.reference)
    calib, held = train_test_split(
        ext, test_size=1 - args.calibration_fraction, stratify=ext["label"], random_state=args.seed
    )

    results: dict = {
        "calibration_images": len(calib),
        "evaluation_images": len(held),
        "seed": args.seed,
    }
    fig, axes = plt.subplots(1, len(TASKS), figsize=(10, 4.4))
    for ax, (task, spec) in zip(axes, TASKS.items(), strict=True):

        def arrays(frame: pd.DataFrame, spec: dict = spec) -> tuple[np.ndarray, np.ndarray]:
            """Binary labels and summed task probability for one prediction table."""
            labels = spec["label"](frame["label"].to_numpy())
            return labels, frame[spec["score"]].sum(axis=1).to_numpy()

        y_ref, s_ref = arrays(ref)
        y_cal, s_cal = arrays(calib)
        y_eval, s_eval = arrays(held)
        target = operating_point(y_ref, s_ref, 0.5)["sensitivity"]
        new_threshold = threshold_for_sensitivity(y_cal, s_cal, target)

        results[task] = {
            "target_sensitivity_from_reference": target,
            "reference_at_default": operating_point(y_ref, s_ref, 0.5),
            "external_auc": float(roc_auc_score(y_eval, s_eval)),
            "external_at_default": operating_point(y_eval, s_eval, 0.5),
            "external_recalibrated": operating_point(y_eval, s_eval, new_threshold),
        }
        r = results[task]
        default, recal = r["external_at_default"], r["external_recalibrated"]
        logger.info(
            f"{task}: AUC {r['external_auc']:.3f} | default sens {default['sensitivity']:.3f} "
            f"spec {default['specificity']:.3f} -> threshold {new_threshold:.3f} "
            f"sens {recal['sensitivity']:.3f} spec {recal['specificity']:.3f}"
        )

        fpr, tpr, _ = roc_curve(y_eval, s_eval)
        ax.plot(
            fpr, tpr, color="#0f625b", lw=1.8, label=f"External ROC (AUC {r['external_auc']:.3f})"
        )
        for key, marker, label in [
            ("external_at_default", "o", "Default threshold 0.5"),
            ("external_recalibrated", "s", f"Recalibrated threshold {new_threshold:.3f}"),
        ]:
            p = r[key]
            ax.plot(1 - p["specificity"], p["sensitivity"], marker, ms=8, label=label)
        ax.plot([0, 1], [0, 1], color="#bbb", lw=0.8, ls="--")
        ax.set_title(
            "Any DR" if task == "any_dr" else "Referable DR (grade 2 or worse)", fontsize=10
        )
        ax.set_xlabel("1 − specificity")
        ax.set_ylabel("Sensitivity")
        ax.legend(fontsize=7.5, loc="lower right")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "recalibration_roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    write_json(results, out / "recalibration.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
