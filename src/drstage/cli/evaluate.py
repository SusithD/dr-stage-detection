"""Evaluate trained models and ensembles on held-out datasets.

For every model and dataset the test set is predicted once per TTA setting; ensembles reuse those
predictions. Each (model, dataset, TTA) combination gets a full report bundle, and all results are
collected in ``comparison.csv``.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch

from drstage.data.dataset import CachedFundusDataset, load_split
from drstage.data.loaders import build_dataloader
from drstage.data.registry import CLASS_NAMES
from drstage.data.transforms import build_eval_transforms
from drstage.evaluation.metrics import compute_metrics
from drstage.evaluation.plots import plot_training_curves
from drstage.evaluation.predict import Predictions, ensemble, run_inference
from drstage.evaluation.report import write_comparison, write_eval_bundle
from drstage.inference.predictor import Predictor, load_predictor
from drstage.training.amp import resolve_amp
from drstage.utils.config import load_config
from drstage.utils.logging import get_logger
from drstage.utils.seed import seed_everything

logger = logging.getLogger("drstage.evaluate")


def resolve_device(device: str) -> torch.device:
    """Resolve 'auto' to the best available device."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def predict_dataset(
    predictor: Predictor,
    dataset_cfg: dict,
    cfg: dict,
    device: torch.device,
    tta: bool,
) -> tuple[Predictions, Path]:
    """Predict one dataset with one model, reading its preprocessing from the checkpoint.

    Returns:
        (predictions, cache_root) where cache_root holds the images the model saw.
    """
    cache_root = (
        Path(cfg["processed_root"])
        / "images"
        / dataset_cfg["dataset"]
        / predictor.method
        / str(predictor.preprocess_size)
    )
    frame = load_split(Path(dataset_cfg["csv"]), dataset_cfg.get("split"))
    transform = build_eval_transforms(
        predictor.img_size, predictor.normalize_mean, predictor.normalize_std
    )
    dataset = CachedFundusDataset(frame, cache_root, predictor.img_size, transform)
    loader = build_dataloader(dataset, cfg["batch_size"], False, cfg["num_workers"], device)
    preds = run_inference(
        predictor.model,
        loader,
        device,
        amp=resolve_amp(device, {"dtype": cfg.get("amp", "auto")}),
        tta=tta,
        head=predictor.head,
        limit_batches=cfg.get("limit_batches", 0),
    )
    return preds, cache_root


def summarise(model: str, dataset: str, tta: bool, metrics: dict) -> dict:
    """One comparison.csv row."""
    return {
        "model": model,
        "dataset": dataset,
        "tta": tta,
        "n": metrics["n"],
        "accuracy": metrics["multiclass"]["accuracy"],
        "qwk": metrics["multiclass"]["qwk"],
        "macro_f1": metrics["multiclass"]["macro_f1"],
        "dr_auc": metrics["binary"]["roc_auc"],
        "dr_sensitivity": metrics["binary"]["sensitivity"],
        "dr_specificity": metrics["binary"]["specificity"],
        "referable_auc": metrics["referable"]["roc_auc"],
        "referable_sensitivity": metrics["referable"]["sensitivity"],
        "referable_specificity": metrics["referable"]["specificity"],
    }


def report(
    name: str,
    dataset: str,
    tta: bool,
    preds: Predictions,
    cache_root: Path,
    out_dir: Path,
    topk: int,
) -> dict:
    """Compute metrics, write the report bundle and return the comparison row."""
    metrics = compute_metrics(
        preds.label, preds.grade, preds.class_probs, preds.prob_dr, list(CLASS_NAMES)
    )
    bundle_dir = out_dir / name / dataset / ("tta" if tta else "no_tta")
    write_eval_bundle(bundle_dir, preds, metrics, list(CLASS_NAMES), cache_root, topk)
    row = summarise(name, dataset, tta, metrics)
    logger.info(
        f"{name} | {dataset} | tta={tta} | qwk {row['qwk']:.4f} acc {row['accuracy']:.4f} "
        f"macroF1 {row['macro_f1']:.4f} DR-AUC {row['dr_auc']:.4f}"
    )
    return row


def main(argv: list[str] | None = None) -> int:
    """Evaluate every configured model and ensemble on every configured dataset."""
    parser = argparse.ArgumentParser(description="Evaluate DR models and ensembles")
    parser.add_argument("--config", required=True, help="evaluation config YAML")
    parser.add_argument("-o", dest="overrides", action="append", default=[], help="key=value")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    get_logger("drstage", log_file=out_dir / "evaluate.log")
    seed_everything(42, deterministic=False)
    device = resolve_device(cfg.get("device", "auto"))

    tta_settings = [True, False] if cfg.get("also_without_tta") and cfg["tta"] else [cfg["tta"]]
    topk = cfg.get("topk_errors", 20)
    rows: list[dict] = []
    cache: dict[tuple[str, str, bool], tuple[Predictions, Path]] = {}
    heads: dict[str, str] = {}

    for model_cfg in cfg["models"]:
        name = model_cfg["name"]
        predictor = load_predictor(model_cfg["checkpoint"], device=device, tta=False)
        heads[name] = predictor.head
        logger.info(f"loaded {name}: {predictor.info()}")

        if cfg.get("training_curves"):
            history = Path(predictor.checkpoint_path).parent.parent / "history.csv"
            if history.exists():
                plot_training_curves(pd.read_csv(history), out_dir / name / "training_curves.png")

        for ds in cfg["datasets"]:
            for tta in tta_settings:
                preds, cache_root = predict_dataset(predictor, ds, cfg, device, tta)
                cache[(name, ds["name"], tta)] = (preds, cache_root)
                rows.append(report(name, ds["name"], tta, preds, cache_root, out_dir, topk))

        del predictor
        if device.type == "mps":
            torch.mps.empty_cache()

    for ens in cfg.get("ensembles", []):
        members = ens["members"]
        member_heads = {heads[m] for m in members}
        # mixed heads average class probabilities, so decode them with the argmax rule
        head = member_heads.pop() if len(member_heads) == 1 else "ce"
        for ds in cfg["datasets"]:
            for tta in tta_settings:
                parts = [cache[(m, ds["name"], tta)] for m in members]
                preds = ensemble([p for p, _ in parts], head)
                rows.append(report(ens["name"], ds["name"], tta, preds, parts[0][1], out_dir, topk))

    write_comparison(rows, out_dir / "comparison.csv")
    logger.info(f"evaluation complete: {out_dir / 'comparison.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
