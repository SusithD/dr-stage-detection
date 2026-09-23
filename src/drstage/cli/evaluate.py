"""CLI for comprehensive model evaluation."""

import argparse
import logging
from pathlib import Path

import torch

from drstage.data.dataset import CachedFundusDataset, load_split
from drstage.data.loaders import build_dataloader
from drstage.data.transforms import build_eval_transforms
from drstage.evaluation.metrics import compute_metrics
from drstage.evaluation.predict import ensemble, run_inference
from drstage.evaluation.report import write_comparison, write_eval_bundle
from drstage.inference.predictor import load_predictor
from drstage.models.backbone import normalization_stats
from drstage.models.factory import build_model
from drstage.utils.config import load_config
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
    """Main evaluation CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(description="Evaluate DR stage detection models")
    parser.add_argument("--config", required=True, help="Eval config YAML file")
    parser.add_argument("-o", nargs="+", default=[], help="Overrides: key=value")
    args = parser.parse_args(argv)

    config = load_config(args.config)

    for override in args.o:
        parts = override.split("=", 1)
        if len(parts) == 2:
            key, value = parts
            config[key] = value

    out_dir = Path(config.get("out_dir", "runs/eval"))
    device = resolve_device(config.get("device", "auto"))
    processed_root = Path(config.get("processed_root", "data/processed"))

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    seed_everything(42, deterministic=False)

    models_cfg = config.get("models", [])
    datasets_cfg = config.get("datasets", [])
    ensembles_cfg = config.get("ensembles", [])
    tta = config.get("tta", True)
    also_without_tta = config.get("also_without_tta", False)
    batch_size = config.get("batch_size", 16)
    num_workers = config.get("num_workers", 0)
    topk_errors = config.get("topk_errors", 20)
    limit_batches = config.get("limit_batches", 0)

    loaded_models = {}
    for model_cfg in models_cfg:
        name = model_cfg["name"]
        checkpoint_spec = model_cfg["checkpoint"]

        try:
            predictor = load_predictor(checkpoint_spec, device=device)
            loaded_models[name] = predictor
            logger.info(f"Loaded model {name} from {checkpoint_spec}")
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")
            continue

    comparison_rows = []

    for model_cfg in models_cfg:
        model_name = model_cfg["name"]
        if model_name not in loaded_models:
            continue

        predictor = loaded_models[model_name]
        model_info = predictor.info()
        backbone_name = model_info["name"]
        head = model_info["head"]
        img_size = model_info["image_size"]
        method = model_info["preprocessing"]

        mean, std = model_info.get("mean"), model_info.get("std")
        if not mean or not std:
            try:
                model = build_model(
                    {
                        "backbone": backbone_name,
                        "head": head,
                        "num_classes": 5,
                        "dropout": 0.0,
                        "drop_path_rate": 0.0,
                    },
                    pretrained=False,
                )
                mean, std = normalization_stats(model.backbone)
            except Exception:
                mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

        eval_transform = build_eval_transforms(img_size, mean, std)

        for dataset_cfg in datasets_cfg:
            dataset_name = dataset_cfg["name"]
            csv_path = Path(dataset_cfg["csv"])
            split = dataset_cfg.get("split", "test")
            dataset_key = dataset_cfg.get("dataset", "ddr")

            try:
                split_df = load_split(csv_path, split)
                cache_root = processed_root / "images" / dataset_key / method / str(img_size)

                dataset = CachedFundusDataset(split_df, cache_root, img_size, eval_transform)
                loader = build_dataloader(dataset, batch_size, False, num_workers, device)

                logger.info(f"Evaluating {model_name} on {dataset_name}")

                predictions = run_inference(
                    predictor.model,
                    loader,
                    device,
                    amp=None,
                    tta=tta,
                    head=head,
                    limit_batches=limit_batches,
                )

                class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
                metrics = compute_metrics(
                    predictions.label,
                    predictions.grade,
                    predictions.class_probs,
                    predictions.prob_dr,
                    class_names,
                )

                model_out_dir = out_dir / model_name / dataset_name
                write_eval_bundle(
                    model_out_dir, predictions, metrics, class_names, cache_root, topk_errors
                )

                row = {
                    "model": model_name,
                    "dataset": dataset_name,
                    "tta": tta,
                    "accuracy": metrics["multiclass"]["accuracy"],
                    "qwk": metrics["multiclass"]["qwk"],
                    "macro_f1": metrics["multiclass"]["macro_f1"],
                    "auc": metrics["binary"]["roc_auc"],
                    "sensitivity": metrics["binary"]["sensitivity"],
                    "specificity": metrics["binary"]["specificity"],
                    "referable_auc": metrics["referable"]["roc_auc"],
                }
                comparison_rows.append(row)

                if also_without_tta and tta:
                    predictions_no_tta = run_inference(
                        predictor.model,
                        loader,
                        device,
                        amp=None,
                        tta=False,
                        head=head,
                        limit_batches=limit_batches,
                    )
                    metrics_no_tta = compute_metrics(
                        predictions_no_tta.label,
                        predictions_no_tta.grade,
                        predictions_no_tta.class_probs,
                        predictions_no_tta.prob_dr,
                        class_names,
                    )
                    row_no_tta = {
                        "model": model_name,
                        "dataset": dataset_name,
                        "tta": False,
                        "accuracy": metrics_no_tta["multiclass"]["accuracy"],
                        "qwk": metrics_no_tta["multiclass"]["qwk"],
                        "macro_f1": metrics_no_tta["multiclass"]["macro_f1"],
                        "auc": metrics_no_tta["binary"]["roc_auc"],
                        "sensitivity": metrics_no_tta["binary"]["sensitivity"],
                        "specificity": metrics_no_tta["binary"]["specificity"],
                        "referable_auc": metrics_no_tta["referable"]["roc_auc"],
                    }
                    comparison_rows.append(row_no_tta)

            except Exception as e:
                logger.error(f"Failed to evaluate {model_name} on {dataset_name}: {e}")
                continue

    for ensemble_cfg in ensembles_cfg:
        ensemble_name = ensemble_cfg["name"]
        member_names = ensemble_cfg["members"]

        members = [loaded_models[name] for name in member_names if name in loaded_models]
        if len(members) < len(member_names):
            logger.warning(f"Could not load all members for ensemble {ensemble_name}")

        if len(members) == 0:
            continue

        predictor = members[0]
        model_info = predictor.info()
        head = model_info["head"]

        for dataset_cfg in datasets_cfg:
            dataset_name = dataset_cfg["name"]
            csv_path = Path(dataset_cfg["csv"])
            split = dataset_cfg.get("split", "test")
            dataset_key = dataset_cfg.get("dataset", "ddr")

            try:
                all_predictions = []

                for member in members:
                    member_info = member.info()
                    img_size = member_info["image_size"]
                    method = member_info["preprocessing"]
                    backbone_name = member_info["name"]

                    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
                    eval_transform = build_eval_transforms(img_size, mean, std)

                    split_df = load_split(csv_path, split)
                    cache_root = processed_root / "images" / dataset_key / method / str(img_size)
                    dataset = CachedFundusDataset(split_df, cache_root, img_size, eval_transform)
                    loader = build_dataloader(dataset, batch_size, False, num_workers, device)

                    predictions = run_inference(
                        member.model,
                        loader,
                        device,
                        amp=None,
                        tta=tta,
                        head=head,
                        limit_batches=limit_batches,
                    )
                    all_predictions.append(predictions)

                ensembled = ensemble(all_predictions, head)

                class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
                metrics = compute_metrics(
                    ensembled.label,
                    ensembled.grade,
                    ensembled.class_probs,
                    ensembled.prob_dr,
                    class_names,
                )

                model_out_dir = out_dir / ensemble_name / dataset_name
                write_eval_bundle(
                    model_out_dir, ensembled, metrics, class_names, cache_root, topk_errors
                )

                row = {
                    "model": ensemble_name,
                    "dataset": dataset_name,
                    "tta": tta,
                    "accuracy": metrics["multiclass"]["accuracy"],
                    "qwk": metrics["multiclass"]["qwk"],
                    "macro_f1": metrics["multiclass"]["macro_f1"],
                    "auc": metrics["binary"]["roc_auc"],
                    "sensitivity": metrics["binary"]["sensitivity"],
                    "specificity": metrics["binary"]["specificity"],
                    "referable_auc": metrics["referable"]["roc_auc"],
                }
                comparison_rows.append(row)

            except Exception as e:
                logger.error(f"Failed to evaluate ensemble {ensemble_name} on {dataset_name}: {e}")
                continue

    if comparison_rows:
        write_comparison(comparison_rows, out_dir / "comparison.csv")

    logger.info(f"Evaluation complete. Results in {out_dir}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
