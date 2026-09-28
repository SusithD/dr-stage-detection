"""Data preparation pipeline: manifest -> hash -> dedup -> split -> cache -> report.

Every stage reads its inputs from and writes its outputs to ``paths.processed_root`` so that
stages can be re-run independently, e.g. ``--stage cache --methods clahe_l``.
"""

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import pandas as pd

from drstage.data.cache import build_cache
from drstage.data.dedup import (
    compute_hashes,
    cross_dataset_overlap,
    cross_split_leakage,
    duplicate_example_pairs,
    find_duplicate_groups,
    resolve_duplicates,
    summarise_groups,
)
from drstage.data.registry import load_manifest, spec_from_config
from drstage.data.split import check_split, split_summary, stratified_group_split
from drstage.utils.config import load_config
from drstage.utils.io import read_json, write_csv, write_json
from drstage.utils.logging import get_logger
from drstage.utils.seed import seed_everything
from drstage.visualization.plots import (
    plot_class_distribution,
    plot_dup_examples,
    plot_image_size_hist,
    plot_preprocess_examples,
    plot_split_distribution,
)

STAGES = ("manifest", "hash", "dedup", "split", "cache", "report")
KEEP_COLUMNS = ["image_id", "path", "label", "dup_group", "width", "height"]

logger = logging.getLogger("drstage.prepare_data")


class Paths:
    """Locations of every artefact produced by the pipeline."""

    def __init__(self, root: Path):
        """Remember the root folder all artefacts are written under."""
        self.root = root
        self.reports = root / "reports"
        self.splits = root / "splits"

    def manifest(self, ds: str) -> Path:
        """Manifest CSV for a dataset."""
        return self.root / "manifests" / f"{ds}_raw.csv"

    def hashes(self, ds: str) -> Path:
        """Perceptual-hash CSV for a dataset."""
        return self.root / "hashes" / f"{ds}_phash.csv"

    def kept(self, ds: str) -> Path:
        """Images kept after de-duplication."""
        return self.root / "dedup" / f"{ds}_kept.csv"

    def dropped(self, ds: str) -> Path:
        """Images dropped by de-duplication, with the reason."""
        return self.root / "dedup" / f"{ds}_dropped.csv"

    @property
    def ddr_split(self) -> Path:
        """DDR train/val/test split CSV."""
        return self.splits / "ddr_split.csv"

    @property
    def aptos_test(self) -> Path:
        """Cleaned APTOS external test CSV."""
        return self.splits / "aptos_test.csv"

    @property
    def dedup_report(self) -> Path:
        """JSON summary of the duplicate audit."""
        return self.reports / "dedup_report.json"


def available_datasets(cfg: dict, requested: list[str]) -> list[str]:
    """Filter requested datasets to those whose label CSV exists."""
    present = []
    for ds in requested:
        if Path(cfg["datasets"][ds]["csv"]).exists():
            present.append(ds)
        else:
            logger.warning(f"{ds}: label CSV not found, skipping")
    return present


def stage_manifest(cfg: dict, paths: Paths, datasets: list[str], overwrite: bool) -> None:
    """Build the manifest of labelled images for each dataset."""
    for ds in datasets:
        if paths.manifest(ds).exists() and not overwrite:
            logger.info(f"{ds}: manifest exists, skipping")
            continue
        manifest = load_manifest(spec_from_config(ds, cfg))
        write_csv(manifest, paths.manifest(ds))
        logger.info(f"{ds}: manifest with {len(manifest)} labelled images")


def stage_hash(cfg: dict, paths: Paths, datasets: list[str], overwrite: bool) -> None:
    """Compute a perceptual hash for every image in each manifest."""
    for ds in datasets:
        if paths.hashes(ds).exists() and not overwrite:
            logger.info(f"{ds}: hashes exist, skipping")
            continue
        manifest = pd.read_csv(paths.manifest(ds))
        hashes = compute_hashes(manifest, cfg, cfg["preprocess"]["num_workers"])
        write_csv(hashes, paths.hashes(ds))
        logger.info(f"{ds}: hashed {len(hashes)} images")


def _dedup_dataset(cfg: dict, paths: Paths, ds: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Group near-duplicates in one dataset and keep one image per label-consistent group."""
    manifest = pd.read_csv(paths.manifest(ds))
    hashes = pd.read_csv(paths.hashes(ds))
    groups = find_duplicate_groups(hashes, cfg["dedup"]["near_dup_threshold"])
    df = manifest.merge(groups, on="image_id", how="inner")
    kept, dropped = resolve_duplicates(df[["image_id", "label", "dup_group"]])
    kept = df[df["image_id"].isin(kept["image_id"])][KEEP_COLUMNS]
    summary = summarise_groups(groups, manifest[["image_id", "label"]])
    summary["n_unhashed"] = int(len(manifest) - len(df))
    return kept.reset_index(drop=True), dropped, summary


def stage_dedup(cfg: dict, paths: Paths, datasets: list[str]) -> None:
    """Drop near-duplicate and conflicting images, and APTOS images that overlap DDR."""
    report: dict = {}
    for ds in datasets:
        kept, dropped, summary = _dedup_dataset(cfg, paths, ds)
        if ds == "aptos" and "ddr" in datasets:
            overlap = cross_dataset_overlap(
                pd.read_csv(paths.hashes("ddr")),
                pd.read_csv(paths.hashes("aptos")),
                cfg["dedup"]["near_dup_threshold"],
            )
            overlap_ids = set(overlap["b_id"])
            moved = kept[kept["image_id"].isin(overlap_ids)].assign(reason="overlaps_ddr")
            dropped = pd.concat([dropped, moved[["image_id", "label", "dup_group", "reason"]]])
            kept = kept[~kept["image_id"].isin(overlap_ids)]
            summary["cross_dataset_pairs_with_ddr"] = int(len(overlap))

        summary["n_kept"] = int(len(kept))
        summary["n_dropped"] = (
            {k: int(v) for k, v in dropped["reason"].value_counts().items()} if len(dropped) else {}
        )
        write_csv(kept, paths.kept(ds))
        write_csv(dropped, paths.dropped(ds))
        report[ds] = summary
        logger.info(f"{ds}: kept {len(kept)}, dropped {summary['n_dropped']}")
    write_json(report, paths.dedup_report)


def stage_split(cfg: dict, paths: Paths, datasets: list[str]) -> bool:
    """Split DDR into train/val/test and export the cleaned APTOS external test set."""
    ok = True
    if "ddr" in datasets:
        kept = pd.read_csv(paths.kept("ddr"))
        split = stratified_group_split(kept, cfg["split"]["ratios"], cfg["seed"])
        check_split(split)
        leakage = cross_split_leakage(split)
        if leakage:
            logger.error(f"{leakage} duplicate groups span more than one split")
            ok = False
        write_csv(split[KEEP_COLUMNS[:3] + ["split"] + KEEP_COLUMNS[3:]], paths.ddr_split)

        summary = split_summary(split)
        summary["seed"] = cfg["seed"]
        summary["ddr_split_sha256"] = hashlib.sha256(paths.ddr_split.read_bytes()).hexdigest()
        write_json(summary, paths.splits / "split_summary.json")

        report = read_json(paths.dedup_report)
        report["ddr"]["cross_split_groups"] = leakage
        write_json(report, paths.dedup_report)
        logger.info(f"ddr: split counts {summary['split_counts']}")

    if "aptos" in datasets:
        write_csv(pd.read_csv(paths.kept("aptos")), paths.aptos_test)
        logger.info(f"aptos: external test set written to {paths.aptos_test}")
    return ok


def stage_cache(
    cfg: dict, paths: Paths, datasets: list[str], methods: list[str], overwrite: bool
) -> None:
    """Preprocess and cache images for each requested method."""
    sources = {"ddr": paths.ddr_split, "aptos": paths.aptos_test}
    for ds in datasets:
        frame = pd.read_csv(sources[ds])
        for method in methods:
            result = build_cache(
                frame,
                ds,
                method,
                cfg["preprocess"],
                paths.root,
                cfg["preprocess"]["num_workers"],
                overwrite,
            )
            n_err = int((result["error"] != "").sum())
            n_fb = int(result["crop_fallback"].sum())
            logger.info(
                f"{ds}/{method}: {len(result)} images, {n_err} errors, {n_fb} crop fallbacks"
            )


def stage_report(cfg: dict, paths: Paths, datasets: list[str]) -> None:
    """Write the data-audit figures used in the report."""
    frames = {ds: pd.read_csv(paths.manifest(ds)) for ds in datasets}
    plot_class_distribution(
        {ds.upper(): f for ds, f in frames.items()}, paths.reports / "fig_class_distribution.png"
    )
    if "ddr" not in datasets:
        return

    split = pd.read_csv(paths.ddr_split)
    plot_split_distribution(split, paths.reports / "fig_split_distribution.png")
    plot_image_size_hist(frames["ddr"], paths.reports / "fig_image_size_hist.png")

    train = split[split["split"] == "train"]
    samples = train.groupby("label", group_keys=False).sample(n=1, random_state=cfg["seed"])
    plot_preprocess_examples(
        samples.sort_values("label"),
        cfg["preprocess"],
        paths.reports / "fig_preprocess_examples.png",
    )

    hashes = pd.read_csv(paths.hashes("ddr"))
    groups = find_duplicate_groups(hashes, cfg["dedup"]["near_dup_threshold"])
    max_pairs = cfg["dedup"]["max_examples_figure"]
    pairs = duplicate_example_pairs(groups, hashes, frames["ddr"], max_pairs)
    if pairs:
        plot_dup_examples(pairs, paths.reports / "fig_dup_examples.png", max_pairs)
    logger.info(f"figures written to {paths.reports}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line options for the data pipeline."""
    parser = argparse.ArgumentParser(description="Prepare DDR and APTOS data for DR grading")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--stage", default="all", choices=("all",) + STAGES)
    parser.add_argument("--datasets", default="ddr,aptos", help="comma-separated dataset names")
    parser.add_argument("--methods", default=None, help="comma-separated preprocessing methods")
    parser.add_argument("--overwrite", action="store_true", help="recompute cached outputs")
    parser.add_argument("-o", dest="overrides", action="append", default=[], help="key.sub=value")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the requested data-preparation stages in order."""
    args = parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg["seed"], deterministic=False)
    paths = Paths(Path(cfg["paths"]["processed_root"]))
    get_logger("drstage", log_file=paths.reports / "prepare_data.log")

    datasets = available_datasets(cfg, [d.strip() for d in args.datasets.split(",")])
    methods = args.methods.split(",") if args.methods else cfg["preprocess"]["methods"]
    stages = STAGES if args.stage == "all" else (args.stage,)

    ok = True
    for stage in stages:
        logger.info(f"=== {stage} ===")
        if stage == "manifest":
            stage_manifest(cfg, paths, datasets, args.overwrite)
        elif stage == "hash":
            stage_hash(cfg, paths, datasets, args.overwrite)
        elif stage == "dedup":
            stage_dedup(cfg, paths, datasets)
        elif stage == "split":
            ok = stage_split(cfg, paths, datasets) and ok
        elif stage == "cache":
            stage_cache(cfg, paths, datasets, methods, args.overwrite)
        elif stage == "report":
            stage_report(cfg, paths, datasets)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
