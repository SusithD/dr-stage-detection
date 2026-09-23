# DR Stage Detection

Detection and ICDR stage grading (0 no DR, 1 mild, 2 moderate, 3 severe, 4 proliferative) of
diabetic retinopathy from colour fundus photographs, using a transfer-learned CNN with an ordinal
(CORN) output head, plus the RetinaScan web prototype.

- **Training data:** DDR (Li et al., 2019; 13,673 images from 147 hospitals), Kaggle mirror
  `mariaherrerot/ddrdataset`, split 6:2:3 into train/validation/test.
- **External test:** APTOS 2019 labelled images, Kaggle mirror `mariaherrerot/aptos2019`, never
  used for training or model selection.
- **Main model:** ConvNeXt-Base (ImageNet-22k pretrained), compared against ResNet50,
  EfficientNet-B0 and EfficientNet-B5 (Noisy Student).

## Setup

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), Node 20+ (web app only) and a Kaggle API
token in `~/.kaggle/`.

```bash
uv sync
scripts/download_data.sh
```

## Prepare data

```bash
uv run python -m drstage.cli.prepare_data --config configs/data.yaml --methods ben_graham
```

Stages run in order and can be re-run individually with `--stage`:

| Stage | Output |
|---|---|
| `manifest` | `data/processed/manifests/<dataset>_raw.csv` |
| `hash` | perceptual hashes for duplicate detection |
| `dedup` | near-duplicate groups; label-conflicting groups dropped, APTOS images overlapping DDR dropped |
| `split` | `splits/ddr_split.csv` (stratified 6:2:3), `splits/aptos_test.csv` |
| `cache` | preprocessed 512 px images per method (`ben_graham`, `clahe_l`, `none`) |
| `report` | `reports/` figures and `dedup_report.json` |

Config values can be overridden, e.g. `-o preprocess.num_workers=4`.

## Train

```bash
uv run python -m drstage.cli.train --config configs/train/final_convnext_b_512.yaml
```

Each run writes `runs/<exp_name>/<timestamp>/` with the resolved config, `history.csv`,
`checkpoints/best.pt` (EMA weights plus everything needed for inference) and `last.pt`
(resume with `--resume <run dir>`). Experiment configs in `configs/train/` inherit from
`base.yaml`:

| Config | Purpose |
|---|---|
| `arch_*_384.yaml` | architecture comparison at 384 px (`scripts/run_arch_comparison.sh`) |
| `abl_*_384.yaml` | output head, preprocessing and sampler ablations |
| `lr_*_384.yaml` | learning-rate sweep |
| `final_convnext_b_512.yaml`, `final_effnet_b5_456.yaml` | final models |

## Evaluate and explain

```bash
uv run python -m drstage.cli.evaluate --config configs/eval/final_models.yaml
uv run python -m drstage.cli.gradcam --checkpoint runs/final_convnext_b_512 \
  --dataset ddr --split test --out runs/final_convnext_b_512/gradcam
```

Evaluation reports accuracy, per-class precision/recall/F1, quadratic weighted kappa, confusion
matrices, DR-vs-no-DR and referable-DR sensitivity/specificity/ROC-AUC, flip test-time
augmentation, ensembles and error analysis for the DDR test set and APTOS.

## Web app

```bash
cd frontend && npm install && npm run build && cd ..
uv run python -m drstage.cli.serve --checkpoint runs/final_convnext_b_512
```

Open http://127.0.0.1:8000. Without `--checkpoint` the app runs in a clearly labelled demo mode.
For frontend development run `npm run dev` in `frontend/` alongside the server.

## Development

```bash
uv run pytest
uv run ruff check src tests && uv run ruff format --check src tests
```
