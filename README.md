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

## Results

| Model (DDR test set, 3,381 images) | QWK | Accuracy | Macro-F1 | DR sensitivity / specificity |
|---|---|---|---|---|
| ConvNeXt-Base 512 px, cross-entropy, flip TTA | 0.938 | 90.4% | 0.766 | 95.4% / 96.3% |
| Ensemble of cross-entropy and CORN heads | 0.940 | 90.6% | 0.774 | 95.6% / 96.7% |

On the external APTOS 2019 set the model keeps an any-DR AUC of 0.99 but is overconfident;
recalibrating the threshold on 20% of APTOS gives 97.5% sensitivity at 99.0% specificity on the
remaining 80%.

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
uv run python -m drstage.cli.train --config configs/train/final_convnext_b_512_ce.yaml
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
| `final_convnext_b_512_ce.yaml`, `final_convnext_b_512.yaml` | final models (cross-entropy and CORN heads) |

## Evaluate and explain

```bash
uv run python -m drstage.cli.evaluate --config configs/eval/final_models.yaml
uv run python -m drstage.cli.gradcam --checkpoint runs/final_convnext_b_512_ce \
  --predictions runs/eval/final_models/convnext_b_512_ce/ddr_test/no_tta/predictions.csv \
  --split-csv data/processed/splits/ddr_split.csv --out runs/eval/gradcam_ddr
uv run python -m drstage.cli.recalibrate \
  --predictions runs/eval/final_models/ensemble_ce_corn/aptos_ext/tta/predictions.csv \
  --reference runs/eval/final_models/ensemble_ce_corn/ddr_test/tta/predictions.csv \
  --out runs/eval/recalibration/ensemble_ce_corn
```

Evaluation reports accuracy, per-class precision/recall/F1, quadratic weighted kappa, confusion
matrices, DR-vs-no-DR and referable-DR sensitivity/specificity/ROC-AUC, flip test-time
augmentation, ensembles and error analysis for the DDR test set and APTOS.

## Web app

```bash
cd frontend && npm install && npm run build && cd ..
uv run python -m drstage.cli.serve --checkpoint runs/final_convnext_b_512_ce
```

Open http://127.0.0.1:8000. Without `--checkpoint` the app runs in a clearly labelled demo mode.
For frontend development run `npm run dev` in `frontend/` alongside the server.

### Live deployment

RetinaScan is hosted on Hugging Face Spaces at **https://susith-retinascan.hf.space**
([Space page](https://huggingface.co/spaces/Susith/retinascan)). The Space serves the same
FastAPI app and React build with half-precision weights on CPU (no TTA, about 2–3 s per
photograph). `scripts/build_space.sh` assembles the upload: `build/space/` for a Docker Space
(`deploy/Dockerfile`) and `build/space_gradio/` for a Gradio-runtime Space (`deploy/gradio/`).

## Development

```bash
uv run pytest
uv run ruff check src tests && uv run ruff format --check src tests
```
