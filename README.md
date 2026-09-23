# DR Stage Detection

Diabetic Retinopathy grading from fundus photographs using deep learning.

## Setup

### Requirements
- Python 3.12+
- uv package manager

### Installation

```bash
git clone <repo>
cd dr-stage-detection
uv sync
```

## Data

### Download

Download the DDR and APTOS datasets:

```bash
scripts/download_data.sh
```

This requires the Kaggle CLI to be configured with credentials.

### Directory Structure

After download, the data should be organized as:

```
data/
  raw/
    ddr/
      DR_grading.csv
      DR_grading/
        DR_grading/
          <image files>
    aptos/
      train.csv
      train_images/
        <image files>
```

## Prepare Data

Run the full data preparation pipeline:

```bash
uv run python -m drstage.cli.prepare_data --config configs/data.yaml
```

### Pipeline Stages

- `manifest`: Load and validate raw datasets
- `hash`: Compute perceptual hashes
- `dedup`: Identify and remove duplicates
- `split`: Stratified split into train/val/test
- `cache`: Preprocess and cache images
- `report`: Generate analysis figures

Run specific stages:

```bash
uv run python -m drstage.cli.prepare_data --stage manifest
uv run python -m drstage.cli.prepare_data --stage cache --methods ben_graham,clahe_l
```

### Configuration

Configuration is in `configs/data.yaml`. Override values on the command line:

```bash
uv run python -m drstage.cli.prepare_data -o preprocess.size=384 -o seed=123
```

## Output Layout

After preparation, processed data is at `data/processed/`:

```
data/processed/
  manifests/
    ddr_raw.csv, aptos_raw.csv
  hashes/
    ddr_phash.csv, aptos_phash.csv
  images/
    <dataset>/<method>/<size>/
      <image_id>.png
  splits/
    ddr_split.csv, aptos_test.csv, split_summary.json
  reports/
    dedup_report.json, prepare_data.log
    fig_class_distribution.png, fig_split_distribution.png
    fig_image_size_hist.png, fig_preprocess_examples.png
    fig_dup_examples.png
```

## Testing

```bash
uv run pytest tests/
```

## Linting and Formatting

```bash
uv run ruff check src/
uv run ruff format src/
```
