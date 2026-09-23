#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

uvx kaggle datasets download mariaherrerot/ddrdataset -p data/raw/ddr --unzip
uvx kaggle datasets download mariaherrerot/aptos2019 -p data/raw/aptos --unzip
rm -f data/raw/aptos/aptos2019.zip data/raw/ddr/ddrdataset.zip

# The APTOS mirror splits the labelled competition training set into three folders; the whole
# set is an external test set here, so merge the labels and link every image into one folder.
cd data/raw/aptos
mkdir -p all_images
for split in train_images val_images test_images; do
  (cd all_images && ln -sf "../$split/$split/"*.png .)
done
uv run python -c "import pandas as pd; pd.concat(pd.read_csv(f) for f in ['train_1.csv', 'valid.csv', 'test.csv']).to_csv('all_labels.csv', index=False)"

echo "Data download complete: $(ls all_images | wc -l | tr -d ' ') APTOS images linked"
