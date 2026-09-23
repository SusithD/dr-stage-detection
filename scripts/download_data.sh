#!/bin/bash
set -e

uvx kaggle datasets download mariaherrerot/ddrdataset -p data/raw/ddr --unzip
uvx kaggle datasets download mariaherrerot/aptos2019 -p data/raw/aptos --unzip

echo "Data download complete"
