#!/bin/bash
set -uo pipefail

cd "$(dirname "$0")/.."
mkdir -p runs

for exp in arch_convnext_b_384 arch_resnet50_384 arch_effnet_b0_384 arch_effnet_b5_384; do
  echo "$(date '+%F %T') start $exp" | tee -a runs/arch_comparison.log
  caffeinate -dims uv run python -m drstage.cli.train --config "configs/train/$exp.yaml" \
    >> "runs/$exp.stdout.log" 2>&1
  echo "$(date '+%F %T') end $exp (exit $?)" | tee -a runs/arch_comparison.log
done
