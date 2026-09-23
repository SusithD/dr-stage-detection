#!/bin/bash
set -uo pipefail

cd "$(dirname "$0")/.."
mkdir -p runs

if [ "${1:-}" = "--after-arch" ]; then
  until grep -q "end arch_effnet_b5_384" runs/arch_comparison.log 2>/dev/null; do sleep 60; done
fi

for exp in abl_head_ce_384 abl_prep_clahe_384 abl_prep_none_384 abl_sampler_384 \
  lr_convnext_5e-5_384 lr_convnext_2e-4_384; do
  echo "$(date '+%F %T') start $exp" | tee -a runs/ablations.log
  caffeinate -dims uv run python -m drstage.cli.train --config "configs/train/$exp.yaml" \
    >> "runs/$exp.stdout.log" 2>&1
  echo "$(date '+%F %T') end $exp (exit $?)" | tee -a runs/ablations.log
done
