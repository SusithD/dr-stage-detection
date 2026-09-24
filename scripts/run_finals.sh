#!/bin/bash
set -uo pipefail

cd "$(dirname "$0")/.."
mkdir -p runs

if [ "${1:-}" = "--after-ablations" ]; then
  until grep -q "end lr_convnext_2e-4_384" runs/ablations.log 2>/dev/null; do sleep 60; done
fi

for exp in final_convnext_b_512 final_convnext_b_512_ce; do
  echo "$(date '+%F %T') start $exp" | tee -a runs/finals.log
  caffeinate -dims uv run python -m drstage.cli.train --config "configs/train/$exp.yaml" \
    >> "runs/$exp.stdout.log" 2>&1
  echo "$(date '+%F %T') end $exp (exit $?)" | tee -a runs/finals.log
done
