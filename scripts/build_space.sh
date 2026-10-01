#!/bin/bash
# Assemble build/space/: everything the Hugging Face Docker Space needs to run RetinaScan.
set -euo pipefail

cd "$(dirname "$0")/.."
CHECKPOINT="${1:-runs/final_convnext_b_512_ce}"
OUT=build/space

rm -rf "$OUT"
mkdir -p "$OUT/model" "$OUT/samples" "$OUT/configs" "$OUT/frontend"

cp deploy/Dockerfile deploy/requirements.txt deploy/README.md "$OUT/"
rsync -a --exclude __pycache__ src "$OUT/"
cp configs/data.yaml "$OUT/configs/"

(cd frontend && npm run build >/dev/null)
cp -R frontend/dist "$OUT/frontend/dist"

# half-precision weights halve the download; the predictor casts them back to float32 on load
uv run python - "$CHECKPOINT" "$OUT" <<'PY'
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch

from drstage.api.app import select_samples
from drstage.training.checkpoint import resolve_checkpoint

checkpoint, out = resolve_checkpoint(sys.argv[1]), Path(sys.argv[2])
state = torch.load(checkpoint, map_location="cpu", weights_only=False)
state["model_state"] = {k: v.half() if v.is_floating_point() else v for k, v in state["model_state"].items()}
for key in ("config", "ema_state", "optimizer_state", "scheduler_state", "scaler_state", "history"):
    state.pop(key, None)
torch.save(state, out / "model/best_fp16.pt")

rows = []
for image_id, row in select_samples(Path("data/processed/splits/ddr_split.csv")).iterrows():
    src = Path(row.path)
    dst = out / "samples" / f"{image_id}{src.suffix}"
    shutil.copy(src, dst)
    rows.append({"image_id": image_id, "path": f"/app/samples/{dst.name}", "label": row.label, "split": "test"})
pd.DataFrame(rows).to_csv(out / "samples/samples.csv", index=False)
print(f"model {(out / 'model/best_fp16.pt').stat().st_size / 1e6:.0f} MB, {len(rows)} samples")
PY

du -sh "$OUT"
