"""Hugging Face Space launcher: unpack the RetinaScan bundle and serve the FastAPI app.

The Space uses the Gradio SDK only as a Python runtime; the UI is our own React app served by
FastAPI on port 7860. spaces must be imported before torch on ZeroGPU hardware.
"""

import os
import sys
import zipfile
from pathlib import Path

import pandas as pd
import spaces

ROOT = Path(__file__).resolve().parent
APP = Path("/tmp/retinascan")

if not (APP / "src").exists():
    zipfile.ZipFile(ROOT / "bundle.zip").extractall(APP)

# the sample list was written with container paths; point it at the unpacked copies
samples = pd.read_csv(APP / "samples/samples.csv")
samples["path"] = [str(APP / "samples" / Path(p).name) for p in samples["path"]]
samples.to_csv(APP / "samples/samples_local.csv", index=False)

os.environ.update(
    {
        "DRSTAGE_CHECKPOINT": str(ROOT / "best_fp16.pt"),
        "DRSTAGE_DEVICE": "cpu",
        "DRSTAGE_TTA": "0",
        "DRSTAGE_SPLIT_CSV": str(APP / "samples/samples_local.csv"),
        "DRSTAGE_FRONTEND_DIST": str(APP / "frontend/dist"),
        "MPLCONFIGDIR": "/tmp/matplotlib",
    }
)
sys.path.insert(0, str(APP / "src"))


@spaces.GPU
def _gpu_available() -> bool:
    """ZeroGPU expects at least one GPU function; inference itself runs on the CPU."""
    return True


if __name__ == "__main__":
    import uvicorn

    # ZeroGPU sends its startup report from a hook on gr.Blocks.launch, which a plain FastAPI
    # server never calls, so trigger it directly (only present on ZeroGPU hardware)
    from spaces import zero

    if hasattr(zero, "startup"):
        zero.startup()

    from drstage.api.app import create_app
    from drstage.utils.logging import get_logger

    get_logger("drstage")
    uvicorn.run(create_app(), host="0.0.0.0", port=7860)
