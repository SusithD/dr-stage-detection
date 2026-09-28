"""FastAPI application serving the RetinaScan API and the built frontend."""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from starlette.concurrency import run_in_threadpool

from drstage.api.imaging import decode_upload, heatmap_to_rgba, thumbnail_jpeg, to_png_data_url
from drstage.api.predictors import Predictor, load_predictor
from drstage.api.quality import assess_quality, is_gradable
from drstage.api.schemas import (
    HealthResponse,
    ModelInfo,
    PredictImages,
    PredictResponse,
    QualityCheckOut,
    QualityReport,
    Sample,
)
from drstage.data.registry import CLASS_NAMES

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
SAMPLES_PER_GRADE = 2
SAMPLE_SEED = 7


@dataclass
class AppState:
    """Objects shared by request handlers for the lifetime of the server."""

    predictor: Predictor | None
    samples: pd.DataFrame = field(default_factory=pd.DataFrame)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def select_samples(split_csv: Path) -> pd.DataFrame:
    """Pick a fixed set of test-split images, SAMPLES_PER_GRADE per grade, indexed by image_id."""
    if not split_csv.exists():
        logger.warning(f"split CSV {split_csv} not found; the sample gallery will be empty")
        return pd.DataFrame(columns=["label", "path"])
    df = pd.read_csv(split_csv)
    if "split" in df.columns:
        df = df[df["split"] == "test"]
    shuffled = df.sort_values("image_id").sample(frac=1.0, random_state=SAMPLE_SEED)
    picked = shuffled.groupby("label").head(SAMPLES_PER_GRADE)
    return picked.set_index("image_id")[["label", "path"]].sort_values("label")


def _sample_path(state: AppState, sample_id: str) -> Path:
    """Resolve a gallery sample id to its image file, or raise a 404 with a clear message."""
    if sample_id not in state.samples.index:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No sample with id '{sample_id}'.")
    path = Path(state.samples.at[sample_id, "path"])
    if not path.exists():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"The image file for '{sample_id}' is missing."
        )
    return path


def create_app(
    predictor: Predictor | None = None,
    split_csv: Path | None = None,
    frontend_dist: Path | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    Args:
        predictor: Predictor to serve; loaded from DRSTAGE_CHECKPOINT (demo if unset) when None.
        split_csv: Split CSV used for the sample gallery (default DRSTAGE_SPLIT_CSV).
        frontend_dist: Built frontend directory served at "/" (default DRSTAGE_FRONTEND_DIST).

    Returns:
        Configured FastAPI application.
    """
    split_csv = split_csv or Path(
        os.getenv("DRSTAGE_SPLIT_CSV", "data/processed/splits/ddr_split.csv")
    )
    frontend_dist = frontend_dist or Path(os.getenv("DRSTAGE_FRONTEND_DIST", "frontend/dist"))
    state = AppState(predictor=predictor, samples=select_samples(split_csv))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        """Load the predictor once at start-up (the demo predictor if no checkpoint is set)."""
        if state.predictor is None:
            state.predictor = load_predictor(
                os.getenv("DRSTAGE_CHECKPOINT"), os.getenv("DRSTAGE_DEVICE", "auto")
            )
        mode = "demo" if state.predictor.demo else "model"
        logger.info(f"RetinaScan ready ({mode} mode, {len(state.samples)} samples)")
        yield

    app = FastAPI(
        title="RetinaScan", docs_url="/api/docs", openapi_url="/api/openapi.json", lifespan=lifespan
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Report whether a trained model is loaded, which one, and on which device."""
        info = state.predictor.info()
        return HealthResponse(
            status="ok",
            demo=state.predictor.demo,
            device=info.get("device", "cpu") if info else "cpu",
            model=ModelInfo(**{k: info[k] for k in ModelInfo.model_fields}) if info else None,
        )

    @app.post("/api/predict", response_model=PredictResponse)
    async def predict(file: UploadFile) -> PredictResponse:
        """Grade one uploaded photograph: grades, probabilities, images and quality checks."""
        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                "The file is larger than 20 MB. Upload a smaller JPG or PNG photograph.",
            )
        try:
            image = decode_upload(data)
        except ValueError as e:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from e

        start = time.perf_counter()
        async with state.lock:
            try:
                pred = await run_in_threadpool(state.predictor.predict, image)
                checks = await run_in_threadpool(assess_quality, image)
            except Exception as e:
                logger.exception("prediction failed")
                raise HTTPException(
                    status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "The model could not analyse this image. Upload a colour fundus photograph.",
                ) from e
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"predict {file.filename!r}: grade {pred.grade} in {elapsed_ms:.0f} ms")

        return PredictResponse(
            demo=state.predictor.demo,
            grade=pred.grade,
            grade_name=pred.grade_name,
            dr_detected=pred.grade >= 1,
            prob_dr=pred.prob_dr,
            referable=pred.grade >= 2,
            class_probs=pred.class_probs,
            images=PredictImages(
                original=to_png_data_url(pred.original),
                enhanced=to_png_data_url(pred.enhanced),
                heatmap=to_png_data_url(heatmap_to_rgba(pred.heatmap)),
            ),
            quality=QualityReport(
                gradable=is_gradable(checks),
                checks=[QualityCheckOut(**vars(c)) for c in checks],
            ),
            inference_ms=round(elapsed_ms, 1),
        )

    @app.get("/api/samples", response_model=list[Sample])
    async def samples() -> list[Sample]:
        """List the fixed gallery of held-out test photographs with their reference grades."""
        return [
            Sample(
                id=sid,
                label=int(row.label),
                grade_name=CLASS_NAMES[int(row.label)],
                thumbnail=f"/api/samples/{sid}/thumbnail",
            )
            for sid, row in state.samples.iterrows()
        ]

    @app.get("/api/samples/{sample_id}/thumbnail")
    async def sample_thumbnail(sample_id: str) -> Response:
        """Return a 256 px circle-cropped JPEG thumbnail of a gallery photograph."""
        jpeg = await run_in_threadpool(thumbnail_jpeg, _sample_path(state, sample_id), 256)
        return Response(jpeg, media_type="image/jpeg")

    @app.get("/api/samples/{sample_id}/image")
    async def sample_image(sample_id: str) -> FileResponse:
        """Return the original file of a gallery photograph so the UI can analyse it."""
        return FileResponse(_sample_path(state, sample_id))

    @app.get("/{path:path}", include_in_schema=False)
    async def frontend(path: str) -> FileResponse:
        """Serve the built app with an index.html fallback, refusing paths outside it."""
        if path.startswith("api/") or not (frontend_dist / "index.html").exists():
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found.")
        root = frontend_dist.resolve()
        target = (root / path).resolve()
        if target.is_file() and target.is_relative_to(root):
            return FileResponse(target)
        return FileResponse(root / "index.html")

    return app
