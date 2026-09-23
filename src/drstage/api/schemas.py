"""Pydantic models for API request/response contracts."""

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Model metadata."""

    name: str
    head: str
    image_size: int
    preprocessing: str
    checkpoint: str


class HealthResponse(BaseModel):
    """GET /api/health response."""

    status: str = "ok"
    demo: bool
    device: str
    model: ModelInfo | None


class PredictImages(BaseModel):
    """Images returned by /api/predict."""

    original: str = Field(..., description="Data URL: circle-cropped, resized, no enhancement")
    enhanced: str = Field(..., description="Data URL: preprocessed image the model saw")
    heatmap: str = Field(..., description="Data URL: RGBA Grad-CAM")


class PredictResponse(BaseModel):
    """POST /api/predict response."""

    demo: bool
    grade: int = Field(..., ge=0, le=4, description="ICDR grade 0-4")
    grade_name: str
    dr_detected: bool = Field(..., description="prob_dr >= 0.5")
    prob_dr: float = Field(..., ge=0, le=1, description="Probability of any retinopathy")
    referable: bool = Field(..., description="grade >= 2")
    class_probs: list[float] = Field(..., description="5 probabilities summing to ~1")
    images: PredictImages
    inference_ms: float


class Sample(BaseModel):
    """Sample from test split."""

    id: str
    label: int = Field(..., ge=0, le=4)
    grade_name: str
    thumbnail: str = Field(..., description="URL to /api/samples/{id}/thumbnail")
