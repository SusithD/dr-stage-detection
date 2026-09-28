"""Request and response schema definitions for the API."""

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Backbone, head type, image size and preprocessing method."""

    name: str
    head: str
    image_size: int
    preprocessing: str
    checkpoint: str


class HealthResponse(BaseModel):
    """Server status: whether a trained model is loaded and what device it's on."""

    status: str = "ok"
    demo: bool
    device: str
    model: ModelInfo | None


class PredictImages(BaseModel):
    """Original, enhanced and Grad-CAM heatmap as PNG data URLs."""

    original: str = Field(..., description="Data URL: circle-cropped, resized, no enhancement")
    enhanced: str = Field(..., description="Data URL: preprocessed image the model saw")
    heatmap: str = Field(..., description="Data URL: RGBA Grad-CAM")


class QualityCheckOut(BaseModel):
    """Outcome of one image quality check: exposure, contrast, focus or field."""

    key: str
    label: str
    status: str = Field(..., description="ok, warn or fail")
    value: float
    message: str


class QualityReport(BaseModel):
    """Summary of image quality checks; gradable if none fail."""

    gradable: bool
    checks: list[QualityCheckOut]


class PredictResponse(BaseModel):
    """Prediction: grade, probabilities, image visualisations and quality checks."""

    demo: bool
    grade: int = Field(..., ge=0, le=4, description="ICDR grade 0-4")
    grade_name: str
    dr_detected: bool = Field(..., description="prob_dr >= 0.5")
    prob_dr: float = Field(..., ge=0, le=1, description="Probability of any retinopathy")
    referable: bool = Field(..., description="grade >= 2")
    class_probs: list[float] = Field(..., description="5 probabilities summing to ~1")
    images: PredictImages
    quality: QualityReport
    inference_ms: float


class Sample(BaseModel):
    """Gallery sample: true grade, name and thumbnail URL."""

    id: str
    label: int = Field(..., ge=0, le=4)
    grade_name: str
    thumbnail: str = Field(..., description="URL to /api/samples/{id}/thumbnail")
