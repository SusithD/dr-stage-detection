"""Request and response schema definitions for the API."""

from typing import Literal

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
    assistant: bool = False


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


class Attention(BaseModel):
    """Grad-CAM attention region and spread information."""

    region: str = Field(
        ...,
        description="Image region holding most of the Grad-CAM weight, e.g. 'upper left'",
    )
    spread: float = Field(
        ...,
        ge=0,
        le=1,
        description="Fraction of the image where the heatmap is at least half its peak",
    )


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
    attention: Attention
    inference_ms: float


class Sample(BaseModel):
    """Gallery sample: true grade, name and thumbnail URL."""

    id: str
    label: int = Field(..., ge=0, le=4)
    grade_name: str
    thumbnail: str = Field(..., description="URL to /api/samples/{id}/thumbnail")


class CheckContext(BaseModel):
    """Outcome of a single quality or diagnostic check."""

    label: str
    status: Literal["ok", "warn", "fail"]
    message: str


class ReviewContext(BaseModel):
    """Expert review of a prediction with decision and rationale."""

    decision: Literal["confirmed", "changed"]
    grade: int = Field(..., ge=0, le=4)
    reason: str = ""
    note: str = ""


class EyeContext(BaseModel):
    """Prediction and context for one eye including checks, attention and optional review."""

    side: Literal["right", "left"]
    grade: int = Field(..., ge=0, le=4)
    grade_name: str
    class_probs: list[float] = Field(..., min_length=5, max_length=5)
    prob_dr: float = Field(..., ge=0, le=1)
    referable: bool
    gradable: bool
    checks: list[CheckContext]
    attention: Attention
    reference_grade: int | None = Field(None, ge=0, le=4)
    review: ReviewContext | None = None


class VisitContext(BaseModel):
    """Complete context for a visit including both eyes and overall outcome."""

    eyes: list[EyeContext] = Field(max_length=2)
    outcome_title: str
    outcome_detail: str


class AssistantMessage(BaseModel):
    """A single message in the assistant conversation."""

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)


class AssistantRequest(BaseModel):
    """Request to the assistant with conversation history and visit context."""

    messages: list[AssistantMessage] = Field(..., min_length=1, max_length=20)
    visit: VisitContext


class AssistantResponse(BaseModel):
    """Response from the assistant with reply and tools used."""

    reply: str
    tools_used: list[str]
