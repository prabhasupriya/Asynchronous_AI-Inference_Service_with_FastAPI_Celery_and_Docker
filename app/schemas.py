"""
Pydantic models used for request validation and response serialization.
"""
from typing import Optional
from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    """Incoming payload for both /predict_sync and /predict_async."""
    text: str = Field(..., min_length=1, description="The text to classify")


class InferenceResponse(BaseModel):
    """Result of a completed inference."""
    prediction: str
    confidence: float


class TaskIdResponse(BaseModel):
    """Returned immediately after dispatching a background task."""
    task_id: str


class TaskStatusResponse(BaseModel):
    """Returned by GET /status/{task_id}."""
    task_id: str
    status: str  # PENDING | STARTED | SUCCESS | FAILURE | RETRY
    result: Optional[InferenceResponse] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
