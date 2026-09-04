"""
FastAPI application entrypoint.

Exposes:
    POST /predict_sync      -> blocking inference, returns result immediately
    POST /predict_async     -> dispatches a Celery background task, returns task_id
    GET  /status/{task_id}  -> polls Celery/Redis for task state + result
    GET  /health            -> basic liveness/readiness probe
"""
import logging
import os
from contextlib import asynccontextmanager

from celery.result import AsyncResult
from fastapi import Depends, FastAPI, HTTPException

from app.config import get_settings
from app.dependencies import ensure_model_loaded
from app.schemas import (
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    TaskIdResponse,
    TaskStatusResponse,
)
from ml import inference
from worker.celery_app import celery_app
from worker.tasks import predict_task

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup: load ML artifacts exactly once into memory ---
    settings = get_settings()
    logger.info("Application startup: loading ML model and vectorizer")
    try:
        inference.load_models(settings.model_path, settings.vectorizer_path)
    except FileNotFoundError as exc:
        # Fail loudly and early rather than serving requests with no model.
        logger.error("Failed to load ML artifacts on startup: %s", exc)
        raise
    yield
    # --- Shutdown: nothing to clean up for this simple service ---
    logger.info("Application shutdown")


app = FastAPI(
    title="AI Inference API",
    description="Asynchronous AI inference microservice using FastAPI, Celery, and Redis.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health():
    """Basic liveness probe used by Docker healthchecks."""
    return HealthResponse(status="ok")


@app.post("/predict_sync", response_model=InferenceResponse, tags=["Inference"])
async def predict_sync(
    request: InferenceRequest,
    _: None = Depends(ensure_model_loaded),
):
    """
    Runs inference synchronously and returns the result in the same
    HTTP response. Suitable for lightweight/low-latency models.
    """
    try:
        result = inference.run_inference(request.text)
        return InferenceResponse(**result)
    except inference.ModelNotLoadedError as exc:
        logger.error("Model not loaded during /predict_sync: %s", exc)
        raise HTTPException(status_code=500, detail="Model is not available.") from exc
    except Exception as exc:
        logger.error("Unexpected error during /predict_sync: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal inference error.") from exc


@app.post("/predict_async", response_model=TaskIdResponse, status_code=202, tags=["Inference"])
async def predict_async(request: InferenceRequest):
    """
    Dispatches the inference to a Celery background worker and returns
    immediately with a task_id the client can use to poll /status.
    """
    try:
        task = predict_task.delay(request.text)
        logger.info("Dispatched async task %s", task.id)
        return TaskIdResponse(task_id=task.id)
    except Exception as exc:
        logger.error("Failed to dispatch async task: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to dispatch background task.") from exc


@app.get("/status/{task_id}", response_model=TaskStatusResponse, tags=["Inference"])
async def get_status(task_id: str):
    """
    Queries the Celery/Redis result backend for the current state of a
    previously dispatched task.
    """
    task_result = AsyncResult(task_id, app=celery_app)
    state = task_result.state

    response = TaskStatusResponse(task_id=task_id, status=state)

    if state == "SUCCESS":
        response.result = InferenceResponse(**task_result.result)
    elif state == "FAILURE":
        # Never let a worker-side exception crash the API response —
        # safely stringify it and omit the result field.
        response.result = None
        response.error = str(task_result.result) if task_result.result else "Task failed."
    # PENDING / STARTED / RETRY -> result stays None, which is expected.

    return response
