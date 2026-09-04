"""
Background inference task executed by Celery workers.

The model/vectorizer are loaded once per worker PROCESS (not once per
task invocation) via a `worker_process_init` signal, with a lazy
fallback inside the task itself in case that signal hasn't fired yet
(e.g. when running the task function directly in tests).
"""
import logging
import os

from celery.signals import worker_process_init

from app.config import get_settings
from ml import inference
from worker.celery_app import celery_app

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def _load_models_if_needed() -> None:
    if not inference.is_loaded():
        settings = get_settings()
        inference.load_models(settings.model_path, settings.vectorizer_path)


@worker_process_init.connect
def _on_worker_process_init(**kwargs):
    """
    Fired once when a Celery worker (child) process starts up. This is
    the correct hook for "load once per worker process" — NOT inside the
    task body on every call.
    """
    logger.info("Worker process starting — loading ML artifacts into memory")
    _load_models_if_needed()


@celery_app.task(bind=True, name="tasks.predict_async")
def predict_task(self, text: str) -> dict:
    """
    Runs sentiment/intent classification in the background.

    Returns a plain dict (JSON-serializable) which Celery stores in the
    Redis result backend: {"prediction": str, "confidence": float}
    """
    task_id = self.request.id
    extra = {"task_id": task_id}

    logger.info("Task %s started for input of length %d", task_id, len(text), extra=extra)

    try:
        _load_models_if_needed()
        result = inference.run_inference(text)
        logger.info(
            "Task %s completed successfully: prediction=%s confidence=%.4f",
            task_id, result["prediction"], result["confidence"], extra=extra,
        )
        return result
    except Exception as exc:
        logger.error("Task %s failed: %s", task_id, exc, exc_info=True, extra=extra)
        # Re-raise so Celery marks the task state as FAILURE and stores
        # the exception in the result backend for /status to report.
        raise
