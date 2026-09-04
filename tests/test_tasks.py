"""
Unit tests for the Celery background task and the underlying ML
inference logic.

We set CELERY_TASK_ALWAYS_EAGER=true so `.delay()` / `.apply()` execute
the task synchronously, in-process, without needing a live Redis broker.
"""
import os

os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "true")

import pytest

from app.config import get_settings
from ml import inference
from worker.celery_app import celery_app
from worker.tasks import predict_task


@pytest.fixture(autouse=True)
def ensure_models_loaded():
    """Make sure artifacts are loaded before any inference-related test."""
    if not inference.is_loaded():
        settings = get_settings()
        inference.load_models(settings.model_path, settings.vectorizer_path)
    yield


def test_run_inference_returns_expected_shape():
    result = inference.run_inference("I absolutely love this")
    assert set(result.keys()) == {"prediction", "confidence"}
    assert isinstance(result["prediction"], str)
    assert 0.0 <= result["confidence"] <= 1.0


def test_run_inference_raises_on_empty_text():
    with pytest.raises(ValueError):
        inference.run_inference("   ")


def test_predict_task_apply_executes_synchronously_and_succeeds():
    """
    `.apply()` runs the task eagerly in the current process and returns
    an EagerResult we can inspect exactly like a real AsyncResult.
    """
    async_result = predict_task.apply(args=["This is a wonderful experience"])
    assert async_result.successful()
    result = async_result.get()
    assert "prediction" in result
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_task_direct_call_matches_run_inference():
    """
    Calling the underlying task function directly (bypassing Celery's
    machinery entirely) should produce the same result as calling
    run_inference() directly, proving the task is a thin wrapper.
    """
    text = "Terrible, I want a refund"
    direct = inference.run_inference(text)
    via_task = predict_task.run(text)
    assert direct == via_task


def test_predict_task_delay_with_eager_mode_returns_result():
    assert celery_app.conf.task_always_eager is True
    async_result = predict_task.delay("It is fine, nothing special")
    assert async_result.state == "SUCCESS"
    result = async_result.result
    assert result["prediction"] in {"positive", "negative", "neutral"}


def test_predict_task_failure_is_reported_as_failure_state(monkeypatch):
    """
    Simulate an internal failure (e.g. corrupt input handling) and verify
    the task correctly surfaces the exception rather than silently
    swallowing it or returning a bad result.

    Note: with `task_eager_propagates=True` (set for local/test eager
    mode), `.apply()` re-raises immediately instead of returning an
    EagerResult in FAILURE state. In a real deployment (non-eager, real
    broker), the same exception instead marks the task's Redis-backed
    state as FAILURE, which `/status/{task_id}` reports without crashing
    the API (see test_api.py::test_status_failure_state_does_not_crash_api).
    """
    def boom(_text):
        raise RuntimeError("simulated inference crash")

    monkeypatch.setattr(inference, "run_inference", boom)

    with pytest.raises(RuntimeError, match="simulated inference crash"):
        predict_task.apply(args=["trigger failure"])
