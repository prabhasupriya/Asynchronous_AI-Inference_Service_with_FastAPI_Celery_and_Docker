"""
Unit tests for the FastAPI layer.

`predict_task.delay(...)` is monkeypatched so these tests never require a
real Redis broker to be running — they exercise the HTTP contract only.
Run `docker-compose up` (or a local Redis) for full end-to-end testing.
"""
import os
from unittest.mock import MagicMock

os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "true")

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------
# /predict_sync
# ---------------------------------------------------------------------

def test_predict_sync_returns_200_and_valid_schema(client):
    response = client.post("/predict_sync", json={"text": "I love this product"})
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert "confidence" in body
    assert isinstance(body["prediction"], str)
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_sync_missing_text_returns_422(client):
    response = client.post("/predict_sync", json={})
    assert response.status_code == 422


def test_predict_sync_empty_text_returns_422(client):
    response = client.post("/predict_sync", json={"text": ""})
    assert response.status_code == 422


def test_predict_sync_wrong_type_returns_422(client):
    response = client.post("/predict_sync", json={"text": 12345})
    assert response.status_code == 422


# ---------------------------------------------------------------------
# /predict_async
# ---------------------------------------------------------------------

def test_predict_async_dispatches_task_and_returns_task_id(client, monkeypatch):
    fake_result = MagicMock()
    fake_result.id = "fake-task-id-123"

    monkeypatch.setattr(
        "app.main.predict_task.delay",
        MagicMock(return_value=fake_result),
    )

    response = client.post("/predict_async", json={"text": "This is terrible"})
    assert response.status_code == 202
    body = response.json()
    assert body["task_id"] == "fake-task-id-123"


def test_predict_async_missing_text_returns_422(client):
    response = client.post("/predict_async", json={})
    assert response.status_code == 422


# ---------------------------------------------------------------------
# /status/{task_id}
# ---------------------------------------------------------------------

def test_status_unknown_task_returns_pending(client, monkeypatch):
    fake_async_result = MagicMock()
    fake_async_result.state = "PENDING"
    fake_async_result.result = None

    monkeypatch.setattr(
        "app.main.AsyncResult",
        MagicMock(return_value=fake_async_result),
    )

    response = client.get("/status/nonexistent-task-id")
    assert response.status_code == 200
    body = response.json()
    assert body["task_id"] == "nonexistent-task-id"
    assert body["status"] == "PENDING"
    assert body["result"] is None


def test_status_success_state_includes_result(client, monkeypatch):
    fake_async_result = MagicMock()
    fake_async_result.state = "SUCCESS"
    fake_async_result.result = {"prediction": "positive", "confidence": 0.91}

    monkeypatch.setattr(
        "app.main.AsyncResult",
        MagicMock(return_value=fake_async_result),
    )

    response = client.get("/status/some-task-id")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "SUCCESS"
    assert body["result"]["prediction"] == "positive"
    assert body["result"]["confidence"] == 0.91


def test_status_failure_state_does_not_crash_api(client, monkeypatch):
    fake_async_result = MagicMock()
    fake_async_result.state = "FAILURE"
    fake_async_result.result = ValueError("something went wrong")

    monkeypatch.setattr(
        "app.main.AsyncResult",
        MagicMock(return_value=fake_async_result),
    )

    response = client.get("/status/broken-task-id")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "FAILURE"
    assert body["result"] is None
    assert "something went wrong" in body["error"]


# ---------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
