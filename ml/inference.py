"""
Core machine learning logic: loading serialized artifacts and running
inference. This module is intentionally framework-agnostic — both the
FastAPI process and the Celery worker process import from here so the
exact same code path is used regardless of how a prediction was triggered.
"""
import logging
import pickle
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level globals holding the loaded artifacts.
# Each process (the FastAPI process, and each Celery worker process) has
# its own copy of these globals — that's expected and desired, since
# loading is cheap and avoids any cross-process shared state.
_model = None
_vectorizer = None


class ModelNotLoadedError(RuntimeError):
    """Raised when run_inference() is called before load_models()."""


def load_models(model_path: str, vectorizer_path: str) -> None:
    """
    Load the pickled classifier and vectorizer into memory.

    This should be called exactly once per process (FastAPI startup via
    lifespan, or Celery worker process init via a signal/first-use check).
    Re-loading is idempotent but wasteful, so callers should guard with
    `is_loaded()`.
    """
    global _model, _vectorizer

    model_file = Path(model_path)
    vectorizer_file = Path(vectorizer_path)

    if not model_file.exists():
        raise FileNotFoundError(
            f"Model artifact not found at '{model_path}'. "
            "Run `python -m ml.train` to generate it."
        )
    if not vectorizer_file.exists():
        raise FileNotFoundError(
            f"Vectorizer artifact not found at '{vectorizer_path}'. "
            "Run `python -m ml.train` to generate it."
        )

    with open(model_file, "rb") as f:
        _model = pickle.load(f)
    with open(vectorizer_file, "rb") as f:
        _vectorizer = pickle.load(f)

    logger.info("ML artifacts loaded successfully (model=%s, vectorizer=%s)",
                model_path, vectorizer_path)


def is_loaded() -> bool:
    """Returns True if both artifacts are currently loaded in this process."""
    return _model is not None and _vectorizer is not None


def get_loaded_model():
    return _model


def run_inference(text: str) -> dict:
    """
    Run the full inference pipeline on a single piece of text.

    Returns:
        {"prediction": <label str>, "confidence": <float 0..1>}
    """
    if not is_loaded():
        raise ModelNotLoadedError(
            "Model/vectorizer not loaded. Call load_models() first."
        )

    cleaned = text.strip().lower()
    if not cleaned:
        raise ValueError("Input text is empty after preprocessing.")

    features = _vectorizer.transform([cleaned])
    probabilities = _model.predict_proba(features)[0]
    classes = _model.classes_

    best_index = probabilities.argmax()
    prediction = str(classes[best_index])
    confidence = float(probabilities[best_index])

    return {"prediction": prediction, "confidence": round(confidence, 6)}
