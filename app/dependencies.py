"""
FastAPI dependency-injection helpers.

Keeping these separate from main.py makes it easy to override them in
tests (e.g. FastAPI's `app.dependency_overrides`).
"""
from app.config import Settings, get_settings
from ml import inference


def get_ml_settings() -> Settings:
    """Dependency wrapper around the cached Settings object."""
    return get_settings()


def ensure_model_loaded() -> None:
    """
    Dependency that guarantees the model is loaded before a request is
    served. Under normal operation the lifespan handler already loads the
    model at startup, so this is a cheap idempotent safety net.
    """
    if not inference.is_loaded():
        settings = get_settings()
        inference.load_models(settings.model_path, settings.vectorizer_path)
