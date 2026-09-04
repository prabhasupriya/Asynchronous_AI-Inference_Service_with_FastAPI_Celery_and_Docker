"""
Centralized application configuration.

All runtime configuration is read from environment variables (optionally
loaded from a local .env file). Nothing is hardcoded here — this module
only defines defaults that are safe for local development.
"""
import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- App metadata ---
    app_name: str = "AI Inference API"
    log_level: str = "INFO"

    # --- Redis / Celery ---
    redis_url: str = "redis://localhost:6379/0"

    # --- ML artifact paths ---
    model_path: str = os.path.join("ml", "artifacts", "model.pkl")
    vectorizer_path: str = os.path.join("ml", "artifacts", "vectorizer.pkl")

    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", protected_namespaces=("settings_",)
    )


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor so we parse env vars only once."""
    return Settings()
