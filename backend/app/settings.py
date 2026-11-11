"""Backend settings, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every URL, path, key and threshold the backend uses. Nothing is hardcoded elsewhere."""

    model_config = SettingsConfigDict(extra="ignore", protected_namespaces=())

    model_server_url: str = "http://localhost:5001"
    mlflow_tracking_uri: str = "http://localhost:5000"
    model_name: str = "acepredict"
    acepredict_root: Path = Path(os.environ.get("ACEPREDICT_ROOT", "."))
    snapshot_dir: Path | None = None
    runtime_dir: Path | None = None
    admin_api_key: str = "change-me-admin-key"
    webhook_token: str = "change-me-webhook-token"
    request_timeout_s: float = 10.0
    log_level: str = "INFO"
    monitor_interval_s: float = 30.0
    live_window: int = 200
    drift_min_samples: int = 50
    auto_retrain_cooldown_s: int = 1800
    max_batch: int = 500
    cors_origins: str = "*"

    @property
    def snapshot_path(self) -> Path:
        return (self.snapshot_dir or self.acepredict_root / "artifacts" / "snapshot").resolve()

    @property
    def runtime_path(self) -> Path:
        return (self.runtime_dir or self.acepredict_root / "runtime").resolve()

    @property
    def repo_path(self) -> Path:
        return self.acepredict_root.resolve()
