"""Wires the services together once per application (simple dependency container)."""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from fastapi import Header, Request

from acepredict.exceptions import AuthError
from app.clients.model_server import ModelServerClient
from app.services.monitor import MonitorService
from app.services.pipeline import PipelineService
from app.services.prediction import PredictionService
from app.services.registry import RegistryService
from app.services.snapshot import SnapshotStore
from app.settings import Settings


@dataclass
class Services:
    settings: Settings
    store: SnapshotStore
    model: ModelServerClient
    monitor: MonitorService
    registry: RegistryService
    pipeline: PipelineService
    prediction: PredictionService

    @classmethod
    def build(cls, settings: Settings, model: ModelServerClient | None = None,
              registry: RegistryService | None = None) -> Services:
        store = SnapshotStore(settings.snapshot_path)
        model = model or ModelServerClient(settings.model_server_url, settings.request_timeout_s)
        monitor = MonitorService(settings.runtime_path, settings.live_window,
                                 settings.drift_min_samples)
        registry = registry or RegistryService(settings.mlflow_tracking_uri, settings.model_name)
        pipeline = PipelineService(settings.repo_path, settings.runtime_path,
                                   settings.auto_retrain_cooldown_s)
        prediction = PredictionService(store, model, monitor, registry)
        return cls(settings, store, model, monitor, registry, pipeline, prediction)


def get_services(request: Request) -> Services:
    return request.app.state.services


def require_admin(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    """Admin endpoints need the X-API-Key header to match ADMIN_API_KEY."""
    expected = request.app.state.services.settings.admin_api_key
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise AuthError("Missing or wrong admin API key (X-API-Key header).")


def require_webhook_token(request: Request, authorization: str | None = Header(default=None)):
    """Alertmanager authenticates with 'Authorization: Bearer <WEBHOOK_TOKEN>'."""
    expected = request.app.state.services.settings.webhook_token
    token = (authorization or "").removeprefix("Bearer ").strip()
    if not token or not secrets.compare_digest(token, expected):
        raise AuthError("Missing or wrong webhook token.")
