"""Liveness and readiness probes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.container import Services, get_services
from app.schemas import Health, Ready

router = APIRouter(tags=["health"])


@router.get("/health", response_model=Health, summary="Liveness probe")
def health() -> Health:
    return Health()


@router.get("/ready", response_model=Ready, summary="Readiness: snapshot loaded and model up",
            responses={503: {"description": "Not ready (same body)"}})
def ready(svc: Services = Depends(get_services)):
    from fastapi.responses import JSONResponse

    loaded = svc.store.ready
    up = svc.model.ping()
    version = svc.registry.model_info().get("champion_version")
    body = Ready(ready=bool(loaded and up), snapshot_loaded=loaded,
                 snapshot_date=svc.store.meta.get("snapshot_date") if loaded else None,
                 model_server=up, model_version=version)
    return JSONResponse(body.model_dump(), status_code=200 if body.ready else 503)
