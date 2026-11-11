"""Model info, pipeline status, admin actions and the Alertmanager webhook."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Body, Depends

from app.container import Services, get_services, require_admin, require_webhook_token
from app.metrics import ALERTS_RECEIVED, PIPELINE_TRIGGERS
from app.schemas import (
    AliasResponse,
    ErrorResponse,
    ModelInfo,
    PipelineStatus,
    PromoteRequest,
    RunAccepted,
    RunRequest,
    WebhookResponse,
)

log = logging.getLogger(__name__)
router = APIRouter()
AUTH = {401: {"model": ErrorResponse}}


@router.get("/model/info", response_model=ModelInfo, tags=["model"],
            summary="Champion model version, run, commit and test metrics")
def model_info(svc: Services = Depends(get_services)):
    return svc.registry.model_info(use_cache=False)


@router.get("/pipeline/status", response_model=PipelineStatus, tags=["pipeline"],
            summary="Stage status, runs, DAG, errors and throughput for the console")
def pipeline_status(svc: Services = Depends(get_services)):
    return {**svc.pipeline.status(), "mlflow_runs": svc.registry.recent_runs(10)}


@router.post("/admin/pipeline/run", response_model=RunAccepted, tags=["admin"],
             responses={**AUTH, 409: {"model": ErrorResponse}}, status_code=202,
             dependencies=[Depends(require_admin)], summary="Queue a pipeline run (dvc repro)")
def run_pipeline(body: RunRequest = Body(default_factory=RunRequest),
                 svc: Services = Depends(get_services)):
    PIPELINE_TRIGGERS.labels(body.reason).inc()
    return svc.pipeline.trigger(body.reason)


@router.post("/admin/model/promote", response_model=AliasResponse, tags=["admin"],
             responses={**AUTH, 502: {"model": ErrorResponse}},
             dependencies=[Depends(require_admin)], summary="Make a model version the champion")
def promote(body: PromoteRequest, svc: Services = Depends(get_services)):
    return svc.registry.promote(body.version)


@router.post("/admin/model/rollback", response_model=AliasResponse, tags=["admin"],
             responses={**AUTH, 502: {"model": ErrorResponse}},
             dependencies=[Depends(require_admin)], summary="Restore the previous champion")
def rollback(svc: Services = Depends(get_services)):
    return svc.registry.rollback()


@router.post("/admin/snapshot/reload", tags=["admin"], responses=AUTH,
             dependencies=[Depends(require_admin)], summary="Reload the serving snapshot now")
def reload_snapshot(svc: Services = Depends(get_services)):
    return {"reloaded": svc.store.reload(force=True), "meta": svc.store.meta}


@router.post("/webhooks/alertmanager", response_model=WebhookResponse, tags=["admin"],
             responses=AUTH, dependencies=[Depends(require_webhook_token)],
             summary="Alertmanager webhook: retrain on drift alerts")
def alertmanager_webhook(payload: dict = Body(...), svc: Services = Depends(get_services)):
    firing = [a for a in payload.get("alerts", []) if a.get("status") == "firing"]
    names = [a.get("labels", {}).get("alertname", "unknown") for a in firing]
    for n in names:
        ALERTS_RECEIVED.labels(n).inc()
    retrain = [a for a in firing if a.get("labels", {}).get("action") == "retrain"]
    if not retrain:
        return {"accepted": True, "triggered": False, "detail": "no retrain alert firing"}
    reason = f"alert:{retrain[0]['labels'].get('alertname', 'drift')}"
    triggered = svc.pipeline.auto_trigger(reason)
    if triggered:
        PIPELINE_TRIGGERS.labels(reason).inc()
    log.info("alert webhook %s -> triggered=%s", names, triggered)
    return {"accepted": True, "triggered": triggered,
            "detail": "pipeline queued" if triggered else "cooldown active or pipeline busy"}
