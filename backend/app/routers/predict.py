"""Predictions, batch predictions, and match outcomes for live monitoring."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from acepredict.exceptions import AcePredictError, InvalidMatchupError
from app.container import Services, get_services
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ErrorResponse,
    MonitoringSummary,
    OutcomeRequest,
    OutcomeResponse,
    PredictRequest,
    PredictResponse,
)
from app.services.monitor import UnknownRequestError

router = APIRouter(tags=["predictions"])
ERRORS = {code: {"model": ErrorResponse} for code in (400, 404, 502, 503)}


class UnknownRequest(AcePredictError):
    code = "UNKNOWN_REQUEST"
    http_status = 404


@router.post("/predict", response_model=PredictResponse, responses=ERRORS,
             summary="Win probability for player 1 vs player 2")
def predict(body: PredictRequest, request: Request, svc: Services = Depends(get_services)):
    return svc.prediction.predict(body, request.state.request_id)


@router.post("/predict/batch", response_model=BatchPredictResponse, responses=ERRORS,
             summary="Several predictions in one call (one model-server round trip)")
def predict_batch(body: BatchPredictRequest, request: Request,
                  svc: Services = Depends(get_services)):
    if len(body.items) > svc.settings.max_batch:
        raise InvalidMatchupError(f"At most {svc.settings.max_batch} items per batch.",
                                  code="BATCH_TOO_LARGE")
    rid = request.state.request_id
    return {"request_id": rid, "results": svc.prediction.predict_batch(body.items, rid)}


@router.post("/outcomes", response_model=OutcomeResponse, responses={404: {"model": ErrorResponse}},
             summary="Report the real result of a predicted match (live accuracy)")
def outcome(body: OutcomeRequest, svc: Services = Depends(get_services)):
    try:
        live = svc.monitor.record_outcome(body.request_id, body.p1_won)
    except UnknownRequestError as exc:
        raise UnknownRequest(f"No prediction with request_id '{body.request_id}'.") from exc
    return {"recorded": True, "live_accuracy": live["accuracy"],
            "live_log_loss": live["log_loss"], "window": live["window"]}


@router.get("/monitoring/summary", response_model=MonitoringSummary,
            summary="Live accuracy and drift (also exported to Prometheus)")
def monitoring_summary(svc: Services = Depends(get_services)):
    return svc.monitor.summary()
