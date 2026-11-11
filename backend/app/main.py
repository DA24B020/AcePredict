"""FastAPI application factory.

Run with:  uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import AsyncIterator
from datetime import date, datetime

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from acepredict import __version__
from acepredict.exceptions import AcePredictError
from acepredict.logging_conf import request_id_var, setup_logging
from app.container import Services
from app.metrics import INVALID_REQUESTS, MODEL_INFO, SNAPSHOT_AGE_DAYS, SNAPSHOT_READY
from app.routers import health, ops, players, predict
from app.settings import Settings

log = logging.getLogger("acepredict.api")


def _refresh_gauges(svc: Services) -> None:
    """Periodic work: drift, model-server health, snapshot age, champion version."""
    ready = svc.store.ready
    SNAPSHOT_READY.set(1 if ready else 0)
    if ready and svc.store.meta.get("snapshot_date"):
        d = datetime.strptime(svc.store.meta["snapshot_date"], "%Y-%m-%d").date()
        SNAPSHOT_AGE_DAYS.set((date.today() - d).days)
    svc.model.ping(max_age_s=0)
    version = svc.registry.model_info(use_cache=False).get("champion_version")
    MODEL_INFO.clear()
    if version:
        MODEL_INFO.labels(version).set(1)
    svc.monitor.refresh_drift(svc.store.reference if ready else None)
    svc.monitor.update_live_metrics()


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    settings = settings or Settings()
    setup_logging(settings.log_level)
    svc = services or Services.build(settings)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async def loop() -> None:
            while True:
                try:
                    await asyncio.to_thread(_refresh_gauges, svc)
                except Exception:
                    log.exception("background monitoring refresh failed")
                await asyncio.sleep(settings.monitor_interval_s)

        task = asyncio.create_task(loop())
        log.info("AcePredict API %s started", __version__)
        yield
        task.cancel()

    app = FastAPI(
        title="AcePredict API",
        version=__version__,
        description="Tennis match win probabilities. The frontend talks only to this API.",
        lifespan=lifespan,
    )
    app.state.services = svc
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins.split(","),
                       allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        return response

    def _rid(request: Request) -> str | None:
        return getattr(request.state, "request_id", None)

    @app.exception_handler(AcePredictError)
    async def handle_known(request: Request, exc: AcePredictError):
        INVALID_REQUESTS.labels(exc.code).inc()
        log.warning("%s: %s", exc.code, exc.message)
        return JSONResponse(status_code=exc.http_status,
                            content={"error_code": exc.code, "message": exc.message,
                                     "request_id": _rid(request)})

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request, exc: RequestValidationError):
        INVALID_REQUESTS.labels("VALIDATION_ERROR").inc()
        details = [{"field": ".".join(str(x) for x in e.get("loc", [])), "problem": e.get("msg")}
                   for e in exc.errors()]
        return JSONResponse(status_code=400,
                            content={"error_code": "VALIDATION_ERROR",
                                     "message": "Some inputs are invalid.",
                                     "request_id": _rid(request), "details": details})

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception):
        log.exception("unexpected error")
        return JSONResponse(status_code=500,
                            content={"error_code": "INTERNAL_ERROR",
                                     "message": "Something went wrong on our side.",
                                     "request_id": _rid(request)})

    for r in (health.router, players.router, predict.router, ops.router):
        app.include_router(r)
    Instrumentator(excluded_handlers=["/metrics", "/health"]).instrument(app).expose(
        app, include_in_schema=False)
    return app
