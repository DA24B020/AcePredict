"""The frontend's only link to the system: REST calls to BACKEND_URL (configurable)."""

from __future__ import annotations

import os
from typing import Any

import requests

from metrics import BACKEND_ERRORS

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = float(os.environ.get("BACKEND_TIMEOUT_S", "20"))

FRIENDLY = {
    "SNAPSHOT_NOT_READY": "The app has no data yet. Ask an admin to run the pipeline "
                          "(Pipeline console page).",
    "MODEL_UNAVAILABLE": "The prediction model is starting up or not trained yet. "
                         "Please try again in a minute.",
    "MODEL_SERVER_ERROR": "The prediction model had a problem. Please try again.",
    "PLAYER_NOT_FOUND": "That player could not be found. Please pick again from the list.",
    "UNAUTHORIZED": "The admin key is missing or wrong.",
    "PIPELINE_BUSY": "A pipeline run is already in progress. Wait for it to finish.",
}


def _error(resp: requests.Response | None, exc: Exception | None = None) -> str:
    BACKEND_ERRORS.inc()
    if resp is None:
        return ("Cannot reach the AcePredict service. Check that the backend is running "
                f"({BACKEND_URL}) and press Retry.")
    try:
        body = resp.json()
        code = body.get("error_code", "")
        return FRIENDLY.get(code, body.get("message", f"Error {resp.status_code}"))
    except ValueError:
        return f"Unexpected error from the service (HTTP {resp.status_code})."


def call(method: str, path: str, *, params: dict | None = None, json: Any = None,
         headers: dict | None = None) -> tuple[Any | None, str | None]:
    """Returns (data, None) on success or (None, friendly error message)."""
    try:
        resp = requests.request(method, f"{BACKEND_URL}{path}", params=params, json=json,
                                headers=headers, timeout=TIMEOUT)
    except requests.RequestException as exc:
        return None, _error(None, exc)
    if resp.status_code >= 400:
        return None, _error(resp)
    return resp.json(), None


def get(path: str, **params: Any):
    return call("GET", path, params={k: v for k, v in params.items() if v is not None})


def post(path: str, body: Any = None, api_key: str | None = None):
    headers = {"X-API-Key": api_key} if api_key else None
    return call("POST", path, json=body, headers=headers)
