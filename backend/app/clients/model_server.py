"""Client for the MLflow scoring server (`mlflow models serve`): /ping and /invocations."""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
import requests

from acepredict.exceptions import ModelServerError, ModelUnavailableError
from app.metrics import MODEL_SERVER_ERRORS, MODEL_SERVER_LATENCY, MODEL_SERVER_UP

log = logging.getLogger(__name__)


class ModelServerClient:
    """Talks to the model server over REST; the URL comes from MODEL_SERVER_URL."""

    def __init__(self, base_url: str, timeout_s: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self._ping_cache: tuple[float, bool] = (0.0, False)

    def ping(self, max_age_s: float = 5.0) -> bool:
        ts, ok = self._ping_cache
        if time.monotonic() - ts < max_age_s:
            return ok
        try:
            ok = self.session.get(f"{self.base_url}/ping", timeout=3).status_code == 200
        except requests.RequestException:
            ok = False
        self._ping_cache = (time.monotonic(), ok)
        MODEL_SERVER_UP.set(1 if ok else 0)
        return ok

    def predict(self, features: pd.DataFrame) -> list[dict[str, Any]]:
        """POST the feature rows; returns one {prob_a, explanation} dict per row."""
        payload = {"dataframe_split": {"columns": list(features.columns),
                                       "data": features.astype(float).values.tolist()}}
        t0 = time.perf_counter()
        try:
            resp = self.session.post(f"{self.base_url}/invocations", json=payload,
                                     timeout=self.timeout_s)
        except (requests.ConnectionError, requests.Timeout) as exc:
            MODEL_SERVER_ERRORS.labels("unreachable").inc()
            MODEL_SERVER_UP.set(0)
            raise ModelUnavailableError(
                "The prediction model is not available yet. If this is a fresh install, "
                "run the pipeline once from the Pipeline console."
            ) from exc
        finally:
            MODEL_SERVER_LATENCY.observe(time.perf_counter() - t0)
        if resp.status_code != 200:
            MODEL_SERVER_ERRORS.labels(f"http_{resp.status_code}").inc()
            log.error("model server error %s: %s", resp.status_code, resp.text[:300])
            raise ModelServerError(f"model server returned HTTP {resp.status_code}")
        preds = resp.json().get("predictions", [])
        if len(preds) != len(features):
            MODEL_SERVER_ERRORS.labels("bad_response").inc()
            raise ModelServerError("model server returned an unexpected number of predictions")
        return preds
