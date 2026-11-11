"""MonitorService: prediction log, live accuracy on real outcomes, and feature drift (PSI).

* Every prediction is kept in memory (bounded) and appended to runtime/predictions.jsonl.
* When a match result arrives (POST /outcomes), rolling accuracy / log-loss / Brier over
  the last ``live_window`` outcomes are updated: concept-drift monitoring on real labels.
* Periodically, PSI between recent request features and the training reference is
  computed: data-drift monitoring. Both are exported as Prometheus gauges.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from collections import OrderedDict, deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from acepredict.monitoring.drift import DRIFT_FEATURES, psi_report
from app.metrics import (
    DRIFT_SAMPLE_SIZE,
    FEATURE_PSI,
    LIVE_ACCURACY,
    LIVE_BRIER,
    LIVE_LOG_LOSS,
    LIVE_OUTCOMES,
    LIVE_WINDOW_SIZE,
)

log = logging.getLogger(__name__)
MAX_PENDING = 20_000


class UnknownRequestError(Exception):
    """An outcome was posted for a request_id this backend never served."""


class MonitorService:
    def __init__(self, runtime_dir: Path, live_window: int, drift_min_samples: int) -> None:
        self.runtime = Path(runtime_dir)
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.log_file = self.runtime / "predictions.jsonl"
        self._lock = threading.Lock()
        self.pending: OrderedDict[str, float] = OrderedDict()
        self.recent: deque[dict[str, float]] = deque(maxlen=2000)
        self.outcomes: deque[tuple[float, int]] = deque(maxlen=live_window)
        self.drift_min_samples = drift_min_samples
        self.last_psi: dict[str, float] = {}
        self.total_logged = 0

    def _append(self, record: dict[str, Any]) -> None:
        try:
            with open(self.log_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:  # logging must not break serving
            log.warning("could not write prediction log: %s", exc)

    def record_prediction(self, request_id: str, features: dict[str, float], prob: float,
                          info: dict[str, Any]) -> None:
        with self._lock:
            self.pending[request_id] = prob
            while len(self.pending) > MAX_PENDING:
                self.pending.popitem(last=False)
            self.recent.append({k: features[k] for k in DRIFT_FEATURES if k in features})
            self.total_logged += 1
        self._append({"ts": datetime.now(UTC).isoformat(), "type": "prediction",
                      "request_id": request_id, "prob": prob, **info})

    def record_outcome(self, request_id: str, p1_won: bool) -> dict[str, Any]:
        with self._lock:
            if request_id not in self.pending:
                raise UnknownRequestError(request_id)
            prob = self.pending.pop(request_id)
            self.outcomes.append((prob, 1 if p1_won else 0))
        LIVE_OUTCOMES.inc()
        self._append({"ts": datetime.now(UTC).isoformat(), "type": "outcome",
                      "request_id": request_id, "p1_won": p1_won})
        return self.update_live_metrics()

    def update_live_metrics(self) -> dict[str, Any]:
        with self._lock:
            data = list(self.outcomes)
        if not data:
            return {"accuracy": None, "log_loss": None, "brier": None, "window": 0}
        eps = 1e-12
        acc = sum((p > 0.5) == (y == 1) for p, y in data) / len(data)
        ll = -sum(y * math.log(max(p, eps)) + (1 - y) * math.log(max(1 - p, eps))
                  for p, y in data) / len(data)
        brier = sum((p - y) ** 2 for p, y in data) / len(data)
        LIVE_ACCURACY.set(acc)
        LIVE_LOG_LOSS.set(ll)
        LIVE_BRIER.set(brier)
        LIVE_WINDOW_SIZE.set(len(data))
        return {"accuracy": round(acc, 4), "log_loss": round(ll, 4), "brier": round(brier, 4),
                "window": len(data)}

    def refresh_drift(self, reference: pd.DataFrame | None) -> dict[str, float]:
        """Recompute PSI of recent requests vs the training reference (if enough samples)."""
        with self._lock:
            sample = list(self.recent)
        DRIFT_SAMPLE_SIZE.set(len(sample))
        if reference is None or len(sample) < self.drift_min_samples:
            return self.last_psi
        report = psi_report(reference, pd.DataFrame(sample))
        for feat, value in report.items():
            FEATURE_PSI.labels(feat).set(value)
        self.last_psi = report
        return report

    def summary(self) -> dict[str, Any]:
        live = self.update_live_metrics()
        return {
            "predictions_logged": self.total_logged,
            "outcomes_in_window": live["window"],
            "live_accuracy": live["accuracy"],
            "live_log_loss": live["log_loss"],
            "live_brier": live["brier"],
            "psi": self.last_psi,
            "drift_sample_size": len(self.recent),
        }
