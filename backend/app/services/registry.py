"""RegistryService: champion info, promotion/rollback and recent MLflow runs."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from acepredict.exceptions import RegistryError
from acepredict.models.registry import (
    CHALLENGER,
    CHAMPION,
    PREVIOUS,
    get_alias_version,
    promote,
    rollback,
)

log = logging.getLogger(__name__)
CACHE_S = 15.0


def _iso(ms: int | None) -> str | None:
    if not ms:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat(timespec="seconds")


class RegistryService:
    def __init__(self, tracking_uri: str, model_name: str) -> None:
        self.client = MlflowClient(tracking_uri=tracking_uri)
        self.name = model_name
        self._cache: tuple[float, dict[str, Any]] | None = None

    def model_info(self, use_cache: bool = True) -> dict[str, Any]:
        if use_cache and self._cache and time.monotonic() - self._cache[0] < CACHE_S:
            return self._cache[1]
        info: dict[str, Any] = {"model_name": self.name, "champion_version": None,
                                "test_metrics": {}}
        try:
            champ = get_alias_version(self.client, self.name, CHAMPION)
            prev = get_alias_version(self.client, self.name, PREVIOUS)
            chall = get_alias_version(self.client, self.name, CHALLENGER)
            info["previous_version"] = str(prev.version) if prev else None
            info["challenger_version"] = str(chall.version) if chall else None
            if champ is not None:
                run = self.client.get_run(champ.run_id)
                info.update({
                    "champion_version": str(champ.version),
                    "run_id": champ.run_id,
                    "model_type": (champ.tags or {}).get("model_type"),
                    "git_commit": run.data.tags.get("git_commit"),
                    "created_at": _iso(champ.creation_timestamp),
                    "test_metrics": {k: round(v, 4) for k, v in run.data.metrics.items()
                                     if k.startswith("test_")},
                })
        except Exception as exc:  # registry unreachable: degrade, do not fail the API
            log.warning("model registry not reachable: %s", exc)
        self._cache = (time.monotonic(), info)
        return info

    def promote(self, version: str) -> dict[str, Any]:
        try:
            res = promote(self.client, self.name, version, reason="console")
        except MlflowException as exc:
            raise RegistryError(f"could not promote version {version}: {exc.message}") from exc
        self._cache = None
        return res

    def rollback(self) -> dict[str, Any]:
        try:
            res = rollback(self.client, self.name)
        except MlflowException as exc:
            raise RegistryError(f"rollback failed: {exc.message}") from exc
        self._cache = None
        return res

    def recent_runs(self, limit: int = 10) -> list[dict[str, Any]]:
        try:
            exps = self.client.search_experiments()
            ids = {e.experiment_id: e.name for e in exps}
            runs = self.client.search_runs(list(ids), order_by=["attributes.start_time DESC"],
                                           max_results=limit)
        except Exception as exc:
            log.warning("could not list MLflow runs: %s", exc)
            return []
        out = []
        keep = ("val_log_loss", "test_log_loss", "test_accuracy", "replication_accuracy",
                "val_accuracy")
        for r in runs:
            start, end = r.info.start_time, r.info.end_time
            out.append({
                "run_id": r.info.run_id,
                "name": r.info.run_name,
                "experiment": ids.get(r.info.experiment_id, r.info.experiment_id),
                "status": r.info.status,
                "start_time": _iso(start),
                "duration_s": round((end - start) / 1000, 1) if start and end else None,
                "metrics": {k: round(v, 4) for k, v in r.data.metrics.items() if k in keep},
            })
        return out
