"""MLflow helpers: connect to the tracking server and build the reproducibility tags."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import mlflow
import requests

from acepredict.config import Paths, git_commit, tracking_uri

log = logging.getLogger(__name__)


def init_mlflow(experiment: str) -> str:
    """Point MLflow at the tracking server and select ``experiment``. Fails fast if the
    server is unreachable, with a message that says what to start."""
    uri = tracking_uri()
    mlflow.set_tracking_uri(uri)
    if uri.startswith("http"):
        try:
            requests.get(uri.rstrip("/") + "/health", timeout=10).raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                f"MLflow tracking server not reachable at {uri} ({exc}). "
                "Start it with `docker compose up -d mlflow` or set MLFLOW_TRACKING_URI."
            ) from exc
    mlflow.set_experiment(experiment)
    return uri


def file_md5(path: Path) -> str:
    """MD5 of a file or of every file in a directory (stable order)."""
    h = hashlib.md5()
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    for f in files:
        if f.exists():
            h.update(f.read_bytes())
    return h.hexdigest()


def reproducibility_tags(paths: Paths, params: dict[str, Any]) -> dict[str, str]:
    """Tags that let any run be reproduced: Git commit, dvc.lock hash, data version."""
    lock = paths.root / "dvc.lock"
    return {
        "git_commit": git_commit(paths.root),
        "dvc_lock_md5": file_md5(lock) if lock.exists() else "none",
        "data_source": str(params["data"].get("source")),
        "data_ref": str(params["data"].get("sackmann_ref")),
    }
