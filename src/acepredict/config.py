"""Central configuration: repository paths and the pipeline parameters in params.yaml.

Every path and tunable value comes from here, the environment, or params.yaml;
nothing else in the code base hardcodes a location or a threshold.
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    """Return the repository root (``ACEPREDICT_ROOT`` or the current directory)."""
    return Path(os.environ.get("ACEPREDICT_ROOT", Path.cwd())).resolve()


class Paths:
    """All file-system locations used by the pipeline, relative to the repo root."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or repo_root()).resolve()
        self.raw = self.root / "data" / "raw"
        self.interim = self.root / "data" / "interim"
        self.features = self.root / "data" / "features"
        self.reports = self.root / "reports"
        self.artifacts = self.root / "artifacts"
        self.snapshot = self.artifacts / "snapshot"
        self.runtime = Path(os.environ.get("ACEPREDICT_RUNTIME", self.root / "runtime")).resolve()
        self.logs = self.runtime / "logs"

        # Canonical files
        self.matches = self.interim / "matches.parquet"
        self.live_matches = self.interim / "live_matches.parquet"
        self.player_match = self.interim / "player_match.parquet"
        self.elo_history = self.features / "elo_history.parquet"
        self.elo_state = self.features / "elo_state.parquet"
        self.train_table = self.features / "train_table.parquet"

    def ensure(self) -> Paths:
        """Create every directory the pipeline writes to."""
        for d in (
            self.raw,
            self.interim,
            self.features,
            self.reports,
            self.artifacts,
            self.runtime,
            self.logs,
            self.logs / "pipeline",
        ):
            d.mkdir(parents=True, exist_ok=True)
        return self


@lru_cache(maxsize=4)
def _load_params_cached(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_params(path: str | Path | None = None) -> dict[str, Any]:
    """Load params.yaml (``ACEPREDICT_PARAMS`` overrides the location)."""
    p = Path(path or os.environ.get("ACEPREDICT_PARAMS", repo_root() / "params.yaml")).resolve()
    return _load_params_cached(str(p))


def tracking_uri() -> str:
    """MLflow tracking URI from the environment (defaults to a local server)."""
    return os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def pushgateway_url() -> str | None:
    """Prometheus Pushgateway address, or None when metrics pushing is disabled."""
    url = os.environ.get("PUSHGATEWAY_URL", "").strip()
    return url or None


def git_commit(root: Path | None = None) -> str:
    """Current Git commit hash, or 'unknown' outside a Git repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root or repo_root(),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"
