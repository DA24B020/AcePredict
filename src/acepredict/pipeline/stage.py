"""Common wrapper for every DVC stage: logging, timing, status file and metrics.

Each stage records its status in runtime/stage_status.json (read by the pipeline
console) and pushes duration / rows / throughput to the Prometheus Pushgateway.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from acepredict.config import Paths, load_params
from acepredict.logging_conf import setup_logging
from acepredict.monitoring.pushgw import push_stage_metrics


@dataclass
class StageResult:
    """What a stage reports back: row counts plus any extra numbers worth exporting."""

    rows_in: int = 0
    rows_out: int = 0
    extra: dict[str, float] = field(default_factory=dict)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON via a temp file + rename so readers never see a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path, default: Any = None) -> Any:
    """Read JSON, returning ``default`` if the file is missing or unreadable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def update_stage_status(paths: Paths, stage: str, **fields: Any) -> None:
    """Merge ``fields`` into this stage's entry in runtime/stage_status.json."""
    status_file = paths.runtime / "stage_status.json"
    data = read_json(status_file, default={}) or {}
    entry = data.get(stage, {})
    entry.update(fields)
    data[stage] = entry
    write_json_atomic(status_file, data)


def run_stage(name: str, fn: Callable[[Paths, dict[str, Any]], StageResult]) -> None:
    """Run one stage function with logging, timing, status tracking and metrics."""
    paths = Paths().ensure()
    params = load_params()
    setup_logging(os.environ.get("LOG_LEVEL", "INFO"), paths.logs / "pipeline" / f"{name}.log")
    log = logging.getLogger(f"acepredict.pipeline.{name}")
    log.info("stage started", extra={"stage": name, "event": "start"})
    update_stage_status(paths, name, status="running", started_at=_now(), error=None)
    t0 = time.perf_counter()
    try:
        result = fn(paths, params)
    except Exception as exc:
        duration = time.perf_counter() - t0
        log.exception("stage failed", extra={"stage": name, "event": "failed"})
        update_stage_status(
            paths,
            name,
            status="failed",
            finished_at=_now(),
            duration_s=round(duration, 2),
            error=f"{type(exc).__name__}: {exc}"[:800],
        )
        push_stage_metrics(name, duration, 0, 0, success=False)
        sys.exit(1)
    duration = time.perf_counter() - t0
    rps = result.rows_out / duration if duration > 0 else 0.0
    update_stage_status(
        paths,
        name,
        status="succeeded",
        finished_at=_now(),
        duration_s=round(duration, 2),
        rows_in=result.rows_in,
        rows_out=result.rows_out,
        rows_per_s=round(rps, 1),
        extra=result.extra,
    )
    push_stage_metrics(name, duration, result.rows_in, result.rows_out, True, result.extra)
    log.info(
        "stage finished: %d rows in, %d rows out, %.1f s, %.0f rows/s",
        result.rows_in,
        result.rows_out,
        duration,
        rps,
        extra={"stage": name, "event": "finish", "duration_s": round(duration, 2)},
    )
