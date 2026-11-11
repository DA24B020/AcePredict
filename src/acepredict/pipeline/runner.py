"""Pipeline runner service: executes `dvc repro` when a trigger file appears.

Triggers are written to runtime/triggers/ by the backend (console "Run pipeline" button,
admin API, or an Alertmanager drift alert). One run at a time (lock file); every run is
recorded in runtime/pipeline_runs.json with its log in runtime/logs/run_<id>.log.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from acepredict.config import Paths
from acepredict.logging_conf import setup_logging
from acepredict.monitoring.pushgw import push_pipeline_run
from acepredict.pipeline.stage import read_json, write_json_atomic

log = logging.getLogger("acepredict.pipeline.runner")
MAX_RUNS_KEPT = 50


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def record_run(paths: Paths, run: dict) -> None:
    runs_file = paths.runtime / "pipeline_runs.json"
    runs = read_json(runs_file, default=[]) or []
    runs = [r for r in runs if r.get("run_id") != run["run_id"]] + [run]
    write_json_atomic(runs_file, runs[-MAX_RUNS_KEPT:])


def execute(paths: Paths, trigger: dict, command: list[str]) -> dict:
    """Run one pipeline execution and return its run record."""
    run_id = trigger.get("run_id") or datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    log_path = paths.logs / f"run_{run_id}.log"
    run = {"run_id": run_id, "reason": trigger.get("reason", "manual"), "status": "running",
           "started_at": _now(), "command": " ".join(command), "log": str(log_path.name)}
    record_run(paths, run)
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.run(command, cwd=paths.root, stdout=fh, stderr=subprocess.STDOUT,
                              check=False)
    duration = time.perf_counter() - t0
    run.update({"status": "succeeded" if proc.returncode == 0 else "failed",
                "finished_at": _now(), "duration_s": round(duration, 1),
                "return_code": proc.returncode})
    record_run(paths, run)
    push_pipeline_run(run["reason"], proc.returncode == 0, duration)
    log.info("pipeline run %s %s in %.1f s", run_id, run["status"], duration)
    return run


def main() -> None:
    paths = Paths().ensure()
    setup_logging(os.environ.get("LOG_LEVEL", "INFO"), paths.logs / "pipeline" / "runner.log")
    triggers = paths.runtime / "triggers"
    triggers.mkdir(parents=True, exist_ok=True)
    lock = paths.runtime / "pipeline.lock"
    lock.unlink(missing_ok=True)  # a lock left by a crashed container is stale
    command = os.environ.get("PIPELINE_COMMAND", "dvc repro").split()
    poll = float(os.environ.get("RUNNER_POLL_SECONDS", "5"))
    log.info("pipeline runner watching %s (command: %s)", triggers, " ".join(command))
    while True:
        pending = sorted(triggers.glob("*.json"))
        if pending:
            first: Path = pending[0]
            try:
                trigger = json.loads(first.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                trigger = {"reason": "unknown"}
            first.unlink(missing_ok=True)
            lock.write_text(json.dumps({"run_id": trigger.get("run_id"), "since": _now()}))
            try:
                execute(paths, trigger, command)
            finally:
                lock.unlink(missing_ok=True)
        time.sleep(poll)


if __name__ == "__main__":
    main()
