"""PipelineService: pipeline status for the console and run triggering.

The DAG is derived from dvc.yaml itself (deps -> outs), so the console always shows the
real pipeline. Stage status comes from runtime/stage_status.json written by every stage;
runs come from runtime/pipeline_runs.json written by the pipeline runner.
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from acepredict.exceptions import PipelineBusyError

STATUS_COLOURS = {"succeeded": "#C8E6C9", "failed": "#FFCDD2", "running": "#FFF59D"}


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _paths(entries: list[Any] | None) -> list[str]:
    out = []
    for e in entries or []:
        out.extend(e.keys() if isinstance(e, dict) else [e])
    return [str(p) for p in out]


class PipelineService:
    def __init__(self, repo_dir: Path, runtime_dir: Path, cooldown_s: int) -> None:
        self.repo = Path(repo_dir)
        self.runtime = Path(runtime_dir)
        self.triggers = self.runtime / "triggers"
        self.cooldown_s = cooldown_s

    # ---- DAG -----------------------------------------------------------------------------
    def stages(self) -> list[dict[str, Any]]:
        spec = yaml.safe_load((self.repo / "dvc.yaml").read_text(encoding="utf-8")) or {}
        out = []
        for name, st in (spec.get("stages") or {}).items():
            outs = _paths(st.get("outs")) + _paths(st.get("metrics")) + _paths(st.get("plots"))
            out.append({"name": name, "deps": _paths(st.get("deps")), "outs": outs})
        return out

    def edges(self) -> list[tuple[str, str]]:
        stages = self.stages()
        edges = []
        for producer in stages:
            for consumer in stages:
                if producer is consumer:
                    continue
                if any(d == o or d.startswith(o.rstrip("/") + "/")
                       for o in producer["outs"] for d in consumer["deps"]):
                    edges.append((producer["name"], consumer["name"]))
        return edges

    def dag_dot(self, status: dict[str, Any]) -> str:
        lines = ['digraph pipeline {', '  rankdir=LR;',
                 '  node [shape=box, style="rounded,filled", fontname="Helvetica", '
                 'fillcolor="#ECEFF1"];']
        for st in self.stages():
            s = status.get(st["name"], {}).get("status", "never run")
            colour = STATUS_COLOURS.get(s, "#ECEFF1")
            lines.append(f'  "{st["name"]}" [fillcolor="{colour}", '
                         f'label="{st["name"]}\\n{s}"];')
        for a, b in self.edges():
            lines.append(f'  "{a}" -> "{b}";')
        lines.append("}")
        return "\n".join(lines)

    # ---- status --------------------------------------------------------------------------
    def is_running(self) -> bool:
        return (self.runtime / "pipeline.lock").exists()

    def queued(self) -> int:
        return len(list(self.triggers.glob("*.json"))) if self.triggers.exists() else 0

    def status(self) -> dict[str, Any]:
        stage_status = _read_json(self.runtime / "stage_status.json", {}) or {}
        stages = []
        for st in self.stages():
            s = stage_status.get(st["name"], {})
            stages.append({"name": st["name"], "status": s.get("status", "never run"),
                           **{k: s.get(k) for k in ("started_at", "finished_at", "duration_s",
                                                    "rows_in", "rows_out", "rows_per_s",
                                                    "error")}})
        runs = list(reversed(_read_json(self.runtime / "pipeline_runs.json", []) or []))[:20]
        tail: list[str] = []
        failed = next((r for r in runs if r.get("status") == "failed"), None)
        if failed and failed.get("log"):
            log_path = self.runtime / "logs" / str(failed["log"])
            if log_path.exists():
                tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        reports = self.repo / "reports"
        validation = _read_json(reports / "data_validation.json", None)
        if validation:
            validation = {k: v for k, v in validation.items() if k != "per_file"}
        return {
            "running": self.is_running(),
            "queued": self.queued(),
            "stages": stages,
            "runs": [{k: r.get(k) for k in ("run_id", "reason", "status", "started_at",
                                            "finished_at", "duration_s")} for r in runs],
            "dag_dot": self.dag_dot(stage_status),
            "failed_log_tail": tail,
            "tuning": _read_json(reports / "elo_tuning.json", None),
            "validation": validation,
            "metrics": _read_json(reports / "metrics.json", None),
        }

    # ---- triggering ----------------------------------------------------------------------
    def trigger(self, reason: str) -> dict[str, str]:
        if self.is_running() or self.queued() > 0:
            raise PipelineBusyError("A pipeline run is already running or queued.")
        self.triggers.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        queued_at = datetime.now(UTC).isoformat(timespec="seconds")
        tmp = self.triggers / f".{run_id}.tmp"
        tmp.write_text(json.dumps({"run_id": run_id, "reason": reason, "queued_at": queued_at}))
        os.replace(tmp, self.triggers / f"{run_id}.json")
        return {"run_id": run_id, "queued_at": queued_at}

    def auto_trigger(self, reason: str) -> bool:
        """Trigger from an alert unless one was started within the cooldown window."""
        marker = self.runtime / "last_auto_trigger"
        if marker.exists() and time.time() - marker.stat().st_mtime < self.cooldown_s:
            return False
        try:
            self.trigger(reason)
        except PipelineBusyError:
            return False
        marker.touch()
        return True
