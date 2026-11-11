"""Publish batch-job metrics to the Prometheus Pushgateway.

Pipeline stages are short-lived, so Prometheus cannot scrape them; they push their
metrics to the Pushgateway instead and Prometheus scrapes the gateway.
"""

from __future__ import annotations

import logging
import time

from prometheus_client import CollectorRegistry, Gauge, pushadd_to_gateway

from acepredict.config import pushgateway_url

log = logging.getLogger(__name__)


def push_stage_metrics(
    stage: str,
    duration_s: float,
    rows_in: int,
    rows_out: int,
    success: bool,
    extra: dict[str, float] | None = None,
) -> None:
    """Push one stage's duration, row counts, throughput and status (best effort)."""
    url = pushgateway_url()
    if not url:
        return
    reg = CollectorRegistry()
    labels = ["stage"]
    Gauge("acepredict_stage_duration_seconds", "Stage wall time", labels, registry=reg).labels(
        stage
    ).set(duration_s)
    Gauge("acepredict_stage_rows_in", "Rows read by the stage", labels, registry=reg).labels(
        stage
    ).set(rows_in)
    Gauge("acepredict_stage_rows_out", "Rows written by the stage", labels, registry=reg).labels(
        stage
    ).set(rows_out)
    Gauge(
        "acepredict_stage_rows_per_second", "Stage throughput (rows out / s)", labels, registry=reg
    ).labels(stage).set(rows_out / duration_s if duration_s > 0 else 0.0)
    Gauge("acepredict_stage_status", "1 = last run succeeded, 0 = failed", labels,
          registry=reg).labels(stage).set(1 if success else 0)
    Gauge("acepredict_stage_last_run_timestamp", "Unix time of last run", labels,
          registry=reg).labels(stage).set(time.time())
    if success:
        Gauge("acepredict_stage_last_success_timestamp", "Unix time of last success", labels,
              registry=reg).labels(stage).set(time.time())
    for name, value in (extra or {}).items():
        Gauge(f"acepredict_{name}", f"Stage metric {name}", labels, registry=reg).labels(stage).set(
            float(value)
        )
    try:
        pushadd_to_gateway(
            url, job="acepredict_pipeline", grouping_key={"stage": stage}, registry=reg, timeout=5
        )
    except Exception as exc:  # metrics must never break the pipeline
        log.warning("could not push stage metrics: %s", exc)


def push_pipeline_run(reason: str, success: bool, duration_s: float) -> None:
    """Push the outcome of a whole `dvc repro` run triggered by the pipeline runner."""
    url = pushgateway_url()
    if not url:
        return
    reg = CollectorRegistry()
    Gauge("acepredict_pipeline_last_run_success", "1 = last run succeeded", registry=reg).set(
        1 if success else 0
    )
    Gauge("acepredict_pipeline_last_run_duration_seconds", "Duration", registry=reg).set(duration_s)
    Gauge("acepredict_pipeline_last_run_timestamp", "Unix time", registry=reg).set(time.time())
    Gauge("acepredict_pipeline_last_run_info", "Reason of last run", ["reason"],
          registry=reg).labels(reason).set(1)
    try:
        pushadd_to_gateway(url, job="acepredict_pipeline_runner", registry=reg, timeout=5)
    except Exception as exc:
        log.warning("could not push pipeline-run metrics: %s", exc)
