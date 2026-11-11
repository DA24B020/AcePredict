"""Stage `register`: promotion gate (challenger -> champion only if not worse)."""

from __future__ import annotations

from typing import Any

from mlflow.tracking import MlflowClient

from acepredict.config import Paths
from acepredict.models.registry import promotion_gate
from acepredict.pipeline.stage import StageResult, read_json, run_stage, write_json_atomic
from acepredict.tracking import init_mlflow


def register(paths: Paths, params: dict[str, Any]) -> StageResult:
    train = read_json(paths.reports / "train.json")
    metrics = read_json(paths.reports / "metrics.json")
    if not train or not metrics:
        raise FileNotFoundError("train.json / metrics.json missing; run train and evaluate")
    init_mlflow(params["mlflow"]["experiment"])
    decision = promotion_gate(
        MlflowClient(),
        params["registry"]["name"],
        train["version"],
        float(metrics["test"]["log_loss"]),
        float(params["registry"].get("gate_tolerance", 0.0)),
    )
    write_json_atomic(paths.reports / "promotion.json", decision)
    return StageResult(rows_in=1, rows_out=1,
                       extra={"promoted": 1.0 if decision["promoted"] else 0.0,
                              "champion_version": float(decision["champion_version_after"])})


def main() -> None:
    run_stage("register", register)


if __name__ == "__main__":
    main()
