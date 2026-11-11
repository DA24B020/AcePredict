"""Stage `evaluate`: test-set metrics, baselines, reliability curve, per-segment metrics."""

from __future__ import annotations

from typing import Any

import pandas as pd

from acepredict.config import Paths
from acepredict.models.evaluate import evaluate_run
from acepredict.pipeline.stage import StageResult, read_json, run_stage, write_json_atomic
from acepredict.tracking import init_mlflow


def evaluate(paths: Paths, params: dict[str, Any]) -> StageResult:
    summary = read_json(paths.reports / "train.json")
    if not summary:
        raise FileNotFoundError("reports/train.json missing; run the train stage first")
    table = pd.read_parquet(paths.train_table)
    init_mlflow(params["mlflow"]["experiment"])
    result = evaluate_run(summary, table, paths.reports)
    write_json_atomic(paths.reports / "metrics.json", result)
    n_test = int(result["test"]["n"])
    return StageResult(rows_in=n_test, rows_out=n_test,
                       extra={"test_log_loss": result["test"]["log_loss"],
                              "test_accuracy": result["test"]["accuracy"]})


def main() -> None:
    run_stage("evaluate", evaluate)


if __name__ == "__main__":
    main()
