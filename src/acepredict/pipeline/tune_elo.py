"""Stage `tune_elo`: Ray Tune search over the Elo constants."""

from __future__ import annotations

from typing import Any

import pandas as pd

from acepredict.config import Paths, tracking_uri
from acepredict.features.splits import resolve_splits
from acepredict.pipeline.stage import StageResult, run_stage, write_json_atomic
from acepredict.tracking import init_mlflow
from acepredict.tuning.tune_elo import run_tuning


def tune_elo(paths: Paths, params: dict[str, Any]) -> StageResult:
    matches = pd.read_parquet(paths.matches).sort_values("order_idx").reset_index(drop=True)
    splits = resolve_splits(matches["year"].unique().tolist(), params)
    init_mlflow(params["mlflow"]["tuning_experiment"])  # fail fast if MLflow is down
    summary = run_tuning(
        matches,
        params,
        splits["val"],
        paths.runtime / "ray_results",
        tracking_uri(),
        params["mlflow"]["tuning_experiment"],
    )
    write_json_atomic(paths.reports / "elo_tuning.json", summary)
    return StageResult(
        rows_in=len(matches) * summary["trials"],
        rows_out=summary["trials"],
        extra={"tune_trials_per_minute": summary["trials_per_minute"],
               "tune_best_val_log_loss": summary["best_val_log_loss"]},
    )


def main() -> None:
    run_stage("tune_elo", tune_elo)


if __name__ == "__main__":
    main()
