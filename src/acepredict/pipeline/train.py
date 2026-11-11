"""Stage `train`: fit candidates, track in MLflow, register the best as `challenger`."""

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd

from acepredict.config import Paths
from acepredict.features.splits import resolve_splits
from acepredict.models.train import train_and_register
from acepredict.pipeline.stage import StageResult, read_json, run_stage, write_json_atomic
from acepredict.tracking import init_mlflow, reproducibility_tags

_ARGS: argparse.Namespace | None = None


def train(paths: Paths, params: dict[str, Any]) -> StageResult:
    table = pd.read_parquet(paths.train_table)
    splits = resolve_splits(table["year"].unique().tolist(), params)
    candidates = list(params["model"]["candidates"])
    if _ARGS is not None and _ARGS.candidates:
        candidates = [c.strip() for c in _ARGS.candidates.split(",") if c.strip()]
    elo_params = read_json(paths.features / "elo_params.json", default={}) or {}
    init_mlflow(params["mlflow"]["experiment"])
    summary = train_and_register(
        table, params, candidates, reproducibility_tags(paths, params), elo_params, splits
    )
    write_json_atomic(paths.reports / "train.json", summary)
    n_train = int((table["split"] == "train").sum())
    return StageResult(rows_in=n_train, rows_out=n_train,
                       extra={"train_val_log_loss": summary["val"]["log_loss"]})


def main() -> None:
    global _ARGS
    parser = argparse.ArgumentParser(description="Train and register AcePredict models")
    parser.add_argument("--candidates", default="", help="comma-separated, e.g. logreg,lightgbm")
    _ARGS = parser.parse_args()
    run_stage("train", train)


if __name__ == "__main__":
    main()
