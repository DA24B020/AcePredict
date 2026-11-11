"""Stage `features`: build the symmetric training table with time-based split labels."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from acepredict.config import Paths
from acepredict.features.build import build_training_table
from acepredict.features.splits import resolve_splits
from acepredict.pipeline.stage import StageResult, read_json, run_stage, write_json_atomic

log = logging.getLogger(__name__)


def features(paths: Paths, params: dict[str, Any]) -> StageResult:
    matches = pd.read_parquet(paths.matches)
    player_match = pd.read_parquet(paths.player_match)
    elo_hist = pd.read_parquet(paths.elo_history)
    elo_params = read_json(paths.features / "elo_params.json", default={}) or {}
    weight = float(elo_params.get("surface_weight", params["elo"]["surface_weight"]))
    splits = resolve_splits(matches["year"].unique().tolist(), params)
    table = build_training_table(
        matches, player_match, elo_hist, splits, weight, int(params["features"]["seed"])
    )
    table.to_parquet(paths.train_table, index=False)
    counts = table["split"].value_counts().to_dict()
    write_json_atomic(paths.reports / "splits.json", {"years": splits, "rows": counts,
                                                      "label_balance": float(table["y"].mean())})
    log.info("features: %d rows, splits=%s", len(table), counts)
    return StageResult(rows_in=len(matches), rows_out=len(table))


def main() -> None:
    run_stage("features", features)


if __name__ == "__main__":
    main()
