"""Stage `elo`: compute pre/post-match Elo for every match and the replication check.

Replication check: the Elo-only prediction is scored on 2014 ATP tour-level matches,
the season used by a published comparison of tennis forecasting models, so the number
can be compared with the literature.
"""

from __future__ import annotations

import logging
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from acepredict.config import Paths
from acepredict.pipeline.stage import StageResult, read_json, run_stage, write_json_atomic
from acepredict.ratings.elo import EloParams, EloSystem
from acepredict.tracking import init_mlflow, reproducibility_tags

log = logging.getLogger(__name__)

REPLICATION_YEAR = 2014
REPLICATION_TOUR = "atp"


def chosen_params(paths: Paths, params: dict[str, Any]) -> tuple[EloParams, str]:
    """Tuned constants when available and enabled, otherwise params.yaml."""
    tuned = read_json(paths.reports / "elo_tuning.json", default=None)
    if params["elo"].get("use_tuned", True) and tuned and "best_params" in tuned:
        return EloParams.from_dict(tuned["best_params"]), "ray_tune"
    return EloParams.from_dict(params["elo"]), "params.yaml"


def replication_metrics(matches: pd.DataFrame, hist: pd.DataFrame) -> dict[str, Any]:
    """Accuracy / log-loss / Brier of Elo-only predictions on 2014 ATP matches."""
    mask = (matches["year"] == REPLICATION_YEAR) & (matches["tour"] == REPLICATION_TOUR)
    if not mask.any():
        return {"available": False, "reason": f"no {REPLICATION_TOUR} {REPLICATION_YEAR} data"}
    p = np.clip(hist.loc[mask.to_numpy(), "p_winner"].to_numpy(), 1e-12, 1 - 1e-12)
    return {
        "available": True,
        "season": f"{REPLICATION_TOUR.upper()} {REPLICATION_YEAR}",
        "matches": int(mask.sum()),
        "accuracy": round(float(np.mean(p > 0.5) + 0.5 * np.mean(p == 0.5)), 4),
        "log_loss": round(float(-np.mean(np.log(p))), 4),
        "brier": round(float(np.mean((1 - p) ** 2)), 4),
        "published_reference_accuracy": 0.70,
    }


def elo(paths: Paths, params: dict[str, Any]) -> StageResult:
    matches = pd.read_parquet(paths.matches).sort_values("order_idx").reset_index(drop=True)
    p, source = chosen_params(paths, params)
    system = EloSystem(p).fit(matches)
    hist = system.history(matches)
    state = system.state()
    hist.to_parquet(paths.elo_history, index=False)
    state.to_parquet(paths.elo_state, index=False)
    write_json_atomic(paths.features / "elo_params.json", {"source": source, **p.to_dict()})

    rep = replication_metrics(matches, hist)
    rep["params_source"] = source
    rep["params"] = p.to_dict()
    write_json_atomic(paths.reports / "elo_replication.json", rep)

    init_mlflow(params["mlflow"]["experiment"])
    with mlflow.start_run(run_name="elo-replication"):
        mlflow.set_tags({**reproducibility_tags(paths, params), "stage": "elo"})
        mlflow.log_params({f"elo_{k}": v for k, v in p.to_dict().items()})
        mlflow.log_param("elo_params_source", source)
        if rep.get("available"):
            mlflow.log_metrics({f"replication_{k}": rep[k]
                                for k in ("accuracy", "log_loss", "brier")})
        top = state.sort_values("elo", ascending=False).head(20)
        mlflow.log_text(top.to_csv(index=False), "elo_top20.csv")
    log.info("elo: %d matches, %d players, replication=%s", len(matches), len(state), rep)
    return StageResult(rows_in=len(matches), rows_out=len(hist))


def main() -> None:
    run_stage("elo", elo)


if __name__ == "__main__":
    main()
