"""Stage `transform`: run the Spark job (see acepredict.spark_jobs.transform)."""

from __future__ import annotations

from typing import Any

from acepredict.config import Paths
from acepredict.pipeline.stage import StageResult, run_stage
from acepredict.spark_jobs.transform import run_transform


def transform(paths: Paths, params: dict[str, Any]) -> StageResult:
    counts = run_transform(
        paths.raw,
        {"matches": paths.matches, "live": paths.live_matches, "player_match": paths.player_match},
        params,
    )
    return StageResult(
        rows_in=counts["rows_in"],
        rows_out=counts["matches"] + counts["player_rows"],
        extra={"transform_matches": counts["matches"], "transform_live_matches": counts["live"]},
    )


def main() -> None:
    run_stage("transform", transform)


if __name__ == "__main__":
    main()
