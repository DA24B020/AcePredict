"""Stage `validate`: data-quality gate on the raw CSVs.

Critical problems (missing columns, empty files, too many invalid rows) raise
DataValidationError and stop `dvc repro`. Row-level problems are counted in
reports/data_validation.json; the Spark transform drops exactly those rows.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from acepredict.config import Paths
from acepredict.data.schema import (
    AGE_RANGE,
    HEIGHT_RANGE,
    MATCH_COLUMNS,
    REQUIRED_COLUMNS,
    SURFACE_MAP,
    WALKOVER_PATTERN,
)
from acepredict.exceptions import DataValidationError
from acepredict.pipeline.stage import StageResult, run_stage, write_json_atomic

log = logging.getLogger(__name__)


def validate_frame(df: pd.DataFrame, name: str = "frame") -> dict[str, Any]:
    """Check one raw match table and return a report dict (no rows are modified)."""
    report: dict[str, Any] = {"file": name, "rows": int(len(df)), "critical": []}
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        report["critical"].append(f"missing required columns: {missing}")
        return report
    if len(df) == 0:
        report["critical"].append("file has no rows")
        return report
    # Informational only: the Spark transform reads columns by name, not position.
    report["column_order_differs"] = list(df.columns[: len(MATCH_COLUMNS)]) != MATCH_COLUMNS

    null_ids = df["winner_id"].isna() | df["loser_id"].isna()
    same_player = (~null_ids) & (df["winner_id"].astype(str) == df["loser_id"].astype(str))
    dates = pd.to_datetime(df["tourney_date"].astype(str).str[:8], format="%Y%m%d", errors="coerce")
    bad_date = dates.isna()
    walkover = df["score"].astype(str).str.contains(WALKOVER_PATTERN, regex=True, na=False)
    dup = df.duplicated(subset=["tourney_id", "match_num"], keep="first")
    unknown_surface = ~df["surface"].isin(list(SURFACE_MAP))
    ages = pd.to_numeric(pd.concat([df.get("winner_age"), df.get("loser_age")]), errors="coerce")
    heights = pd.to_numeric(pd.concat([df.get("winner_ht"), df.get("loser_ht")]), errors="coerce")

    dropped = null_ids | same_player | bad_date | walkover | dup
    report.update(
        {
            "null_player_ids": int(null_ids.sum()),
            "same_winner_and_loser": int(same_player.sum()),
            "unparseable_dates": int(bad_date.sum()),
            "walkovers": int(walkover.sum()),
            "duplicate_match_keys": int(dup.sum()),
            "unknown_surface (mapped to Hard)": int(unknown_surface.sum()),
            "age_out_of_range (set null)": int(
                ((ages < AGE_RANGE[0]) | (ages > AGE_RANGE[1])).sum()
            ),
            "height_out_of_range (set null)": int(
                ((heights < HEIGHT_RANGE[0]) | (heights > HEIGHT_RANGE[1])).sum()
            ),
            "rows_dropped": int(dropped.sum()),
            "invalid_fraction": round(float((dropped & ~walkover).mean()), 5),
        }
    )
    return report


def validate(paths: Paths, params: dict[str, Any]) -> StageResult:
    max_invalid = float(params["validation"]["max_invalid_fraction"])
    files = sorted(paths.raw.glob("*/*.csv"))
    if not files:
        raise DataValidationError("no raw CSV files found; run the ingest stage first")
    reports, total_rows, total_dropped = [], 0, 0
    critical: list[str] = []
    for f in files:
        df = pd.read_csv(f, dtype=str, low_memory=False)
        rep = validate_frame(df, f"{f.parent.name}/{f.name}")
        reports.append(rep)
        total_rows += rep["rows"]
        total_dropped += rep.get("rows_dropped", 0)
        critical += [f"{rep['file']}: {c}" for c in rep["critical"]]
        if rep.get("invalid_fraction", 0) > max_invalid:
            critical.append(
                f"{rep['file']}: invalid fraction {rep['invalid_fraction']:.3f} > {max_invalid}"
            )
    summary = {
        "files": len(files),
        "rows": total_rows,
        "rows_dropped": total_dropped,
        "rows_kept": total_rows - total_dropped,
        "critical_failures": critical,
        "passed": not critical,
        "per_file": reports,
    }
    write_json_atomic(paths.reports / "data_validation.json", summary)
    if critical:
        raise DataValidationError("; ".join(critical[:5]))
    log.info("validation passed: %d files, %d rows, %d dropped", len(files), total_rows,
             total_dropped)
    return StageResult(rows_in=total_rows, rows_out=total_rows - total_dropped)


def main() -> None:
    run_stage("validate", validate)


if __name__ == "__main__":
    main()
