"""Stage `ingest`: download the yearly match CSVs (or generate the synthetic sample).

Files land in data/raw/<tour>/<tour>_matches_<year>.csv. The commit (``data.sackmann_ref``)
is part of the URL, so pinning it in params.yaml pins the exact data version.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import requests

from acepredict.config import Paths
from acepredict.data.sample import write_sample
from acepredict.data.schema import raw_file_name
from acepredict.exceptions import DataSourceError
from acepredict.pipeline.stage import StageResult, run_stage, write_json_atomic

log = logging.getLogger(__name__)


def file_url(params: dict[str, Any], tour: str, year: int) -> str:
    """URL of one yearly CSV at the configured Git ref."""
    base = params["data"][f"base_url_{tour}"].rstrip("/")
    ref = params["data"].get("sackmann_ref", "master")
    return f"{base}/{ref}/{raw_file_name(tour, year)}"


def download(url: str, dest: Path, timeout: float, retries: int = 3) -> int | None:
    """Download ``url`` to ``dest``. Returns bytes written, or None on HTTP 404."""
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return len(resp.content)
        except requests.RequestException as exc:
            last_exc = exc
            log.warning("download attempt %d/%d failed for %s: %s", attempt, retries, url, exc)
            time.sleep(2 * attempt)
    raise DataSourceError(f"could not download {url}: {last_exc}")


def ingest(paths: Paths, params: dict[str, Any]) -> StageResult:
    data = params["data"]
    tours = list(data["tours"])
    start, end = int(data["start_year"]), int(data["end_year"])
    for tour in tours:
        for old in (paths.raw / tour).glob("*.csv") if (paths.raw / tour).exists() else []:
            old.unlink()

    files, total_bytes, missing = 0, 0, []
    if data.get("source", "github") == "sample":
        written = write_sample(paths.raw, tours, start, end)
        files = len(written)
        total_bytes = sum(p.stat().st_size for p in written)
    else:
        for tour in tours:
            (paths.raw / tour).mkdir(parents=True, exist_ok=True)
            tour_files = 0
            for year in range(start, end + 1):
                url = file_url(params, tour, year)
                n = download(url, paths.raw / tour / raw_file_name(tour, year),
                             timeout=float(data.get("download_timeout_s", 60)))
                if n is None:
                    missing.append(f"{tour}:{year}")
                    log.warning("not published yet (404): %s", url)
                    continue
                files += 1
                tour_files += 1
                total_bytes += n
            if tour_files == 0:
                raise DataSourceError(f"no files could be downloaded for tour '{tour}'")

    rows = 0
    for csv in paths.raw.glob("*/*.csv"):
        with open(csv, encoding="utf-8", errors="replace") as fh:
            rows += max(0, sum(1 for _ in fh) - 1)
    write_json_atomic(
        paths.reports / "ingest_summary.json",
        {"source": data.get("source"), "ref": data.get("sackmann_ref"), "files": files,
         "bytes": total_bytes, "rows": rows, "missing_years": missing},
    )
    log.info("ingested %d files, %.1f MB, %d rows", files, total_bytes / 1e6, rows)
    return StageResult(rows_in=0, rows_out=rows, extra={"ingest_files": files,
                                                        "ingest_bytes": total_bytes})


def main() -> None:
    run_stage("ingest", ingest)


if __name__ == "__main__":
    main()
