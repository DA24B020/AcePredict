"""Replay held-out matches against the running API, like a live season.

For every match (in date order) it asks POST /predict for a win probability, then reports
the real result with POST /outcomes. That feeds the live accuracy / log-loss gauges and the
drift (PSI) check that Prometheus and Grafana show.

    python scripts/replay_season.py                     # normal replay
    python scripts/replay_season.py --inject-drift      # grass + best-of-5 only -> PSI alert
    python scripts/replay_season.py --flip-outcomes 0.7 # report wrong winners -> log-loss alert

Player order (who is "player 1") is randomised so the winner is not always first.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
TOP_TIER = {"M", "PM", "P", "F", "W"}


def level_of(code: str) -> str:
    if code == "G":
        return "grand_slam"
    return "masters" if code in TOP_TIER else "tour"


def round_of(name: str) -> str:
    return name if name in ("QF", "SF", "F") else "early"


def load_matches(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.empty:
        fallback = ROOT / "data" / "interim" / "matches.parquet"
        print(f"{path.name} is empty (no matches after data.holdout_from were published yet); "
              f"replaying the most recent year of {fallback.name} instead.")
        df = pd.read_parquet(fallback)
        df = df[df["year"] == df["year"].max()]
    return df.sort_values(["match_date", "tour", "match_num"]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default=os.environ.get("ACEPREDICT_BASE_URL",
                                                         "http://localhost:8000"))
    ap.add_argument("--file", type=Path, default=ROOT / "data/interim/live_matches.parquet")
    ap.add_argument("--limit", type=int, default=400, help="max matches to replay")
    ap.add_argument("--rate", type=float, default=5.0, help="matches per second")
    ap.add_argument("--inject-drift", action="store_true",
                    help="send every match as grass, best-of-5 (input drift)")
    ap.add_argument("--flip-outcomes", type=float, default=0.0,
                    help="share of results reported the wrong way round (concept drift)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    df = load_matches(args.file).head(args.limit)
    session = requests.Session()
    url = args.base_url.rstrip("/")
    stats = {"sent": 0, "ok": 0, "skipped": 0, "errors": 0, "correct": 0}
    pause = 1.0 / args.rate if args.rate > 0 else 0.0

    for row in df.itertuples(index=False):
        swap = rng.random() < 0.5
        p1, p2 = (row.loser_key, row.winner_key) if swap else (row.winner_key, row.loser_key)
        p1_won = not swap
        body = {
            "tour": row.tour, "player1_id": p1, "player2_id": p2,
            "surface": "Grass" if args.inject_drift else row.surface,
            "best_of": 5 if args.inject_drift else int(row.best_of),
            "tourney_level": level_of(str(row.tourney_level)),
            "round": round_of(str(row.round)),
            "match_date": str(pd.Timestamp(row.match_date).date()),
        }
        stats["sent"] += 1
        try:
            r = session.post(f"{url}/predict", json=body, timeout=10)
        except requests.RequestException as exc:
            stats["errors"] += 1
            print(f"request failed: {exc}", file=sys.stderr)
            time.sleep(1)
            continue
        if r.status_code == 404:  # a player new this season is not in the snapshot yet
            stats["skipped"] += 1
            continue
        if r.status_code != 200:
            stats["errors"] += 1
            print(f"{r.status_code}: {r.text[:200]}", file=sys.stderr)
            continue
        pred = r.json()
        stats["ok"] += 1
        stats["correct"] += int((pred["p1_win_prob"] >= 0.5) == p1_won)
        reported = (not p1_won) if rng.random() < args.flip_outcomes else p1_won
        out = session.post(f"{url}/outcomes",
                           json={"request_id": pred["request_id"], "p1_won": reported},
                           timeout=10).json()
        if stats["ok"] % 25 == 0:
            print(f"{stats['ok']:4d} predicted | live accuracy {out.get('live_accuracy')} "
                  f"| live log-loss {out.get('live_log_loss')} | window {out.get('window')}")
        if pause:
            time.sleep(pause)

    acc = stats["correct"] / stats["ok"] if stats["ok"] else float("nan")
    print(f"done: {stats} | accuracy on true results {acc:.3f}")
    return 0 if stats["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
