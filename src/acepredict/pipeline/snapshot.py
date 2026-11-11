"""Stage `snapshot`: everything the API needs at serving time, as small Parquet files.

artifacts/snapshot/
  players.parquet       latest attributes per player (Elo, surface Elo, form, serve stats...)
  h2h.parquet           wins of player p over player q, for every pair that has met
  matches_lite.parquet  match list for recent results and head-to-head meetings
  elo_monthly.parquet   month-end Elo history per player (profile chart)
  reference.parquet     sample of training features: the drift-detection reference
  meta.json             snapshot date, Elo constants, data version, champion version
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from acepredict.config import Paths
from acepredict.data.schema import SURFACES
from acepredict.features.build import FEATURES
from acepredict.pipeline.stage import StageResult, read_json, run_stage


def player_long(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per player per match with the attributes known after that match."""
    def side(me: str, pre: str, won: int) -> pd.DataFrame:
        d = pd.DataFrame({
            "player_key": matches[f"{me}_key"], "order_idx": matches["order_idx"],
            "match_date": matches["match_date"], "tour": matches["tour"],
            "surface": matches["surface"], "won": won, "name": matches[f"{me}_name"],
            "hand": matches[f"{me}_hand"], "ht": matches[f"{me}_ht"],
            "ioc": matches[f"{me}_ioc"], "age": matches[f"{me}_age"],
            "rank": matches[f"{me}_rank"], "rank_points": matches[f"{me}_rank_points"],
            "svpt": matches[f"{pre}_svpt"], "ace": matches[f"{pre}_ace"],
            "first_in": matches[f"{pre}_1stIn"], "first_won_n": matches[f"{pre}_1stWon"],
            "bp_saved_n": matches[f"{pre}_bpSaved"], "bp_faced": matches[f"{pre}_bpFaced"],
        })  # fmt: skip
        return d

    long = pd.concat([side("winner", "w", 1), side("loser", "l", 0)], ignore_index=True)
    long["match_date"] = pd.to_datetime(long["match_date"])
    long["ace_rate"] = long["ace"] / long["svpt"].where(long["svpt"] > 0)
    long["first_won"] = long["first_won_n"] / long["first_in"].where(long["first_in"] > 0)
    long["bp_saved"] = long["bp_saved_n"] / long["bp_faced"].where(long["bp_faced"] > 0)
    return long.sort_values(["player_key", "order_idx"]).reset_index(drop=True)


def build_players(long: pd.DataFrame, state: pd.DataFrame, snapshot_date: pd.Timestamp,
                  form_window: int, serve_window: int) -> pd.DataFrame:
    g = long.groupby("player_key", sort=False)
    last = g.tail(1).set_index("player_key")
    out = state.set_index("player_key").copy()
    for col in ("tour", "name", "hand", "ht", "ioc", "age", "rank", "rank_points"):
        out[col] = last[col]
    out["last_match_date"] = last["match_date"]
    out["age"] = out["age"] + (snapshot_date - out["last_match_date"]).dt.days / 365.25
    out["form10"] = long.groupby("player_key").tail(form_window).groupby("player_key")["won"].mean()
    recent = long[long["match_date"] > snapshot_date - pd.Timedelta(days=365)]
    out["form52"] = recent.groupby("player_key")["won"].mean()
    out["matches14"] = (
        long[long["match_date"] >= snapshot_date - pd.Timedelta(days=14)]
        .groupby("player_key").size()
    )
    out["matches14"] = out["matches14"].fillna(0)
    sform = (long.groupby(["player_key", "surface"]).tail(form_window)
             .groupby(["player_key", "surface"])["won"].mean().unstack())
    for s in SURFACES:
        out[f"sform_{s.lower()}"] = sform[s] if s in sform.columns else np.nan
    serve = (long.groupby("player_key").tail(serve_window).groupby("player_key")
             [["ace_rate", "first_won", "bp_saved"]].mean())
    out = out.join(serve)
    out["wins"] = g["won"].sum()
    out["losses"] = g["won"].count() - out["wins"]
    out["first_year"] = g["match_date"].min().dt.year
    out["active"] = out["last_match_date"] >= snapshot_date - pd.Timedelta(days=365)
    out["player_id"] = out.index.str.split("_", n=1).str[1]
    out["last_match_date"] = out["last_match_date"].dt.strftime("%Y-%m-%d")
    return out.reset_index()


def build_elo_monthly(matches: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    m = matches[["match_id", "order_idx", "match_date", "surface", "winner_key", "loser_key"]]
    m = m.merge(hist[["match_id", "w_elo_post", "l_elo_post", "w_selo_post", "l_selo_post"]],
                on="match_id")
    rows = pd.concat([
        pd.DataFrame({"player_key": m["winner_key"], "order_idx": m["order_idx"],
                      "match_date": m["match_date"], "surface": m["surface"],
                      "elo": m["w_elo_post"], "selo": m["w_selo_post"]}),
        pd.DataFrame({"player_key": m["loser_key"], "order_idx": m["order_idx"],
                      "match_date": m["match_date"], "surface": m["surface"],
                      "elo": m["l_elo_post"], "selo": m["l_selo_post"]}),
    ], ignore_index=True).sort_values(["player_key", "order_idx"])
    for s in SURFACES:
        rows[f"elo_{s.lower()}"] = rows["selo"].where(rows["surface"] == s)
    cols = [f"elo_{s.lower()}" for s in SURFACES]
    rows[cols] = rows.groupby("player_key")[cols].ffill()
    rows["month"] = pd.to_datetime(rows["match_date"]).dt.strftime("%Y-%m")
    monthly = rows.groupby(["player_key", "month"], sort=True)[["elo", *cols]].last()
    return monthly.round(1).reset_index()


def snapshot(paths: Paths, params: dict[str, Any]) -> StageResult:
    matches = pd.read_parquet(paths.matches).sort_values("order_idx").reset_index(drop=True)
    hist = pd.read_parquet(paths.elo_history)
    state = pd.read_parquet(paths.elo_state)
    table = pd.read_parquet(paths.train_table)
    elo_params = read_json(paths.features / "elo_params.json", default={}) or {}
    promotion = read_json(paths.reports / "promotion.json", default={}) or {}
    matches["match_date"] = pd.to_datetime(matches["match_date"])
    snapshot_date = matches["match_date"].max()

    long = player_long(matches)
    players = build_players(long, state, snapshot_date, int(params["features"]["form_window"]),
                            int(params["features"]["serve_window"]))
    h2h = (matches.groupby(["winner_key", "loser_key"]).size().rename("wins").reset_index()
           .rename(columns={"winner_key": "p", "loser_key": "q"}))
    lite = matches[["match_id", "tour", "match_date", "year", "tourney_name", "tourney_level",
                    "surface", "round", "winner_key", "loser_key", "winner_name", "loser_name",
                    "score"]].copy()
    lite["match_date"] = lite["match_date"].dt.strftime("%Y-%m-%d")
    monthly = build_elo_monthly(matches, hist)
    train = table[table["split"] == "train"]
    n_ref = min(int(params["features"]["reference_sample"]), len(train))
    reference = train.sample(n=n_ref, random_state=int(params["features"]["seed"]))[FEATURES]

    tmp = paths.artifacts / "snapshot.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    players.to_parquet(tmp / "players.parquet", index=False)
    h2h.to_parquet(tmp / "h2h.parquet", index=False)
    lite.to_parquet(tmp / "matches_lite.parquet", index=False)
    monthly.to_parquet(tmp / "elo_monthly.parquet", index=False)
    reference.to_parquet(tmp / "reference.parquet", index=False)
    meta = {
        "snapshot_date": snapshot_date.strftime("%Y-%m-%d"),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "players": int(len(players)),
        "active_players": int(players["active"].sum()),
        "matches": int(len(matches)),
        "elo_params": elo_params,
        "surface_weight": float(elo_params.get("surface_weight",
                                               params["elo"]["surface_weight"])),
        "holdout_from": params["data"].get("holdout_from"),
        "data_source": params["data"].get("source"),
        "data_ref": params["data"].get("sackmann_ref"),
        "champion_version": promotion.get("champion_version_after"),
        "features": [*FEATURES],
    }
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
    shutil.rmtree(paths.snapshot, ignore_errors=True)
    tmp.rename(paths.snapshot)
    return StageResult(rows_in=len(matches), rows_out=len(players) + len(lite) + len(monthly))


def main() -> None:
    run_stage("snapshot", snapshot)


if __name__ == "__main__":
    main()
