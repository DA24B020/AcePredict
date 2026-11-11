"""SnapshotStore: in-memory view of artifacts/snapshot, hot-reloaded when the pipeline
writes a new snapshot (meta.json changes)."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from acepredict.exceptions import PlayerNotFoundError, SnapshotNotReadyError

log = logging.getLogger(__name__)
CHECK_EVERY_S = 5.0


def _clean(v: Any) -> Any:
    """NaN/NaT -> None so values serialise to JSON."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if v is pd.NaT:
        return None
    return v


class SnapshotStore:
    """Loads the Parquet snapshot and answers player, head-to-head and feature lookups."""

    def __init__(self, snapshot_dir: Path) -> None:
        self.dir = Path(snapshot_dir)
        self._lock = threading.RLock()
        self._mtime: float | None = None
        self._last_check = 0.0
        self.meta: dict[str, Any] = {}
        self.players: pd.DataFrame | None = None
        self.h2h: dict[tuple[str, str], int] = {}
        self.matches: pd.DataFrame | None = None
        self.elo_monthly: pd.DataFrame | None = None
        self.reference: pd.DataFrame | None = None

    # ---- loading -------------------------------------------------------------------------
    def reload(self, force: bool = False) -> bool:
        """(Re)load when meta.json changed. Keeps the old snapshot if the new one is broken."""
        meta_path = self.dir / "meta.json"
        with self._lock:
            self._last_check = time.monotonic()
            if not meta_path.exists():
                return self.players is not None
            mtime = meta_path.stat().st_mtime
            if not force and self._mtime == mtime:
                return True
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                players = pd.read_parquet(self.dir / "players.parquet").set_index(
                    "player_key", drop=False
                )
                h2h_df = pd.read_parquet(self.dir / "h2h.parquet")
                matches = pd.read_parquet(self.dir / "matches_lite.parquet")
                monthly = pd.read_parquet(self.dir / "elo_monthly.parquet")
                reference = pd.read_parquet(self.dir / "reference.parquet")
            except Exception as exc:
                log.error("snapshot reload failed, keeping previous one: %s", exc)
                return self.players is not None
            self.meta, self.players, self.matches = meta, players, matches
            self.h2h = {(p, q): int(w) for p, q, w in h2h_df[["p", "q", "wins"]].itertuples(
                index=False)}
            self.elo_monthly, self.reference = monthly, reference
            self._mtime = mtime
            log.info("snapshot loaded: %s players, data to %s", len(players),
                     meta.get("snapshot_date"))
            return True

    def _maybe_reload(self) -> None:
        if time.monotonic() - self._last_check >= CHECK_EVERY_S or self.players is None:
            self.reload()

    @property
    def ready(self) -> bool:
        self._maybe_reload()
        return self.players is not None

    def require(self) -> None:
        if not self.ready:
            raise SnapshotNotReadyError(
                "No serving snapshot yet. Run the pipeline from the Pipeline console first."
            )

    @property
    def surface_weight(self) -> float:
        return float(self.meta.get("surface_weight", 0.5))

    # ---- players -------------------------------------------------------------------------
    def summary(self, row: pd.Series) -> dict[str, Any]:
        rank = _clean(row.get("rank"))
        return {
            "id": row["player_key"],
            "name": str(row["name"]),
            "tour": row["tour"],
            "country": _clean(row.get("ioc")),
            "hand": _clean(row.get("hand")),
            "elo": round(float(row["elo"]), 1),
            "elo_hard": round(float(row["elo_hard"]), 1),
            "elo_clay": round(float(row["elo_clay"]), 1),
            "elo_grass": round(float(row["elo_grass"]), 1),
            "rank": int(rank) if rank is not None else None,
            "matches": int(row["matches_played"]),
            "active": bool(row["active"]),
            "last_match_date": _clean(row.get("last_match_date")),
        }

    def get(self, player_key: str) -> pd.Series:
        self.require()
        assert self.players is not None
        try:
            return self.players.loc[player_key]
        except KeyError as exc:
            raise PlayerNotFoundError(f"unknown player id '{player_key}'") from exc

    def search(self, q: str | None, tour: str | None, limit: int, active_only: bool) -> list:
        self.require()
        assert self.players is not None
        df = self.players
        if tour:
            df = df[df["tour"] == tour]
        if active_only:
            df = df[df["active"]]
        if q:
            df = df[df["name"].str.contains(q, case=False, regex=False, na=False)]
        df = df.sort_values("elo", ascending=False).head(limit)
        return [self.summary(r) for _, r in df.iterrows()]

    def h2h_record(self, p: str, q: str) -> tuple[int, int]:
        return self.h2h.get((p, q), 0), self.h2h.get((q, p), 0)

    def meetings(self, p: str, q: str, limit: int = 20) -> list[dict[str, Any]]:
        assert self.matches is not None
        m = self.matches
        mask = ((m["winner_key"] == p) & (m["loser_key"] == q)) | (
            (m["winner_key"] == q) & (m["loser_key"] == p))
        rows = m[mask].sort_values("match_date", ascending=False).head(limit)
        return [{"date": r.match_date, "tournament": r.tourney_name, "surface": r.surface,
                 "round": _clean(r.round), "winner_id": r.winner_key, "score": _clean(r.score)}
                for r in rows.itertuples(index=False)]

    def recent_matches(self, p: str, limit: int = 10) -> list[dict[str, Any]]:
        assert self.matches is not None
        m = self.matches
        rows = m[(m["winner_key"] == p) | (m["loser_key"] == p)].sort_values(
            "match_date", ascending=False).head(limit)
        out = []
        for r in rows.itertuples(index=False):
            won = r.winner_key == p
            out.append({"date": r.match_date, "tournament": r.tourney_name, "surface": r.surface,
                        "round": _clean(r.round), "won": bool(won),
                        "opponent": r.loser_name if won else r.winner_name,
                        "opponent_id": r.loser_key if won else r.winner_key,
                        "score": _clean(r.score)})
        return out

    def elo_history(self, p: str) -> list[dict[str, Any]]:
        assert self.elo_monthly is not None
        rows = self.elo_monthly[self.elo_monthly["player_key"] == p]
        return [{k: _clean(v) for k, v in r.items() if k != "player_key"}
                for r in rows.to_dict(orient="records")]

    # ---- features ------------------------------------------------------------------------
    def feature_attrs(self, row: pd.Series, surface: str, match_date: date, opp: str) -> dict:
        """Player attributes in the exact form the feature builder expects."""
        s = surface.lower()
        last = _clean(row.get("last_match_date"))
        days_since = None
        if last:
            days_since = (match_date - datetime.strptime(last, "%Y-%m-%d").date()).days
        return {
            "elo": float(row["elo"]),
            "selo": float(row[f"elo_{s}"]),
            "n": float(row["matches_played"]),
            "rank": _clean(row.get("rank")),
            "rank_points": _clean(row.get("rank_points")),
            "age": _clean(row.get("age")),
            "ht": _clean(row.get("ht")),
            "left": 1.0 if row.get("hand") == "L" else 0.0,
            "form10": _clean(row.get("form10")),
            "form52": _clean(row.get("form52")),
            "sform": _clean(row.get(f"sform_{s}")),
            "days_since": days_since,
            "matches14": _clean(row.get("matches14")),
            "ace_rate": _clean(row.get("ace_rate")),
            "first_won": _clean(row.get("first_won")),
            "bp_saved": _clean(row.get("bp_saved")),
            "h2h_wins": float(self.h2h.get((row["player_key"], opp), 0)),
        }
