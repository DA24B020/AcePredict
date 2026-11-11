"""Elo ratings with a decaying K-factor and separate per-surface ratings.

    expected(A beats B) = 1 / (1 + 10 ** ((R_B - R_A) / 400))
    R_A <- R_A + K_A * (S_A - E_A),     K_A = c / (m_A + offset) ** shape

m_A is the number of matches A has played, so newcomers move fast and veterans slowly
(FiveThirtyEight-style). Each player also has one rating per surface; the prediction
uses the blend  w * overall + (1 - w) * surface.

Elo is inherently sequential (every match depends on all earlier ones), so this is a
single pass over chronologically ordered matches rather than a Spark job.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from acepredict.data.schema import SURFACE_INDEX, SURFACES


@dataclass(frozen=True)
class EloParams:
    """Tunable Elo constants (tuned by Ray Tune in the tune_elo stage)."""

    initial: float = 1500.0
    k_c: float = 250.0
    k_offset: float = 5.0
    k_shape: float = 0.4
    surface_weight: float = 0.5

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EloParams:
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: float(v) for k, v in d.items() if k in keys})

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def expected_score(r_a: float | np.ndarray, r_b: float | np.ndarray) -> float | np.ndarray:
    """Probability that a player rated r_a beats one rated r_b."""
    return 1.0 / (1.0 + np.power(10.0, (np.asarray(r_b) - np.asarray(r_a)) / 400.0))


def k_factor(matches_played: int | np.ndarray, p: EloParams) -> float | np.ndarray:
    """Decaying K: c / (m + offset) ** shape."""
    return p.k_c / np.power(np.asarray(matches_played, dtype=float) + p.k_offset, p.k_shape)


def blended(overall: float | np.ndarray, surface: float | np.ndarray, w: float):
    """Blend of overall and surface rating used for predictions."""
    return w * np.asarray(overall) + (1.0 - w) * np.asarray(surface)


def elo_pass(
    winners: np.ndarray,
    losers: np.ndarray,
    surfaces: np.ndarray,
    n_players: int,
    p: EloParams,
) -> dict[str, np.ndarray]:
    """Process matches in order. Returns pre- and post-match ratings plus final state.

    Arguments are integer arrays: player indices for winner/loser and surface index.
    """
    n = len(winners)
    init, c, off, sh = p.initial, p.k_c, p.k_offset, p.k_shape
    r = [init] * n_players
    m = [0] * n_players
    rs = [[init] * n_players for _ in SURFACES]
    ms = [[0] * n_players for _ in SURFACES]
    out = {k: np.empty(n) for k in ("w_elo", "l_elo", "w_selo", "l_selo",
                                    "w_elo_post", "l_elo_post", "w_selo_post", "l_selo_post")}
    out["w_n"] = np.empty(n, dtype=np.int64)
    out["l_n"] = np.empty(n, dtype=np.int64)
    w_list, l_list, s_list = winners.tolist(), losers.tolist(), surfaces.tolist()
    for i in range(n):
        a, b, s = w_list[i], l_list[i], s_list[i]
        ra, rb, sa, sb = r[a], r[b], rs[s][a], rs[s][b]
        out["w_elo"][i], out["l_elo"][i], out["w_selo"][i], out["l_selo"][i] = ra, rb, sa, sb
        out["w_n"][i], out["l_n"][i] = m[a], m[b]
        e = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        ka = c / (m[a] + off) ** sh
        kb = c / (m[b] + off) ** sh
        r[a] = ra + ka * (1.0 - e)
        r[b] = rb - kb * (1.0 - e)
        es = 1.0 / (1.0 + 10.0 ** ((sb - sa) / 400.0))
        ksa = c / (ms[s][a] + off) ** sh
        ksb = c / (ms[s][b] + off) ** sh
        rs[s][a] = sa + ksa * (1.0 - es)
        rs[s][b] = sb - ksb * (1.0 - es)
        m[a] += 1
        m[b] += 1
        ms[s][a] += 1
        ms[s][b] += 1
        out["w_elo_post"][i], out["l_elo_post"][i] = r[a], r[b]
        out["w_selo_post"][i], out["l_selo_post"][i] = rs[s][a], rs[s][b]
    out["final_elo"] = np.asarray(r)
    out["final_n"] = np.asarray(m)
    for j, name in enumerate(SURFACES):
        out[f"final_elo_{name.lower()}"] = np.asarray(rs[j])
        out[f"final_n_{name.lower()}"] = np.asarray(ms[j])
    return out


def winner_probability(res: dict[str, np.ndarray], w: float) -> np.ndarray:
    """Pre-match probability the eventual winner had, from the blended ratings."""
    return expected_score(blended(res["w_elo"], res["w_selo"], w),
                          blended(res["l_elo"], res["l_selo"], w))


def log_loss_of_winner_probs(p_win: np.ndarray) -> float:
    """Log-loss when every prediction is scored against 'the winner won'."""
    eps = 1e-12
    return float(-np.mean(np.log(np.clip(p_win, eps, 1 - eps))))


class EloSystem:
    """Object-oriented wrapper: encodes players, runs the pass, exposes history and state."""

    def __init__(self, params: EloParams) -> None:
        self.params = params
        self.players: list[str] = []
        self.index: dict[str, int] = {}
        self.result: dict[str, np.ndarray] | None = None

    def encode(self, matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        both = pd.concat([matches["winner_key"], matches["loser_key"]], ignore_index=True)
        keys = pd.unique(both)
        self.players = list(keys)
        self.index = {k: i for i, k in enumerate(self.players)}
        w = matches["winner_key"].map(self.index).to_numpy(dtype=np.int64)
        lo = matches["loser_key"].map(self.index).to_numpy(dtype=np.int64)
        s = matches["surface"].map(SURFACE_INDEX).fillna(0).to_numpy(dtype=np.int64)
        return w, lo, s

    def fit(self, matches: pd.DataFrame) -> EloSystem:
        """Run over matches (must already be sorted by order_idx)."""
        w, lo, s = self.encode(matches)
        self.result = elo_pass(w, lo, s, len(self.players), self.params)
        return self

    def history(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Per-match pre- and post-match ratings, aligned with ``matches``."""
        assert self.result is not None, "call fit() first"
        cols = ["w_elo", "l_elo", "w_selo", "l_selo", "w_n", "l_n",
                "w_elo_post", "l_elo_post", "w_selo_post", "l_selo_post"]
        hist = pd.DataFrame({c: self.result[c] for c in cols})
        hist.insert(0, "match_id", matches["match_id"].to_numpy())
        hist["p_winner"] = winner_probability(self.result, self.params.surface_weight)
        return hist

    def state(self) -> pd.DataFrame:
        """Final rating of every player after the last match."""
        assert self.result is not None, "call fit() first"
        df = pd.DataFrame({"player_key": self.players, "elo": self.result["final_elo"],
                           "matches_played": self.result["final_n"]})
        for name in SURFACES:
            low = name.lower()
            df[f"elo_{low}"] = self.result[f"final_elo_{low}"]
            df[f"matches_{low}"] = self.result[f"final_n_{low}"]
        return df


def predict_from_ratings(
    elo_a: float, elo_b: float, selo_a: float, selo_b: float, w: float
) -> float:
    """Elo-only win probability of A over B (used by the API's 'Elo only' comparison)."""
    ba = w * elo_a + (1 - w) * selo_a
    bb = w * elo_b + (1 - w) * selo_b
    return 1.0 / (1.0 + math.pow(10.0, (bb - ba) / 400.0))
