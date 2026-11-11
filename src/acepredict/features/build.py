"""Model features for a match between player A and player B.

``pair_features`` is the ONLY function that turns player attributes into model inputs.
Training (on historical pre-match attributes) and the API (on the current snapshot) both
call it, so there is no train/serve skew.

Features are of two kinds:
* antisymmetric: A-minus-B differences. Swapping A and B negates them.
* symmetric: match context (surface, best-of, level...). Swapping leaves them unchanged.
The served model averages f(A,B) and 1 - f(B,A), so P(A wins) + P(B wins) = 1 exactly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from acepredict.data.schema import GRAND_SLAM_LEVELS, TOP_TIER_LEVELS
from acepredict.features.splits import split_of
from acepredict.ratings.elo import blended, expected_score

PLAYER_ATTRS = [
    "elo", "selo", "n", "rank", "rank_points", "age", "ht", "left", "form10", "form52",
    "sform", "days_since", "matches14", "ace_rate", "first_won", "bp_saved", "h2h_wins",
]  # fmt: skip
CONTEXT_ATTRS = ["surface", "best_of", "tourney_level", "round_order", "tour", "h2h_matches"]

ANTISYMMETRIC = [
    "elo_diff", "selo_diff", "elo_prob_c", "rank_log_ratio", "rank_points_diff", "exp_diff",
    "age_diff", "height_diff", "lefty_diff", "form10_diff", "form52_diff", "sform_diff",
    "days_since_diff", "matches14_diff", "ace_diff", "first_won_diff", "bp_saved_diff",
    "h2h_diff",
]  # fmt: skip
SYMMETRIC = ["is_clay", "is_grass", "best_of5", "level_slam", "level_top", "late_round",
             "tour_wta", "h2h_total"]  # fmt: skip
FEATURES = ANTISYMMETRIC + SYMMETRIC

# Plain-English names used in the API's "top reasons".
FEATURE_LABELS = {
    "elo_diff": "Overall Elo rating", "selo_diff": "Surface Elo rating",
    "elo_prob_c": "Blended Elo win chance", "rank_log_ratio": "ATP/WTA ranking",
    "rank_points_diff": "Ranking points", "exp_diff": "Career experience",
    "age_diff": "Age", "height_diff": "Height", "lefty_diff": "Left-handedness",
    "form10_diff": "Form (last 10 matches)", "form52_diff": "Form (last 12 months)",
    "sform_diff": "Form on this surface", "days_since_diff": "Days since last match",
    "matches14_diff": "Matches in last 14 days (fatigue)", "ace_diff": "Ace rate",
    "first_won_diff": "First-serve points won", "bp_saved_diff": "Break points saved",
    "h2h_diff": "Head-to-head record", "is_clay": "Clay court", "is_grass": "Grass court",
    "best_of5": "Best of five sets", "level_slam": "Grand Slam", "level_top": "Top-tier event",
    "late_round": "Late round", "tour_wta": "WTA tour", "h2h_total": "Previous meetings",
}  # fmt: skip

UNRANKED = 2000.0
MAX_DAYS = 365.0


def _col(df: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(default, index=df.index, dtype=float)


def pair_features(df: pd.DataFrame, surface_weight: float) -> pd.DataFrame:
    """Turn a_* / b_* player attributes plus match context into the model's feature frame."""
    a = {k: _col(df, f"a_{k}") for k in PLAYER_ATTRS}
    b = {k: _col(df, f"b_{k}") for k in PLAYER_ATTRS}
    surface = df["surface"].astype(str)
    level = df["tourney_level"].astype(str)
    blend_a = blended(a["elo"], a["selo"], surface_weight)
    blend_b = blended(b["elo"], b["selo"], surface_weight)
    out = pd.DataFrame(index=df.index)
    out["elo_diff"] = a["elo"] - b["elo"]
    out["selo_diff"] = a["selo"] - b["selo"]
    out["elo_prob_c"] = expected_score(blend_a, blend_b) - 0.5
    out["rank_log_ratio"] = np.log(b["rank"].fillna(UNRANKED).clip(lower=1)) - np.log(
        a["rank"].fillna(UNRANKED).clip(lower=1)
    )
    out["rank_points_diff"] = np.log1p(a["rank_points"].fillna(0).clip(lower=0)) - np.log1p(
        b["rank_points"].fillna(0).clip(lower=0)
    )
    out["exp_diff"] = np.log1p(a["n"].fillna(0)) - np.log1p(b["n"].fillna(0))
    out["age_diff"] = a["age"] - b["age"]
    out["height_diff"] = a["ht"] - b["ht"]
    out["lefty_diff"] = a["left"].fillna(0) - b["left"].fillna(0)
    out["form10_diff"] = a["form10"] - b["form10"]
    out["form52_diff"] = a["form52"] - b["form52"]
    out["sform_diff"] = a["sform"] - b["sform"]
    out["days_since_diff"] = a["days_since"].fillna(MAX_DAYS).clip(0, MAX_DAYS) - b[
        "days_since"
    ].fillna(MAX_DAYS).clip(0, MAX_DAYS)
    out["matches14_diff"] = a["matches14"].fillna(0) - b["matches14"].fillna(0)
    out["ace_diff"] = a["ace_rate"] - b["ace_rate"]
    out["first_won_diff"] = a["first_won"] - b["first_won"]
    out["bp_saved_diff"] = a["bp_saved"] - b["bp_saved"]
    out["h2h_diff"] = a["h2h_wins"].fillna(0) - b["h2h_wins"].fillna(0)
    out["is_clay"] = (surface == "Clay").astype(float)
    out["is_grass"] = (surface == "Grass").astype(float)
    out["best_of5"] = (_col(df, "best_of", 3).fillna(3) >= 5).astype(float)
    out["level_slam"] = level.isin(GRAND_SLAM_LEVELS).astype(float)
    out["level_top"] = level.isin(TOP_TIER_LEVELS).astype(float)
    out["late_round"] = (_col(df, "round_order", 5).fillna(5) >= 6).astype(float)
    out["tour_wta"] = (df["tour"].astype(str) == "wta").astype(float)
    out["h2h_total"] = _col(df, "h2h_matches", 0).fillna(0)
    # Missing on one side -> "no difference". No NaN ever reaches the model.
    return out[FEATURES].astype(float).fillna(0.0)


def swap_players(features: pd.DataFrame) -> pd.DataFrame:
    """Features of the same match with A and B exchanged."""
    swapped = features.copy()
    swapped[ANTISYMMETRIC] = -swapped[ANTISYMMETRIC]
    return swapped


def build_training_table(
    matches: pd.DataFrame,
    player_match: pd.DataFrame,
    elo_hist: pd.DataFrame,
    splits: dict[str, list[int]],
    surface_weight: float,
    seed: int,
) -> pd.DataFrame:
    """One row per historical match: features, label and split, with players randomly
    assigned to slots A and B (raw data lists the winner first, which would leak the label)."""
    m = matches.merge(elo_hist, on="match_id", how="inner", validate="one_to_one")
    pm = player_match.set_index(["match_id", "player_key"])
    pm_cols = ["form10", "form52", "sform", "days_since", "matches14", "ace_rate", "first_won",
               "bp_saved", "h2h_wins", "h2h_matches"]  # fmt: skip
    w_feats = pm.reindex(pd.MultiIndex.from_arrays([m["match_id"], m["winner_key"]]))[pm_cols]
    l_feats = pm.reindex(pd.MultiIndex.from_arrays([m["match_id"], m["loser_key"]]))[pm_cols]

    side = {
        "w": {
            "elo": m["w_elo"], "selo": m["w_selo"], "n": m["w_n"], "rank": m["winner_rank"],
            "rank_points": m["winner_rank_points"], "age": m["winner_age"], "ht": m["winner_ht"],
            "left": (m["winner_hand"] == "L").astype(float),
            **{c: w_feats[c].to_numpy() for c in pm_cols if c != "h2h_matches"},
        },
        "l": {
            "elo": m["l_elo"], "selo": m["l_selo"], "n": m["l_n"], "rank": m["loser_rank"],
            "rank_points": m["loser_rank_points"], "age": m["loser_age"], "ht": m["loser_ht"],
            "left": (m["loser_hand"] == "L").astype(float),
            **{c: l_feats[c].to_numpy() for c in pm_cols if c != "h2h_matches"},
        },
    }  # fmt: skip
    rng = np.random.default_rng(seed)
    flip = rng.random(len(m)) < 0.5  # True -> A is the loser
    frame = pd.DataFrame(
        {
            "surface": m["surface"].to_numpy(),
            "best_of": m["best_of"].to_numpy(),
            "tourney_level": m["tourney_level"].to_numpy(),
            "round_order": m["round_order"].to_numpy(),
            "tour": m["tour"].to_numpy(),
            "h2h_matches": w_feats["h2h_matches"].to_numpy(),
        }
    )
    for attr in PLAYER_ATTRS:
        wv = np.asarray(side["w"][attr], dtype=float)
        lv = np.asarray(side["l"][attr], dtype=float)
        frame[f"a_{attr}"] = np.where(flip, lv, wv)
        frame[f"b_{attr}"] = np.where(flip, wv, lv)
    feats = pair_features(frame, surface_weight)
    feats["y"] = (~flip).astype(int)
    feats["year"] = m["year"].to_numpy()
    feats["match_id"] = m["match_id"].to_numpy()
    feats["order_idx"] = m["order_idx"].to_numpy()
    feats["tour"] = m["tour"].to_numpy()
    feats["surface"] = m["surface"].to_numpy()
    feats["tourney_level"] = m["tourney_level"].to_numpy()
    feats["a_rank"] = frame["a_rank"].to_numpy()
    feats["b_rank"] = frame["b_rank"].to_numpy()
    feats["split"] = [split_of(int(y), splits) for y in feats["year"]]
    return feats.sort_values("order_idx").reset_index(drop=True)


def context_frame(
    a: dict[str, Any], b: dict[str, Any], context: dict[str, Any]
) -> pd.DataFrame:
    """One-row frame of a_* / b_* attributes + context, ready for ``pair_features``."""
    row: dict[str, Any] = dict(context)
    for attr in PLAYER_ATTRS:
        row[f"a_{attr}"] = a.get(attr)
        row[f"b_{attr}"] = b.get(attr)
    return pd.DataFrame([row])
