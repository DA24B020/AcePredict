"""Feature engineering and the symmetric model wrapper (TC-FEAT-*)."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from acepredict.features.build import (
    ANTISYMMETRIC,
    FEATURES,
    PLAYER_ATTRS,
    SYMMETRIC,
    context_frame,
    pair_features,
    swap_players,
)
from acepredict.features.splits import resolve_splits, split_of
from acepredict.models.pyfunc import AcePredictModel

CTX = {"surface": "Clay", "tourney_level": "G", "best_of": 5, "round_order": 7, "tour": "atp",
       "h2h_matches": 3}


def player(elo: float, rank: float) -> dict:
    d = {k: 0.5 for k in PLAYER_ATTRS}
    d.update({"elo": elo, "selo": elo + 20, "rank": rank, "rank_points": 1000, "n": 200,
              "age": 25, "ht": 185, "left": 0, "days_since": 10, "h2h_wins": 1})
    return d


def test_feature_lists_cover_every_feature_once():
    assert sorted(ANTISYMMETRIC + SYMMETRIC) == sorted(FEATURES)
    assert len(set(FEATURES)) == len(FEATURES)


def test_pair_features_has_no_nan_and_right_columns():
    a, b = player(1800, 5), {"elo": 1600}  # b mostly missing
    feats = pair_features(context_frame(a, b, CTX), 0.5)
    assert list(feats.columns) == FEATURES
    assert not feats.isna().any().any()


def test_swapping_players_negates_antisymmetric_and_keeps_context():
    fa = pair_features(context_frame(player(1800, 5), player(1600, 40), CTX), 0.5)
    fb = pair_features(context_frame(player(1600, 40), player(1800, 5), CTX), 0.5)
    pd.testing.assert_frame_equal(swap_players(fa), fb, check_exact=False, atol=1e-9)
    assert fa["is_clay"].iloc[0] == 1 and fa["level_slam"].iloc[0] == 1


@pytest.fixture()
def fitted_model():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, len(FEATURES))), columns=FEATURES)
    y = (X["elo_diff"] + 0.3 * rng.normal(size=400) > 0).astype(int)
    est = Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler()),
                    ("clf", LogisticRegression())]).fit(X, y)
    return AcePredictModel(est, "logreg"), X


def test_pyfunc_probabilities_are_exactly_symmetric(fitted_model):
    model, X = fitted_model
    p_ab = model.predict(None, X.head(50))["prob_a"].to_numpy()
    p_ba = model.predict(None, swap_players(X.head(50)))["prob_a"].to_numpy()
    np.testing.assert_allclose(p_ab + p_ba, 1.0, atol=1e-9)


def test_pyfunc_explanations_are_top_factors(fitted_model):
    model, X = fitted_model
    out = model.predict(None, X.head(3))
    factors = json.loads(out["explanation"].iloc[0])
    assert 1 <= len(factors) <= 3
    assert {"feature", "label", "contribution", "favours"} <= set(factors[0])


def test_splits_use_config_when_years_exist():
    params = {"splits": {"train_start": 2000, "train_end": 2021, "val_years": [2022, 2023],
                         "test_years": [2024, 2025]}}
    s = resolve_splits(list(range(1995, 2026)), params)
    assert s["test"] == [2024, 2025] and s["val"] == [2022, 2023]
    assert max(s["train"]) == 2021 and min(s["train"]) == 2000
    assert split_of(1996, s) == "none"


def test_splits_fall_back_when_recent_seasons_missing():
    params = {"splits": {"train_start": 2000, "train_end": 2021, "val_years": [2022, 2023],
                         "test_years": [2024, 2025]}}
    s = resolve_splits(list(range(2000, 2024)), params)  # 2024/25 not published
    assert s["test"] == [2022, 2023] and s["val"] == [2020, 2021]
    assert max(s["train"]) < 2020
