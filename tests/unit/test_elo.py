"""Elo rating maths (TC-ELO-*)."""

import numpy as np
import pandas as pd
import pytest

from acepredict.ratings.elo import (
    EloParams,
    EloSystem,
    elo_pass,
    expected_score,
    k_factor,
    predict_from_ratings,
)

P = EloParams.from_dict({"initial": 1500, "k_c": 250, "k_offset": 5, "k_shape": 0.4,
                         "surface_weight": 0.5})


def test_expected_score_equal_ratings_is_half():
    assert expected_score(1500, 1500) == pytest.approx(0.5)


def test_expected_score_is_complementary():
    assert expected_score(1700, 1500) + expected_score(1500, 1700) == pytest.approx(1.0)


def test_400_points_is_ten_to_one():
    assert expected_score(1900, 1500) == pytest.approx(10 / 11)


def test_k_factor_decreases_with_experience():
    ks = k_factor(np.array([0, 10, 100, 1000]), P)
    assert np.all(np.diff(ks) < 0)


def test_winner_gains_and_loser_drops_same_amount_for_newcomers():
    res = elo_pass(np.array([0]), np.array([1]), np.array([0]), 2, P)
    gain = res["w_elo_post"][0] - 1500
    loss = 1500 - res["l_elo_post"][0]
    assert gain > 0 and gain == pytest.approx(loss)


def test_pre_match_rating_never_includes_the_match_itself():
    """Leakage guard: the rating used for match i is the one before match i."""
    res = elo_pass(np.array([0, 0]), np.array([1, 1]), np.array([0, 0]), 2, P)
    assert res["w_elo"][0] == 1500
    assert res["w_elo"][1] == pytest.approx(res["w_elo_post"][0])
    assert res["w_n"][1] == 1


def test_surface_rating_only_moves_on_that_surface():
    res = elo_pass(np.array([0]), np.array([1]), np.array([1]), 2, P)  # clay
    assert res["final_elo_clay"][0] > 1500
    assert res["final_elo_hard"][0] == 1500


def test_elo_system_state_and_history():
    matches = pd.DataFrame({"match_id": ["m1", "m2", "m3"],
                            "winner_key": ["a", "a", "b"], "loser_key": ["b", "c", "c"],
                            "surface": ["Hard", "Clay", "Grass"]})
    system = EloSystem(P).fit(matches)
    state = system.state().set_index("player_key")
    hist = system.history(matches)
    assert state.loc["a", "matches_played"] == 2
    assert state.loc["a", "elo"] > state.loc["c", "elo"]
    assert len(hist) == 3 and hist["p_winner"].between(0, 1).all()


def test_predict_from_ratings_symmetric():
    p = predict_from_ratings(1600, 1500, 1650, 1450, 0.5)
    q = predict_from_ratings(1500, 1600, 1450, 1650, 0.5)
    assert p + q == pytest.approx(1.0) and p > 0.5
