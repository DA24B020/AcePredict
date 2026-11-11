"""Data validation, data schema helpers, metrics and drift (TC-DATA-*, TC-MON-*)."""

import numpy as np
import pandas as pd
import pytest

from acepredict.data.sample import generate_tour
from acepredict.data.schema import player_key, raw_file_name, round_order
from acepredict.models.metrics import expected_calibration_error, prob_metrics
from acepredict.monitoring.drift import psi, psi_report
from acepredict.pipeline.validate import validate_frame


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    frames = generate_tour("atp", 2015, 2016, seed=1)
    return pd.concat(frames.values() if isinstance(frames, dict) else frames, ignore_index=True)


def test_sample_data_passes_validation(raw):
    rep = validate_frame(raw, "sample")
    assert rep["critical"] == []
    assert rep["invalid_fraction"] < 0.05


def test_validation_flags_missing_columns(raw):
    rep = validate_frame(raw.drop(columns=["winner_id"]), "broken")
    assert rep["critical"] and "winner_id" in rep["critical"][0]


def test_validation_counts_bad_rows(raw):
    bad = raw.copy()
    bad.loc[bad.index[:5], "loser_id"] = bad.loc[bad.index[:5], "winner_id"]
    bad.loc[bad.index[5:8], "tourney_date"] = "notadate"
    rep = validate_frame(bad, "bad")
    assert rep["same_winner_and_loser"] >= 5
    assert rep["unparseable_dates"] >= 3


def test_schema_helpers():
    assert player_key("atp", 104925) == "atp_104925"
    assert player_key("wta", "200001.0") == "wta_200001"
    assert raw_file_name("wta", 2014) == "wta_matches_2014.csv"
    assert round_order("F") > round_order("SF") > round_order("R32")


def test_prob_metrics_perfect_and_coinflip():
    y = np.array([1, 0, 1, 0])
    good = prob_metrics(y, np.array([0.99, 0.01, 0.99, 0.01]))
    coin = prob_metrics(y, np.full(4, 0.5))
    assert good["accuracy"] == 1.0 and good["log_loss"] < 0.05
    assert coin["log_loss"] == pytest.approx(np.log(2), rel=1e-3)


def test_ece_of_calibrated_predictions_is_small():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=20000)
    y = (rng.uniform(size=20000) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.02


def test_psi_same_distribution_is_near_zero_and_shift_is_large():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=5000)
    assert psi(ref, rng.normal(size=5000)) < 0.02
    assert psi(ref, rng.normal(loc=1.5, size=5000)) > 0.2


def test_psi_report_only_uses_shared_columns():
    ref = pd.DataFrame({"elo_diff": np.arange(100.0), "is_grass": np.zeros(100)})
    cur = pd.DataFrame({"elo_diff": np.arange(100.0)})
    rep = psi_report(ref, cur)
    assert set(rep) == {"elo_diff"}
