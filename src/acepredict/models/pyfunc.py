"""The served model: an MLflow pyfunc wrapping the classifier with symmetry and explanations.

Input:  a DataFrame with the columns in ``FEATURES`` (player A vs player B).
Output: a DataFrame with
          prob_a       P(A wins), averaged with 1 - P(B wins | swapped) so that
                       P(A beats B) + P(B beats A) = 1 exactly
          explanation  JSON list of the 3 features that moved the prediction most
"""

from __future__ import annotations

import json
from typing import Any

import joblib
import mlflow.pyfunc
import numpy as np
import pandas as pd

from acepredict.features.build import FEATURE_LABELS, FEATURES, swap_players


def contributions(estimator: Any, model_type: str, X: pd.DataFrame) -> np.ndarray:
    """Per-feature contribution to the log-odds of A winning, one row per input row."""
    if model_type == "lightgbm":
        contrib = estimator.predict(X, pred_contrib=True)
        return np.asarray(contrib)[:, : len(FEATURES)]
    # logistic regression pipeline: impute -> scale -> clf ; contribution = z * coef
    z = estimator.named_steps["scale"].transform(estimator.named_steps["impute"].transform(X))
    return z * estimator.named_steps["clf"].coef_[0]


def top_factors(contrib_row: np.ndarray, k: int = 3) -> list[dict[str, Any]]:
    """The k largest absolute contributions, in plain words."""
    order = np.argsort(-np.abs(contrib_row))[:k]
    return [
        {
            "feature": FEATURES[i],
            "label": FEATURE_LABELS.get(FEATURES[i], FEATURES[i]),
            "contribution": round(float(contrib_row[i]), 4),
            "favours": "player1" if contrib_row[i] > 0 else "player2",
        }
        for i in order
        if abs(contrib_row[i]) > 1e-9
    ]


class AcePredictModel(mlflow.pyfunc.PythonModel):
    """Symmetric win-probability model with built-in explanations."""

    def __init__(self, estimator: Any = None, model_type: str | None = None) -> None:
        self.estimator = estimator
        self.model_type = model_type

    def load_context(self, context: Any) -> None:
        self.estimator = joblib.load(context.artifacts["estimator"])
        with open(context.artifacts["meta"], encoding="utf-8") as fh:
            self.model_type = json.load(fh)["model_type"]

    def _proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(X)[:, 1]

    def predict(self, context: Any, model_input: pd.DataFrame, params: dict | None = None):
        X = model_input[FEATURES].astype(float).fillna(0.0)
        p_ab = self._proba(X)
        p_ba = self._proba(swap_players(X))
        prob = (p_ab + (1.0 - p_ba)) / 2.0
        contrib = contributions(self.estimator, self.model_type, X)
        expl = [json.dumps(top_factors(row)) for row in contrib]
        return pd.DataFrame({"prob_a": prob, "explanation": expl})
