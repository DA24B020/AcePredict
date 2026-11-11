"""Data-drift statistics: Population Stability Index (PSI) per feature.

PSI = sum over bins of (cur% - ref%) * ln(cur% / ref%), with bins taken from reference
quantiles. Rule of thumb: < 0.1 stable, 0.1-0.2 moderate shift, > 0.2 significant drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Features watched for drift (continuous, informative, and comparable across requests).
DRIFT_FEATURES = [
    "elo_diff", "selo_diff", "elo_prob_c", "rank_log_ratio", "form10_diff", "age_diff",
    "exp_diff", "days_since_diff", "is_clay", "is_grass", "best_of5", "tour_wta",
]  # fmt: skip

EPS = 1e-4


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index of ``current`` against ``reference``."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    uniq = np.unique(ref)
    if len(uniq) <= bins:  # categorical / binary feature: one bin per value
        values = np.union1d(uniq, np.unique(cur))
        ref_p = np.array([(ref == v).mean() for v in values])
        cur_p = np.array([(cur == v).mean() for v in values])
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        ref_p = np.histogram(ref, edges)[0] / len(ref)
        cur_p = np.histogram(cur, edges)[0] / len(cur)
    ref_p = np.clip(ref_p, EPS, None)
    cur_p = np.clip(cur_p, EPS, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def psi_report(reference: pd.DataFrame, current: pd.DataFrame,
               features: list[str] | None = None, bins: int = 10) -> dict[str, float]:
    """PSI for each watched feature present in both frames."""
    feats = [f for f in (features or DRIFT_FEATURES)
             if f in reference.columns and f in current.columns]
    return {f: round(psi(reference[f].to_numpy(), current[f].to_numpy(), bins), 4) for f in feats}
