"""Evaluate the challenger through the exact pyfunc that will be served, against baselines."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from acepredict.features.build import FEATURES  # noqa: E402
from acepredict.models.metrics import prob_metrics, reliability_table  # noqa: E402

log = logging.getLogger(__name__)


def baseline_probs(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Elo-only probability and 'higher-ranked player wins'."""
    elo = 0.5 + df["elo_prob_c"].to_numpy()
    ra = df["a_rank"].fillna(2000).to_numpy()
    rb = df["b_rank"].fillna(2000).to_numpy()
    rank = np.where(ra < rb, 0.65, np.where(ra > rb, 0.35, 0.5))  # rank only gives a direction
    return {"elo_only": elo, "higher_rank": rank}


def segment_metrics(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    rows = []
    for col in ("surface", "tour", "tourney_level"):
        for value, idx in df.groupby(col).groups.items():
            mask = df.index.isin(idx)
            if mask.sum() >= 20:
                m = prob_metrics(df.loc[mask, "y"].to_numpy(), p[mask])
                rows.append({"segment": col, "value": value, **{k: round(v, 4) if
                             isinstance(v, float) else v for k, v in m.items()}})
    return pd.DataFrame(rows)


def evaluate_run(train_summary: dict[str, Any], table: pd.DataFrame, reports: Path) -> dict:
    model = mlflow.pyfunc.load_model(train_summary["model_uri"])
    out: dict[str, Any] = {"model_version": train_summary["version"],
                           "model_type": train_summary["best_candidate"]}
    test = table[table["split"] == "test"].reset_index(drop=True)
    val = table[table["split"] == "val"].reset_index(drop=True)
    pred_test = model.predict(test[FEATURES])["prob_a"].to_numpy()
    pred_val = model.predict(val[FEATURES])["prob_a"].to_numpy()
    out["test"] = prob_metrics(test["y"].to_numpy(), pred_test)
    out["val"] = prob_metrics(val["y"].to_numpy(), pred_val)
    out["baselines"] = {n: prob_metrics(test["y"].to_numpy(), b)
                        for n, b in baseline_probs(test).items()}
    out["log_loss_gain_vs_elo"] = out["baselines"]["elo_only"]["log_loss"] - out["test"]["log_loss"]
    out["beats_baselines"] = all(out["test"]["log_loss"] <= b["log_loss"]
                                 for b in out["baselines"].values())

    rel = reliability_table(test["y"].to_numpy(), pred_test)
    rel.to_csv(reports / "reliability.csv", index=False)
    seg = segment_metrics(test, pred_test)
    seg.to_csv(reports / "segments.csv", index=False)

    client = MlflowClient()
    run_id = train_summary["run_id"]
    for split in ("test", "val"):
        for k, v in out[split].items():
            if k != "n" and np.isfinite(v):
                client.log_metric(run_id, f"{split}_{k}", float(v))
    for name, m in out["baselines"].items():
        client.log_metric(run_id, f"baseline_{name}_test_log_loss", m["log_loss"])
        client.log_metric(run_id, f"baseline_{name}_test_accuracy", m["accuracy"])
    with tempfile.TemporaryDirectory() as tmp:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
        ax.plot(rel["predicted"], rel["observed"], "o-", label="AcePredict")
        ax.set_xlabel("Predicted P(player A wins)")
        ax.set_ylabel("Observed win rate")
        ax.set_title("Reliability on test seasons")
        ax.legend()
        fig.tight_layout()
        fig.savefig(Path(tmp) / "reliability.png", dpi=120)
        plt.close(fig)
        client.log_artifact(run_id, str(Path(tmp) / "reliability.png"))
        cm = pd.crosstab(test["y"], (pred_test > 0.5).astype(int),
                         rownames=["actual"], colnames=["predicted"])
        (Path(tmp) / "confusion_matrix.csv").write_text(cm.to_csv())
        client.log_artifact(run_id, str(Path(tmp) / "confusion_matrix.csv"))
        client.log_artifact(run_id, str(reports / "segments.csv"))
    reg = train_summary["registered_model"]
    client.set_model_version_tag(reg, train_summary["version"], "test_log_loss",
                                 f"{out['test']['log_loss']:.5f}")
    log.info("test log-loss %.4f (elo-only %.4f), accuracy %.4f", out["test"]["log_loss"],
             out["baselines"]["elo_only"]["log_loss"], out["test"]["accuracy"])
    return out
