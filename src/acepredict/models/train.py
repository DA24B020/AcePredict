"""Train the candidate models, track them in MLflow, register the best as `challenger`.

MLflow tracking goes beyond autolog: data/commit tags, split years, feature-list hash,
validation metrics per candidate, feature importance and the packaged pyfunc model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from acepredict.features.build import FEATURES
from acepredict.models.metrics import prob_metrics
from acepredict.models.pyfunc import AcePredictModel

log = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).resolve().parents[1]  # src/acepredict -> shipped with the model


def make_estimator(name: str, cfg: dict[str, Any], seed: int):
    """Build an unfitted candidate model."""
    if name == "logreg":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=float(cfg["C"]), max_iter=int(cfg["max_iter"]))),
            ]
        )
    if name == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=int(cfg["n_estimators"]),
            learning_rate=float(cfg["learning_rate"]),
            num_leaves=int(cfg["num_leaves"]),
            min_child_samples=int(cfg["min_child_samples"]),
            random_state=seed,
            deterministic=True,
            force_row_wise=True,
            n_jobs=2,
            verbose=-1,
        )
    raise ValueError(f"unknown candidate model: {name}")


def feature_importance(estimator: Any, name: str) -> pd.DataFrame:
    if name == "lightgbm":
        imp = estimator.booster_.feature_importance(importance_type="gain")
    else:
        imp = np.abs(estimator.named_steps["clf"].coef_[0])
    return (pd.DataFrame({"feature": FEATURES, "importance": imp})
            .sort_values("importance", ascending=False))


def train_and_register(
    table: pd.DataFrame,
    params: dict[str, Any],
    candidates: list[str],
    tags: dict[str, str],
    elo_params: dict[str, Any],
    splits: dict[str, list[int]],
) -> dict[str, Any]:
    """Fit every candidate (nested MLflow runs), keep the best on validation log-loss,
    log it as a pyfunc model and register it with alias `challenger`."""
    seed = int(params["features"]["seed"])
    train = table[table["split"] == "train"]
    val = table[table["split"] == "val"]
    X_tr, y_tr = train[FEATURES], train["y"].to_numpy()
    X_val, y_val = val[FEATURES], val["y"].to_numpy()
    feature_hash = hashlib.md5(",".join(FEATURES).encode()).hexdigest()[:12]
    reg_name = params["registry"]["name"]

    with mlflow.start_run(run_name="train") as parent:
        mlflow.set_tags({**tags, "stage": "train"})
        mlflow.log_params(
            {
                "candidates": ",".join(candidates),
                "n_features": len(FEATURES),
                "feature_list_md5": feature_hash,
                "train_years": f"{min(splits['train'])}-{max(splits['train'])}",
                "val_years": ",".join(map(str, splits["val"])),
                "test_years": ",".join(map(str, splits["test"])),
                "train_rows": len(train),
                "val_rows": len(val),
                **{f"elo_{k}": v for k, v in elo_params.items() if k != "source"},
            }
        )
        results: dict[str, dict[str, Any]] = {}
        for name in candidates:
            with mlflow.start_run(run_name=name, nested=True) as child:
                mlflow.autolog(log_models=False, silent=True)
                est = make_estimator(name, params["model"][name], seed)
                est.fit(X_tr, y_tr)
                mlflow.autolog(disable=True)
                m = prob_metrics(y_val, est.predict_proba(X_val)[:, 1])
                mlflow.log_metrics({f"val_{k}": v for k, v in m.items() if k != "n"})
                mlflow.set_tag("candidate", name)
                imp = feature_importance(est, name)
                mlflow.log_text(imp.to_csv(index=False), "feature_importance.csv")
                results[name] = {"estimator": est, "val": m, "run_id": child.info.run_id}
                log.info("candidate %s: val log-loss %.4f, accuracy %.4f", name,
                         m["log_loss"], m["accuracy"])

        best = min(results, key=lambda n: results[n]["val"]["log_loss"])
        chosen = results[best]
        mlflow.set_tag("best_candidate", best)
        mlflow.log_metrics({f"val_{k}": v for k, v in chosen["val"].items() if k != "n"})
        for name, r in results.items():
            mlflow.log_metric(f"val_log_loss_{name}", r["val"]["log_loss"])

        with tempfile.TemporaryDirectory() as tmp:
            est_path = Path(tmp) / "estimator.joblib"
            meta_path = Path(tmp) / "meta.json"
            joblib.dump(chosen["estimator"], est_path)
            meta_path.write_text(json.dumps({"model_type": best, "features": FEATURES,
                                             "elo_params": elo_params}))
            example = X_val.head(5)
            wrapper = AcePredictModel(chosen["estimator"], best)
            signature = infer_signature(example, wrapper.predict(None, example))
            info = mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=AcePredictModel(),
                artifacts={"estimator": str(est_path), "meta": str(meta_path)},
                code_paths=[str(PACKAGE_DIR)],
                signature=signature,
                input_example=example,
                registered_model_name=reg_name,
                pip_requirements=["mlflow==2.22.1", "scikit-learn==1.5.2", "lightgbm==4.5.0",
                                  "pandas==2.2.3", "numpy==1.26.4", "joblib"],
            )
        version = str(info.registered_model_version)
        client = MlflowClient()
        client.set_registered_model_alias(reg_name, "challenger", version)
        client.set_model_version_tag(reg_name, version, "git_commit", tags.get("git_commit", ""))
        client.set_model_version_tag(reg_name, version, "model_type", best)
        best_imp = feature_importance(chosen["estimator"], best)
        mlflow.log_text(best_imp.to_csv(index=False), "feature_importance.csv")
        return {
            "run_id": parent.info.run_id,
            "model_uri": f"runs:/{parent.info.run_id}/model",
            "registered_model": reg_name,
            "version": version,
            "best_candidate": best,
            "val": chosen["val"],
            "candidates": {n: r["val"] for n, r in results.items()},
            "candidate_run_ids": {n: r["run_id"] for n, r in results.items()},
        }
