"""Ray Tune search over the Elo constants (K-factor shape and the surface blend weight).

Every trial replays the whole match history with its own constants and is scored by
log-loss on the validation seasons. Trials are independent, which is exactly the kind of
work Ray parallelises across CPU cores.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from acepredict.ratings.elo import EloParams, EloSystem, elo_pass, winner_probability

log = logging.getLogger(__name__)


def prepare_arrays(matches: pd.DataFrame, val_years: list[int]) -> dict[str, Any]:
    """Encode matches once; the arrays are shared with every trial via Ray's object store."""
    system = EloSystem(EloParams())
    w, lo, s = system.encode(matches)
    return {
        "w": w,
        "l": lo,
        "s": s,
        "n_players": len(system.players),
        "val_mask": matches["year"].isin(val_years).to_numpy(),
    }


def score_params(arrays: dict[str, Any], p: EloParams) -> dict[str, float]:
    """Validation log-loss and accuracy of the Elo-only prediction for one parameter set."""
    res = elo_pass(arrays["w"], arrays["l"], arrays["s"], arrays["n_players"], p)
    probs = winner_probability(res, p.surface_weight)[arrays["val_mask"]]
    probs = np.clip(probs, 1e-12, 1 - 1e-12)
    return {
        "val_log_loss": float(-np.mean(np.log(probs))),
        "val_accuracy": float(np.mean(probs > 0.5)),
    }


def _trainable(config: dict[str, float], arrays: dict[str, Any] | None = None) -> None:
    from ray import train

    p = EloParams(
        initial=config["initial"],
        k_c=config["k_c"],
        k_offset=config["k_offset"],
        k_shape=config["k_shape"],
        surface_weight=config["surface_weight"],
    )
    train.report(score_params(arrays, p))


def run_tuning(
    matches: pd.DataFrame,
    params: dict[str, Any],
    val_years: list[int],
    storage_dir: Path,
    tracking_uri: str | None,
    experiment: str,
) -> dict[str, Any]:
    """Run the Ray Tune search and return the best constants plus throughput numbers."""
    import ray
    from ray import train, tune

    tp = params["tune"]
    arrays = prepare_arrays(matches, val_years)
    baseline = score_params(arrays, EloParams.from_dict(params["elo"]))

    ray.init(
        num_cpus=int(tp["num_cpus"]),
        include_dashboard=bool(tp.get("dashboard", False)),
        dashboard_host="0.0.0.0",
        object_store_memory=int(tp.get("object_store_mb", 400)) * 1024 * 1024,
        ignore_reinit_error=True,
        log_to_driver=False,
        configure_logging=False,
    )
    callbacks = []
    if tracking_uri:
        from ray.air.integrations.mlflow import MLflowLoggerCallback

        callbacks.append(
            MLflowLoggerCallback(
                tracking_uri=tracking_uri, experiment_name=experiment, save_artifact=False
            )
        )
    space = {
        "initial": float(params["elo"]["initial"]),
        "k_c": tune.uniform(*tp["c_range"]),
        "k_offset": tune.uniform(*tp["offset_range"]),
        "k_shape": tune.uniform(*tp["shape_range"]),
        "surface_weight": tune.uniform(*tp["weight_range"]),
    }
    t0 = time.perf_counter()
    try:
        tuner = tune.Tuner(
            tune.with_resources(tune.with_parameters(_trainable, arrays=arrays), {"cpu": 1}),
            param_space=space,
            tune_config=tune.TuneConfig(
                metric="val_log_loss", mode="min", num_samples=int(tp["num_samples"])
            ),
            run_config=train.RunConfig(
                name="elo_tuning",
                storage_path=str(storage_dir.resolve()),
                callbacks=callbacks,
                verbose=0,
            ),
        )
        results = tuner.fit()
        best = results.get_best_result()
        trials = results.get_dataframe()
    finally:
        ray.shutdown()
    seconds = time.perf_counter() - t0
    n = len(trials)
    best_params = {k: float(v) for k, v in best.config.items()}
    log.info("ray tune: %d trials in %.1f s, best val log-loss %.4f (default %.4f)",
             n, seconds, best.metrics["val_log_loss"], baseline["val_log_loss"])
    return {
        "best_params": best_params,
        "best_val_log_loss": float(best.metrics["val_log_loss"]),
        "best_val_accuracy": float(best.metrics["val_accuracy"]),
        "default_params_val_log_loss": baseline["val_log_loss"],
        "trials": n,
        "seconds": round(seconds, 2),
        "trials_per_minute": round(n / seconds * 60.0, 2) if seconds > 0 else 0.0,
        "val_years": val_years,
    }
