"""Custom Prometheus metrics exposed on the backend's /metrics endpoint.

HTTP request counts and latency histograms per endpoint come from
prometheus-fastapi-instrumentator; these add the application- and model-level signals.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

PREDICTIONS = Counter(
    "acepredict_predictions_total", "Predictions served", ["tour", "surface"]
)
PREDICTED_PROB = Histogram(
    "acepredict_predicted_probability",
    "Distribution of P(player 1 wins)",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
INVALID_REQUESTS = Counter(
    "acepredict_invalid_requests_total", "Rejected requests by error code", ["code"]
)
MODEL_SERVER_LATENCY = Histogram(
    "acepredict_model_server_latency_seconds",
    "Latency of calls from the backend to the MLflow model server",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)
MODEL_SERVER_ERRORS = Counter(
    "acepredict_model_server_errors_total", "Failed calls to the model server", ["kind"]
)
MODEL_SERVER_UP = Gauge("acepredict_model_server_up", "1 if the model server answers /ping")
SNAPSHOT_READY = Gauge("acepredict_snapshot_ready", "1 if the serving snapshot is loaded")
SNAPSHOT_AGE_DAYS = Gauge(
    "acepredict_snapshot_data_age_days", "Days between today and the newest match in the snapshot"
)
MODEL_INFO = Gauge("acepredict_model_info", "Champion model version being served", ["version"])
LIVE_OUTCOMES = Counter("acepredict_live_outcomes_total", "Match results received for predictions")
LIVE_ACCURACY = Gauge("acepredict_live_accuracy", "Rolling accuracy on recent real outcomes")
LIVE_LOG_LOSS = Gauge("acepredict_live_log_loss", "Rolling log-loss on recent real outcomes")
LIVE_BRIER = Gauge("acepredict_live_brier", "Rolling Brier score on recent real outcomes")
LIVE_WINDOW_SIZE = Gauge("acepredict_live_window_size", "Outcomes in the rolling window")
FEATURE_PSI = Gauge(
    "acepredict_feature_psi", "Population Stability Index vs training data", ["feature"]
)
DRIFT_SAMPLE_SIZE = Gauge("acepredict_drift_sample_size", "Recent requests used for PSI")
ALERTS_RECEIVED = Counter(
    "acepredict_alerts_received_total", "Alertmanager notifications received", ["alertname"]
)
PIPELINE_TRIGGERS = Counter(
    "acepredict_pipeline_triggers_total", "Pipeline runs requested", ["reason"]
)
