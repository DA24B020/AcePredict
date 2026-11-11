"""Generate the Grafana dashboard JSON files (run once; output is committed)."""
import json
from pathlib import Path

DS = {"type": "prometheus", "uid": "prometheus"}
OUT = Path(__file__).resolve().parents[1] / "monitoring" / "grafana" / "dashboards"


def panel(pid, title, exprs, x, y, w=12, h=8, kind="timeseries", unit=None, thresholds=None):
    p = {"id": pid, "type": kind, "title": title, "datasource": DS,
         "gridPos": {"x": x, "y": y, "w": w, "h": h},
         "targets": [{"refId": chr(65 + i), "expr": e, "legendFormat": lf, "datasource": DS}
                     for i, (e, lf) in enumerate(exprs)],
         "fieldConfig": {"defaults": {}, "overrides": []}, "options": {}}
    if unit:
        p["fieldConfig"]["defaults"]["unit"] = unit
    if thresholds:
        p["fieldConfig"]["defaults"]["thresholds"] = {"mode": "absolute", "steps": [
            {"color": "green", "value": None}] + [{"color": c, "value": v} for v, c in thresholds]}
        if kind == "timeseries":
            p["fieldConfig"]["defaults"]["custom"] = {"thresholdsStyle": {"mode": "line"}}
    return p


def dash(uid, title, panels):
    return {"uid": uid, "title": title, "schemaVersion": 39, "version": 1, "editable": True,
            "refresh": "5s", "time": {"from": "now-30m", "to": "now"}, "tags": ["acepredict"],
            "panels": panels}


app = dash("acepredict-app", "AcePredict · Application", [
    panel(1, "Requests per second by endpoint",
          [('sum by (handler) (rate(http_requests_total{job="backend"}[1m]))', "{{handler}}")], 0, 0),
    panel(2, "/predict latency (p50, p95)", [
        ('histogram_quantile(0.5, sum by (le) (rate(http_request_duration_seconds_bucket{job="backend",handler="/predict"}[1m])))', "p50"),
        ('histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket{job="backend",handler="/predict"}[1m])))', "p95")],
          12, 0, unit="s", thresholds=[(0.3, "red")]),
    panel(3, "Error rate (5xx share)", [
        ('sum(rate(http_requests_total{job="backend",status=~"5.."}[1m])) / clamp_min(sum(rate(http_requests_total{job="backend"}[1m])), 1e-9)', "5xx share")],
          0, 8, unit="percentunit", thresholds=[(0.05, "red")]),
    panel(4, "Rejected requests by reason",
          [("sum by (code) (rate(acepredict_invalid_requests_total[1m]))", "{{code}}")], 12, 8),
    panel(5, "Predictions per second by tour / surface",
          [("sum by (tour, surface) (rate(acepredict_predictions_total[1m]))", "{{tour}} {{surface}}")], 0, 16),
    panel(6, "Model-server call latency p95 and errors", [
        ("histogram_quantile(0.95, sum by (le) (rate(acepredict_model_server_latency_seconds_bucket[1m])))", "p95 latency (s)"),
        ("sum(rate(acepredict_model_server_errors_total[1m]))", "errors/s")], 12, 16),
    panel(7, "Frontend page views / predict clicks", [
        ("sum by (page) (rate(acepredict_frontend_page_views_total[5m]))", "{{page}}"),
        ("rate(acepredict_frontend_predict_clicks_total[5m])", "predict clicks")], 0, 24),
    panel(8, "Service up", [('up', "{{job}}")], 12, 24, kind="stat"),
])

model = dash("acepredict-model", "AcePredict · Model quality & drift", [
    panel(1, "Live accuracy (real results)", [("acepredict_live_accuracy", "accuracy")], 0, 0, w=6, kind="stat", unit="percentunit"),
    panel(2, "Live log-loss", [("acepredict_live_log_loss", "log-loss")], 6, 0, w=6, kind="stat", thresholds=[(0.69, "red")]),
    panel(3, "Live Brier score", [("acepredict_live_brier", "brier")], 12, 0, w=6, kind="stat"),
    panel(4, "Champion version", [("acepredict_model_info", "v{{version}}")], 18, 0, w=6, kind="stat"),
    panel(5, "Live accuracy and log-loss over time", [
        ("acepredict_live_accuracy", "accuracy"), ("acepredict_live_log_loss", "log-loss")], 0, 8),
    panel(6, "Feature drift: PSI per feature (red line = 0.2)",
          [("acepredict_feature_psi", "{{feature}}")], 12, 8, thresholds=[(0.2, "red")]),
    panel(7, "Distribution of predicted probabilities", [
        ("sum by (le) (rate(acepredict_predicted_probability_bucket[5m]))", "{{le}}")], 0, 16, kind="bargauge"),
    panel(8, "Results received / drift sample size", [
        ("acepredict_live_window_size", "outcomes in window"), ("acepredict_drift_sample_size", "requests for PSI")], 12, 16),
    panel(9, "Alerts received and retraining triggers", [
        ("sum by (alertname) (acepredict_alerts_received_total)", "alert {{alertname}}"),
        ("sum by (reason) (acepredict_pipeline_triggers_total)", "trigger {{reason}}")], 0, 24, w=24),
])

pipeline = dash("acepredict-pipeline", "AcePredict · Pipeline", [
    panel(1, "Stage duration (s)", [("acepredict_stage_duration_seconds", "{{stage}}")], 0, 0, kind="bargauge", unit="s"),
    panel(2, "Stage throughput (rows/s)", [("acepredict_stage_rows_per_second", "{{stage}}")], 12, 0, kind="bargauge"),
    panel(3, "Stage status (1 = ok, 0 = failed)", [("acepredict_stage_status", "{{stage}}")], 0, 8, kind="stat", thresholds=[(1, "green")]),
    panel(4, "Rows written per stage", [("acepredict_stage_rows_out", "{{stage}}")], 12, 8, kind="bargauge"),
    panel(5, "Ray Tune trials per minute / best val log-loss", [
        ("acepredict_tune_trials_per_minute", "trials/min"), ("acepredict_tune_best_val_log_loss", "best val log-loss")], 0, 16, kind="stat"),
    panel(6, "Last pipeline run", [
        ("acepredict_pipeline_last_run_success", "success (1/0)"),
        ("acepredict_pipeline_last_run_duration_seconds", "duration (s)")], 12, 16, kind="stat"),
    panel(7, "Test log-loss / accuracy of latest model", [
        ("acepredict_test_log_loss", "test log-loss"), ("acepredict_test_accuracy", "test accuracy")], 0, 24, kind="stat"),
    panel(8, "MLflow server request rate", [
        ("sum(rate(mlflow_http_request_total[1m]))", "requests/s")], 12, 24),
])

infra = dash("acepredict-infra", "AcePredict · Infrastructure", [
    panel(1, "VM CPU busy", [('1 - avg(rate(node_cpu_seconds_total{mode="idle"}[1m]))', "cpu")], 0, 0, unit="percentunit"),
    panel(2, "VM memory used", [("1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes", "memory")], 12, 0, unit="percentunit"),
    panel(3, "VM disk used", [('1 - node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}', "disk")], 0, 8, kind="stat", unit="percentunit", thresholds=[(0.9, "red")]),
    panel(4, "Container memory (MB)", [('sum by (name) (container_memory_working_set_bytes{name!=""}) / 1e6', "{{name}}")], 12, 8),
    panel(5, "Container CPU (cores)", [('sum by (name) (rate(container_cpu_usage_seconds_total{name!=""}[1m]))', "{{name}}")], 0, 16),
    panel(6, "Targets up", [("up", "{{job}}")], 12, 16, kind="stat"),
])

for d in (app, model, pipeline, infra):
    (OUT / f"{d['uid']}.json").write_text(json.dumps(d, indent=2))
    print("wrote", d["uid"])
