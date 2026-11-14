# AcePredict — Low-Level Design

## 1. Code layout

```
src/acepredict/            core library (importable; used by the pipeline, backend and model server)
  config.py                Paths, load_params(), tracking URI, git_commit()
  logging_conf.py          JSON log formatter, request-id context var, rotating file handler
  exceptions.py            AcePredictError hierarchy (code + HTTP status)
  tracking.py              init_mlflow() health check, reproducibility tags
  data/schema.py           49 Sackmann columns, surfaces, round order, player_key()
  data/sample.py           synthetic data generator (offline mode)
  spark_jobs/transform.py  Spark cleaning + rolling window features
  ratings/elo.py           EloParams, k_factor, elo_pass, EloSystem, predict_from_ratings
  tuning/tune_elo.py       Ray Tune search over Elo parameters (MLflowLoggerCallback)
  features/build.py        feature lists, pair_features, swap_players, build_training_table
  features/splits.py       time-based splits with fallback
  models/train.py          estimators, nested MLflow runs, pyfunc logging, registration
  models/pyfunc.py         AcePredictModel (symmetric, explanations)
  models/evaluate.py       test metrics, baselines, reliability, segments, plots
  models/registry.py       aliases, promote, rollback, promotion_gate
  models/metrics.py        log-loss, Brier, ECE, ROC-AUC, reliability table
  monitoring/pushgw.py     Pushgateway metrics for stages and runner
  monitoring/drift.py      PSI, psi_report, DRIFT_FEATURES
  serving/serve_champion.py supervisor for `mlflow models serve` (alias polling)
  pipeline/<stage>.py      one module per DVC stage; pipeline/stage.py = common wrapper
  pipeline/runner.py       trigger-file watcher, lock, `dvc repro`, run history
backend/app/               FastAPI gateway (settings, schemas, services, clients, routers, metrics)
frontend/                  Streamlit app (api_client.py, ui.py, metrics.py, Predict_a_Match.py, pages/)
```

**Style.** Functional core (pure functions for Elo, features, metrics, PSI), object-oriented
services at the edges (`SnapshotStore`, `MonitorService`, `PipelineService`, `RegistryService`,
`PredictionService`, `ModelServerClient`). PEP 8 enforced by `ruff` (line length 100) in CI and in
the pre-commit hook. All modules, classes and public functions have docstrings.

## 2. Pipeline stages (dvc.yaml)

| Stage | Module | Inputs | Outputs | Notes |
|---|---|---|---|---|
| ingest | `pipeline.ingest` | params `data.*` | `data/raw/{atp,wta}/*.csv` | HTTP download with retries; 404 years skipped; fails if a tour has no files |
| validate | `pipeline.validate` | `data/raw` | `reports/data_validation.json` | schema, required fields, dates, winner ≠ loser; fails if > 5 % invalid |
| transform | `pipeline.transform` → `spark_jobs.transform` | `data/raw`, report | `data/interim/{matches,live_matches,player_match}.parquet` | Spark windows `rowsBetween(-N,-1)` / `rangeBetween` on ordinal → no leakage |
| tune_elo | `pipeline.tune_elo` → `tuning.tune_elo` | matches | `reports/elo_tuning.json` | Ray Tune, 16 trials, 4 CPUs, best by validation log-loss |
| elo | `pipeline.elo` | matches, tuning | `data/features/elo_history.parquet`, `elo_state.parquet`, `elo_params.json`, `reports/elo_replication.json` (metric) | pre-match ratings per match; ATP 2014 replication |
| features | `pipeline.features` | elo history, player_match | `data/features/train_table.parquet`, `reports/splits.json` | seeded A/B flip so the label is balanced |
| train | `pipeline.train` | train table, params `model.*` | `reports/train.json` | nested MLflow runs per candidate; best → pyfunc → registry, alias `challenger` |
| evaluate | `pipeline.evaluate` | model, train table | `reports/metrics.json` (metric), `reliability.csv` (plot), `segments.csv` | baselines, plots logged to MLflow |
| register | `pipeline.register` | train.json, metrics.json | `reports/promotion.json` | promotion gate |
| snapshot | `pipeline.snapshot` | all of the above | `artifacts/snapshot/` | players, h2h, matches_lite, elo_monthly, reference, meta.json; atomic rename |

Every stage runs through `pipeline.stage.run_stage()`, which logs start/finish as JSON, measures
duration and rows/s, writes `runtime/stage_status.json`, pushes metrics to the Pushgateway, and exits
with code 1 on failure (DVC then stops and the run is marked failed).

## 3. Core algorithms

**Elo update** (`ratings/elo.py`):
```
E_w   = 1 / (1 + 10^((R_l − R_w)/400))
K(m)  = c / (m + offset)^shape          m = matches played before this match
R_w' = R_w + K(m_w)·(1 − E_w);  R_l' = R_l − K(m_l)·(1 − E_w)
```
The same update runs on the surface rating of that surface only. Blended probability uses
w·overall + (1 − w)·surface ratings.

**Symmetric pyfunc** (`models/pyfunc.py`): for a feature row x and its swap s(x)
(antisymmetric features negated, symmetric kept): `prob_a = (f(x) + 1 − f(s(x))) / 2`.
Explanations: logistic regression uses coefficient × standardised value; LightGBM uses
`pred_contrib`. The top 3 by absolute contribution are returned as JSON.

**PSI** (`monitoring/drift.py`): bins from reference quantiles (binary features use 2 bins);
`PSI = Σ (p_cur − p_ref) · ln(p_cur / p_ref)` with ε = 1e-4.

**Promotion gate** (`models/registry.py`): promote the challenger if there is no champion, or if
`round(challenger_ll, 5) ≤ round(champion_ll, 5) + tolerance`. Otherwise keep the champion.
Promotion moves the old champion to the `previous` alias, so rollback restores it.

## 4. REST API (backend, port 8000)

All errors have the body `{"error_code": str, "message": str, "request_id": str, "details": any}`.
Every response carries the header `X-Request-ID`. Interactive docs: `/docs`; spec: `docs/openapi.json`.

| Method | Path | Input | Output (200) | Errors |
|---|---|---|---|---|
| GET | `/health` | – | `{"status":"ok"}` | – |
| GET | `/ready` | – | `{ready, snapshot_loaded, snapshot_date, model_server, model_version}` | 503 when not ready |
| GET | `/players` | query `q`, `tour` (atp/wta), `limit` (≤ 200), `active_only` | list of `PlayerSummary {id, name, tour, country, elo, rank, ...}` sorted by Elo | 400, 503 |
| GET | `/players/{player_id}` | path id (e.g. `atp_104925`) | `PlayerDetail` (summary + surface Elos, form, `elo_history`, `recent_matches`) | 404 `PLAYER_NOT_FOUND` |
| GET | `/h2h` | query `p1`, `p2` | `{p1_wins, p2_wins, meetings[]}` | 404, 400 |
| POST | `/predict` | `PredictRequest {tour, player1_id, player2_id, surface=Hard\|Clay\|Grass, best_of=3\|5, tourney_level=tour\|masters\|grand_slam, round=early\|QF\|SF\|F, match_date?}` | `PredictResponse {request_id, player1, player2, surface, p1_win_prob, p2_win_prob, elo_only_p1_prob, model_version, top_factors[], warnings[], head_to_head}` | 400 `VALIDATION_ERROR` / `INVALID_MATCHUP`, 404, 502 `MODEL_SERVER_ERROR`, 503 `MODEL_UNAVAILABLE` / `SNAPSHOT_NOT_READY` |
| POST | `/predict/batch` | `{items: [PredictRequest]}` (≤ max_batch) | `{results: [PredictResponse]}` — one model call | as above |
| POST | `/outcomes` | `{request_id, p1_won}` | `{recorded, live_accuracy, live_log_loss, window}` | 404 unknown request |
| GET | `/monitoring/summary` | – | predictions logged, live metrics, PSI per feature, drift sample size | – |
| GET | `/model/info` | – | champion / previous / challenger versions, run id, model type, git commit, test metrics | 502 `REGISTRY_ERROR` |
| GET | `/pipeline/status` | – | stage table, DAG (`dag_dot`), recent runs, failed log tail, lock state | – |
| POST | `/admin/pipeline/run` | header `X-API-Key`; `{reason?}` | 202 `{run_id, queued_at}` | 401 `AUTH_ERROR`, 409 `PIPELINE_BUSY` |
| POST | `/admin/model/promote` | `X-API-Key`; `{version}` | `{champion_version, previous_version}` | 401, 502 |
| POST | `/admin/model/rollback` | `X-API-Key` | same | 401, 502 |
| POST | `/admin/snapshot/reload` | `X-API-Key` | snapshot meta | 401, 503 |
| POST | `/webhooks/alertmanager` | header `Authorization: Bearer <token>`; Alertmanager payload | `{triggered, run_id?, reason}` | 401 |
| GET | `/metrics` | – | Prometheus text format | – |

**Model server** (port 5001, MLflow scoring protocol): `GET /ping`; `POST /invocations` with
`{"dataframe_split": {"columns": [26 feature names], "data": [[...]]}}` → `{"predictions":
[{"prob_a": float, "explanation": "<json list>"}]}`.

## 5. Backend internals
* `Services.build()` wires everything once: `SnapshotStore` (reloads when `meta.json` mtime changes),
  `ModelServerClient` (requests session, 10 s timeout, ping cache 5 s), `RegistryService` (15 s
  cache), `MonitorService` (pending predictions in an LRU, outcome window, PSI), `PipelineService`
  (trigger files, cooldown), `PredictionService`.
* A background task runs every `MONITOR_INTERVAL_S` (30 s): snapshot age, model-server health,
  champion version, PSI refresh, live metrics.
* Middleware adds a request ID, logs one JSON line per request, and maps `AcePredictError` subclasses
  to HTTP codes; pydantic validation errors become 400 `VALIDATION_ERROR`.

## 6. Configuration
* `params.yaml` — every pipeline parameter (data source, years, Spark, splits, Elo, Ray Tune ranges,
  features, model hyperparameters, registry, MLflow experiments).
* `.env` — service settings: UID/GID, admin key, webhook token, monitor interval, live window, drift
  minimum sample, retrain cooldown, thread limits, model-server workers, public URLs.

## 7. Metrics exported (selection)

| Metric | Source | Meaning |
|---|---|---|
| `http_requests_total`, `http_request_duration_seconds` | backend | per-endpoint rate and latency |
| `acepredict_predictions_total{tour,surface}` | backend | prediction count |
| `acepredict_model_server_latency_seconds`, `_errors_total`, `_up` | backend | model-server health |
| `acepredict_live_accuracy`, `_live_log_loss`, `_live_brier`, `_live_window_size` | backend | live quality on real outcomes |
| `acepredict_feature_psi{feature}`, `acepredict_drift_sample_size` | backend | input drift |
| `acepredict_alerts_received_total`, `acepredict_pipeline_triggers_total` | backend | closed loop |
| `acepredict_stage_duration_seconds`, `_rows_per_second`, `_status{stage}` | pushgateway | pipeline throughput and failures |
| `acepredict_tune_trials_per_minute`, `acepredict_test_log_loss` | pushgateway | tuning speed, model quality |
| `acepredict_frontend_page_views_total`, `_predict_clicks_total` | frontend | UI usage |
| `mlflow_http_request_total` | mlflow | tracking server load |

## 8. Alert rules (`monitoring/prometheus/alert_rules.yml`)
BackendDown, ModelServerDown, HighErrorRate (> 5 % 5xx), HighLatencyP95 (/predict > 300 ms),
FeatureDrift (max PSI > 0.2, `action=retrain`), LiveLogLossHigh (> 0.69 with ≥ 50 outcomes,
`action=retrain`), PipelineStageFailed, DiskAlmostFull (> 90 %).
