# AcePredict — Architecture

![Architecture](architecture.png)

*(Diagram source: `docs/architecture.mmd`, rendered with Mermaid CLI.)*

## 1. What the system does
AcePredict gives the probability that tennis player 1 beats player 2 (ATP or WTA) on a chosen surface,
event level and round. It combines Elo ratings (overall and per surface, with parameters tuned by Ray
Tune) with a classifier trained on match features. The whole lifecycle — data ingestion, training,
registration, serving, monitoring and retraining — runs as 11 Docker Compose services on a single
Ubuntu VM, with no cloud services.

## 2. Blocks

| Block | Container | Technology | Responsibility |
|---|---|---|---|
| UI | `frontend` | Streamlit | 6 pages: Predict, Player Profile, Head-to-Head, Pipeline Console, Monitoring, Help. Talks **only** to the backend over REST (`BACKEND_URL`). |
| API gateway | `backend` | FastAPI + Uvicorn | Validation, feature assembly from the snapshot, calls the model server, records predictions/outcomes, live metrics, PSI drift, admin + webhook endpoints, `/metrics`. |
| Model serving | `model-server` | `mlflow models serve` | Serves `models:/acepredict@champion` at `/invocations`. A supervisor (`serve_champion.py`) polls the alias every 30 s and restarts the server on a new champion (hot swap). |
| Experiment tracking + registry | `mlflow` | MLflow server (SQLite + artifact proxy) | Runs, params, metrics, artifacts, model versions, aliases `champion` / `challenger` / `previous`. |
| ML pipeline | `pipeline-runner` | DVC + PySpark + Ray Tune + scikit-learn/LightGBM | Watches `runtime/triggers/`; runs `dvc repro` (10 stages) under a lock; pushes stage metrics to the Pushgateway. |
| Batch metrics | `pushgateway` | Prometheus Pushgateway | Holds per-stage duration / rows / rows-per-second / status pushed by the batch jobs. |
| Metrics store | `prometheus` | Prometheus | Scrapes every service every 5 s; evaluates alert rules. |
| Alert routing | `alertmanager` | Alertmanager | Routes `action="retrain"` alerts to the backend webhook (Bearer token). |
| Dashboards | `grafana` | Grafana (provisioned) | 4 dashboards: Application, Model quality & drift, Pipeline, Infrastructure. Embedded in the UI. |
| Host / container metrics | `node-exporter`, `cadvisor` | — | VM CPU / memory / disk; per-container CPU / memory. |

## 3. Main flows

**Prediction (online).** Browser → `frontend` → `POST /predict` on `backend` → the backend looks up both
players in the in-memory snapshot (`artifacts/snapshot/*.parquet`), builds the 26 features → `POST
/invocations` on `model-server` → the symmetric pyfunc returns `prob_a` and the top-3 factors → the
backend adds the Elo-only probability, head-to-head record and warnings, logs the request
(`runtime/predictions.jsonl`) and returns JSON.

**Feedback.** When the real result is known, the client calls `POST /outcomes` with the `request_id`.
The backend updates live accuracy / log-loss / Brier over the last 200 outcomes (Prometheus gauges).

**Training (batch).** A trigger (UI button, `scripts/run_pipeline.sh`, or an alert) writes
`runtime/triggers/<id>.json`. The `pipeline-runner` takes the lock and runs `dvc repro`:
`ingest → validate → transform (Spark) → tune_elo (Ray Tune) → elo → features → train → evaluate →
register → snapshot`. DVC skips any stage whose dependencies and parameters are unchanged.
`register` promotes the new version to `champion` only if its test log-loss is no worse
(promotion gate). `snapshot` writes the serving tables; the backend hot-reloads them when
`meta.json` changes.

**Closed loop.** The backend computes PSI between recent request features and the training reference
every 30 s → Prometheus alert `FeatureDrift` (max PSI > 0.2 for 1 min) or `LiveLogLossHigh` →
Alertmanager → `POST /webhooks/alertmanager` → trigger file (30 min cooldown) → retraining run.

## 4. Design principles
* **Loose coupling.** The frontend and backend are separate images; the only link is the REST URL in
  an environment variable. The backend reaches the model only over HTTP (`MODEL_SERVER_URL`), so a
  model can be replaced without touching the API.
* **Reproducibility.** Every MLflow run is tagged with `git_commit`, `dvc_lock_md5`, `data_source` and
  `data_ref`. `git checkout <commit> && dvc pull && dvc repro` rebuilds any version. Seeds are fixed
  (A/B flip, LightGBM, Ray Tune).
* **No leakage.** Rolling features use windows that end at the previous match; Elo uses pre-match
  ratings; the splits are by time (train 2000–2021, validation 2022–2023, test 2024–2025, live 2026+).
* **Observability everywhere.** All application services expose `/metrics`; batch stages push to the
  Pushgateway; host and container metrics come from node-exporter and cAdvisor.
* **Fail safe.** The backend returns typed errors (`error_code`, `message`, `request_id`) and stays
  up when the model server is down (`/ready` = 503, UI shows a friendly message).

## 5. Deployment
All services are defined in `docker-compose.yml`. Two images are built: `acepredict-ml` (Python 3.11,
Java 17 for Spark, all ML libraries — shared by mlflow, model-server, backend and pipeline-runner)
and `acepredict-frontend` (Streamlit only). Python dependencies are pinned through `constraints.txt`.
The backend and pipeline-runner run as the VM user's UID so files they write stay owned by the user.
Resource limits fit a 6-vCPU / 12 GB VM: the pipeline-runner is capped at 6 GB, the other containers at
0.5–1.5 GB each.
