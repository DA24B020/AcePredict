# AcePredict — High-Level Design

## 1. Problem and objective
Predict P(player 1 beats player 2) for professional tennis matches (ATP and WTA) before the match,
using only public historical results. Secondary goal: run the full model lifecycle with AIOps
practices — versioned data and models, tracked experiments, automated pipeline, containerised
serving, monitoring, alerting and automatic retraining.

**Baseline being replicated.** Surface-aware Elo with decaying K-factor (FiveThirtyEight-style),
which is reported to pick about 70 % of 2014 ATP winners. Our replication: **68.9 %** accuracy
(log-loss 0.586) on 2,491 ATP 2014 matches with Ray-Tune-tuned parameters.

## 2. Data
* **Source:** Jeff Sackmann's `tennis_atp` / `tennis_wta` yearly match CSVs (CC BY-NC-SA 4.0), read
  from the public mirror `Aneeshers/tennis-sackmann-archive` (the original repository returns 404).
  The URL template and Git ref are in `params.yaml` (`data.base_url_*`, `data.sackmann_ref`).
* **Volume (current run):** 72 files, 42 MB, 210,382 rows (1991–2026) → 186,566 valid matches,
  4,751 players.
* **Validation:** schema (49 columns), required fields, date parsing, winner ≠ loser, surface values;
  walkovers dropped. The stage fails if more than 5 % of rows are invalid.

## 3. Modelling approach
1. **Elo ratings** — overall and per surface (hard / clay / grass); K = c / (matches + offset)^shape;
   the blended rating is w · overall + (1 − w) · surface. Parameters c, offset, shape and w are tuned by
   **Ray Tune** (16 trials) on validation log-loss.
2. **Features (26)** — 18 antisymmetric player-1-minus-player-2 differences (overall Elo, surface
   Elo, blended Elo probability, log rank ratio, rank points, experience, age, height, left-handedness,
   form over the last 10 matches / 12 months / this surface, days since last match, matches in the
   last 14 days, ace rate, first-serve points won, break points saved, head-to-head wins) + 8
   symmetric context flags (clay, grass, best of 5, Grand Slam, top-tier event, late round, WTA,
   number of previous meetings). Rolling statistics are computed with **Spark** window
   functions.
3. **Classifiers** — logistic regression and LightGBM, both trained; the one with the lower
   validation log-loss is chosen. The model is wrapped in a **symmetric** MLflow pyfunc:
   p = (f(A,B) + 1 − f(B,A)) / 2, so swapping the players always gives complementary probabilities.
4. **Evaluation** — test log-loss, accuracy, Brier, ECE, ROC-AUC, reliability table and per-segment
   metrics, compared with two baselines (Elo only, higher-ranked player wins).

Current champion (v1, logistic regression), test set 2024–2025 (10,785 matches):

| Model | Log-loss | Accuracy | ROC-AUC |
|---|---|---|---|
| AcePredict v1 | **0.613** | **65.6 %** | **0.719** |
| Elo only | 0.631 | 64.0 % | 0.702 |
| Higher-ranked wins | 0.657 | 63.5 % | 0.635 |

## 4. System components (summary)
See `architecture.md` for the diagram. Components: Streamlit UI, FastAPI backend, MLflow model server,
MLflow tracking server + registry, DVC pipeline runner (Spark, Ray), Prometheus, Pushgateway,
Alertmanager, Grafana, node-exporter, cAdvisor.

## 5. Key design decisions

| Decision | Choice | Reason |
|---|---|---|
| Pipeline orchestration | DVC stages run by a small runner service | Caching (unchanged stages skipped), `dvc.lock` as an exact record, DAG view, no extra scheduler on a small VM. |
| Distributed processing | Spark (local[4]) for rolling features | Window functions over ~370k player-match rows; scales to a cluster by changing `SPARK_MASTER`. |
| Hyperparameter search | Ray Tune for Elo parameters | Parallel trials (4 CPUs); each trial logged to MLflow; dashboard on :8265. |
| Serving | `mlflow models serve` behind a FastAPI gateway | The model is served straight from the registry by alias; the gateway adds validation, features, explanations, monitoring and errors. |
| Model promotion | Alias-based (`champion`/`challenger`/`previous`) with a test log-loss gate | Safe hot swap and one-click rollback; no redeploy needed. |
| Batch job metrics | Pushgateway | Pipeline stages are short-lived processes. |
| Retraining trigger | Alertmanager webhook → backend → trigger file | Reuses the same queue and lock as manual runs; 30 min cooldown prevents loops. |
| Online drift | PSI on 12 key features, 2,000-request window | Cheap; interpretable; alert threshold 0.2 is the usual "significant shift" level. |
| Config | `params.yaml` (pipeline), `.env` (services) | Every tunable value is outside the code; changes are tracked by Git/DVC. |

## 6. Non-functional requirements and results

| Requirement | Target | Measured on the VM |
|---|---|---|
| Single prediction latency | < 200 ms | 65–90 ms (backend + model server) |
| Throughput | Report | 10.5 predictions/s at 4 concurrent clients, 0 errors |
| p95 latency under load | < 1 s at 4 clients | 584 ms |
| Full pipeline (real data) | < 30 min | ≈ 5 min (ingest 36 s, Spark 82 s, Ray Tune 43 s, train 21 s) |
| Retrain after a parameter change | — | ≈ 1 min (data stages skipped by DVC) |
| Resources | 6 vCPU, 12 GB RAM, < 10 GB disk | Fits; images ≈ 3.5 GB |
| Availability | Services restart on failure | `restart: unless-stopped`; health checks on mlflow, backend, frontend |

## 7. Security and privacy
Public sports data only; no personal user data is stored (prediction logs hold player IDs and
probabilities). Admin endpoints need an `X-API-Key`; the alert webhook needs a Bearer token; both
come from `.env`, which is not committed. All services run locally on the VM.

## 8. Risks and mitigations
* **Data source disappears** (happened with the original repository) → URL is configurable; a
  synthetic data generator (`data.source: sample`) runs the whole system offline.
* **Dependency drift** → `constraints.txt` pins every package version for both images.
* **Retraining loops** → cooldown, lock file, and DVC skipping unchanged work.
* **Worse model deployed** → promotion gate on test log-loss; rollback in one click.
