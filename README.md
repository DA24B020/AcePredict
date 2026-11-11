# AcePredict — tennis match win-probability prediction (DA3408 final project)

Elo ratings (overall + per surface, tuned with Ray Tune) + an ML model (logistic regression /
LightGBM, tracked in MLflow) predict P(player 1 beats player 2) for ATP/WTA matches, using
Jeff Sackmann's open `tennis_atp` / `tennis_wta` CSVs. Built and run with AIOps tooling:
Spark, Ray, DVC, Git LFS, MLflow (tracking, registry, serving), FastAPI, Streamlit,
Prometheus, Pushgateway, Alertmanager, Grafana, Docker Compose.

## Run it on the VM (Ubuntu, Docker)

```bash
unzip acepredict.zip && cd acepredict
chmod +x scripts/*.sh
scripts/setup.sh          # once: checks tools, stops old grafana, .env, builds images, git+lfs+dvc init
scripts/start.sh          # starts all 11 containers, prints URLs
scripts/run_pipeline.sh   # full pipeline: ingest → … → register → snapshot (first run ~15–25 min)
```
Then open http://localhost:8501. Once the pipeline finishes, the model server picks up the
`champion` model on its own and the backend reloads the data snapshot.

| Script | What it does |
|---|---|
| `scripts/setup.sh` | one-time setup (needs `docker`, `git`; installs `git-lfs` if it's missing) |
| `scripts/start.sh` / `scripts/stop.sh [--wipe]` | start / stop the stack (`--wipe` also deletes MLflow/Grafana volumes) |
| `scripts/run_pipeline.sh [reason]` | queue a `dvc repro` run on the pipeline-runner and follow its log |
| `scripts/version.sh "msg"` | `dvc push` + git commit (dvc.lock, params, metrics) |
| `scripts/ci.sh [--fast]` | ruff → pytest (+ test report) → dvc repro → metrics diff → dag → push |
| `scripts/replay.sh [--inject-drift] [--flip-outcomes 0.7]` | replay the held-out 2026 matches as live traffic |
| `scripts/load_test.sh --seconds 30 --concurrency 8` | throughput + p95 latency of /predict |
| `scripts/use_sample_data.sh sample\|github` | switch to synthetic data (offline, ~3 min pipeline) or back |
| `scripts/update_data.sh` | re-download the CSVs and rerun what changed |
| `scripts/logs.sh [service]` | follow container logs |

## URLs
App :8501 · API docs :8000/docs · MLflow :5000 · Grafana :3000 (admin/admin) · Prometheus :9090 ·
Alertmanager :9093 · Pushgateway :9091 · Ray dashboard :8265 and Spark UI :4040 (only while those stages run).
Admin key for the console buttons: `ADMIN_API_KEY` in `.env` (default `change-me-admin-key`).

## Demo flow (viva)
1. **Predict** page: pick two players, a surface and a round → see the probability, the Elo-only probability, the top 3 factors, and the head-to-head record.
2. **Pipeline Console**: the DAG coloured by stage status, per-stage duration and rows/s, past runs, the failing stage's log tail, MLflow runs, and the Run / Promote / Rollback buttons.
3. `scripts/replay.sh` → Grafana *Model quality & drift* shows live accuracy and log-loss.
4. `scripts/replay.sh --inject-drift` → PSI goes above 0.2 → the `FeatureDrift` alert fires → Alertmanager calls the webhook → a retraining run starts on its own (it shows in the console).
5. Reproduce any model: MLflow run tags hold `git_commit` and `dvc_lock_md5` → `git checkout <commit> && dvc pull && dvc repro`.

## Layout
`src/acepredict/` pipeline, Spark job, Elo, Ray Tune, features, models, registry, monitoring ·
`backend/app/` FastAPI gateway · `frontend/` Streamlit UI (REST only, `BACKEND_URL`) ·
`docker/`, `docker-compose.yml` · `monitoring/` Prometheus, alert rules, Alertmanager, Grafana ·
`dvc.yaml`, `params.yaml`, `MLproject` · `tests/` unit + integration · `scripts/`.

## Notes
* If Sackmann hasn't published 2025/2026 files yet, ingest skips them (it logs a 404), the splits fall back to the last available seasons, and the replay uses the latest season.
* Disk: the images take ~3.5 GB and the data plus the DVC cache ~0.5 GB. RAM: the pipeline-runner is capped at 6 GB.
* Pin exact data: set `data.sackmann_ref` in `params.yaml` to a commit SHA.
