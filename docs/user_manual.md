# AcePredict — User Manual

AcePredict tells you how likely each player is to win a tennis match. You don't need to know
anything about machine learning to use it.

## 1. Opening the app
Open **http://localhost:8501** in a web browser on the VM. The left sidebar shows:
* **Model vX · data to <date>** — the app is ready.
* **Not ready** — the model or data is still loading (wait a minute), or the system has never been
  trained (ask the administrator to run the pipeline, see §7).
* **Service unreachable** — the system is switched off (administrator: `scripts/start.sh`).

## 2. Predict a match (first page)
1. Choose **ATP** (men) or **WTA** (women).
2. Type part of a player's name in **Player 1** and pick from the list. Do the same for **Player 2**.
   The two players must be different (the button stays grey until they are).
3. Choose the **surface** (Hard, Clay, Grass). Optionally choose the **event level** (Tour, Masters,
   Grand Slam), **round** (Early, QF, SF, Final) and **best of 3 / 5** sets.
4. Press **Predict**.

**Reading the result**
* The two large numbers are each player's chance of winning. They always add up to 100 %.
* **Elo ratings alone would say…** — the simpler rating-only estimate, for comparison.
* **Why?** — the three factors that moved the prediction most, and which player each one favours
  (for example "Surface Elo rating — favours Sinner").
* **Head-to-head** — how many times each has beaten the other.
* **Yellow notes** — warnings when data is thin: few recorded matches, a long break since the last
  match, or few matches on this surface. Treat those predictions with more caution.

## 3. Player Profile
Pick a tour and a player to see: current overall and surface ratings, ranking, recent form, a chart
of the Elo rating over time, and the last 10 matches.

## 4. Head to Head
Pick two players to see their full record against each other: total wins, wins by surface and the
list of previous meetings with scores.

## 5. Pipeline Console (for the administrator)
* **Pipeline graph** — the 10 steps from downloading data to publishing the model, coloured green
  (done), blue (running), red (failed) or grey (not run).
* **Stage table** — when each step last ran, how long it took and how many rows per second it
  processed.
* **Runs** — history of training runs and why each started (manual, button, or an automatic alert).
* **If a step failed** — the last lines of its log are shown here.
* **Recent MLflow runs** — experiments and their scores; link to MLflow for details.
* **Admin actions** (enter the admin key from `.env`):
  * **Run pipeline** — retrain with the latest data and settings.
  * **Promote** — make a specific model version the live one.
  * **Roll back** — return to the previous live version.

## 6. Monitoring
Shows the live numbers: predictions served, accuracy and log-loss on matches whose real results were
reported, and data drift scores (how different today's requests are from the training data; above
0.2 means a significant shift). The embedded Grafana panels show request rate, latency, errors and
model quality over time. Full dashboards: **http://localhost:3000** (folder "AcePredict").

## 7. Administrator quick reference

| Task | Command (in `~/acepredict` on the VM) |
|---|---|
| First-time setup | `scripts/setup.sh` |
| Start / stop | `scripts/start.sh` / `scripts/stop.sh` |
| Train or retrain | `scripts/run_pipeline.sh "reason"` |
| Save a version (data + Git) | `scripts/version.sh "message"` |
| Run all checks (lint, tests, pipeline) | `scripts/ci.sh` (`--fast` = lint + tests only) |
| Replay the 2026 season as live traffic | `scripts/replay.sh` |
| Simulate drift (tests the automatic retraining) | `scripts/replay.sh --inject-drift` |
| Load test | `scripts/load_test.sh --concurrency 4 --p95-limit-ms 1000` |
| Container logs | `scripts/logs.sh backend` (or any service name) |

Other tools: API documentation **http://localhost:8000/docs**, MLflow **http://localhost:5000**,
Prometheus **http://localhost:9090**, Alertmanager **http://localhost:9093**.

## 8. Glossary
* **Elo rating** — a skill score that rises after wins and falls after losses; beating a strong
  player gains more. New players start at 1500; the best are above 2000.
* **Surface Elo** — the same score counted only on one surface.
* **Form** — share of recent matches won.
* **Log-loss** — how wrong the probabilities are on average (lower is better; 0.69 is a coin flip).
* **Drift (PSI)** — how much the incoming requests differ from the training data.
* **Champion** — the model version currently used for predictions.

## 9. Troubleshooting

| Message | What to do |
|---|---|
| "Cannot reach the AcePredict service" | Administrator runs `scripts/start.sh`. |
| "The prediction model is starting up / not available" | Wait 1 minute and retry. If it persists: `scripts/logs.sh model-server`. |
| "Unknown player" | Choose the player from the list instead of typing an ID. |
| "Both players must be on the same tour" | Pick two ATP or two WTA players. |
| Browser shows "Connection failed with status 404" | The app was restarted; close the tab and open http://localhost:8501 again. |
| A pipeline step is red | Open the Pipeline Console; read the log tail; fix and press Run pipeline. |
