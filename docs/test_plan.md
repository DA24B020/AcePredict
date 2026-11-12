# AcePredict — Test Plan and Test Cases

## 1. Scope
In scope: data validation, Elo and feature logic, model wrapper, metrics and drift maths, REST API
(including errors and security), frontend REST client, the DVC pipeline end to end, model
registry/promotion, serving, monitoring and alerting, the automatic retraining loop, performance.
Out of scope: accuracy targets beyond beating the baselines; browser compatibility beyond Chrome/Firefox.

## 2. Test levels and tools

| Level | What | Tool / command | Where |
|---|---|---|---|
| Static | PEP 8, imports, unused code | `ruff check` | CI (`scripts/ci.sh`), pre-commit |
| Unit | pure functions + API with fakes | `pytest` (45 tests, `tests/unit/`) | inside the `pipeline-runner` image |
| Integration | API against the running stack | `pytest -m integration` (`tests/integration/`) | VM, stack running |
| Pipeline | all 10 DVC stages on real data | `scripts/run_pipeline.sh`, `dvc metrics show/diff` | VM |
| System / E2E | UI flows, replay, drift → retrain | manual checklist (§5), `scripts/replay.sh` | VM |
| Performance | throughput and latency | `scripts/load_test.sh` | VM |

The unit tests use a small fixture snapshot (`tests/fixtures/snapshot/`), a fake model server
(logistic of the Elo difference, can be switched "down") and a fake registry, so they run in a few
seconds without Docker, MLflow or the network.

## 3. Entry and exit criteria
* **Entry:** images build; `ruff` passes; the stack starts (`scripts/start.sh` reports `/health` OK).
* **Exit:** all acceptance criteria in §4 met; no open defect of severity High.

## 4. Acceptance criteria

| ID | Criterion | Threshold |
|---|---|---|
| AC-1 | Unit tests | 100 % pass |
| AC-2 | Lint | 0 ruff errors |
| AC-3 | Pipeline | all 10 stages succeed on real data; `dvc.lock` committed; data pushed to the DVC remote |
| AC-4 | Model quality | test log-loss lower than both baselines (Elo only, higher-ranked wins) |
| AC-5 | Elo replication | ATP 2014 accuracy within 3 points of the published ≈ 70 % |
| AC-6 | Symmetry | P(A beats B) + P(B beats A) = 1 (± 1e-9) |
| AC-7 | Latency | single /predict < 200 ms; p95 < 1 s at 4 concurrent clients |
| AC-8 | Reliability | 0 errors (non-2xx) during the load test |
| AC-9 | Errors | every invalid input returns a typed JSON error with a request ID; no 500s |
| AC-10 | Monitoring | every service is an `up` target in Prometheus; the 4 Grafana dashboards show data |
| AC-11 | Closed loop | injected drift → `FeatureDrift` firing → retraining run started automatically within 5 min |
| AC-12 | Promotion safety | a challenger that is worse on test log-loss is not promoted; rollback restores the previous champion |
| AC-13 | Usability | a first-time user can get a prediction in under 1 minute using only the Help page |

## 5. Test cases

### 5.1 Unit (automated, `tests/unit/`)

| ID | Test | Expected |
|---|---|---|
| TC-ELO-01 | `test_expected_score_equal_ratings_is_half` | E = 0.5 |
| TC-ELO-02 | `test_expected_score_is_complementary` | E(a,b) + E(b,a) = 1 |
| TC-ELO-03 | `test_400_points_is_ten_to_one` | E = 10/11 |
| TC-ELO-04 | `test_k_factor_decreases_with_experience` | K strictly decreasing |
| TC-ELO-05 | `test_winner_gains_and_loser_drops_same_amount_for_newcomers` | zero-sum update |
| TC-ELO-06 | `test_pre_match_rating_never_includes_the_match_itself` | no leakage |
| TC-ELO-07 | `test_surface_rating_only_moves_on_that_surface` | other surfaces unchanged |
| TC-ELO-08 | `test_elo_system_state_and_history` | counts, ordering, probabilities in [0,1] |
| TC-ELO-09 | `test_predict_from_ratings_symmetric` | complementary |
| TC-FEAT-01 | `test_feature_lists_cover_every_feature_once` | 26 unique features |
| TC-FEAT-02 | `test_pair_features_has_no_nan_and_right_columns` | missing inputs → 0, column order fixed |
| TC-FEAT-03 | `test_swapping_players_negates_antisymmetric_and_keeps_context` | swap rule holds |
| TC-FEAT-04 | `test_pyfunc_probabilities_are_exactly_symmetric` | AC-6 |
| TC-FEAT-05 | `test_pyfunc_explanations_are_top_factors` | 1–3 factors with label/contribution/favours |
| TC-FEAT-06 | `test_splits_use_config_when_years_exist` | configured years used |
| TC-FEAT-07 | `test_splits_fall_back_when_recent_seasons_missing` | falls back to latest years, no overlap |
| TC-DATA-01 | `test_sample_data_passes_validation` | no critical issues |
| TC-DATA-02 | `test_validation_flags_missing_columns` | critical issue names the column |
| TC-DATA-03 | `test_validation_counts_bad_rows` | bad rows counted |
| TC-DATA-04 | `test_schema_helpers` | player keys, file names, round order |
| TC-MON-01 | `test_prob_metrics_perfect_and_coinflip` | log-loss ≈ 0 / ln 2 |
| TC-MON-02 | `test_ece_of_calibrated_predictions_is_small` | ECE < 0.02 |
| TC-MON-03 | `test_psi_same_distribution_is_near_zero_and_shift_is_large` | < 0.02 and > 0.2 |
| TC-MON-04 | `test_psi_report_only_uses_shared_columns` | ignores absent features |
| TC-API-01 | `test_health_and_ready` | 200, ready = true |
| TC-API-02 | `test_ready_is_503_when_model_server_down` | 503, ready = false |
| TC-API-03 | `test_player_search_sorted_by_elo` | filtered by tour, sorted |
| TC-API-04 | `test_player_profile` | history and recent matches present |
| TC-API-05 | `test_unknown_player_is_404_with_error_code` | 404 PLAYER_NOT_FOUND + request_id |
| TC-API-06 | `test_predict_returns_complementary_probabilities` | p1 + p2 = 1, factors, version |
| TC-API-07 | `test_bad_predict_requests_are_400` (3 cases) | same player / bad surface / missing field → 400 |
| TC-API-08 | `test_predict_across_tours_is_rejected` | 400 INVALID_MATCHUP |
| TC-API-09 | `test_predict_when_model_down_is_503` | 503 MODEL_UNAVAILABLE |
| TC-API-10 | `test_batch_uses_one_model_call` | 4 results, 1 model call |
| TC-API-11 | `test_outcomes_update_live_accuracy` | window grows; unknown id → 404 |
| TC-API-12 | `test_admin_endpoints_need_api_key` | 401 without key, 202 with, 409 when busy |
| TC-API-13 | `test_promote_and_rollback` | aliases move both ways |
| TC-API-14 | `test_alertmanager_webhook_triggers_retraining` | 401 without token; triggered with |
| TC-API-15 | `test_pipeline_status_has_dag` | DAG contains ingest…snapshot |
| TC-API-16 | `test_metrics_endpoint_exposes_custom_metrics` | custom + HTTP metrics present |
| TC-UI-01 | `test_backend_url_comes_from_env` | URL configurable (loose coupling) |
| TC-UI-02 | `test_unreachable_backend_gives_friendly_error` | friendly message, no crash |
| TC-UI-03 | `test_error_codes_map_to_plain_messages` | error code → plain text |

### 5.2 Integration (automated, stack running: `ACEPREDICT_BASE_URL=http://backend:8000 pytest -m integration`)

| ID | Test | Expected |
|---|---|---|
| TC-E2E-01 | `test_ready` | ready = true |
| TC-E2E-02 | `test_end_to_end_prediction_and_outcome` | 200 prediction; outcome recorded |
| TC-E2E-03 | `test_model_info_has_champion` | champion version set |

### 5.3 System and pipeline (manual / scripted)

| ID | Steps | Expected | Criterion |
|---|---|---|---|
| TC-SYS-01 | `scripts/run_pipeline.sh` on a fresh stack | 10 stages succeed; v1 champion; `/ready` true | AC-3 |
| TC-SYS-02 | `dvc metrics show` | test log-loss < both baselines | AC-4 |
| TC-SYS-03 | read `reports/elo_replication.json` | accuracy within 3 points of 0.70 | AC-5 |
| TC-SYS-04 | run the pipeline again without changes | all stages skipped | caching |
| TC-SYS-05 | change `model.logreg.C`, run | only train → snapshot rerun; v2 registered; gate decision in `promotion.json` | AC-12 |
| TC-SYS-06 | Pipeline Console → Promote v2, then Roll back | champion 2 then 1; model server switches within 60 s | AC-12 |
| TC-SYS-07 | `scripts/replay.sh` | live accuracy / log-loss panels fill in Grafana | AC-10 |
| TC-SYS-08 | `scripts/replay.sh --inject-drift`, wait 3 min | PSI(is_grass) > 0.2; `FeatureDrift` firing; run with reason `alert:FeatureDrift` | AC-11 |
| TC-SYS-09 | `scripts/load_test.sh --concurrency 4 --p95-limit-ms 1000` | passed = true | AC-7, AC-8 |
| TC-SYS-10 | Prometheus → Status → Targets | all targets UP | AC-10 |
| TC-SYS-11 | `docker compose stop model-server`, use the UI | friendly "model not available" message; `ModelServerDown` alert after 2 min; restart recovers | AC-9 |
| TC-SYS-12 | set an invalid base URL in `params.yaml`, run | `ingest` fails; console shows the failed stage and log tail; `PipelineStageFailed` alert | robustness |
| TC-SYS-13 | `git checkout <commit> && dvc pull && dvc repro` | same `metrics.json` as the MLflow run with that `git_commit` tag | reproducibility |

### 5.4 UI (manual)

| ID | Steps | Expected |
|---|---|---|
| TC-UI-10 | Predict: choose ATP, two players, Clay, Grand Slam, F → Predict | two probabilities summing to 100 %, Elo-only comparison, 3 factors, head-to-head |
| TC-UI-11 | Predict with the same player twice | Predict button disabled |
| TC-UI-12 | Player Profile for a top player | Elo history chart, surface ratings, recent matches |
| TC-UI-13 | Head-to-Head for two players | record and list of meetings |
| TC-UI-14 | Pipeline Console | DAG coloured by status, stage durations and rows/s, run history, MLflow runs |
| TC-UI-15 | Monitoring page | summary numbers + embedded Grafana panels |
| TC-UI-16 | Stop the backend, reload the UI | sidebar shows "Service unreachable", no stack trace |

## 6. Defect handling
Defects found during testing are logged with severity (High / Medium / Low), the failing test ID, the
Git commit and the run log. A fix is accepted when the failing test passes and all other tests still
pass. Defects found so far are listed in `test_report.md`.
