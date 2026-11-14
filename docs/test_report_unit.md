# AcePredict — Test Report

Generated 2026-11-14 13:25 UTC from `junit.xml`.

**Total 45 · passed 45 · failed 0 · skipped 0**

## Load test

270 requests in 30.2 s with 4 clients: **9.0 req/s**, p50 430.5 ms, p95 734.3 ms (limit 1000 ms) → PASS

## Test cases

| Module | Test | Result | Time (s) |
|---|---|---|---|
| tests.unit.test_api | test_health_and_ready | passed | 6.18 |
| tests.unit.test_api | test_ready_is_503_when_model_server_down | passed | 0.93 |
| tests.unit.test_api | test_player_search_sorted_by_elo | passed | 1.09 |
| tests.unit.test_api | test_player_profile | passed | 1.77 |
| tests.unit.test_api | test_unknown_player_is_404_with_error_code | passed | 1.10 |
| tests.unit.test_api | test_predict_returns_complementary_probabilities | passed | 1.81 |
| tests.unit.test_api | test_bad_predict_requests_are_400[payload0-INVALID_MATCHUP] | passed | 0.75 |
| tests.unit.test_api | test_bad_predict_requests_are_400[payload1-VALIDATION_ERROR] | passed | 0.26 |
| tests.unit.test_api | test_bad_predict_requests_are_400[payload2-VALIDATION_ERROR] | passed | 0.28 |
| tests.unit.test_api | test_predict_across_tours_is_rejected | passed | 0.28 |
| tests.unit.test_api | test_predict_when_model_down_is_503 | passed | 0.23 |
| tests.unit.test_api | test_batch_uses_one_model_call | passed | 0.43 |
| tests.unit.test_api | test_outcomes_update_live_accuracy | passed | 0.62 |
| tests.unit.test_api | test_admin_endpoints_need_api_key | passed | 0.23 |
| tests.unit.test_api | test_promote_and_rollback | passed | 0.22 |
| tests.unit.test_api | test_alertmanager_webhook_triggers_retraining | passed | 0.27 |
| tests.unit.test_api | test_pipeline_status_has_dag | passed | 0.35 |
| tests.unit.test_api | test_metrics_endpoint_exposes_custom_metrics | passed | 0.29 |
| tests.unit.test_elo | test_expected_score_equal_ratings_is_half | passed | 0.00 |
| tests.unit.test_elo | test_expected_score_is_complementary | passed | 0.00 |
| tests.unit.test_elo | test_400_points_is_ten_to_one | passed | 0.00 |
| tests.unit.test_elo | test_k_factor_decreases_with_experience | passed | 0.00 |
| tests.unit.test_elo | test_winner_gains_and_loser_drops_same_amount_for_newcomers | passed | 0.00 |
| tests.unit.test_elo | test_pre_match_rating_never_includes_the_match_itself | passed | 0.00 |
| tests.unit.test_elo | test_surface_rating_only_moves_on_that_surface | passed | 0.00 |
| tests.unit.test_elo | test_elo_system_state_and_history | passed | 0.02 |
| tests.unit.test_elo | test_predict_from_ratings_symmetric | passed | 0.00 |
| tests.unit.test_features | test_feature_lists_cover_every_feature_once | passed | 0.00 |
| tests.unit.test_features | test_pair_features_has_no_nan_and_right_columns | passed | 0.06 |
| tests.unit.test_features | test_swapping_players_negates_antisymmetric_and_keeps_context | passed | 0.12 |
| tests.unit.test_features | test_pyfunc_probabilities_are_exactly_symmetric | passed | 0.08 |
| tests.unit.test_features | test_pyfunc_explanations_are_top_factors | passed | 0.04 |
| tests.unit.test_features | test_splits_use_config_when_years_exist | passed | 0.00 |
| tests.unit.test_features | test_splits_fall_back_when_recent_seasons_missing | passed | 0.00 |
| tests.unit.test_frontend_client | test_backend_url_comes_from_env | passed | 0.02 |
| tests.unit.test_frontend_client | test_unreachable_backend_gives_friendly_error | passed | 0.01 |
| tests.unit.test_frontend_client | test_error_codes_map_to_plain_messages | passed | 0.01 |
| tests.unit.test_validate_and_metrics | test_sample_data_passes_validation | passed | 0.35 |
| tests.unit.test_validate_and_metrics | test_validation_flags_missing_columns | passed | 0.01 |
| tests.unit.test_validate_and_metrics | test_validation_counts_bad_rows | passed | 0.04 |
| tests.unit.test_validate_and_metrics | test_schema_helpers | passed | 0.00 |
| tests.unit.test_validate_and_metrics | test_prob_metrics_perfect_and_coinflip | passed | 0.03 |
| tests.unit.test_validate_and_metrics | test_ece_of_calibrated_predictions_is_small | passed | 0.01 |
| tests.unit.test_validate_and_metrics | test_psi_same_distribution_is_near_zero_and_shift_is_large | passed | 0.01 |
| tests.unit.test_validate_and_metrics | test_psi_report_only_uses_shared_columns | passed | 0.01 |
