"""Backend REST API with a fake model server and registry (TC-API-*)."""

import pytest


def two_players(client, tour="atp"):
    rows = client.get("/players", params={"tour": tour, "limit": 5}).json()
    return rows[0]["id"], rows[1]["id"]


def test_health_and_ready(app_client):
    assert app_client.get("/health").json() == {"status": "ok"}
    ready = app_client.get("/ready")
    assert ready.status_code == 200 and ready.json()["ready"] is True


def test_ready_is_503_when_model_server_down(app_client):
    app_client.fake_model.up = False
    r = app_client.get("/ready")
    assert r.status_code == 503 and r.json()["ready"] is False


def test_player_search_sorted_by_elo(app_client):
    rows = app_client.get("/players", params={"tour": "wta", "limit": 10}).json()
    assert rows and all(r["tour"] == "wta" for r in rows)
    elos = [r["elo"] for r in rows]
    assert elos == sorted(elos, reverse=True)


def test_player_profile(app_client):
    pid, _ = two_players(app_client)
    body = app_client.get(f"/players/{pid}").json()
    assert body["id"] == pid and body["elo_history"] and "recent_matches" in body


def test_unknown_player_is_404_with_error_code(app_client):
    r = app_client.get("/players/atp_999999999")
    assert r.status_code == 404
    assert r.json()["error_code"] == "PLAYER_NOT_FOUND" and r.json()["request_id"]


def test_predict_returns_complementary_probabilities(app_client):
    a, b = two_players(app_client)
    r = app_client.post("/predict", json={"tour": "atp", "player1_id": a, "player2_id": b,
                                          "surface": "Clay"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["p1_win_prob"] + body["p2_win_prob"] == pytest.approx(1.0)
    assert 0 < body["elo_only_p1_prob"] < 1
    assert body["model_version"] == "3" and body["top_factors"]


@pytest.mark.parametrize("payload,code", [
    ({"tour": "atp", "player1_id": "SAME", "player2_id": "SAME"}, "INVALID_MATCHUP"),
    ({"tour": "atp", "player1_id": "atp_1", "player2_id": "atp_2", "surface": "Ice"},
     "VALIDATION_ERROR"),
    ({"tour": "atp", "player1_id": "a"}, "VALIDATION_ERROR"),
])
def test_bad_predict_requests_are_400(app_client, payload, code):
    if payload.get("player1_id") == "SAME":
        a, _ = two_players(app_client)
        payload = {**payload, "player1_id": a, "player2_id": a}
    r = app_client.post("/predict", json=payload)
    assert r.status_code == 400 and r.json()["error_code"] == code


def test_predict_across_tours_is_rejected(app_client):
    a, _ = two_players(app_client, "atp")
    w, _ = two_players(app_client, "wta")
    r = app_client.post("/predict", json={"tour": "atp", "player1_id": a, "player2_id": w})
    assert r.status_code == 400


def test_predict_when_model_down_is_503(app_client):
    a, b = two_players(app_client)
    app_client.fake_model.up = False
    r = app_client.post("/predict", json={"tour": "atp", "player1_id": a, "player2_id": b})
    assert r.status_code == 503 and r.json()["error_code"] == "MODEL_UNAVAILABLE"


def test_batch_uses_one_model_call(app_client):
    a, b = two_players(app_client)
    items = [{"tour": "atp", "player1_id": a, "player2_id": b}] * 4
    before = app_client.fake_model.calls
    r = app_client.post("/predict/batch", json={"items": items})
    assert r.status_code == 200 and len(r.json()["results"]) == 4
    assert app_client.fake_model.calls == before + 1


def test_outcomes_update_live_accuracy(app_client):
    a, b = two_players(app_client)
    for _ in range(6):
        rid = app_client.post("/predict", json={"tour": "atp", "player1_id": a,
                                                "player2_id": b}).json()["request_id"]
        out = app_client.post("/outcomes", json={"request_id": rid, "p1_won": True}).json()
    assert out["recorded"] and out["window"] == 6 and out["live_accuracy"] is not None
    assert app_client.post("/outcomes", json={"request_id": "nope", "p1_won": True}).status_code \
        == 404
    summary = app_client.get("/monitoring/summary").json()
    assert summary


def test_admin_endpoints_need_api_key(app_client):
    assert app_client.post("/admin/pipeline/run", json={}).status_code == 401
    r = app_client.post("/admin/pipeline/run", json={"reason": "test"},
                        headers={"X-API-Key": "test-admin"})
    assert r.status_code == 202
    busy = app_client.post("/admin/pipeline/run", json={}, headers={"X-API-Key": "test-admin"})
    assert busy.status_code == 409 and busy.json()["error_code"] == "PIPELINE_BUSY"


def test_promote_and_rollback(app_client):
    h = {"X-API-Key": "test-admin"}
    assert app_client.post("/admin/model/promote", json={"version": "4"},
                           headers=h).json()["champion_version"] == "4"
    assert app_client.post("/admin/model/rollback", headers=h).json()["champion_version"] == "3"


def test_alertmanager_webhook_triggers_retraining(app_client):
    alert = {"status": "firing", "alerts": [{"status": "firing",
                                             "labels": {"alertname": "FeatureDrift",
                                                        "action": "retrain"}}]}
    assert app_client.post("/webhooks/alertmanager", json=alert).status_code == 401
    r = app_client.post("/webhooks/alertmanager", json=alert,
                        headers={"Authorization": "Bearer test-hook"})
    assert r.status_code == 200 and r.json()["triggered"] is True


def test_pipeline_status_has_dag(app_client):
    body = app_client.get("/pipeline/status").json()
    assert "ingest" in body["dag_dot"] and "snapshot" in body["dag_dot"]


def test_metrics_endpoint_exposes_custom_metrics(app_client):
    a, b = two_players(app_client)
    app_client.post("/predict", json={"tour": "atp", "player1_id": a, "player2_id": b})
    text = app_client.get("/metrics").text
    assert "acepredict_predictions_total" in text and "http_request_duration_seconds" in text
