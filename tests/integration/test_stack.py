"""End-to-end checks against the running docker compose stack (TC-E2E-*).

    ACEPREDICT_BASE_URL=http://localhost:8000 pytest -m integration
"""

import os

import pytest
import requests

pytestmark = pytest.mark.integration
BASE = os.environ.get("ACEPREDICT_BASE_URL", "http://localhost:8000").rstrip("/")


def test_ready():
    body = requests.get(f"{BASE}/ready", timeout=10).json()
    assert body["ready"], body


def test_end_to_end_prediction_and_outcome():
    players = requests.get(f"{BASE}/players", params={"tour": "atp", "limit": 2}, timeout=10).json()
    body = {"tour": "atp", "player1_id": players[0]["id"], "player2_id": players[1]["id"]}
    pred = requests.post(f"{BASE}/predict", json=body, timeout=10)
    assert pred.status_code == 200, pred.text
    out = requests.post(f"{BASE}/outcomes", json={"request_id": pred.json()["request_id"],
                                                  "p1_won": True}, timeout=10)
    assert out.json()["recorded"]


def test_model_info_has_champion():
    assert requests.get(f"{BASE}/model/info", timeout=10).json()["champion_version"]
