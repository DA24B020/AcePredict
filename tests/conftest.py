"""Shared fixtures: a tiny snapshot, a fake model server and a fake MLflow registry."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"


class FakeModelServer:
    """Stands in for `mlflow models serve`: logistic on the Elo difference."""

    def __init__(self, up: bool = True) -> None:
        self.up = up
        self.calls = 0

    def ping(self, max_age_s: float = 5.0) -> bool:
        return self.up

    def predict(self, features: pd.DataFrame) -> list[dict]:
        from acepredict.exceptions import ModelUnavailableError

        if not self.up:
            raise ModelUnavailableError("model server down (test)")
        self.calls += 1
        p = 1.0 / (1.0 + np.exp(-features["elo_diff"].to_numpy() / 200.0))
        return [{"prob_a": float(x),
                 "explanation": '[{"feature": "elo_diff", "label": "Elo rating difference", '
                                '"contribution": 0.5, "favours": "player1"}]'} for x in p]


class FakeRegistry:
    def __init__(self) -> None:
        self.champion = "3"
        self.previous = "2"

    def model_info(self, use_cache: bool = True) -> dict:
        return {"model_name": "acepredict", "champion_version": self.champion,
                "previous_version": self.previous, "test_metrics": {"log_loss": 0.6}}

    def promote(self, version: str) -> dict:
        self.previous, self.champion = self.champion, version
        return {"champion_version": self.champion, "previous_version": self.previous}

    def rollback(self) -> dict:
        self.champion, self.previous = self.previous, None
        return {"champion_version": self.champion, "previous_version": None}

    def recent_runs(self, limit: int = 10) -> list:
        return []


@pytest.fixture()
def snapshot_dir(tmp_path: Path) -> Path:
    dst = tmp_path / "snapshot"
    shutil.copytree(FIXTURES / "snapshot", dst)
    return dst


@pytest.fixture()
def players(snapshot_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(snapshot_dir / "players.parquet")


@pytest.fixture()
def app_client(tmp_path: Path, snapshot_dir: Path):
    """FastAPI TestClient wired to the fixture snapshot and the fakes."""
    from fastapi.testclient import TestClient

    from app.container import Services
    from app.main import create_app
    from app.settings import Settings

    repo = Path(__file__).resolve().parents[1]
    settings = Settings(acepredict_root=str(repo), snapshot_dir=str(snapshot_dir),
                        runtime_dir=str(tmp_path / "runtime"), admin_api_key="test-admin",
                        webhook_token="test-hook", monitor_interval_s=3600,
                        drift_min_samples=5, live_window=50)
    model, registry = FakeModelServer(), FakeRegistry()
    services = Services.build(settings, model=model, registry=registry)
    app = create_app(settings, services)
    with TestClient(app) as client:
        client.fake_model = model
        client.services = services
        yield client
