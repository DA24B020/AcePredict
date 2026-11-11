"""Request and response models. They generate the OpenAPI spec served at /docs."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field

Tour = Literal["atp", "wta"]
Surface = Literal["Hard", "Clay", "Grass"]
Level = Literal["grand_slam", "masters", "tour"]
Round = Literal["early", "QF", "SF", "F"]


class ErrorResponse(BaseModel):
    error_code: str = Field(examples=["PLAYER_NOT_FOUND"])
    message: str
    request_id: str | None = None
    details: Any | None = None


class Health(BaseModel):
    status: Literal["ok"] = "ok"


class Ready(BaseModel):
    ready: bool
    snapshot_loaded: bool
    snapshot_date: str | None
    model_server: bool
    model_version: str | None


class PlayerSummary(BaseModel):
    id: str = Field(description="Player key, e.g. 'atp_104925'")
    name: str
    tour: Tour
    country: str | None = None
    hand: str | None = None
    elo: float
    elo_hard: float
    elo_clay: float
    elo_grass: float
    rank: int | None = None
    matches: int
    active: bool
    last_match_date: str | None = None


class EloPoint(BaseModel):
    month: str
    elo: float
    elo_hard: float | None = None
    elo_clay: float | None = None
    elo_grass: float | None = None


class MatchResult(BaseModel):
    date: str
    tournament: str
    surface: str
    round: str | None
    opponent: str
    opponent_id: str
    won: bool
    score: str | None


class PlayerDetail(PlayerSummary):
    age: float | None = None
    height_cm: float | None = None
    wins: int
    losses: int
    form_last10: float | None = None
    form_last52w: float | None = None
    elo_history: list[EloPoint]
    recent_matches: list[MatchResult]


class Meeting(BaseModel):
    date: str
    tournament: str
    surface: str
    round: str | None
    winner_id: str
    score: str | None


class H2HResponse(BaseModel):
    player1: PlayerSummary
    player2: PlayerSummary
    p1_wins: int
    p2_wins: int
    meetings: list[Meeting]


class PredictRequest(BaseModel):
    tour: Tour
    player1_id: str = Field(min_length=3, max_length=40, examples=["atp_104925"])
    player2_id: str = Field(min_length=3, max_length=40, examples=["atp_106421"])
    surface: Surface = "Hard"
    best_of: Literal[3, 5] = 3
    tourney_level: Level = "tour"
    round: Round = "early"
    match_date: date | None = Field(default=None, description="Defaults to today")


class Factor(BaseModel):
    feature: str
    label: str
    contribution: float
    favours: Literal["player1", "player2"]


class PredictResponse(BaseModel):
    request_id: str
    player1: PlayerSummary
    player2: PlayerSummary
    surface: Surface
    p1_win_prob: float
    p2_win_prob: float
    elo_only_p1_prob: float
    model_version: str | None
    top_factors: list[Factor]
    warnings: list[str]
    head_to_head: dict[str, int]


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest] = Field(min_length=1)


class BatchPredictResponse(BaseModel):
    request_id: str
    results: list[PredictResponse]


class OutcomeRequest(BaseModel):
    request_id: str = Field(description="request_id returned by /predict")
    p1_won: bool


class OutcomeResponse(BaseModel):
    recorded: bool
    live_accuracy: float | None
    live_log_loss: float | None
    window: int


class ModelInfo(BaseModel):
    model_name: str
    champion_version: str | None
    previous_version: str | None = None
    challenger_version: str | None = None
    run_id: str | None = None
    model_type: str | None = None
    git_commit: str | None = None
    created_at: str | None = None
    test_metrics: dict[str, float] = {}


class StageStatus(BaseModel):
    name: str
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float | None = None
    rows_in: int | None = None
    rows_out: int | None = None
    rows_per_s: float | None = None
    error: str | None = None


class PipelineRun(BaseModel):
    run_id: str
    reason: str
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float | None = None


class MlflowRun(BaseModel):
    run_id: str
    name: str | None
    experiment: str
    status: str
    start_time: str | None
    duration_s: float | None
    metrics: dict[str, float]


class PipelineStatus(BaseModel):
    running: bool
    queued: int
    stages: list[StageStatus]
    runs: list[PipelineRun]
    dag_dot: str
    failed_log_tail: list[str]
    tuning: dict[str, Any] | None
    validation: dict[str, Any] | None
    metrics: dict[str, Any] | None
    mlflow_runs: list[MlflowRun]


class RunRequest(BaseModel):
    reason: str = Field(default="manual", max_length=60)


class RunAccepted(BaseModel):
    run_id: str
    queued_at: str


class PromoteRequest(BaseModel):
    version: str = Field(min_length=1, max_length=10)


class AliasResponse(BaseModel):
    champion_version: str
    previous_version: str | None


class WebhookResponse(BaseModel):
    accepted: bool
    triggered: bool
    detail: str


class MonitoringSummary(BaseModel):
    predictions_logged: int
    outcomes_in_window: int
    live_accuracy: float | None
    live_log_loss: float | None
    live_brier: float | None
    psi: dict[str, float]
    drift_sample_size: int
    drift_threshold: float = 0.2
