"""PredictionService: validate a matchup, build features, call the model server, explain."""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any

import pandas as pd

from acepredict.exceptions import InvalidMatchupError
from acepredict.features.build import FEATURE_LABELS, context_frame, pair_features
from acepredict.ratings.elo import predict_from_ratings
from app.clients.model_server import ModelServerClient
from app.metrics import PREDICTED_PROB, PREDICTIONS
from app.schemas import PredictRequest
from app.services.monitor import MonitorService
from app.services.registry import RegistryService
from app.services.snapshot import SnapshotStore

log = logging.getLogger(__name__)

LEVEL_CODES = {"grand_slam": "G", "masters": "M", "tour": "A"}
ROUND_CODES = {"early": 3, "QF": 6, "SF": 7, "F": 9}
FEW_MATCHES = 20
LONG_BREAK_DAYS = 180


class PredictionService:
    def __init__(self, store: SnapshotStore, model: ModelServerClient,
                 monitor: MonitorService, registry: RegistryService) -> None:
        self.store = store
        self.model = model
        self.monitor = monitor
        self.registry = registry

    def _prepare(self, req: PredictRequest) -> dict[str, Any]:
        self.store.require()
        if req.player1_id == req.player2_id:
            raise InvalidMatchupError("Choose two different players.")
        p1 = self.store.get(req.player1_id)
        p2 = self.store.get(req.player2_id)
        if p1["tour"] != p2["tour"]:
            raise InvalidMatchupError("Both players must be on the same tour (ATP or WTA).")
        if p1["tour"] != req.tour:
            raise InvalidMatchupError(f"These players are on the {p1['tour'].upper()} tour, "
                                      f"not {req.tour.upper()}.")
        match_date = req.match_date or date.today()
        a = self.store.feature_attrs(p1, req.surface, match_date, req.player2_id)
        b = self.store.feature_attrs(p2, req.surface, match_date, req.player1_id)
        w1, w2 = self.store.h2h_record(req.player1_id, req.player2_id)
        context = {"surface": req.surface, "best_of": req.best_of,
                   "tourney_level": LEVEL_CODES[req.tourney_level],
                   "round_order": ROUND_CODES[req.round], "tour": req.tour,
                   "h2h_matches": w1 + w2}
        feats = pair_features(context_frame(a, b, context), self.store.surface_weight)
        warnings = []
        for name, attrs, row in ((p1["name"], a, p1), (p2["name"], b, p2)):
            if attrs["n"] < FEW_MATCHES:
                warnings.append(f"{name} has played only {int(attrs['n'])} recorded matches; "
                                "their rating is uncertain.")
            if attrs["days_since"] is not None and attrs["days_since"] > LONG_BREAK_DAYS:
                warnings.append(f"{name}'s last recorded match was {attrs['days_since']} days "
                                "ago; form data may be stale.")
            if int(row[f"matches_{req.surface.lower()}"]) < 5:
                warnings.append(f"{name} has few matches on {req.surface.lower()} courts.")
        elo_only = predict_from_ratings(a["elo"], b["elo"], a["selo"], b["selo"],
                                        self.store.surface_weight)
        return {"p1": p1, "p2": p2, "features": feats, "warnings": warnings,
                "elo_only": elo_only, "h2h": {"p1_wins": w1, "p2_wins": w2}}

    def _response(self, req: PredictRequest, prep: dict[str, Any], pred: dict[str, Any],
                  request_id: str, version: str | None) -> dict[str, Any]:
        prob = float(pred["prob_a"])
        try:
            factors = json.loads(pred.get("explanation") or "[]")
        except ValueError:
            factors = []
        for f in factors:
            f.setdefault("label", FEATURE_LABELS.get(f.get("feature", ""), f.get("feature")))
        PREDICTIONS.labels(req.tour, req.surface).inc()
        PREDICTED_PROB.observe(prob)
        self.monitor.record_prediction(
            request_id, prep["features"].iloc[0].to_dict(), prob,
            {"tour": req.tour, "surface": req.surface, "player1": req.player1_id,
             "player2": req.player2_id, "model_version": version})
        return {
            "request_id": request_id,
            "player1": self.store.summary(prep["p1"]),
            "player2": self.store.summary(prep["p2"]),
            "surface": req.surface,
            "p1_win_prob": round(prob, 4),
            "p2_win_prob": round(1.0 - prob, 4),
            "elo_only_p1_prob": round(prep["elo_only"], 4),
            "model_version": version,
            "top_factors": factors,
            "warnings": prep["warnings"],
            "head_to_head": prep["h2h"],
        }

    def predict(self, req: PredictRequest, request_id: str) -> dict[str, Any]:
        prep = self._prepare(req)
        pred = self.model.predict(prep["features"])[0]
        version = self.registry.model_info().get("champion_version")
        return self._response(req, prep, pred, request_id, version)

    def predict_batch(self, reqs: list[PredictRequest], request_id: str) -> list[dict[str, Any]]:
        preps = [self._prepare(r) for r in reqs]
        feats = pd.concat([p["features"] for p in preps], ignore_index=True)
        preds = self.model.predict(feats)
        version = self.registry.model_info().get("champion_version")
        return [self._response(r, p, pred, f"{request_id}-{i}", version)
                for i, (r, p, pred) in enumerate(zip(reqs, preps, preds, strict=True))]
