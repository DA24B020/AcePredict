"""Player search, profiles and head-to-head records."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.container import Services, get_services
from app.schemas import ErrorResponse, H2HResponse, PlayerDetail, PlayerSummary, Tour

router = APIRouter(tags=["players"])
ERRORS = {404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}}


@router.get("/players", response_model=list[PlayerSummary], responses=ERRORS,
            summary="Search players (sorted by Elo)")
def list_players(
    q: str | None = Query(default=None, max_length=50, description="Part of the name"),
    tour: Tour | None = None,
    limit: int = Query(default=50, ge=1, le=2000),
    active_only: bool = True,
    svc: Services = Depends(get_services),
):
    return svc.store.search(q, tour, limit, active_only)


@router.get("/players/{player_id}", response_model=PlayerDetail, responses=ERRORS,
            summary="Player profile with Elo history and recent results")
def player_detail(player_id: str, svc: Services = Depends(get_services)):
    row = svc.store.get(player_id)
    base = svc.store.summary(row)

    def num(key: str):
        v = row.get(key)
        return None if v is None or v != v else round(float(v), 3)

    return {
        **base,
        "age": num("age"),
        "height_cm": num("ht"),
        "wins": int(row["wins"]),
        "losses": int(row["losses"]),
        "form_last10": num("form10"),
        "form_last52w": num("form52"),
        "elo_history": svc.store.elo_history(player_id),
        "recent_matches": svc.store.recent_matches(player_id),
    }


@router.get("/h2h", response_model=H2HResponse, responses={400: {"model": ErrorResponse},
                                                           **ERRORS},
            summary="Head-to-head record and past meetings")
def head_to_head(p1: str, p2: str, svc: Services = Depends(get_services)):
    from acepredict.exceptions import InvalidMatchupError

    if p1 == p2:
        raise InvalidMatchupError("Choose two different players.")
    r1, r2 = svc.store.get(p1), svc.store.get(p2)
    w1, w2 = svc.store.h2h_record(p1, p2)
    return {"player1": svc.store.summary(r1), "player2": svc.store.summary(r2),
            "p1_wins": w1, "p2_wins": w2, "meetings": svc.store.meetings(p1, p2)}
