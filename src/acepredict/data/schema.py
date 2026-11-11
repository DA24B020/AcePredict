"""Schema of Jeff Sackmann's match CSVs and the shared cleaning rules.

The same rules are applied by the validation stage (pandas) and the Spark transform,
so the validation report describes exactly what the transform drops.
"""

from __future__ import annotations

MATCH_COLUMNS: list[str] = [
    "tourney_id", "tourney_name", "surface", "draw_size", "tourney_level", "tourney_date",
    "match_num", "winner_id", "winner_seed", "winner_entry", "winner_name", "winner_hand",
    "winner_ht", "winner_ioc", "winner_age", "loser_id", "loser_seed", "loser_entry",
    "loser_name", "loser_hand", "loser_ht", "loser_ioc", "loser_age", "score", "best_of",
    "round", "minutes", "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "w_SvGms", "w_bpSaved", "w_bpFaced", "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon",
    "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced", "winner_rank", "winner_rank_points",
    "loser_rank", "loser_rank_points",
]  # fmt: skip

# Columns the pipeline cannot work without (critical validation check).
REQUIRED_COLUMNS: list[str] = [
    "tourney_id", "tourney_name", "surface", "tourney_level", "tourney_date", "match_num",
    "winner_id", "winner_name", "loser_id", "loser_name", "score", "best_of", "round",
]  # fmt: skip

SURFACES: list[str] = ["Hard", "Clay", "Grass"]
SURFACE_INDEX: dict[str, int] = {s: i for i, s in enumerate(SURFACES)}

# Carpet was discontinued in 2009; it plays closest to indoor hard courts.
SURFACE_MAP: dict[str, str] = {"Hard": "Hard", "Clay": "Clay", "Grass": "Grass", "Carpet": "Hard"}

# Order of rounds inside one tournament. tourney_date is the tournament's start date,
# so this order is what keeps matches in their true sequence.
ROUND_ORDER: dict[str, int] = {
    "Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0,
    "R128": 1, "R64": 2, "R32": 3, "R16": 4, "RR": 5, "ER": 5,
    "QF": 6, "SF": 7, "BR": 8, "F": 9,
}  # fmt: skip
DEFAULT_ROUND_ORDER = 5

# A walkover means no match was played: these rows are dropped.
WALKOVER_PATTERN = r"(?i)(?:W/O|walkover|unfinished)"
# Retirements and defaults were played (partly): kept, but flagged.
RETIRED_PATTERN = r"(?i)(RET|DEF)"

GRAND_SLAM_LEVELS = {"G"}
TOP_TIER_LEVELS = {"M", "PM", "P", "F", "W"}  # ATP Masters, WTA Premier/1000, Finals

AGE_RANGE = (13.0, 50.0)
HEIGHT_RANGE = (140.0, 230.0)


def round_order(round_code: str | None) -> int:
    """Map a round code (e.g. 'QF') to its position in the tournament."""
    if not round_code:
        return DEFAULT_ROUND_ORDER
    return ROUND_ORDER.get(str(round_code).strip(), DEFAULT_ROUND_ORDER)


def player_key(tour: str, player_id: object) -> str:
    """Globally unique player key; ATP and WTA ids are prefixed so they can never collide."""
    pid = str(player_id).strip()
    if pid.endswith(".0"):
        pid = pid[:-2]
    return f"{tour}_{pid}"


def raw_file_name(tour: str, year: int) -> str:
    """File name used by the Sackmann repositories, e.g. atp_matches_2024.csv."""
    return f"{tour}_matches_{year}.csv"
