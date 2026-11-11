"""Synthetic tennis data in exactly the same CSV format as Jeff Sackmann's repositories.

Used for tests, for offline development, and as a fallback when GitHub is unreachable
(set ``data.source: sample`` in params.yaml). Players have a hidden skill that rises and
falls with age plus a per-surface offset, so Elo and the model have real signal to learn.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from acepredict.data.schema import MATCH_COLUMNS, raw_file_name

FIRST = ["Alex", "Ben", "Carlos", "Dan", "Emil", "Felix", "Gael", "Hugo", "Ivan", "Jan", "Kei",
         "Leo", "Marco", "Nico", "Omar", "Pablo", "Rafa", "Sam", "Tomas", "Ugo", "Viktor", "Will",
         "Yuri", "Zed", "Ana", "Bea", "Cara", "Dina", "Elena", "Fia", "Gaby", "Hana", "Iga", "Jas",
         "Kim", "Lia", "Maya", "Nina", "Ola", "Paula"]  # fmt: skip
LAST = ["Arden", "Brook", "Costa", "Dimov", "Evans", "Ferro", "Garin", "Hale", "Ito", "Jarry",
        "Kova", "Lopez", "Moreau", "Novak", "Ortiz", "Petrov", "Quinn", "Rossi", "Silva", "Tanaka",
        "Ueda", "Vargas", "Weber", "Xu", "Young", "Zane", "Berg", "Cruz", "Dahl", "Ek"]  # fmt: skip
COUNTRIES = ["ESP", "FRA", "ITA", "USA", "GER", "ARG", "AUS", "SRB", "RUS", "GBR", "JPN", "IND",
             "CZE", "POL", "CAN", "SUI"]  # fmt: skip


@dataclass(frozen=True)
class Event:
    code: str
    name: str
    surface: str
    level: str
    draw: int
    month: int
    day: int


def _calendar(tour: str) -> list[Event]:
    slam = "G"
    top = "M" if tour == "atp" else "PM"
    small = "A" if tour == "atp" else "I"
    return [
        Event("580", "Australian Open", "Hard", slam, 64, 1, 15),
        Event("403", "Miami Masters", "Hard", top, 32, 3, 20),
        Event("404", "Indian Wells", "Hard", top, 32, 3, 6),
        Event("1536", "Madrid", "Clay", top, 32, 5, 1),
        Event("416", "Rome", "Clay", top, 32, 5, 12),
        Event("520", "Roland Garros", "Clay", slam, 64, 5, 26),
        Event("540", "Wimbledon", "Grass", slam, 64, 6, 30),
        Event("421", "Canada Masters", "Hard", top, 32, 8, 5),
        Event("422", "Cincinnati", "Hard", top, 32, 8, 12),
        Event("560", "US Open", "Hard", slam, 64, 8, 26),
        Event("301", "Auckland", "Hard", small, 16, 1, 3),
        Event("451", "Doha", "Hard", small, 16, 2, 10),
        Event("315", "Rio", "Clay", small, 16, 2, 17),
        Event("321", "Stuttgart", "Grass", small, 16, 6, 10),
        Event("500", "Halle", "Grass", small, 16, 6, 17),
        Event("316", "Bastad", "Clay", small, 16, 7, 14),
        Event("319", "Kitzbuhel", "Clay", small, 16, 7, 28),
        Event("341", "Tokyo", "Hard", small, 16, 10, 2),
        Event("352", "Vienna", "Hard", small, 16, 10, 21),
        Event("329", "Basel", "Carpet", small, 16, 10, 28),
    ]


ROUNDS_BY_DRAW = {
    64: ["R64", "R32", "R16", "QF", "SF", "F"],
    32: ["R32", "R16", "QF", "SF", "F"],
    16: ["R16", "QF", "SF", "F"],
}


class _Player:
    def __init__(self, pid: int, rng: np.random.Generator, start_year: int, end_year: int) -> None:
        self.pid = pid
        self.name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        self.hand = "L" if rng.random() < 0.15 else "R"
        self.ht = int(rng.normal(185, 7))
        self.ioc = str(rng.choice(COUNTRIES))
        self.birth = float(rng.uniform(start_year - 24, end_year - 18))
        self.base = float(rng.normal(0.0, 1.0))
        self.surf = {s: float(rng.normal(0.0, 0.35)) for s in ("Hard", "Clay", "Grass")}

    def age(self, t: float) -> float:
        return t - self.birth

    def skill(self, t: float, surface: str) -> float:
        a = self.age(t)
        surface = "Hard" if surface == "Carpet" else surface
        return self.base - 0.012 * (a - 26.0) ** 2 + self.surf[surface]


def _score(rng: np.random.Generator, best_of: int) -> str:
    sets_needed = 3 if best_of == 5 else 2
    lost = int(rng.integers(0, sets_needed))
    sets = [f"6-{int(rng.integers(0, 5))}" for _ in range(sets_needed)]
    for _ in range(lost):
        sets.insert(int(rng.integers(0, len(sets))), f"{int(rng.integers(0, 5))}-6")
    return " ".join(sets)


def _serve_stats(rng: np.random.Generator, edge: float) -> dict[str, int]:
    svpt = int(max(30, rng.normal(75, 15)))
    first_in = int(svpt * min(0.8, max(0.45, rng.normal(0.62, 0.05))))
    first_won = int(first_in * min(0.9, max(0.5, rng.normal(0.70 + 0.04 * edge, 0.05))))
    second = svpt - first_in
    second_won = int(second * min(0.75, max(0.3, rng.normal(0.50 + 0.03 * edge, 0.06))))
    bp_faced = int(rng.poisson(max(0.5, 6 - 2 * edge)))
    bp_saved = int(rng.binomial(bp_faced, min(0.95, max(0.2, 0.6 + 0.05 * edge))))
    return {
        "ace": int(rng.binomial(svpt, min(0.25, max(0.01, 0.06 + 0.02 * edge)))),
        "df": int(rng.poisson(3)),
        "svpt": svpt,
        "1stIn": first_in,
        "1stWon": first_won,
        "2ndWon": second_won,
        "SvGms": int(max(6, svpt / 6.5)),
        "bpSaved": bp_saved,
        "bpFaced": bp_faced,
    }


def generate_tour(
    tour: str, start_year: int, end_year: int, seed: int = 7, n_players: int = 160
) -> dict[int, pd.DataFrame]:
    """Generate one DataFrame of matches per year for a tour."""
    rng = np.random.default_rng(seed + (0 if tour == "atp" else 1000))
    id_base = 100000 if tour == "atp" else 200000
    players = [_Player(id_base + i, rng, start_year, end_year) for i in range(n_players)]
    out: dict[int, pd.DataFrame] = {}
    for year in range(start_year, end_year + 1):
        t_mid = year + 0.5
        active = [p for p in players if 17.5 <= p.age(t_mid) <= 36.0]
        strength = {p.pid: p.skill(t_mid, "Hard") + rng.normal(0, 0.15) for p in active}
        ranked = sorted(active, key=lambda p: -strength[p.pid])
        rank = {p.pid: i + 1 for i, p in enumerate(ranked)}
        rows: list[dict] = []
        for ev in _calendar(tour):
            t = year + (ev.month - 1) / 12 + ev.day / 365
            pool = ranked[: min(len(ranked), int(ev.draw * 1.5))]
            if len(pool) < ev.draw:
                continue
            entrants = list(rng.choice(pool, size=ev.draw, replace=False))
            rng.shuffle(entrants)
            best_of = 5 if (ev.level == "G" and tour == "atp") else 3
            match_num = 0
            for rnd in ROUNDS_BY_DRAW[ev.draw]:
                winners = []
                for a, b in zip(entrants[0::2], entrants[1::2], strict=True):
                    match_num += 1
                    d = a.skill(t, ev.surface) - b.skill(t, ev.surface)
                    p_a = 1.0 / (1.0 + np.exp(-1.3 * d))
                    w, lsr = (a, b) if rng.random() < p_a else (b, a)
                    winners.append(w)
                    edge = float(np.clip(w.skill(t, ev.surface) - lsr.skill(t, ev.surface), -2, 2))
                    score = _score(rng, best_of)
                    u = rng.random()
                    if u < 0.004:
                        score = "W/O"
                    elif u < 0.02:
                        score = score + " RET"
                    ws, ls = _serve_stats(rng, edge), _serve_stats(rng, -edge)
                    stats_missing = rng.random() < 0.08
                    row = {
                        "tourney_id": f"{year}-{ev.code}",
                        "tourney_name": ev.name,
                        "surface": ev.surface,
                        "draw_size": ev.draw,
                        "tourney_level": ev.level,
                        "tourney_date": int(f"{year}{ev.month:02d}{ev.day:02d}"),
                        "match_num": match_num,
                        "winner_id": w.pid,
                        "winner_seed": "",
                        "winner_entry": "",
                        "winner_name": w.name,
                        "winner_hand": w.hand,
                        "winner_ht": w.ht,
                        "winner_ioc": w.ioc,
                        "winner_age": round(w.age(t), 1),
                        "loser_id": lsr.pid,
                        "loser_seed": "",
                        "loser_entry": "",
                        "loser_name": lsr.name,
                        "loser_hand": lsr.hand,
                        "loser_ht": lsr.ht,
                        "loser_ioc": lsr.ioc,
                        "loser_age": round(lsr.age(t), 1),
                        "score": score,
                        "best_of": best_of,
                        "round": rnd,
                        "minutes": int(rng.normal(100 if best_of == 3 else 160, 25)),
                        "winner_rank": rank.get(w.pid, ""),
                        "winner_rank_points": int(10000 / rank[w.pid]) if w.pid in rank else "",
                        "loser_rank": rank.get(lsr.pid, ""),
                        "loser_rank_points": int(10000 / rank[lsr.pid]) if lsr.pid in rank else "",
                    }
                    for prefix, st in (("w_", ws), ("l_", ls)):
                        for k, v in st.items():
                            row[prefix + k] = "" if stats_missing else v
                    rows.append(row)
                entrants = winners
        out[year] = pd.DataFrame(rows, columns=MATCH_COLUMNS)
    return out


def write_sample(
    raw_dir: Path, tours: list[str], start_year: int, end_year: int, seed: int = 7
) -> list[Path]:
    """Write synthetic CSVs as data/raw/<tour>/<tour>_matches_<year>.csv."""
    written: list[Path] = []
    for tour in tours:
        tour_dir = raw_dir / tour
        tour_dir.mkdir(parents=True, exist_ok=True)
        for year, df in generate_tour(tour, start_year, end_year, seed=seed).items():
            path = tour_dir / raw_file_name(tour, year)
            df.to_csv(path, index=False)
            written.append(path)
    return written
