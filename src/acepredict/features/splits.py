"""Time-based train / validation / test splits (never random: no future leaks into training)."""

from __future__ import annotations

from typing import Any


def resolve_splits(years_available: list[int], params: dict[str, Any]) -> dict[str, list[int]]:
    """Return the years in each split, falling back gracefully when recent years are missing.

    If the configured test years have no data (e.g. a season not yet published), the last
    two available years become the test set and the two before them the validation set.
    """
    sp = params["splits"]
    years = sorted(set(int(y) for y in years_available))
    if not years:
        raise ValueError("no years available to split")
    test = [y for y in sp["test_years"] if y in years]
    val = [y for y in sp["val_years"] if y in years]
    if not test or not val:
        if len(years) < 5:
            raise ValueError(f"need at least 5 seasons of data, have {years}")
        test = years[-2:]
        val = years[-4:-2]
    first_held_out = min(val + test)
    lo, hi = int(sp["train_start"]), int(sp["train_end"])
    train = [y for y in years if lo <= y <= hi and y < first_held_out]
    if not train:
        train = [y for y in years if y < first_held_out][-5:]
    return {"train": train, "val": val, "test": test}


def split_of(year: int, splits: dict[str, list[int]]) -> str:
    """Name of the split a season belongs to ('none' = only used for Elo warm-up)."""
    for name in ("train", "val", "test"):
        if year in splits[name]:
            return name
    return "none"
