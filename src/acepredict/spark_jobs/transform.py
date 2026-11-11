"""Spark transformation: raw yearly CSVs -> ordered matches + per-player rolling features.

Outputs (Parquet):
* matches.parquet        one row per match before ``data.holdout_from``, with a global
                         chronological ``order_idx``
* live_matches.parquet   matches on/after the holdout date (the replayed "live" stream)
* player_match.parquet   one row per player per match with PRE-match features computed
                         by window functions whose frames end one row/day before the
                         current match, so a match never sees its own result (no leakage)
"""

from __future__ import annotations

import logging
import os
from functools import reduce
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from acepredict.data.schema import (
    AGE_RANGE,
    DEFAULT_ROUND_ORDER,
    HEIGHT_RANGE,
    RETIRED_PATTERN,
    ROUND_ORDER,
    SURFACE_MAP,
    WALKOVER_PATTERN,
)

log = logging.getLogger(__name__)

DAY = 1000  # ord_val = day_number * DAY + round_order  -> earlier rounds sort before later ones


def build_spark(params: dict[str, Any]) -> SparkSession:
    """Create the local Spark session configured from params.yaml (env can override master)."""
    sp = params["spark"]
    return (
        SparkSession.builder.appName("acepredict-transform")
        .master(os.environ.get("SPARK_MASTER", sp["master"]))
        .config("spark.driver.memory", sp["driver_memory"])
        .config("spark.sql.shuffle.partitions", str(sp["shuffle_partitions"]))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.ui.port", "4040")
        .getOrCreate()
    )


def _map_expr(mapping: dict[str, Any]):
    return F.create_map(*[F.lit(x) for kv in mapping.items() for x in kv])


def read_raw(spark: SparkSession, raw_dir: Path, tours: list[str]) -> DataFrame:
    """Read every yearly CSV as strings and union by column name (robust to column order)."""
    frames = []
    for tour in tours:
        for f in sorted((raw_dir / tour).glob("*.csv")):
            df = spark.read.option("header", True).option("mode", "PERMISSIVE").csv(str(f))
            frames.append(df.withColumn("tour", F.lit(tour)))
    if not frames:
        raise FileNotFoundError(f"no raw CSVs under {raw_dir}")
    return reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), frames)


def _key(tour_col, id_col):
    return F.concat_ws("_", tour_col, F.regexp_replace(F.trim(id_col), r"\.0$", ""))


def _num(col: str, lo: float | None = None, hi: float | None = None):
    c = F.col(col).cast("double")
    if lo is not None:
        c = F.when((c >= lo) & (c <= hi), c)
    return c


def clean_matches(df: DataFrame, levels_exclude: list[str]) -> DataFrame:
    """Apply the shared cleaning rules (same rules as the validation stage) and order matches."""
    df = (
        df.filter(F.col("winner_id").isNotNull() & F.col("loser_id").isNotNull())
        .filter(F.col("winner_id") != F.col("loser_id"))
        .withColumn("match_date", F.to_date(F.substring("tourney_date", 1, 8), "yyyyMMdd"))
        .filter(F.col("match_date").isNotNull())
        .filter(~F.coalesce(F.col("score"), F.lit("")).rlike(WALKOVER_PATTERN))
        .filter(~F.coalesce(F.col("tourney_level"), F.lit("")).isin(levels_exclude))
        .dropDuplicates(["tour", "tourney_id", "match_num"])
    )
    surface = F.coalesce(_map_expr(SURFACE_MAP)[F.col("surface")], F.lit("Hard"))
    rnd = F.coalesce(_map_expr(ROUND_ORDER)[F.trim(F.col("round"))], F.lit(DEFAULT_ROUND_ORDER))
    df = df.select(
        "tour",
        "tourney_id",
        "tourney_name",
        "tourney_level",
        "match_date",
        F.year("match_date").alias("year"),
        F.datediff("match_date", F.lit("1970-01-01")).alias("day_num"),
        F.col("match_num").cast("int").alias("match_num"),
        surface.alias("surface"),
        F.col("round").alias("round"),
        rnd.cast("int").alias("round_order"),
        F.coalesce(F.col("best_of").cast("int"), F.lit(3)).alias("best_of"),
        F.col("score"),
        F.coalesce(F.col("score"), F.lit("")).rlike(RETIRED_PATTERN).alias("retired"),
        _key(F.col("tour"), F.col("winner_id")).alias("winner_key"),
        _key(F.col("tour"), F.col("loser_id")).alias("loser_key"),
        F.col("winner_name"),
        F.col("loser_name"),
        F.col("winner_hand"),
        F.col("loser_hand"),
        F.col("winner_ioc"),
        F.col("loser_ioc"),
        _num("winner_ht", *HEIGHT_RANGE).alias("winner_ht"),
        _num("loser_ht", *HEIGHT_RANGE).alias("loser_ht"),
        _num("winner_age", *AGE_RANGE).alias("winner_age"),
        _num("loser_age", *AGE_RANGE).alias("loser_age"),
        _num("winner_rank").alias("winner_rank"),
        _num("loser_rank").alias("loser_rank"),
        _num("winner_rank_points").alias("winner_rank_points"),
        _num("loser_rank_points").alias("loser_rank_points"),
        *[_num(f"{p}_{s}").alias(f"{p}_{s}") for p in ("w", "l")
          for s in ("ace", "svpt", "1stIn", "1stWon", "bpSaved", "bpFaced")],
    )
    df = df.withColumn("match_id", F.concat_ws("-", "tour", "tourney_id", F.col("match_num")))
    order = Window.orderBy("match_date", "tourney_id", "round_order", "match_num", "tour")
    return df.withColumn("order_idx", F.row_number().over(order))


def player_long(matches: DataFrame) -> DataFrame:
    """One row per player per match (winner row won=1, loser row won=0) with serve rates."""
    common = ["match_id", "order_idx", "match_date", "day_num", "round_order", "tour", "surface"]

    def side(me: str, opp: str, won: int, pre: str) -> DataFrame:
        return matches.select(
            *common,
            F.col(f"{me}_key").alias("player_key"),
            F.col(f"{opp}_key").alias("opp_key"),
            F.lit(won).alias("won"),
            F.col(f"{pre}_ace").alias("ace"),
            F.col(f"{pre}_svpt").alias("svpt"),
            F.col(f"{pre}_1stIn").alias("first_in"),
            F.col(f"{pre}_1stWon").alias("first_won"),
            F.col(f"{pre}_bpSaved").alias("bp_saved"),
            F.col(f"{pre}_bpFaced").alias("bp_faced"),
        )

    long = side("winner", "loser", 1, "w").unionByName(side("loser", "winner", 0, "l"))
    return (
        long.withColumn("ace_rate", F.when(F.col("svpt") > 0, F.col("ace") / F.col("svpt")))
        .withColumn(
            "first_won_rate", F.when(F.col("first_in") > 0, F.col("first_won") / F.col("first_in"))
        )
        .withColumn(
            "bp_saved_rate", F.when(F.col("bp_faced") > 0, F.col("bp_saved") / F.col("bp_faced"))
        )
        .withColumn("ord_val", F.col("day_num").cast("long") * DAY + F.col("round_order"))
    )


def rolling_features(long: DataFrame, form_window: int, serve_window: int) -> DataFrame:
    """Pre-match rolling features. Every frame ends at -1, i.e. before the current match."""
    by_player = Window.partitionBy("player_key").orderBy("order_idx")
    by_surface = Window.partitionBy("player_key", "surface").orderBy("order_idx")
    by_time = Window.partitionBy("player_key").orderBy("ord_val")
    by_pair = Window.partitionBy("player_key", "opp_key").orderBy("order_idx")
    prev = (-form_window, -1)
    return long.select(
        "match_id",
        "player_key",
        "opp_key",
        "won",
        "order_idx",
        "match_date",
        "surface",
        F.count("won").over(by_player.rowsBetween(Window.unboundedPreceding, -1)).alias("n_prev"),
        F.avg("won").over(by_player.rowsBetween(*prev)).alias("form10"),
        F.avg("won").over(by_surface.rowsBetween(*prev)).alias("sform"),
        F.avg("won").over(by_time.rangeBetween(-365 * DAY, -1)).alias("form52"),
        F.count("won").over(by_time.rangeBetween(-14 * DAY, -1)).alias("matches14"),
        (F.col("day_num") - F.lag("day_num").over(by_player)).alias("days_since"),
        F.avg("ace_rate").over(by_player.rowsBetween(-serve_window, -1)).alias("ace_rate"),
        F.avg("first_won_rate").over(by_player.rowsBetween(-serve_window, -1)).alias("first_won"),
        F.avg("bp_saved_rate").over(by_player.rowsBetween(-serve_window, -1)).alias("bp_saved"),
        F.coalesce(
            F.sum("won").over(by_pair.rowsBetween(Window.unboundedPreceding, -1)), F.lit(0)
        ).alias("h2h_wins"),
        F.count("won").over(by_pair.rowsBetween(Window.unboundedPreceding, -1)).alias(
            "h2h_matches"
        ),
    )


def run_transform(raw_dir: Path, out: dict[str, Path], params: dict[str, Any]) -> dict[str, int]:
    """Run the whole Spark job; returns row counts for throughput reporting."""
    spark = build_spark(params)
    spark.sparkContext.setLogLevel("WARN")
    try:
        raw = read_raw(spark, raw_dir, list(params["data"]["tours"]))
        rows_in = raw.count()
        matches = clean_matches(raw, list(params["data"].get("levels_exclude", []))).cache()
        holdout = params["data"].get("holdout_from")
        if holdout:
            is_live = F.col("match_date") >= F.to_date(F.lit(str(holdout)))
            train_part = matches.filter(~is_live)
            live_part = matches.filter(is_live)
        else:
            train_part = matches
            live_part = matches.limit(0)
        train_part = train_part.cache()
        n_matches = train_part.count()
        n_live = live_part.count()
        train_part.coalesce(1).write.mode("overwrite").parquet(str(out["matches"]))
        live_part.coalesce(1).write.mode("overwrite").parquet(str(out["live"]))

        feats = rolling_features(
            player_long(train_part),
            int(params["features"]["form_window"]),
            int(params["features"]["serve_window"]),
        )
        feats.coalesce(1).write.mode("overwrite").parquet(str(out["player_match"]))
        n_long = spark.read.parquet(str(out["player_match"])).count()
        log.info("spark transform: %d raw rows -> %d matches (+%d live) -> %d player rows",
                 rows_in, n_matches, n_live, n_long)
        return {"rows_in": rows_in, "matches": n_matches, "live": n_live, "player_rows": n_long}
    finally:
        spark.stop()
