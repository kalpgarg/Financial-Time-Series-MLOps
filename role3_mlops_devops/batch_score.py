"""Batch scoring -- stands in for the final task of the Airflow DAG.

Reads the news and OHLCV data produced upstream, runs the shared pipeline over
every symbol present on the latest date, and upserts the results into
``pipeline_predictions``. The real DAG task can call ``run_batch()`` directly
with DataFrames instead of shelling out:

    from batch_score import run_batch
    run_batch(news_df, ohlcv_df, run_id=context["run_id"])

It imports ``app.scoring`` -- the same module ``POST /predict`` uses. One
scoring path, two invocation modes: a single symbol on demand through the API,
all constituents once a day through here.

Usage:
    python batch_score.py \
        --news "Prediction Artifacts/headlines.csv" \
        --ohlcv "Prediction Artifacts/merged_ohlc_15min.csv" \
        --run-id manual__001
"""

import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd

from app import db, scoring

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("batch_score")


def run_batch(
    news_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    run_id: str,
    as_of_date: str | None = None,
) -> List[dict]:
    """Score all symbols and upsert the results.

    The upsert is keyed on (symbol, date), so re-running for the same trading
    day -- an Airflow retry or a manual backfill -- overwrites rather than
    duplicating. The task is safe to run more than once.

    ``as_of_date`` (YYYY-MM-DD) scores that specific trading date instead of the
    latest one in the data (default).
    """
    db.init_db()

    rows = scoring.score(news_df, ohlcv_df, as_of_date=as_of_date)
    for row in rows:
        row["run_id"] = run_id
        logger.info(
            "%s -> %s (confidence=%.4f)",
            row["symbol"],
            row["direction"],
            row["confidence"],
        )

    with db.session_scope() as session:
        written = db.upsert_pipeline_predictions(session, rows)

    logger.info(
        "run %s: %d predictions written (date=%s)",
        run_id, written, as_of_date or "latest",
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Score all symbols for the latest date.")
    parser.add_argument("--news", type=Path, required=True, help="Headlines CSV")
    parser.add_argument("--ohlcv", type=Path, required=True, help="15-min OHLCV CSV")
    parser.add_argument(
        "--run-id",
        default="manual",
        help="Pipeline run identifier (use the Airflow dag_run_id)",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Trading date to score (YYYY-MM-DD). Default: latest date in the data.",
    )
    args = parser.parse_args()

    news_df = pd.read_csv(args.news)
    ohlcv_df = pd.read_csv(args.ohlcv)
    run_batch(news_df, ohlcv_df, run_id=args.run_id, as_of_date=args.date)


if __name__ == "__main__":
    main()
