"""
Trim full datasets down to the minimal date window needed by inference.

Reads the DVC-tracked headlines and OHLCV CSVs, filters to:
  - Headlines: last 10 calendar days (7-day sentiment window + buffer)
  - OHLCV 15-min bars: last 45 calendar days (30-day rolling window + buffer)

Writes trimmed CSVs to data/inference/ so the Docker inference container
receives only the data it actually needs.

Uses PySpark for consistency with the other Spark jobs.

Usage:
    python -m role1_data_engineering.spark.trim_for_inference
"""

import logging
import os
import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ── Resolve project root so shared imports work ──────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import SPARK_MASTER, SPARK_APP_NAME

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("trim_for_inference")

# ── Paths ────────────────────────────────────────────────────────────────────
HEADLINES_CSV = os.path.join(PROJECT_ROOT, "data", "headlines.csv")
OHLCV_CSV = os.path.join(PROJECT_ROOT, "data", "merged_ohlc_15min.csv")
INFERENCE_DIR = os.path.join(PROJECT_ROOT, "data", "inference")

# ── Lookback windows ────────────────────────────────────────────────────────
# pipeline.py uses NEWS_WINDOW_DAYS=6 (7 days inclusive) and 30-day rolling
# features. Extra buffer accounts for weekends and timezone edge cases.
NEWS_LOOKBACK_DAYS = 10
OHLCV_LOOKBACK_DAYS = 45


def _write_single_csv(df, output_path):
    """Write a Spark DataFrame as a single CSV file."""
    tmp_dir = output_path + "_tmp"
    df.coalesce(1).write.csv(tmp_dir, header=True, mode="overwrite")

    part_files = [f for f in os.listdir(tmp_dir) if f.startswith("part-")]
    if part_files:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.move(os.path.join(tmp_dir, part_files[0]), output_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    for path, label in [(HEADLINES_CSV, "headlines"), (OHLCV_CSV, "OHLCV")]:
        if not os.path.exists(path):
            logger.error("%s CSV not found: %s", label, path)
            sys.exit(1)

    os.makedirs(INFERENCE_DIR, exist_ok=True)

    spark = (
        SparkSession.builder
        .master(SPARK_MASTER)
        .appName(f"{SPARK_APP_NAME}_trim_inference")
        .getOrCreate()
    )

    try:
        # ── Trim headlines ───────────────────────────────────────────────
        headlines_df = spark.read.csv(
            HEADLINES_CSV, header=True, inferSchema=True, encoding="UTF-8",
        )
        headlines_count_before = headlines_df.count()

        cutoff_news = (
            datetime.now(UTC) - timedelta(days=NEWS_LOOKBACK_DAYS)
        ).strftime("%Y-%m-%d")

        trimmed_headlines = headlines_df.filter(
            F.col("published_at") >= cutoff_news
        )
        headlines_count_after = trimmed_headlines.count()

        headlines_out = os.path.join(INFERENCE_DIR, "headlines.csv")
        _write_single_csv(trimmed_headlines, headlines_out)
        logger.info(
            "Headlines: %d → %d rows (last %d days) → %s",
            headlines_count_before, headlines_count_after,
            NEWS_LOOKBACK_DAYS, headlines_out,
        )

        # ── Trim OHLCV ───────────────────────────────────────────────────
        ohlcv_df = spark.read.csv(
            OHLCV_CSV, header=True, inferSchema=True, encoding="UTF-8",
        )
        ohlcv_count_before = ohlcv_df.count()

        cutoff_ohlcv = (
            datetime.now(UTC) - timedelta(days=OHLCV_LOOKBACK_DAYS)
        ).strftime("%Y-%m-%d")

        trimmed_ohlcv = ohlcv_df.filter(
            F.col("datetime") >= cutoff_ohlcv
        )
        ohlcv_count_after = trimmed_ohlcv.count()

        ohlcv_out = os.path.join(INFERENCE_DIR, "ohlcv_15min.csv")
        _write_single_csv(trimmed_ohlcv, ohlcv_out)
        logger.info(
            "OHLCV: %d → %d rows (last %d days) → %s",
            ohlcv_count_before, ohlcv_count_after,
            OHLCV_LOOKBACK_DAYS, ohlcv_out,
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
