"""
Merge pre-open data into the 15-minute OHLCV merged CSV.

Reads the day's NSE pre-open CSV (data/preopen_csv/nse_fo_<YYYYMMDD>_preopen.csv),
maps NSE ticker symbols to Stock_name values via stock_list.csv, and appends
a synthetic 09:15 bar to data/ohlc_data/merged_ohlc_15min.csv.

Mapping:
  open = close = high = low = final_price
  volume = final_quantity
  datetime = <today> 09:15:00

Uses PySpark for distributed-ready processing (runs in local[*] mode).

Usage:
    python -m role1_data_engineering.spark.merge_preopen
    python -m role1_data_engineering.spark.merge_preopen --date 20260619
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# ── Resolve project root so shared imports work when running as script ────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import SPARK_MASTER, SPARK_APP_NAME, STOCK_LIST_CSV_PATH, now_local

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("merge_preopen")

PREOPEN_DIR = os.path.join(PROJECT_ROOT, "data", "preopen_csv")
MERGED_15MIN_CSV = os.path.join(PROJECT_ROOT, "data", "merged_ohlc_15min.csv")

OHLCV_SCHEMA = StructType([
    StructField("symbol", StringType(), False),
    StructField("datetime", StringType(), False),
    StructField("open", DoubleType(), False),
    StructField("high", DoubleType(), False),
    StructField("low", DoubleType(), False),
    StructField("close", DoubleType(), False),
    StructField("volume", IntegerType(), False),
])


def merge_preopen(date_str: str | None = None) -> int:
    """Read the pre-open CSV for the given date and append 09:15 bars.

    Args:
        date_str: Date in YYYYMMDD format. Defaults to today (IST).

    Returns:
        Number of rows appended.
    """
    if date_str is None:
        date_str = now_local().strftime("%Y%m%d")

    preopen_filename = f"nse_fo_{date_str}_preopen.csv"
    preopen_path = os.path.join(PREOPEN_DIR, preopen_filename)

    if not os.path.exists(preopen_path):
        logger.error("Pre-open CSV not found: %s", preopen_path)
        return 0

    spark = (
        SparkSession.builder
        .master(SPARK_MASTER)
        .appName(f"{SPARK_APP_NAME}_merge_preopen")
        .getOrCreate()
    )

    try:
        # Read pre-open data
        preopen_df = spark.read.csv(preopen_path, header=True, inferSchema=True, encoding="UTF-8")

        if preopen_df.count() == 0:
            logger.warning("Pre-open CSV is empty: %s", preopen_path)
            return 0

        logger.info("Read %d pre-open records from %s", preopen_df.count(), preopen_path)

        # Read stock_list.csv to build symbol mapping
        stock_list_df = spark.read.csv(STOCK_LIST_CSV_PATH, header=True, inferSchema=True)

        # Build NSE symbol → Stock_name mapping
        # Try Symbol column first, then extract from TradingView_name
        stock_cols = [c.strip() for c in stock_list_df.columns]
        stock_list_df = stock_list_df.toDF(*stock_cols)

        if "Symbol" in stock_cols:
            mapping_df = (
                stock_list_df
                .select(
                    F.trim(F.col("Symbol")).alias("nse_symbol"),
                    F.trim(F.col("Stock_name")).alias("stock_name"),
                )
                .filter(F.col("nse_symbol").isNotNull() & F.col("stock_name").isNotNull())
                .filter(F.col("nse_symbol") != "")
            )
        elif "TradingView_name" in stock_cols:
            mapping_df = (
                stock_list_df
                .filter(F.col("TradingView_name").contains(":"))
                .select(
                    F.split(F.trim(F.col("TradingView_name")), ":").getItem(1).alias("nse_symbol"),
                    F.trim(F.col("Stock_name")).alias("stock_name"),
                )
                .filter(F.col("nse_symbol").isNotNull() & F.col("stock_name").isNotNull())
            )
        else:
            logger.error("stock_list.csv has no Symbol or TradingView_name column")
            return 0

        logger.info("Built symbol map: %d entries", mapping_df.count())

        # Join pre-open with mapping to get Stock_name
        datetime_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} 09:15:00"

        new_bars = (
            preopen_df
            .join(
                mapping_df,
                F.trim(preopen_df["symbol"]) == mapping_df["nse_symbol"],
                "inner",
            )
            .filter(F.col("final_price").isNotNull())
            .select(
                F.col("stock_name").alias("symbol"),
                F.lit(datetime_str).alias("datetime"),
                F.round(F.col("final_price").cast(DoubleType()), 2).alias("open"),
                F.round(F.col("final_price").cast(DoubleType()), 2).alias("high"),
                F.round(F.col("final_price").cast(DoubleType()), 2).alias("low"),
                F.round(F.col("final_price").cast(DoubleType()), 2).alias("close"),
                F.coalesce(F.col("final_quantity").cast(IntegerType()), F.lit(0)).alias("volume"),
            )
        )

        new_count = new_bars.count()
        if new_count == 0:
            logger.warning("No symbols matched between pre-open and stock_list.")
            return 0

        logger.info("Created %d OHLCV rows from pre-open data for %s", new_count, datetime_str)

        # Merge with existing merged_ohlc_15min.csv
        os.makedirs(os.path.dirname(MERGED_15MIN_CSV), exist_ok=True)

        if os.path.exists(MERGED_15MIN_CSV) and os.path.getsize(MERGED_15MIN_CSV) > 0:
            existing_df = spark.read.csv(MERGED_15MIN_CSV, header=True, inferSchema=True, encoding="UTF-8")
            # Remove existing rows for this datetime to avoid duplicates
            existing_filtered = existing_df.filter(F.col("datetime") != datetime_str)
            removed = existing_df.count() - existing_filtered.count()
            if removed > 0:
                logger.info("Removing %d existing rows for datetime=%s", removed, datetime_str)
            combined = existing_filtered.unionByName(new_bars).orderBy("symbol", "datetime")
        else:
            combined = new_bars.orderBy("symbol", "datetime")

        # Write to temp dir, then move the single part file
        tmp_dir = MERGED_15MIN_CSV + "_tmp"
        combined.coalesce(1).write.csv(tmp_dir, header=True, mode="overwrite")

        part_files = [f for f in os.listdir(tmp_dir) if f.startswith("part-")]
        if part_files:
            shutil.move(os.path.join(tmp_dir, part_files[0]), MERGED_15MIN_CSV)
        shutil.rmtree(tmp_dir, ignore_errors=True)

        logger.info("Wrote %d total rows → %s", combined.count(), MERGED_15MIN_CSV)
        return new_count

    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge NSE pre-open data into merged_ohlc_15min.csv"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date in YYYYMMDD format (default: today IST)",
    )
    args = parser.parse_args()
    count = merge_preopen(date_str=args.date)
    if count == 0:
        sys.exit(1)
