"""
PySpark job: reads raw price data from the raw_prices PostgreSQL table,
handles missing values (forward-fill via Window functions), removes
duplicates, and writes the results to the clean_prices table.

Usage:
    spark-submit role1_data_engineering/spark/clean_prices.py
"""

import sys
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

# ── Resolve project root so shared imports work ──────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import (
    JDBC_URL,
    POSTGRES_PASSWORD,
    POSTGRES_USER,
    SPARK_APP_NAME,
    SPARK_MASTER,
)

JDBC_PROPS = {
    "user": POSTGRES_USER,
    "password": POSTGRES_PASSWORD,
    "driver": "org.postgresql.Driver",
}

# Numeric columns eligible for forward-fill
OHLCV_COLS = ["open", "high", "low", "close", "volume", "adjusted_close"]


def forward_fill(df, partition_col, order_col, columns):
    """Forward-fill nulls within each partition, ordered by order_col."""
    window = Window.partitionBy(partition_col).orderBy(order_col).rowsBetween(
        Window.unboundedPreceding, Window.currentRow
    )
    for col_name in columns:
        df = df.withColumn(
            col_name,
            F.last(F.col(col_name), ignorenulls=True).over(window),
        )
    return df


def main():
    spark = (
        SparkSession.builder
        .master(SPARK_MASTER)
        .appName(f"{SPARK_APP_NAME}_clean_prices")
        .getOrCreate()
    )

    # Read raw prices from PostgreSQL
    raw_df = (
        spark.read.jdbc(
            url=JDBC_URL,
            table="raw_prices",
            properties=JDBC_PROPS,
        )
    )

    if raw_df.rdd.isEmpty():
        print("No raw prices to process.")
        spark.stop()
        return

    # Deduplicate by (symbol, date) — keep latest inserted row
    deduped_df = (
        raw_df
        .orderBy(F.col("inserted_at").desc())
        .dropDuplicates(["symbol", "date"])
    )

    # Forward-fill missing OHLCV values within each symbol
    filled_df = forward_fill(deduped_df, "symbol", "date", OHLCV_COLS)

    # Drop rows where close is still null (can't be filled)
    filled_df = filled_df.filter(F.col("close").isNotNull())

    # Select columns matching clean_prices table schema
    output_df = filled_df.select(
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adjusted_close",
    )

    # Write to clean_prices table (overwrite for idempotency)
    output_df.write.jdbc(
        url=JDBC_URL,
        table="clean_prices",
        mode="overwrite",
        properties=JDBC_PROPS,
    )

    print(f"Wrote {output_df.count()} clean price rows to PostgreSQL.")
    spark.stop()


if __name__ == "__main__":
    main()
