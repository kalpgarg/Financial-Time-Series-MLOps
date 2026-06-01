"""
PySpark job: joins clean_headlines with clean_prices on (symbol, date)
to produce a unified feature vector written to the feature_vectors table.

Headlines are aggregated per (symbol, date): count + JSON array of texts.
The output is consumed by Role 2 for model training / inference.

Usage:
    spark-submit role1_data_engineering/spark/join_data.py
"""

import json
import sys
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

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


def main():
    spark = (
        SparkSession.builder
        .master(SPARK_MASTER)
        .appName(f"{SPARK_APP_NAME}_join_data")
        .getOrCreate()
    )

    # Read clean tables from PostgreSQL
    prices_df = spark.read.jdbc(url=JDBC_URL, table="clean_prices", properties=JDBC_PROPS)
    headlines_df = spark.read.jdbc(url=JDBC_URL, table="clean_headlines", properties=JDBC_PROPS)

    if prices_df.rdd.isEmpty():
        print("No clean prices to join. Exiting.")
        spark.stop()
        return

    # Aggregate headlines per (symbol, date):
    #   - headline_count: number of headlines
    #   - headlines_json: JSON array of headline texts
    if not headlines_df.rdd.isEmpty():
        # Extract date from published_at or scraped_at for joining
        # Headlines may not have a proper date column, so we derive from scraped_at
        headlines_agg = (
            headlines_df
            .withColumn("date", F.to_date(F.col("scraped_at")))
            .groupBy("symbol", "date")
            .agg(
                F.count("headline_id").alias("headline_count"),
                F.collect_list("headline").alias("_headlines_list"),
            )
        )

        # Convert list of headlines to JSON string
        to_json_udf = F.udf(lambda lst: json.dumps(lst) if lst else "[]", StringType())
        headlines_agg = headlines_agg.withColumn(
            "headlines_json", to_json_udf(F.col("_headlines_list"))
        ).drop("_headlines_list")

        # Left-join prices with aggregated headlines
        joined_df = prices_df.join(
            headlines_agg,
            on=["symbol", "date"],
            how="left",
        )
    else:
        # No headlines — just add empty columns
        joined_df = (
            prices_df
            .withColumn("headline_count", F.lit(0))
            .withColumn("headlines_json", F.lit("[]"))
        )

    # Fill nulls for stocks with no headlines on a given date
    joined_df = (
        joined_df
        .fillna({"headline_count": 0, "headlines_json": "[]"})
    )

    # Select final output columns
    output_df = joined_df.select(
        "symbol",
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "adjusted_close",
        "headline_count",
        "headlines_json",
    )

    # Write to feature_vectors table (overwrite for idempotency)
    output_df.write.jdbc(
        url=JDBC_URL,
        table="feature_vectors",
        mode="overwrite",
        properties=JDBC_PROPS,
    )

    print(f"Wrote {output_df.count()} feature vectors to PostgreSQL.")
    spark.stop()


if __name__ == "__main__":
    main()
