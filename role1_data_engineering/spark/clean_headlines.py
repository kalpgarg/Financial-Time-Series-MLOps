"""
PySpark job: reads raw headlines from the raw_headlines PostgreSQL table,
cleans text (lowercase, strip HTML entities, deduplicate by headline_id),
and writes the results to the clean_headlines table.

Usage:
    spark-submit role1_data_engineering/spark/clean_headlines.py
"""

import re
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


def clean_text(text: str | None) -> str | None:
    """Lowercase, strip HTML tags/entities, and collapse whitespace."""
    if text is None:
        return None
    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text)           # strip HTML tags
    text = re.sub(r"&[a-z]+;", " ", text)          # strip HTML entities
    text = re.sub(r"\s+", " ", text).strip()       # collapse whitespace
    return text


def main():
    spark = (
        SparkSession.builder
        .master(SPARK_MASTER)
        .appName(f"{SPARK_APP_NAME}_clean_headlines")
        .getOrCreate()
    )

    clean_text_udf = F.udf(clean_text, StringType())

    # Read raw headlines from PostgreSQL
    raw_df = (
        spark.read.jdbc(
            url=JDBC_URL,
            table="raw_headlines",
            properties=JDBC_PROPS,
        )
    )

    if raw_df.rdd.isEmpty():
        print("No raw headlines to process.")
        spark.stop()
        return

    # Clean text fields
    cleaned_df = (
        raw_df
        .withColumn("headline", clean_text_udf(F.col("headline")))
        .withColumn("source", clean_text_udf(F.col("source")))
    )

    # Deduplicate by headline_id (keep first occurrence)
    deduped_df = cleaned_df.dropDuplicates(["headline_id"])

    # Select columns matching clean_headlines table schema
    output_df = deduped_df.select(
        "headline_id",
        "symbol",
        "published_at",
        "source",
        "headline",
        "article_url",
        "scraped_at",
    )

    # Write to clean_headlines table (overwrite for idempotency)
    output_df.write.jdbc(
        url=JDBC_URL,
        table="clean_headlines",
        mode="overwrite",
        properties=JDBC_PROPS,
    )

    print(f"Wrote {output_df.count()} clean headlines to PostgreSQL.")
    spark.stop()


if __name__ == "__main__":
    main()
