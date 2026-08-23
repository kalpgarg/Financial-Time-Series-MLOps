"""
Clean raw scraped headlines and write a deduplicated, cleaned CSV.

Reads data/stock_news/headlines.csv (raw scraper output), applies text
cleaning (lowercase, strip HTML tags/entities, collapse whitespace),
deduplicates by headline_id, and writes the result to data/headlines.csv
(the DVC-tracked file consumed by the inference pipeline).

Uses PySpark for distributed-ready cleaning (runs in local[*] mode).

Usage:
    python -m role1_data_engineering.spark.clean_headlines
"""

import logging
import os
import re
import shutil
import sys
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

# ── Resolve project root so shared imports work ──────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import SPARK_MASTER, SPARK_APP_NAME

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("clean_headlines")

RAW_CSV = os.path.join(PROJECT_ROOT, "data", "stock_news", "headlines.csv")
CLEAN_CSV = os.path.join(PROJECT_ROOT, "data", "headlines.csv")

OUTPUT_COLUMNS = [
    "headline_id",
    "symbol",
    "published_at",
    "source",
    "headline",
    "article_url",
    "scraped_at",
]


def clean_text(text):
    """Lowercase, strip HTML tags/entities, and collapse whitespace."""
    if text is None:
        return None
    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text)           # strip HTML tags
    text = re.sub(r"&[a-z]+;", " ", text)          # strip HTML entities
    text = re.sub(r"\s+", " ", text).strip()       # collapse whitespace
    return text


clean_text_udf = F.udf(clean_text, StringType())


def main():
    if not os.path.exists(RAW_CSV):
        logger.error("Raw headlines CSV not found: %s", RAW_CSV)
        sys.exit(1)

    spark = (
        SparkSession.builder
        .master(SPARK_MASTER)
        .appName(f"{SPARK_APP_NAME}_clean_headlines")
        .getOrCreate()
    )

    try:
        raw_df = spark.read.csv(RAW_CSV, header=True, inferSchema=True, encoding="UTF-8")

        row_count = raw_df.count()
        if row_count == 0:
            logger.warning("No raw headlines to process.")
            spark.stop()
            sys.exit(0)

        logger.info("Read %d raw headlines from %s", row_count, RAW_CSV)

        # Clean text fields
        cleaned_df = raw_df.withColumn(
            "headline", clean_text_udf(F.col("headline"))
        ).withColumn(
            "source", clean_text_udf(F.col("source"))
        )

        # Deduplicate by headline_id (keep first occurrence via row_number)
        from pyspark.sql.window import Window

        window = Window.partitionBy("headline_id").orderBy("scraped_at")
        deduped_df = (
            cleaned_df
            .withColumn("_rn", F.row_number().over(window))
            .filter(F.col("_rn") == 1)
            .drop("_rn")
        )

        # Select output columns (only those present in the data)
        available_cols = [c for c in OUTPUT_COLUMNS if c in deduped_df.columns]
        output_df = deduped_df.select(available_cols)

        final_count = output_df.count()

        # Write to a temp directory, then move the single CSV part file
        tmp_dir = CLEAN_CSV + "_tmp"
        output_df.coalesce(1).write.csv(tmp_dir, header=True, mode="overwrite")

        # Find the part file and move it to the final path
        part_files = [f for f in os.listdir(tmp_dir) if f.startswith("part-")]
        if part_files:
            os.makedirs(os.path.dirname(CLEAN_CSV), exist_ok=True)
            shutil.move(os.path.join(tmp_dir, part_files[0]), CLEAN_CSV)
        shutil.rmtree(tmp_dir, ignore_errors=True)

        logger.info("Wrote %d clean headlines → %s", final_count, CLEAN_CSV)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
