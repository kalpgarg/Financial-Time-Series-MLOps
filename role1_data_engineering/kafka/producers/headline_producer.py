"""
Kafka producer: reads a scraped headlines CSV (produced by headline_scraper.py)
and publishes each row to the Kafka news_features topic.

This script is invoked by Airflow at 9:00 AM IST, after the scraper finishes.

Usage:
    # Full mode (requires a running Kafka broker)
    python -m role1_data_engineering.kafka.producers.headline_producer --csv-path data/stock_news/headlines.csv

    # Dry-run mode (prints messages to stdout, no Kafka needed)
    python -m role1_data_engineering.kafka.producers.headline_producer --csv-path data/stock_news/headlines.csv --dry-run
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# ── Resolve project root so shared imports work when running as script ────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC_NEWS_FEATURES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("headline_producer")


def _get_kafka_producer():
    """Lazy-import and create a KafkaProducer (avoids import error in dry-run)."""
    from kafka import KafkaProducer

    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )


def load_headlines_csv(csv_path: str) -> list[dict]:
    """Read the scraped headlines CSV and return a list of dicts."""
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d headlines from %s", len(df), csv_path)
    return df.to_dict(orient="records")


def produce_headlines(
    records: list[dict],
    dry_run: bool = False,
    producer=None,
) -> int:
    """Publish headline records to Kafka (or stdout in dry-run mode).

    Returns the number of messages produced.
    """
    count = 0
    for record in records:
        if dry_run:
            print(json.dumps(record, indent=2))
        else:
            producer.send(KAFKA_TOPIC_NEWS_FEATURES, value=record)
        count += 1

    if not dry_run and producer:
        producer.flush()

    logger.info("Produced %d headline messages.", count)
    return count


def main(csv_path: str, dry_run: bool = False):
    records = load_headlines_csv(csv_path)
    if not records:
        logger.warning("No headlines found in %s. Nothing to produce.", csv_path)
        return

    producer = None if dry_run else _get_kafka_producer()
    try:
        produce_headlines(records, dry_run=dry_run, producer=producer)
    finally:
        if producer:
            producer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headline CSV → Kafka producer")
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to the scraped headlines CSV file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print messages to stdout instead of sending to Kafka",
    )
    args = parser.parse_args()
    main(csv_path=args.csv_path, dry_run=args.dry_run)
