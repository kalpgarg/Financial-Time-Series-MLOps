"""
Kafka consumer: reads headline messages from the news_features topic and
inserts them into the raw_headlines PostgreSQL table for Spark to process.

Usage:
    # Full mode (requires Kafka + PostgreSQL)
    python -m role1_data_engineering.kafka.consumers.headline_consumer

    # Dry-run mode (prints consumed messages to stdout, no DB writes)
    python -m role1_data_engineering.kafka.consumers.headline_consumer --dry-run
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# ── Resolve project root so shared imports work when running as script ────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC_NEWS_FEATURES,
)
from role1_data_engineering.db.init_db import get_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("headline_consumer")

REQUIRED_FIELDS = {"headline_id", "symbol", "headline"}

INSERT_SQL = """
    INSERT INTO raw_headlines (headline_id, symbol, published_at, source, headline, article_url, scraped_at)
    VALUES (%(headline_id)s, %(symbol)s, %(published_at)s, %(source)s, %(headline)s, %(article_url)s, %(scraped_at)s)
    ON CONFLICT (headline_id) DO NOTHING;
"""


def validate_headline(record: dict) -> bool:
    """Check that required fields are present and non-empty."""
    for field in REQUIRED_FIELDS:
        if not record.get(field):
            logger.warning("Skipping record missing field '%s': %s", field, record)
            return False
    return True


def insert_headline(conn, record: dict):
    """Insert a single headline record into raw_headlines."""
    params = {
        "headline_id": record["headline_id"],
        "symbol": record["symbol"],
        "published_at": record.get("published_at"),
        "source": record.get("source"),
        "headline": record["headline"],
        "article_url": record.get("article_url"),
        "scraped_at": record.get("scraped_at"),
    }
    with conn.cursor() as cur:
        cur.execute(INSERT_SQL, params)
    conn.commit()


def consume(dry_run: bool = False, max_messages: int | None = None):
    """Consume headlines from Kafka and write to PostgreSQL.

    Args:
        dry_run: If True, print messages to stdout instead of writing to DB.
        max_messages: Stop after this many messages (None = run forever).
    """
    from kafka import KafkaConsumer

    consumer = KafkaConsumer(
        KAFKA_TOPIC_NEWS_FEATURES,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        group_id="headline-consumer-group",
        consumer_timeout_ms=10_000 if max_messages else -1,
    )

    conn = None if dry_run else get_connection()
    count = 0

    logger.info(
        "Consuming from topic '%s' (dry_run=%s) ...",
        KAFKA_TOPIC_NEWS_FEATURES,
        dry_run,
    )

    try:
        for message in consumer:
            record = message.value
            if not validate_headline(record):
                continue

            if dry_run:
                print(json.dumps(record, indent=2))
            else:
                insert_headline(conn, record)

            count += 1
            if max_messages and count >= max_messages:
                break
    except KeyboardInterrupt:
        logger.info("Consumer interrupted by user.")
    finally:
        consumer.close()
        if conn:
            conn.close()
        logger.info("Consumed %d headline messages.", count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kafka headline consumer → PostgreSQL")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print messages to stdout instead of writing to PostgreSQL",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="Stop after consuming N messages (default: run forever)",
    )
    args = parser.parse_args()
    consume(dry_run=args.dry_run, max_messages=args.max_messages)
