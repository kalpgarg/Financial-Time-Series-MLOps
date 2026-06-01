"""
Kafka consumer: reads price messages from the market_features topic and
inserts them into the raw_prices PostgreSQL table for Spark to process.

Usage:
    # Full mode (requires Kafka + PostgreSQL)
    python -m role1_data_engineering.kafka.consumers.price_consumer

    # Dry-run mode (prints consumed messages to stdout, no DB writes)
    python -m role1_data_engineering.kafka.consumers.price_consumer --dry-run
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
    KAFKA_TOPIC_MARKET_FEATURES,
)
from role1_data_engineering.db.init_db import get_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("price_consumer")

REQUIRED_FIELDS = {"symbol", "date", "close"}

INSERT_SQL = """
    INSERT INTO raw_prices (symbol, date, open, high, low, close, volume, adjusted_close)
    VALUES (%(symbol)s, %(date)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s, %(adjusted_close)s)
    ON CONFLICT (symbol, date) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume,
        adjusted_close = EXCLUDED.adjusted_close,
        inserted_at = NOW();
"""


def validate_price(record: dict) -> bool:
    """Check that required fields are present and non-empty."""
    for field in REQUIRED_FIELDS:
        if record.get(field) is None:
            logger.warning("Skipping record missing field '%s': %s", field, record)
            return False
    return True


def insert_price(conn, record: dict):
    """Insert a single price record into raw_prices."""
    params = {
        "symbol": record["symbol"],
        "date": record["date"],
        "open": record.get("open"),
        "high": record.get("high"),
        "low": record.get("low"),
        "close": record["close"],
        "volume": record.get("volume"),
        "adjusted_close": record.get("adjusted_close"),
    }
    with conn.cursor() as cur:
        cur.execute(INSERT_SQL, params)
    conn.commit()


def consume(dry_run: bool = False, max_messages: int | None = None):
    """Consume price data from Kafka and write to PostgreSQL.

    Args:
        dry_run: If True, print messages to stdout instead of writing to DB.
        max_messages: Stop after this many messages (None = run forever).
    """
    from kafka import KafkaConsumer

    consumer = KafkaConsumer(
        KAFKA_TOPIC_MARKET_FEATURES,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        group_id="price-consumer-group",
        consumer_timeout_ms=10_000 if max_messages else -1,
    )

    conn = None if dry_run else get_connection()
    count = 0

    logger.info(
        "Consuming from topic '%s' (dry_run=%s) ...",
        KAFKA_TOPIC_MARKET_FEATURES,
        dry_run,
    )

    try:
        for message in consumer:
            record = message.value
            if not validate_price(record):
                continue

            if dry_run:
                print(json.dumps(record, indent=2))
            else:
                insert_price(conn, record)

            count += 1
            if max_messages and count >= max_messages:
                break
    except KeyboardInterrupt:
        logger.info("Consumer interrupted by user.")
    finally:
        consumer.close()
        if conn:
            conn.close()
        logger.info("Consumed %d price messages.", count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kafka price consumer → PostgreSQL")
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
