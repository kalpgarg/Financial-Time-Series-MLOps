"""
Kafka producer: fetches pre-market OHLCV snapshot from Upstox API and
publishes each row to the Kafka market_features topic.

This script is invoked by Airflow at 9:07 AM IST.

Usage:
    # Full mode (requires Upstox API key + running Kafka broker)
    python -m role1_data_engineering.kafka.producers.price_producer

    # Dry-run mode (reads sample_prices.csv, prints to stdout)
    python -m role1_data_engineering.kafka.producers.price_producer --dry-run
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ── Resolve project root so shared imports work when running as script ────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC_MARKET_FEATURES,
    STOCK_LIST_CSV_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("price_producer")

# ── Upstox API configuration ────────────────────────────────────────────────
# TODO: Replace with real Upstox credentials and endpoints
UPSTOX_API_BASE = "https://api.upstox.com/v2"
UPSTOX_ACCESS_TOKEN = None  # Set via env var or OAuth flow

SAMPLE_PRICES_CSV = str(PROJECT_ROOT / "data" / "day1_sample" / "sample_prices.csv")


def _get_kafka_producer():
    """Lazy-import and create a KafkaProducer (avoids import error in dry-run)."""
    from kafka import KafkaProducer

    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )


def fetch_prices_upstox(symbols: list[str]) -> list[dict]:
    """Fetch pre-market OHLCV data from Upstox API.

    TODO: Implement actual Upstox API integration:
        1. Authenticate via OAuth2 (access token)
        2. Call GET /market-quote/ohlc for each instrument key
        3. Map Upstox instrument keys to stock symbols
        4. Return list of PriceRecord-shaped dicts

    For now, this raises NotImplementedError.
    """
    raise NotImplementedError(
        "Upstox API integration not yet implemented. "
        "Use --dry-run to test with sample data."
    )


def load_sample_prices(csv_path: str) -> list[dict]:
    """Load prices from the sample CSV (used in dry-run mode)."""
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d price rows from %s", len(df), csv_path)
    records = []
    for _, row in df.iterrows():
        record = {
            "symbol": row["symbol"],
            "date": row["date"],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": int(row["volume"]),
            "adjusted_close": float(row["adjusted_close"]) if pd.notna(row.get("adjusted_close")) else None,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        records.append(record)
    return records


def produce_prices(
    records: list[dict],
    dry_run: bool = False,
    producer=None,
) -> int:
    """Publish price records to Kafka (or stdout in dry-run mode).

    Returns the number of messages produced.
    """
    count = 0
    for record in records:
        if dry_run:
            print(json.dumps(record, indent=2))
        else:
            producer.send(KAFKA_TOPIC_MARKET_FEATURES, value=record)
        count += 1

    if not dry_run and producer:
        producer.flush()

    logger.info("Produced %d price messages.", count)
    return count


def main(dry_run: bool = False):
    if dry_run:
        records = load_sample_prices(SAMPLE_PRICES_CSV)
    else:
        # Load stock list to get symbols
        stock_df = pd.read_csv(STOCK_LIST_CSV_PATH, sep="\t")
        stock_df.columns = [c.strip() for c in stock_df.columns]
        stock_df = stock_df.dropna(subset=["Stock_name"])
        symbols = stock_df["Stock_name"].str.strip().tolist()
        records = fetch_prices_upstox(symbols)

    if not records:
        logger.warning("No price data available. Nothing to produce.")
        return

    producer = None if dry_run else _get_kafka_producer()
    try:
        produce_prices(records, dry_run=dry_run, producer=producer)
    finally:
        if producer:
            producer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Price data → Kafka producer")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use sample_prices.csv and print to stdout (no Kafka needed)",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
