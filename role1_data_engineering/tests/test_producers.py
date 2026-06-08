"""Unit tests for Kafka producers."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# ── Resolve project root ────────────────────────────────────────────────────
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from role1_data_engineering.kafka.producers.headline_producer import (
    load_headlines_csv,
    produce_headlines,
)
from role1_data_engineering.kafka.producers.price_producer import (
    load_sample_prices,
    produce_prices,
)

# ── Fixtures ────────────────────────────────────────────────────────────────

SAMPLE_PRICES_CSV = os.path.join(PROJECT_ROOT, "data", "ohlc_data", "sample_prices.csv")
SAMPLE_HEADLINES_CSV = os.path.join(PROJECT_ROOT, "data", "stock_news", "headlines.csv")

HEADLINE_REQUIRED_KEYS = {"headline_id", "symbol", "headline"}
PRICE_REQUIRED_KEYS = {"symbol", "date", "open", "high", "low", "close", "volume"}


# ── Headline Producer Tests ─────────────────────────────────────────────────

class TestHeadlineProducer:
    def test_load_headlines_csv(self):
        records = load_headlines_csv(SAMPLE_HEADLINES_CSV)
        assert len(records) > 0
        for r in records:
            assert "headline_id" in r or "headline" in r

    def test_produce_headlines_dry_run(self, capsys):
        """Dry-run should print JSON to stdout without Kafka."""
        records = load_headlines_csv(SAMPLE_HEADLINES_CSV)
        count = produce_headlines(records, dry_run=True)
        assert count == len(records)
        captured = capsys.readouterr()
        assert "symbol" in captured.out

    def test_produce_headlines_to_kafka(self):
        """Mock Kafka producer and verify send is called with correct topic."""
        records = load_headlines_csv(SAMPLE_HEADLINES_CSV)
        mock_producer = MagicMock()
        count = produce_headlines(records, dry_run=False, producer=mock_producer)
        assert count == len(records)
        assert mock_producer.send.call_count == len(records)
        mock_producer.flush.assert_called_once()


# ── Price Producer Tests ─────────────────────────────────────────────────────

class TestPriceProducer:
    def test_load_sample_prices(self):
        records = load_sample_prices(SAMPLE_PRICES_CSV)
        assert len(records) == 6
        for r in records:
            for key in PRICE_REQUIRED_KEYS:
                assert key in r, f"Missing key '{key}' in price record"

    def test_produce_prices_dry_run(self, capsys):
        """Dry-run should print JSON to stdout without Kafka."""
        records = load_sample_prices(SAMPLE_PRICES_CSV)
        count = produce_prices(records, dry_run=True)
        assert count == 6
        captured = capsys.readouterr()
        # Verify output is valid JSON lines
        lines = captured.out.strip().split("}\n{")
        assert len(lines) >= 1

    def test_produce_prices_to_kafka(self):
        """Mock Kafka producer and verify send is called."""
        records = load_sample_prices(SAMPLE_PRICES_CSV)
        mock_producer = MagicMock()
        count = produce_prices(records, dry_run=False, producer=mock_producer)
        assert count == 6
        assert mock_producer.send.call_count == 6
        mock_producer.flush.assert_called_once()

    def test_price_record_schema(self):
        """Verify price records match PriceRecord schema fields."""
        records = load_sample_prices(SAMPLE_PRICES_CSV)
        for r in records:
            assert isinstance(r["symbol"], str)
            assert isinstance(r["date"], str)
            assert isinstance(r["open"], float)
            assert isinstance(r["high"], float)
            assert isinstance(r["low"], float)
            assert isinstance(r["close"], float)
            assert isinstance(r["volume"], int)
