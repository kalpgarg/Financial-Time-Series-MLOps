"""Unit tests for Kafka consumers."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# ── Resolve project root ────────────────────────────────────────────────────
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from role1_data_engineering.kafka.consumers.headline_consumer import (
    validate_headline,
    insert_headline,
)
from role1_data_engineering.kafka.consumers.price_consumer import (
    validate_price,
    insert_price,
)


# ── Headline Consumer Tests ──────────────────────────────────────────────────

class TestHeadlineConsumer:
    def test_validate_headline_valid(self):
        record = {
            "headline_id": "abc123",
            "symbol": "AAPL",
            "headline": "Apple stock rises",
            "source": "reuters",
        }
        assert validate_headline(record) is True

    def test_validate_headline_missing_id(self):
        record = {"symbol": "AAPL", "headline": "Apple stock rises"}
        assert validate_headline(record) is False

    def test_validate_headline_missing_symbol(self):
        record = {"headline_id": "abc123", "headline": "Apple stock rises"}
        assert validate_headline(record) is False

    def test_validate_headline_missing_headline(self):
        record = {"headline_id": "abc123", "symbol": "AAPL"}
        assert validate_headline(record) is False

    def test_insert_headline_calls_execute(self):
        """Mock PostgreSQL connection, verify INSERT is executed."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        record = {
            "headline_id": "abc123",
            "symbol": "AAPL",
            "published_at": "2 days ago",
            "source": "reuters",
            "headline": "Apple stock rises",
            "article_url": "https://example.com",
            "scraped_at": "2026-05-30T00:00:00Z",
        }
        insert_headline(mock_conn, record)
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()


# ── Price Consumer Tests ─────────────────────────────────────────────────────

class TestPriceConsumer:
    def test_validate_price_valid(self):
        record = {
            "symbol": "AAPL",
            "date": "2025-01-02",
            "open": 248.5,
            "high": 252.3,
            "low": 247.8,
            "close": 251.6,
            "volume": 55000000,
        }
        assert validate_price(record) is True

    def test_validate_price_missing_symbol(self):
        record = {"date": "2025-01-02", "close": 251.6}
        assert validate_price(record) is False

    def test_validate_price_missing_close(self):
        record = {"symbol": "AAPL", "date": "2025-01-02"}
        assert validate_price(record) is False

    def test_validate_price_none_close(self):
        record = {"symbol": "AAPL", "date": "2025-01-02", "close": None}
        assert validate_price(record) is False

    def test_insert_price_calls_execute(self):
        """Mock PostgreSQL connection, verify INSERT is executed."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        record = {
            "symbol": "AAPL",
            "date": "2025-01-02",
            "open": 248.5,
            "high": 252.3,
            "low": 247.8,
            "close": 251.6,
            "volume": 55000000,
            "adjusted_close": 251.6,
        }
        insert_price(mock_conn, record)
        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()
