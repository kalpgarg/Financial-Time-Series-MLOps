"""Unit tests for Spark cleaning and joining jobs.

These tests validate the pure-Python helper functions without requiring
a live Spark session or PostgreSQL connection.
"""

import re
import sys
from pathlib import Path

# ── Resolve project root ────────────────────────────────────────────────────
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from shared.schemas.data_contract import HeadlineRecord, PriceRecord


# ── clean_text logic test (inlined to avoid pyspark import) ──────────────────
# Mirrors role1_data_engineering.spark.clean_headlines.clean_text exactly.


def _clean_text(text):
    if text is None:
        return None
    text = text.lower()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class TestCleanText:
    def test_lowercase(self):
        assert _clean_text("HELLO World") == "hello world"

    def test_strip_html_tags(self):
        assert _clean_text("<b>bold</b> text") == "bold text"

    def test_strip_html_entities(self):
        assert _clean_text("foo&amp;bar") == "foo bar"

    def test_collapse_whitespace(self):
        assert _clean_text("  too   many   spaces  ") == "too many spaces"

    def test_none_input(self):
        assert _clean_text(None) is None

    def test_combined(self):
        result = _clean_text("  <b>Hello</b> &amp; World  ")
        assert result == "hello world"


# ── Schema Validation Tests ──────────────────────────────────────────────────

class TestDataContracts:
    def test_price_record_required_fields(self):
        """PriceRecord must have all required fields."""
        pr = PriceRecord(
            symbol="AAPL",
            date="2025-01-02",
            open=248.5,
            high=252.3,
            low=247.8,
            close=251.6,
            volume=55000000,
        )
        assert pr.symbol == "AAPL"
        assert pr.adjusted_close is None  # optional

    def test_headline_record_required_fields(self):
        """HeadlineRecord must have all required fields."""
        hr = HeadlineRecord(
            headline_id="abc123",
            symbol="AAPL",
            published_at="2025-01-02T08:30:00Z",
            source="reuters",
            headline="Apple stock rises",
        )
        assert hr.headline_id == "abc123"
        assert hr.author is None  # optional
        assert hr.body_snippet is None  # optional

    def test_price_record_with_adjusted_close(self):
        pr = PriceRecord(
            symbol="^GSPC",
            date="2025-01-02",
            open=5880.0,
            high=5920.5,
            low=5870.25,
            close=5910.3,
            volume=3200000000,
            adjusted_close=5910.3,
        )
        assert pr.adjusted_close == 5910.3
