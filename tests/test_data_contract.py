"""Tests for the cross-role data contract (shared/schemas/data_contract.py).

Pure-stdlib dataclasses, so these run in any environment.
"""

from datetime import datetime

from shared.schemas.data_contract import (
    HeadlineRecord,
    PredictionRequest,
    PredictionResponse,
    PriceRecord,
)


def test_price_record_fields():
    rec = PriceRecord(
        symbol="RELIANCE", date="2026-07-08",
        open=1500.0, high=1510.0, low=1495.0, close=1505.0, volume=412000,
    )
    assert rec.symbol == "RELIANCE"
    assert rec.close == 1505.0
    assert rec.volume == 412000


def test_headline_record_optionals_default_none():
    rec = HeadlineRecord(
        headline_id="h-1", symbol="RELIANCE",
        published_at="2026-07-08T08:12:00+05:30",
        source="ET", headline="Reliance beats estimates",
    )
    assert rec.author is None
    assert rec.body_snippet is None


def test_prediction_request_defaults_are_independent():
    a = PredictionRequest(symbol="RELIANCE", date="2026-07-08")
    b = PredictionRequest(symbol="TCS", date="2026-07-08")
    assert a.headlines == [] and a.latest_close is None
    # default_factory must give each instance its own list, not a shared one.
    a.headlines.append("x")
    assert b.headlines == []


def test_prediction_response_autofills_timestamp():
    resp = PredictionResponse(
        symbol="RELIANCE", date="2026-07-08",
        direction="high", confidence=0.82, model_version="xgb-1",
    )
    assert resp.timestamp                      # non-empty
    datetime.fromisoformat(resp.timestamp)     # parseable ISO-8601


def test_prediction_response_keeps_explicit_timestamp():
    ts = "2026-07-08T09:30:00"
    resp = PredictionResponse(
        symbol="RELIANCE", date="2026-07-08",
        direction="flat", confidence=0.5, model_version="xgb-1", timestamp=ts,
    )
    assert resp.timestamp == ts
