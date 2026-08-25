"""Serving envelope around :mod:`app.pipeline`.

The pipeline returns raw prediction rows; this layer stamps each with the model
version and a generation timestamp and shapes it into the dict stored in the
database. Both ``app/routes.py`` and ``batch_score.py`` go through here.
"""

import time
from datetime import date as date_type
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import pandas as pd

from app import pipeline
from app.config import MODEL_VERSION

# Re-exported so callers can drive readiness/warmup without importing pipeline.
warmup = pipeline.warmup
is_ready = pipeline.is_ready


def model_version() -> str:
    return MODEL_VERSION


def _to_row(record: Dict) -> Dict:
    """Add the serving envelope (version + UTC timestamp) to one prediction."""
    d = record["date"]
    if isinstance(d, (pd.Timestamp, datetime)):
        d = d.date()
    return {
        "symbol": record["symbol"],
        "date": d,
        "direction": record["direction"],
        "confidence": float(record["confidence"]),
        "prob_negative": float(record["prob_negative"]),
        "prob_neutral": float(record["prob_neutral"]),
        "prob_positive": float(record["prob_positive"]),
        "article_count": int(record["article_count"]),
        "weighted_sentiment": float(record["weighted_sentiment"]),
        "model_version": MODEL_VERSION,
        # Stored timezone-aware in UTC; rendered in IST at the edges.
        "timestamp": datetime.now(timezone.utc),
    }


def score(
    news_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    embed_fn: Optional[Callable] = None,
    as_of_date: Optional[str] = None,
) -> List[Dict]:
    """Run the pipeline and return DB-ready rows, one per symbol.

    ``as_of_date`` (YYYY-MM-DD) scores a specific date instead of the latest one.
    """
    result = pipeline.score_frames(
        news_df, ohlcv_df, embed_fn=embed_fn, as_of_date=as_of_date
    )
    return [_to_row(rec) for rec in result.to_dict(orient="records")]


def score_one(
    symbol: str,
    news_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    embed_fn: Optional[Callable] = None,
) -> tuple[Dict, float]:
    """Score a single symbol for ``POST /predict``.

    Returns ``(row, latency_ms)``. Raises ``LookupError`` if the pipeline does
    not produce a row for the requested symbol.
    """
    started = time.perf_counter()
    rows = score(news_df, ohlcv_df, embed_fn=embed_fn)
    latency_ms = (time.perf_counter() - started) * 1000

    for row in rows:
        if row["symbol"] == symbol:
            return row, latency_ms
    raise LookupError(f"pipeline produced no prediction for {symbol!r}")
