"""
Shared data contract between all three roles.
This defines the exact schema that Role 1 outputs and Role 2 consumes.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ── Schema: Clean Price Data (Role 1 → Role 2) ──────────────────────────────

@dataclass
class PriceRecord:
    """One row of clean OHLCV price data produced by the Spark processor."""
    symbol: str                    # e.g. "^GSPC", "AAPL"
    date: str                      # ISO-8601 date, e.g. "2025-06-01"
    open: float
    high: float
    low: float
    close: float
    volume: int


# ── Schema: Clean News/Headline Data (Role 1 → Role 2) ──────────────────────

@dataclass
class HeadlineRecord:
    """One row of cleaned news headline data."""
    headline_id: str               # unique identifier
    symbol: str                    # related ticker(s), comma-separated
    published_at: str              # ISO-8601 datetime
    source: str                    # e.g. "reuters", "yahoo_finance"
    headline: str                  # cleaned headline text
    author: Optional[str] = None   # author of the article
    body_snippet: Optional[str] = None  # first 500 chars of article body


# ── Schema: Model Prediction (Role 2 → Role 3) ──────────────────────────────

@dataclass
class PredictionRequest:
    """Input payload for the FastAPI prediction endpoint."""
    symbol: str
    date: str                      # prediction target date (market open)
    headlines: list[str] = field(default_factory=list)
    latest_close: Optional[float] = None


@dataclass
class PredictionResponse:
    """Output payload from the FastAPI prediction endpoint."""
    symbol: str
    date: str
    direction: str                 # "high" | "low" | "flat"
    confidence: float              # 0.0 – 1.0
    model_version: str             # MLflow run_id or model alias
    timestamp: str = ""            # ISO-8601 when prediction was made

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
