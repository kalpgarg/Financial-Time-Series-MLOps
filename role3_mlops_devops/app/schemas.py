"""Request/response schemas -- the wire contract agreed with the team.

Validation lives here so a mismatch between the data the pipeline receives and
what the model expects fails loudly at the boundary with a 422, instead of
silently producing a garbage prediction.
"""

from datetime import date as date_type
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# The model's native 3-class output, kept as-is.
Direction = Literal["Negative", "Neutral", "Positive"]


class NewsItem(BaseModel):
    """One headline. Extra CSV columns (article_url, scraped_at) are ignored."""

    headline_id: str
    symbol: str
    published_at: str          # ISO 8601; parsed to UTC in the pipeline
    source: str
    headline: str


class PriceBar(BaseModel):
    """One 15-minute OHLCV bar."""

    symbol: str
    datetime: str
    open: float
    high: float
    low: float
    close: float
    volume: int

    @model_validator(mode="after")
    def check_bar_consistency(self):
        if self.high < self.low:
            raise ValueError("high cannot be lower than low")
        if not (self.low <= self.open <= self.high):
            raise ValueError("open must lie between low and high")
        if not (self.low <= self.close <= self.high):
            raise ValueError("close must lie between low and high")
        if self.volume < 0:
            raise ValueError("volume cannot be negative")
        return self


class PredictRequest(BaseModel):
    """One symbol per request, with enough data for the pipeline to run.

    The model needs history: rolling features span up to 30 trading days, and
    the news window is the last 7 calendar days. Supply the full 15-minute bar
    history you have for the symbol and any recent headlines; the pipeline
    predicts for the latest date present in ``prices``.
    """

    symbol: str = Field(..., min_length=1)
    news: List[NewsItem] = Field(default_factory=list)
    prices: List[PriceBar] = Field(..., min_length=1)

    @model_validator(mode="after")
    def check_symbols_match(self):
        mismatched = {
            item.symbol
            for item in (*self.news, *self.prices)
            if item.symbol != self.symbol
        }
        if mismatched:
            raise ValueError(
                f"all news and price rows must be for {self.symbol!r}; "
                f"also found {sorted(mismatched)}"
            )
        return self


class PredictResponse(BaseModel):
    symbol: str
    date: date_type
    direction: Direction
    confidence: float
    prob_negative: float
    prob_neutral: float
    prob_positive: float
    article_count: int
    weighted_sentiment: float
    model_version: str
    timestamp: datetime
    request_id: str
    latency_ms: float


class StoredPrediction(BaseModel):
    symbol: str
    date: date_type
    direction: Direction
    confidence: float
    prob_negative: float
    prob_neutral: float
    prob_positive: float
    article_count: int
    weighted_sentiment: float
    model_version: str
    timestamp: datetime
    run_id: str

    model_config = {"from_attributes": True}


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str]
    database: str
