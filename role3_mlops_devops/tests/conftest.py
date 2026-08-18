"""Test fixtures.

The tests run against the REAL artifacts (PCA, symbol encoder, feature columns,
XGBoost) but with FinBERT stubbed out -- ``embed_headlines`` is replaced with a
deterministic fake so the suite needs neither the ~440MB weights nor a network
call, yet still exercises the whole feature-engineering + prediction path.

Environment is set at import time, before ``app.config`` is first imported, so
everything runs against a throwaway SQLite file.
"""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_TMP = Path(tempfile.mkdtemp(prefix="direction-tests-"))
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP / 'test.db'}"
os.environ["MODEL_VERSION"] = "test-1.0.0"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app import pipeline  # noqa: E402

# Two real Nifty 50 constituents present in the symbol encoder.
SYMBOL_A = "Reliance Industries"
SYMBOL_B = "Infosys"


def _fake_embed(texts):
    """Deterministic stand-in for FinBERT: correct shapes, no weights."""
    n = len(texts)
    probs = np.tile(np.array([0.2, 0.2, 0.6]), (n, 1))  # positive, negative, neutral
    embs = np.vstack([np.full(768, (i % 7) * 0.01 + 0.001) for i in range(n)])
    id2label = {0: "positive", 1: "negative", 2: "neutral"}
    return probs, embs, id2label


# Patch before the app (and its lifespan warmup) is imported.
pipeline.embed_headlines = _fake_embed
pipeline._load_finbert = lambda: {"tokenizer": None, "model": None, "device": "cpu"}

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import db  # noqa: E402
from app.main import app  # noqa: E402


def make_ohlcv(symbol: str, days: int = 35, base: float = 1500.0) -> pd.DataFrame:
    """Synthetic 15-min bars with enough daily history for rolling features."""
    rows = []
    dates = pd.bdate_range("2026-05-01", periods=days)
    for d in range(days):
        day = dates[d]
        day_base = base + d  # gentle upward drift
        for b, (hh, mm) in enumerate([(9, 15), (9, 30), (9, 45)]):
            o = day_base + b
            c = o + 1.0
            rows.append(
                {
                    "symbol": symbol,
                    "datetime": f"{day.date()} {hh:02d}:{mm:02d}:00",
                    "open": o,
                    "high": c + 1.0,
                    "low": o - 1.0,
                    "close": c,
                    "volume": 100000 + d * 100 + b,
                }
            )
    return pd.DataFrame(rows)


def make_news(symbol: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "headline_id": f"{symbol[:3]}-1",
                "symbol": symbol,
                "published_at": "2026-06-18T08:12:00+05:30",
                "source": "Test Wire",
                "headline": f"{symbol} posts strong quarterly results",
            }
        ]
    )


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def clean_tables():
    db.init_db()
    with db.session_scope() as session:
        session.query(db.PipelinePrediction).delete()
        session.query(db.ApiPrediction).delete()
        session.commit()
    yield


@pytest.fixture
def predict_request():
    """A valid single-symbol POST /predict body."""
    ohlcv = make_ohlcv(SYMBOL_A)
    news = make_news(SYMBOL_A)
    return {
        "symbol": SYMBOL_A,
        "news": news.to_dict(orient="records"),
        "prices": ohlcv.drop(columns=[]).to_dict(orient="records"),
    }
