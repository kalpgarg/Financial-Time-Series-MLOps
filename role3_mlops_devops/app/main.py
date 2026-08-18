"""Application entrypoint.

    uvicorn app.main:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import db, scoring
from app.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables and load the artifacts + FinBERT once at boot, so the first
    # request does not pay for it and a missing artifact surfaces immediately
    # rather than on the first prediction.
    db.init_db()
    try:
        scoring.warmup()
    except FileNotFoundError as exc:
        logger.error("%s -- /predict will return 503 until this is fixed", exc)
    yield


app = FastAPI(
    title="Nifty 50 Direction Prediction Service",
    description=(
        "Serves direction predictions (Negative / Neutral / Positive) with "
        "class probabilities, via a FinBERT + PCA + XGBoost pipeline.\n\n"
        "* `POST /predict` — score one symbol on demand from raw news + OHLCV\n"
        "* `GET /predictions` — read the authoritative daily batch predictions\n"
        "* `GET /metrics` — Prometheus metrics (latency, request count, "
        "class + confidence distributions)"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/", include_in_schema=False)
def root():
    return {"service": "nifty50-direction", "docs": "/docs"}
