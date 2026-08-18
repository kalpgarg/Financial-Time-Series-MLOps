"""HTTP endpoints."""

import logging
import uuid
from datetime import date as date_type
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import text

from app import db, metrics, scoring
from app.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    StoredPrediction,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Label distinguishing on-demand API predictions from any future batch source.
_SOURCE = "api"


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe. Used by the Docker healthcheck and by CI after boot."""
    try:
        with db.session_scope() as session:
            session.execute(text("SELECT 1"))
        database = "ok"
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("health check: database unreachable")
        database = f"error: {exc}"

    return HealthResponse(
        status="ok" if database == "ok" and scoring.is_ready() else "degraded",
        model_loaded=scoring.is_ready(),
        model_version=scoring.model_version(),
        database=database,
    )


@router.get("/metrics", tags=["ops"], include_in_schema=False)
def metrics_endpoint() -> Response:
    """Prometheus scrape endpoint (text exposition format, not JSON)."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(request: PredictRequest) -> PredictResponse:
    """Score one symbol live and record the inference in ``api_predictions``.

    The request carries the symbol's own news and 15-minute OHLCV history; the
    pipeline predicts for the latest date present in ``prices``. Authoritative
    daily predictions come from the batch pipeline via ``GET /predictions``.
    """
    request_id = uuid.uuid4().hex
    news_df = pd.DataFrame([item.model_dump() for item in request.news])
    prices_df = pd.DataFrame([bar.model_dump() for bar in request.prices])

    try:
        row, latency_ms = scoring.score_one(request.symbol, news_df, prices_df)
    except FileNotFoundError as exc:
        metrics.record_error(_SOURCE)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, LookupError) as exc:
        # Bad/insufficient input, or an unknown symbol: a client error.
        metrics.record_error(_SOURCE)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        metrics.record_error(_SOURCE)
        logger.exception("inference failed", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail="inference failed") from exc

    # Prometheus: request count, class distribution, confidence + latency spread.
    metrics.record_success(_SOURCE, row["direction"], row["confidence"], latency_ms)

    # Logged for monitoring: latency, predicted class and confidence per call.
    logger.info(
        "prediction served",
        extra={
            "request_id": request_id,
            "symbol": row["symbol"],
            "direction": row["direction"],
            "confidence": row["confidence"],
            "latency_ms": round(latency_ms, 3),
            "model_version": row["model_version"],
        },
    )

    with db.session_scope() as session:
        db.insert_api_prediction(
            session, {**row, "request_id": request_id, "latency_ms": latency_ms}
        )

    return PredictResponse(
        **row, request_id=request_id, latency_ms=round(latency_ms, 3)
    )


@router.get("/predictions", response_model=List[StoredPrediction], tags=["history"])
def list_predictions(
    date: Optional[date_type] = Query(None, description="Trading date, YYYY-MM-DD"),
    symbol: Optional[str] = Query(None, description="e.g. Reliance Industries"),
    limit: int = Query(100, ge=1, le=1000),
) -> List[StoredPrediction]:
    """Read the authoritative batch predictions. This is what a frontend calls."""
    with db.session_scope() as session:
        rows = db.fetch_pipeline_predictions(
            session, on_date=date, symbol=symbol, limit=limit
        )
    return [StoredPrediction.model_validate(row) for row in rows]
