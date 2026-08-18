"""Database models and session handling.

Two tables, deliberately kept separate:

``pipeline_predictions``
    The authoritative daily record produced by the Airflow batch run. One row
    per (symbol, date), enforced by a unique constraint so a retried task
    updates rather than duplicates. Carries ``run_id`` so every prediction
    traces back to a specific pipeline run. This is what the frontend reads.

``api_predictions``
    Ad-hoc inferences served by ``POST /predict``. Not authoritative, never one
    per day, and kept out of the table above so demo calls cannot pollute the
    history the accuracy metrics are computed from. Carries ``latency_ms`` and
    ``request_id`` for API monitoring.

Both store the model's three class probabilities plus the two headline news
features, so the frontend and monitoring have more than just the winning label.
"""

from datetime import date as date_type
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    select,
)
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.config import DATABASE_URL, SQL_ECHO


class Base(DeclarativeBase):
    pass


class _PredictionMixin:
    """Columns shared by both prediction tables."""

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    date: Mapped[date_type] = mapped_column(Date, nullable=False)
    direction: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    prob_negative: Mapped[float] = mapped_column(Float, nullable=False)
    prob_neutral: Mapped[float] = mapped_column(Float, nullable=False)
    prob_positive: Mapped[float] = mapped_column(Float, nullable=False)
    article_count: Mapped[int] = mapped_column(Integer, nullable=False)
    weighted_sentiment: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class PipelinePrediction(_PredictionMixin, Base):
    __tablename__ = "pipeline_predictions"

    run_id: Mapped[str] = mapped_column(String(128), nullable=False)

    __table_args__ = (
        # Makes the Airflow scoring task idempotent: a retry upserts.
        UniqueConstraint("symbol", "date", name="uq_pipeline_symbol_date"),
        Index("ix_pipeline_date", "date"),
    )


class ApiPrediction(_PredictionMixin, Base):
    __tablename__ = "api_predictions"

    request_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)


# Columns the upsert refreshes on conflict (everything except the key + id).
_UPSERT_FIELDS = (
    "direction",
    "confidence",
    "prob_negative",
    "prob_neutral",
    "prob_positive",
    "article_count",
    "weighted_sentiment",
    "model_version",
    "timestamp",
    "run_id",
)

_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        kwargs: Dict[str, Any] = {"echo": SQL_ECHO, "future": True}
        if DATABASE_URL.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        _engine = create_engine(DATABASE_URL, **kwargs)
    return _engine


def get_sessionmaker():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionLocal


def init_db() -> None:
    """Create tables if absent. Fine at this size; a live Postgres whose schema
    keeps changing would use Alembic migrations instead."""
    Base.metadata.create_all(get_engine())


def session_scope():
    return get_sessionmaker()()


# -- writes ----------------------------------------------------------------


def upsert_pipeline_predictions(session, rows: Iterable[Dict[str, Any]]) -> int:
    """Insert-or-update batch predictions keyed on (symbol, date).

    Written against both dialects so moving SQLite -> Postgres needs no code
    change: both expose ``on_conflict_do_update`` with the same API.
    """
    rows = list(rows)
    if not rows:
        return 0

    dialect = session.get_bind().dialect.name
    insert = postgresql.insert if dialect == "postgresql" else sqlite.insert

    stmt = insert(PipelinePrediction).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["symbol", "date"],
        set_={f: getattr(stmt.excluded, f) for f in _UPSERT_FIELDS},
    )
    session.execute(stmt)
    session.commit()
    return len(rows)


def insert_api_prediction(session, row: Dict[str, Any]) -> None:
    session.add(ApiPrediction(**row))
    session.commit()


# -- reads -----------------------------------------------------------------


def fetch_pipeline_predictions(
    session,
    on_date: Optional[date_type] = None,
    symbol: Optional[str] = None,
    limit: int = 100,
) -> List[PipelinePrediction]:
    stmt = select(PipelinePrediction)
    if on_date is not None:
        stmt = stmt.where(PipelinePrediction.date == on_date)
    if symbol is not None:
        stmt = stmt.where(PipelinePrediction.symbol == symbol)
    stmt = stmt.order_by(
        PipelinePrediction.date.desc(), PipelinePrediction.symbol.asc()
    ).limit(limit)
    return list(session.execute(stmt).scalars())
