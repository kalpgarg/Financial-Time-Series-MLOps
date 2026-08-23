"""Endpoint, pipeline and batch tests (FinBERT stubbed via conftest).

These double as the CI integration check: the pipeline boots the app and runs
this suite, exercising the real PCA/encoder/XGBoost artifacts end to end.
"""

import copy

import pandas as pd

from app import db
from batch_score import run_batch
from tests.conftest import (
    SYMBOL_A,
    SYMBOL_B,
    make_news,
    make_ohlcv,
    needs_artifacts,
)

VALID_DIRECTIONS = {"Negative", "Neutral", "Positive"}


@needs_artifacts
def test_health_reports_model_and_database(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "test-1.0.0"
    assert body["database"] == "ok"


@needs_artifacts
def test_predict_returns_contract_fields(client, predict_request):
    response = client.post("/predict", json=predict_request)
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["symbol"] == SYMBOL_A
    assert body["direction"] in VALID_DIRECTIONS
    assert 0.0 <= body["confidence"] <= 1.0
    probs = [body["prob_negative"], body["prob_neutral"], body["prob_positive"]]
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert abs(sum(probs) - 1.0) < 1e-4          # a proper distribution
    assert body["model_version"] == "test-1.0.0"
    assert body["request_id"]
    assert body["latency_ms"] >= 0


@needs_artifacts
def test_predict_is_recorded_in_api_predictions_only(client, predict_request):
    request_id = client.post("/predict", json=predict_request).json()["request_id"]

    with db.session_scope() as session:
        rows = session.query(db.ApiPrediction).all()
        assert len(rows) == 1
        assert rows[0].request_id == request_id
        assert rows[0].latency_ms >= 0
        # Ad-hoc calls must never leak into the authoritative table.
        assert session.query(db.PipelinePrediction).count() == 0


def test_mismatched_symbol_is_rejected(client, predict_request):
    payload = copy.deepcopy(predict_request)
    payload["prices"][0]["symbol"] = SYMBOL_B
    assert client.post("/predict", json=payload).status_code == 422


def test_inconsistent_ohlc_bar_is_rejected(client, predict_request):
    payload = copy.deepcopy(predict_request)
    payload["prices"][0]["high"] = 1.0  # below low
    assert client.post("/predict", json=payload).status_code == 422


def test_empty_price_list_is_rejected(client, predict_request):
    payload = copy.deepcopy(predict_request)
    payload["prices"] = []
    assert client.post("/predict", json=payload).status_code == 422


@needs_artifacts
def test_unknown_symbol_is_rejected(client):
    """A symbol the encoder never saw must be a client error, not a 500."""
    ohlcv = make_ohlcv("NOT_A_REAL_SYMBOL")
    payload = {
        "symbol": "NOT_A_REAL_SYMBOL",
        "news": [],
        "prices": ohlcv.to_dict(orient="records"),
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
    assert "not seen during training" in resp.json()["detail"]


def _two_symbol_frames():
    ohlcv = pd.concat([make_ohlcv(SYMBOL_A), make_ohlcv(SYMBOL_B, base=1600.0)])
    news = pd.concat([make_news(SYMBOL_A), make_news(SYMBOL_B)])
    return news, ohlcv


@needs_artifacts
def test_batch_scores_all_symbols():
    news, ohlcv = _two_symbol_frames()
    rows = run_batch(news, ohlcv, run_id="run-1")
    assert {r["symbol"] for r in rows} == {SYMBOL_A, SYMBOL_B}
    for r in rows:
        assert r["direction"] in VALID_DIRECTIONS
        assert r["run_id"] == "run-1"


@needs_artifacts
def test_batch_rerun_upserts_instead_of_duplicating():
    """An Airflow retry must not create a second row per symbol/date."""
    news, ohlcv = _two_symbol_frames()
    run_batch(news, ohlcv, run_id="run-1")
    run_batch(news, ohlcv, run_id="run-2")

    with db.session_scope() as session:
        rows = session.query(db.PipelinePrediction).all()
        assert len(rows) == 2                     # one per symbol, not four
        assert {r.run_id for r in rows} == {"run-2"}  # latest run wins


@needs_artifacts
def test_predictions_endpoint_serves_batch_rows(client):
    news, ohlcv = _two_symbol_frames()
    run_batch(news, ohlcv, run_id="run-1")

    rows = client.get("/predictions").json()
    assert {r["symbol"] for r in rows} == {SYMBOL_A, SYMBOL_B}
    assert all(r["run_id"] == "run-1" for r in rows)

    filtered = client.get("/predictions", params={"symbol": SYMBOL_A}).json()
    assert len(filtered) == 1 and filtered[0]["symbol"] == SYMBOL_A
    assert client.get("/predictions", params={"symbol": "NOPE"}).json() == []
