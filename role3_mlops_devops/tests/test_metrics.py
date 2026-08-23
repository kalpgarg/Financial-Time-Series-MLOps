"""Prometheus /metrics endpoint tests (FinBERT stubbed via conftest)."""

from tests.conftest import needs_artifacts

# Both tests issue a prediction, so they need the real model artifacts.
pytestmark = needs_artifacts


def test_metrics_endpoint_exposes_all_four_signals(client, predict_request):
    direction = client.post("/predict", json=predict_request).json()["direction"]

    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]

    body = resp.text
    # The four monitoring signals the project asks for.
    assert "prediction_requests_total" in body          # request count
    assert "inference_latency_seconds" in body          # latency histogram
    assert "predictions_by_direction_total" in body     # class distribution
    assert "prediction_confidence" in body              # confidence spread

    # The served prediction was actually recorded.
    assert f'direction="{direction}"' in body
    assert 'source="api"' in body
    assert 'outcome="success"' in body


def test_rejected_request_increments_error_counter(client):
    # An unknown symbol passes schema validation but fails inside the pipeline,
    # so it reaches the handler's error path (unlike a schema 422, which is
    # rejected before the endpoint runs and is not counted here).
    from tests.conftest import make_ohlcv

    ohlcv = make_ohlcv("NOT_A_REAL_SYMBOL")
    payload = {
        "symbol": "NOT_A_REAL_SYMBOL",
        "news": [],
        "prices": ohlcv.to_dict(orient="records"),
    }
    assert client.post("/predict", json=payload).status_code == 422

    body = client.get("/metrics").text
    assert 'outcome="error"' in body
