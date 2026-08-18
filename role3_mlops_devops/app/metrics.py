"""Prometheus metrics for the prediction service.

Exposed at ``GET /metrics`` for Prometheus to scrape. Covers the four
monitoring signals the project calls for:

* request count       -- ``prediction_requests_total{source,outcome}``
* latency histogram   -- ``inference_latency_seconds{source}``
* class distribution  -- ``predictions_by_direction_total{source,direction}``
* confidence spread   -- ``prediction_confidence{source}``

The ``source`` label distinguishes on-demand API calls (``api``) from the
daily batch, should the batch ever push metrics too. Metrics live in the
default process-global registry, so ``/metrics`` also exposes the standard
process/python collectors for free.

A single uvicorn worker is assumed. Multiple workers would each keep their own
in-memory counters, which needs prometheus_client's multiprocess mode.
"""

from prometheus_client import Counter, Histogram

# Confidence is in [0, 1] and for this model skews very high (~0.99), so the
# upper buckets are deliberately fine-grained to keep the distribution legible.
_CONFIDENCE_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)
# Single-symbol inference runs FinBERT on CPU: sub-second to a few seconds.
_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)

prediction_requests_total = Counter(
    "prediction_requests_total",
    "Total /predict requests handled, by outcome.",
    ["source", "outcome"],
)
predictions_by_direction_total = Counter(
    "predictions_by_direction_total",
    "Predictions served, by predicted class.",
    ["source", "direction"],
)
prediction_confidence = Histogram(
    "prediction_confidence",
    "Confidence score of served predictions.",
    ["source"],
    buckets=_CONFIDENCE_BUCKETS,
)
inference_latency_seconds = Histogram(
    "inference_latency_seconds",
    "Model inference latency in seconds.",
    ["source"],
    buckets=_LATENCY_BUCKETS,
)


def record_success(
    source: str, direction: str, confidence: float, latency_ms: float
) -> None:
    """Record one successful prediction across all four metrics."""
    prediction_requests_total.labels(source, "success").inc()
    predictions_by_direction_total.labels(source, direction).inc()
    prediction_confidence.labels(source).observe(confidence)
    inference_latency_seconds.labels(source).observe(latency_ms / 1000.0)


def record_error(source: str) -> None:
    """Record a failed/rejected request (bad input, missing model, crash)."""
    prediction_requests_total.labels(source, "error").inc()
