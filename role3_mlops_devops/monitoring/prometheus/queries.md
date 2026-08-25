# Prometheus demo queries

Open Prometheus at **http://localhost:9090** → *Graph*, and paste any query below.

These metrics are recorded on **`POST /predict`** (label `source="api"`), so make
a few predictions first to populate them (see "Generate some traffic"). The daily
batch runs in a separate process and is **not** scraped here.

Metric families (from `app/metrics.py`):
`prediction_requests_total{source,outcome}` ·
`predictions_by_direction_total{source,direction}` ·
`prediction_confidence` (histogram) · `inference_latency_seconds` (histogram).

---

## 0. Generate some traffic (so the graphs aren't empty)

```bash
for i in $(seq 1 30); do
  curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' \
    -d @role3_mlops_devops/sample_payloads/predict_request.json > /dev/null
done
```

## 1. Service health

```promql
up{job="stock-direction-api"}                                  # 1 = scrape target healthy
process_resident_memory_bytes{job="stock-direction-api"} / 1e6 # API memory (MB)
rate(process_cpu_seconds_total{job="stock-direction-api"}[5m]) # API CPU (cores)
```

## 2. Request volume & errors (request count)

```promql
sum(prediction_requests_total)                                 # total requests handled
sum by (outcome) (prediction_requests_total)                   # success vs error split
sum(rate(prediction_requests_total[5m]))                       # requests / second (5m)
increase(prediction_requests_total{outcome="error"}[1h])       # errors in the last hour

# error ratio (0-1); returns no data if there has been no traffic
sum(rate(prediction_requests_total{outcome="error"}[5m]))
  / sum(rate(prediction_requests_total[5m]))
```

## 3. Predicted-class distribution

```promql
sum by (direction) (predictions_by_direction_total)            # count per class
topk(1, sum by (direction) (predictions_by_direction_total))   # dominant class

# share of each class (0-1)
sum by (direction) (predictions_by_direction_total)
  / scalar(sum(predictions_by_direction_total))

sum by (direction) (rate(predictions_by_direction_total[15m])) # recent class mix
```

## 4. Confidence distribution

```promql
# average confidence
sum(rate(prediction_confidence_sum[5m]))
  / sum(rate(prediction_confidence_count[5m]))

histogram_quantile(0.5, sum by (le) (rate(prediction_confidence_bucket[5m])))  # median
histogram_quantile(0.9, sum by (le) (rate(prediction_confidence_bucket[5m])))  # p90

# share of low-confidence predictions (<= 0.5)
sum(prediction_confidence_bucket{le="0.5"}) / sum(prediction_confidence_count)

# share of very-high-confidence predictions (> 0.99)
1 - sum(prediction_confidence_bucket{le="0.99"}) / sum(prediction_confidence_count)
```

## 5. Inference latency

```promql
# average latency (seconds)
sum(rate(inference_latency_seconds_sum[5m]))
  / sum(rate(inference_latency_seconds_count[5m]))

histogram_quantile(0.50, sum by (le) (rate(inference_latency_seconds_bucket[5m]))) # p50
histogram_quantile(0.95, sum by (le) (rate(inference_latency_seconds_bucket[5m]))) # p95
histogram_quantile(0.99, sum by (le) (rate(inference_latency_seconds_bucket[5m]))) # p99

# number of requests slower than 2.5s in the last 5m
sum(increase(inference_latency_seconds_count[5m]))
  - sum(increase(inference_latency_seconds_bucket{le="2.5"}[5m]))
```

---

## Notes

- **Short demo?** `rate()`/`histogram_quantile` over `[5m]` need a few scrapes
  (15s interval) with recent traffic. For an instant snapshot after a handful of
  calls, drop `rate()` and use the raw counters, e.g.
  `histogram_quantile(0.95, sum by (le) (inference_latency_seconds_bucket))`.
- Widen or narrow the window (`[5m]`, `[15m]`, `[1h]`) to match how long the
  service has been running.
- Metrics are **in-memory** and reset when the API container restarts.
- All application metrics also carry `job="stock-direction-api"` and an
  `instance` label added by Prometheus; add `{source="api"}` to be explicit.
