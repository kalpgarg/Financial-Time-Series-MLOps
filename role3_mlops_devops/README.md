# Nifty 50 Direction Prediction Service

Serving layer for the MLOps end-term project. Predicts a direction class —
`Negative` / `Neutral` / `Positive` — for each Nifty 50 constituent, with class
probabilities and a confidence score.

This role covers **Deployment (6)**, **Monitoring (7)** and **CI/CD (8)**.
Data ingestion, preprocessing and model training live in Roles 1 and 2.

## Dependencies on other roles

- **Consumes Role 2's artifacts** — `finbert_pca.pkl`, `symbol_encoder.pkl`,
  `feature_columns.pkl`, `xgboost_model.pkl` — read from the DVC-tracked
  `models/` directory (mounted at runtime); serving versions are pinned to match
  the training environment.
- **Consumes Role 1's data** — the cleaned `headlines.csv` and
  `merged_ohlc_15min.csv` feed `batch_score.py`.
- **Invoked by Role 1's inference DAG**, whose `run_inference` task runs this
  service's image (`fints-api:latest`) to score all constituents and write to the
  predictions database.

See the [root README](../README.md) for running the whole system together.

## The model

The prediction is a five-stage pipeline; XGBoost is only the last step. It is a
faithful port of the team's `stock_prediction.ipynb`.

| Stage | Artifact | Role |
|---|---|---|
| FinBERT | `ProsusAI/finbert` (HuggingFace) | Per-headline sentiment probs + 768-d [CLS] embedding |
| PCA | `finbert_pca.pkl` | Reduces embeddings 768 → 50 |
| News aggregation | — | Recency-weighted sentiment + embedding features per symbol |
| OHLCV features | — | ~40 daily features: returns, lags, rolling vol/MA (7/14/30d), volume, 09:15 & first-15-min |
| XGBoost | `xgboost_model.pkl` | 95 features → 3-class prediction + probabilities |

Also required: `symbol_encoder.pkl` (label-encodes the 50 company names, e.g.
"Reliance Industries") and `feature_columns.pkl` (the exact 95-feature order).

> **Version pins matter.** The artifacts were saved with **scikit-learn 1.6.1**
> and **xgboost 3.2.0**. Loading under other versions triggers an
> `InconsistentVersionWarning` and can silently produce wrong results, so these
> are pinned in `requirements.txt`. Do not bump them without re-exported
> artifacts.

## Architecture

Two paths, **one pipeline module** (`app/pipeline.py`).

```
                      ┌──────────────────────────────────────────┐
  Airflow DAG         │  app/pipeline.py                          │
  (daily) ──────────▶ │   FinBERT → PCA → feature eng → XGBoost   │ ◀── POST /predict
  batch_score.py      └──────────────────────────────────────────┘     (one symbol,
  (all constituents)          │                    │                     raw news+OHLCV)
        │                     ▼                    ▼                          │
        ▼            finbert_pca.pkl        xgboost_model.pkl                 ▼
  pipeline_predictions  symbol_encoder.pkl  feature_columns.pkl        api_predictions
  (authoritative)              + FinBERT (baked into image)            (ad-hoc, logged)
        │
        ▼
  GET /predictions ──▶ frontend
```

**Why batch, not request-time inference.** Running FinBERT over every headline
for all constituents is heavy, and the OHLCV features need ≥30 trading days of
history plus a 7-day news window. So predictions are computed once by the daily
pipeline, stored, and served as reads — decoupling model execution from request
serving and leaving a durable, auditable prediction history.

`POST /predict` exists so the model is genuinely *served*, not just executed in
a DAG. It scores a **single symbol** from data supplied in the request. Both
paths call `app/pipeline.py`, so the API and the pipeline cannot compute a
prediction differently.

### Why two tables

| | `pipeline_predictions` | `api_predictions` |
|---|---|---|
| Source | Airflow batch run | `POST /predict` |
| Cardinality | one row per (symbol, date) | unbounded |
| Authoritative | yes — drives the frontend and accuracy metrics | no |
| Extra fields | `run_id` | `latency_ms`, `request_id` |

Both store `direction`, `confidence`, the three class probabilities
(`prob_negative/neutral/positive`), `article_count`, `weighted_sentiment`,
`model_version` and `timestamp`. Ad-hoc calls are kept out of the authoritative
table so demo calls cannot pollute the history accuracy is computed from.
`pipeline_predictions` has a `UNIQUE(symbol, date)` constraint and is written
with an upsert, so an Airflow retry updates rather than duplicates — the scoring
task is safe to run more than once.

## Running with Docker (Point 6)

The compose stack is the intended way to run the service: the FastAPI container
(with FinBERT and the artifacts baked in) plus a Postgres backend. The API waits
for Postgres to become healthy before it starts.

```bash
docker compose up --build           # API at http://localhost:8000/docs
docker compose down                 # stop  (add -v to wipe the database volume)
```

The build installs **CPU-only torch** and downloads the **FinBERT weights**
(~440MB) into the image, so the container runs fully offline and boots without a
network call. Expect an image of ~1.5–2GB and a first-boot warmup of a few
seconds (allowed for by the healthcheck's start period).

To serve retrained artifacts without rebuilding, mount them over the baked-in
ones and bump `MODEL_VERSION` — see the commented block in
[docker-compose.yml](docker-compose.yml).

## Local setup (without Docker)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install torch==2.13.0            # separate on purpose — see requirements.txt
```

> torch is installed separately so it is never pulled as the multi-GB CUDA
> build. On macOS `pip install torch==2.13.0` gives the CPU/MPS wheel; the
> Docker image installs it from PyTorch's CPU-only index.

The 4 pickled artifacts are in [artifacts/](artifacts/). FinBERT downloads from
HuggingFace on first use unless already cached.

> **macOS note:** torch and xgboost each bring an OpenMP runtime, which can
> deadlock. If a run hangs at model load, prefix commands with
> `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1`. The Docker image sets
> `OMP_NUM_THREADS=1` and does not hit this.

## Running (local)

```bash
# API — interactive docs at http://localhost:8000/docs
uvicorn app.main:app --reload

# Batch scoring (what the Airflow task calls) — reads the two data files
python batch_score.py \
  --news "../Prediction Artifacts/headlines.csv" \
  --ohlcv "../Prediction Artifacts/merged_ohlc_15min.csv" \
  --run-id demo__001

# Tests (FinBERT is stubbed; no weights or network needed)
pytest -q
```

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite:///predictions.db` | Swap for `postgresql://…`; the upsert already handles both dialects |
| `ARTIFACTS_DIR` | `artifacts/` | Where the four `.pkl` files live |
| `FINBERT_MODEL` | `ProsusAI/finbert` | Sentiment model id (or local path) |
| `MODEL_VERSION` | `finbert-pca-xgb-1.0.0` | Stamped onto every stored prediction |
| `MARKET_TZ` | `Asia/Kolkata` | Market timezone for trading-date resolution |
| `TRANSFORMERS_OFFLINE` / `HF_HUB_OFFLINE` | unset (local) / `1` (Docker) | Force FinBERT to load from cache |
| `SQL_ECHO` | `false` | Log emitted SQL |

## API

### `GET /health`
Liveness probe — reports model and database status. Used by the Docker
healthcheck and by CI after container boot.

```json
{"status":"ok","model_loaded":true,"model_version":"finbert-pca-xgb-1.0.0","database":"ok"}
```

### `POST /predict`
Scores **one symbol** from data in the request, and records the inference in
`api_predictions`. Supply the symbol's 15-minute OHLCV history (the pipeline
predicts for the latest date present in `prices`) and any recent headlines.

```bash
curl -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d @sample_payloads/predict_request.json
```

Request (`symbol`, `prices` required; `news` optional):

```json
{
  "symbol": "Reliance Industries",
  "news":   [{"headline_id":"h-1","symbol":"Reliance Industries","published_at":"2026-07-07T08:12:00+05:30","source":"ET","headline":"Reliance posts strong results"}],
  "prices": [{"symbol":"Reliance Industries","datetime":"2026-07-08 09:15:00","open":1500.0,"high":1508.5,"low":1498.2,"close":1506.0,"volume":412000}]
}
```

Response:

```json
{
  "symbol":"Reliance Industries","date":"2026-07-08",
  "direction":"Neutral","confidence":0.9993,
  "prob_negative":0.0004,"prob_neutral":0.9993,"prob_positive":0.0003,
  "article_count":24,"weighted_sentiment":0.0466,
  "model_version":"finbert-pca-xgb-1.0.0","timestamp":"2026-07-08T...Z",
  "request_id":"4dfa71...","latency_ms":837.4
}
```

Returns `422` for a symbol/row mismatch, an inconsistent OHLC bar
(`high < low`, close outside range), empty `prices`, insufficient data, or a
symbol the encoder never saw. `503` if the artifacts are missing.

### `GET /predictions`
Reads the authoritative batch predictions. This is the frontend's endpoint.

```bash
curl "localhost:8000/predictions?date=2026-07-08"
curl "localhost:8000/predictions?symbol=Reliance%20Industries&limit=30"
```

## Monitoring (Point 7)

Every `POST /predict` updates Prometheus metrics, exposed at `GET /metrics` in
the standard text exposition format. Four signals cover the model's behaviour:

| Metric | Type | What it shows |
|---|---|---|
| `prediction_requests_total{source,outcome}` | counter | request count + success/error split |
| `inference_latency_seconds{source}` | histogram | inference-latency distribution |
| `predictions_by_direction_total{source,direction}` | counter | class distribution (Negative/Neutral/Positive) |
| `prediction_confidence{source}` | histogram | confidence-score distribution |

`GET /metrics` also exposes the default process/python collectors. Prometheus
scrapes the API every 15s (config in `monitoring/prometheus/prometheus.yml`).
With the stack up:

- Metrics endpoint — http://localhost:8000/metrics
- Prometheus UI — http://localhost:9090

Example PromQL:

```promql
sum by (direction) (predictions_by_direction_total)                              # class mix
histogram_quantile(0.95, sum by (le) (rate(inference_latency_seconds_bucket[5m])))  # p95 latency
sum(rate(prediction_requests_total{outcome="error"}[5m]))                        # error rate
```

> Metrics are per-process and in-memory (single uvicorn worker; multiple workers
> would need prometheus_client's multiprocess mode). The daily batch runs as a
> separate short-lived process, so its predictions aren't scraped here — pushing
> them via a Prometheus Pushgateway is a straightforward addition.

## Folder structure

```
app/
  config.py             Environment-driven configuration (artifact paths, model version)
  pipeline.py           THE scoring pipeline: FinBERT → PCA → features → XGBoost
  scoring.py            Serving envelope (stamps version + timestamp) over pipeline
  schemas.py            Pydantic request/response validation
  db.py                 SQLAlchemy models, upsert, queries
  metrics.py            Prometheus metric definitions + recording helpers
  routes.py             /health, /predict, /predictions, /metrics
  main.py               App factory; warms up artifacts + FinBERT at boot
batch_score.py          Reads news + OHLCV CSVs, scores all symbols (the Airflow task)
monitoring/
  prometheus/prometheus.yml   Prometheus scrape config (targets the API /metrics)
sample_payloads/        Example /predict request (real Reliance data)
tests/                  Endpoint, pipeline, metrics and idempotency tests (FinBERT stubbed)
Dockerfile              CPU torch + baked-in FinBERT, non-root
docker-compose.yml      API + Postgres + Prometheus stack (Points 6 & 7)
```

> The four `.pkl` artifacts are not in this folder — they are read from the
> repo-level `models/` directory (mounted at `/app/models` in Docker; see
> `ARTIFACTS_DIR` in [Configuration](#configuration)).

## Handing over retrained artifacts

Replace the four files in the repo-level `models/` directory and bump
`MODEL_VERSION`. The pipeline expects them to remain compatible with the code in
`app/pipeline.py`:

- `xgboost_model.pkl` — `XGBClassifier`, 95 features, `classes_ == [0,1,2]`
- `feature_columns.pkl` — list of the 95 feature names, in model order
- `symbol_encoder.pkl` — `LabelEncoder` over the 50 company names
- `finbert_pca.pkl` — `PCA`, 768 → 50

Keep them consistent with the pinned `scikit-learn` / `xgboost` versions, or
re-pin to whatever they were re-exported with.

## Known limitations & notes

- **09:15-feature shift + latest-date NaN (partly fixed, one part open).**
  The 09:15 features (`open_915`, `close_915`, `gap_from_prev_close`,
  `first15_return`, `first15_direction`) are built with a **single**-day shift
  to match role2's `feature_engineering.add_next_day_features` (the current
  models are trained that way). The earlier two-day shift — copied from the old
  notebook — was the bug role2 fixed; role3 now matches.
  **Still open:** this function returns the row for the *latest* OHLC date,
  whose 09:15 features are NaN (its D+1 doesn't exist). role2 trains rows as
  "predict D+1 from D's daily features + D+1's 09:15", so to actually use these
  features at serving, role3 should predict the row that carries the most recent
  *real* 09:15 bar (`latest_date - 1`, target `latest_date`) and align the news
  merge accordingly. Until then the served latest-date row still has NaN 09:15
  features, so predictions skew toward `Neutral`. Needs Role 2's serving-intent
  confirmation + a re-validation before changing the date-selection logic.
- **48 of 50 symbols** are scored on the sample data: two have no bar on the
  latest date and are correctly excluded (can't predict without that day's data).
- **XGBoost load warning.** The model was pickled rather than saved via
  `Booster.save_model`, so loading logs a forward-compat warning. Harmless;
  re-export with `save_model` to silence it.
- **SQLite drops timezones** (`DateTime(timezone=True)` is a no-op). Resolves on
  Postgres; no code change needed.
- **`init_db()` creates tables but never migrates them.** Use Alembic once the
  schema changes against a live Postgres.

## Roadmap

- [x] Dockerfile + docker-compose (API + Postgres) — **Point 6**
- [x] Real FinBERT + PCA + XGBoost pipeline integrated and validated end-to-end
- [x] Prometheus `/metrics`: latency histogram, request count, class + confidence distributions — **Point 7**
- [ ] Drift detection against the training-time feature baseline
- [ ] GitHub Actions: lint → tests → build image → boot container → hit `/health` and `/predict` → push
- [ ] Minimal frontend over `GET /predictions`
```