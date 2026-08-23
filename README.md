# Financial Time-Series MLOps

An end-to-end MLOps pipeline that predicts the **next-day market-open price
direction** — Negative / Neutral / Positive (down / flat / up) — for the 50
NSE F&O constituent stocks, using historical **OHLCV price data** and
**financial-news sentiment**.

The model is a five-stage pipeline: **FinBERT** sentiment on headlines → **PCA**
(768→50) → recency-weighted news-feature aggregation → OHLCV technical-indicator
feature engineering → an **XGBoost / LightGBM** classifier producing a
three-class direction with class probabilities. Predictions are produced daily
in batch and served through a containerised **FastAPI** service.

---

## System architecture

The project is split into three roles, glued by a `shared/` module (central
config + cross-role data contracts) and versioned with **DVC**.

| Role | Directory | Responsibility | Details |
|---|---|---|---|
| 1 | `role1_data_engineering/` | Scraping, PySpark cleaning, Airflow orchestration of the daily inference pipeline | [README](role1_data_engineering/README.md) |
| 2 | `role2_ml_modeling/` | Feature engineering, XGBoost/LightGBM training, MLflow tracking | [README](role2_ml_modeling/README.md) |
| 3 | `role3_mlops_devops/` | FastAPI serving, Docker, Prometheus monitoring, CI/CD | [README](role3_mlops_devops/README.md) |

### Data flow

```
Role 1 (ingest + clean)            Role 2 (model)             Role 3 (serve)
scrapers ── PySpark clean/merge ──> feature engineering ──> FastAPI + batch_score
   │            │                      │  train (XGB/LGBM)      │  /predict, /predictions
   └── data/*.csv (DVC) ──────────────┴── models/*.pkl (DVC) ──┴── predictions DB
                                          (MLflow tracking)        (Prometheus /metrics)
```

### Tech stack

Airflow · PySpark · Crawl4AI · tvDatafeed · FinBERT (Transformers/PyTorch) ·
scikit-learn (PCA) · XGBoost · LightGBM · MLflow · DVC (Google Drive remote) ·
FastAPI · Docker / Docker Compose · Prometheus · PostgreSQL · GitHub Actions ·
`uv` (Python 3.12).

### Mandatory-components coverage

| Requirement | Where |
|---|---|
| Version control (Git) | this repository |
| Airflow + one more tool (**PySpark**) | `role1_data_engineering/` (2 DAGs, Spark jobs) |
| Data processing | scraping → Spark cleaning → FinBERT → feature engineering |
| ≥2 models + MLflow | `role2_ml_modeling/` (XGBoost + LightGBM, MLflow) |
| DVC dataset/model versioning | `data/*.dvc`, `models/*.dvc` |
| Deployment (FastAPI + Docker) | `role3_mlops_devops/` |
| Monitoring | Prometheus `/metrics` + prediction logging (Role 3) |
| CI/CD | `.github/workflows/ci.yml` |
| Documentation | this README + per-role READMEs |

---

## Repository layout

```
shared/                     config.py, schemas/data_contract.py
data/                       DVC-tracked CSVs (stock_list.csv committed)
models/                     DVC-tracked model artifacts (.pkl)
role1_data_engineering/     scrapers/, spark/, airflow/dags/, tests/
role2_ml_modeling/          features/, models/, training/, airflow/dags/, notebooks/, tests/
role3_mlops_devops/         app/, batch_score.py, Dockerfile, docker-compose.yml, monitoring/, tests/
tests/                      root-level tests (shared contract + config)
Dockerfile                  multi-stage image (api / airflow / mlflow targets)
docker-compose.yml          whole-project stack
deploy/postgres-init/       creates the airflow + mlflow databases on first boot
requirements-test.txt       light deps to run the test suite (used by CI)
.github/workflows/ci.yml    CI test gate
pyproject.toml, uv.lock     uv-managed environment (Python 3.12)
```

---

## Prerequisites

- **Docker** (Docker Desktop) for the whole-system stack.
- **`uv`** (`pip install uv`) for local development on Python 3.12.
- Access to the **DVC remote** (Google Drive) to pull data + models.

---

## Run the whole system

One command brings up the entire stack — FastAPI service, PostgreSQL, Prometheus,
MLflow, and Airflow (webserver + scheduler) — from the root
`Dockerfile` + `docker-compose.yml`.

```bash
# 1. Pull DVC-tracked data + model artifacts (the API needs models/ to predict)
uv run dvc pull            # or: dvc pull

# 2. Build and start everything
docker compose up --build
```

| Service | URL | Notes |
|---|---|---|
| Airflow | http://localhost:8080 | login **admin / admin** |
| MLflow | http://localhost:5000 | experiment tracking |
| FastAPI (Swagger) | http://localhost:8000/docs | prediction API |
| Prometheus | http://localhost:9090 | scrapes the API `/metrics` |

The API mounts `models/` read-only; without a `dvc pull`, `/predict` returns 503
until the artifacts are present. Stop with `docker compose down` (add `-v` to
wipe the database + MLflow volumes).

### Triggering the pipelines

Both DAGs load automatically in the Airflow UI. To run the inference pipeline on
the existing DVC data (`SKIP_SCRAPERS=1` is set in the stack):

```bash
docker compose exec airflow-scheduler airflow dags unpause financial_ts_inference_pipeline
docker compose exec airflow-scheduler airflow dags trigger financial_ts_inference_pipeline
# then read the results:
curl "localhost:8000/predictions?limit=5"
```

---

## Pipelines

**Inference DAG** — `financial_ts_inference_pipeline` (Role 1), daily on weekdays:

```
[scrape_headlines, scrape_ohlcv] -> scrape_preopen
   -> [spark_clean_headlines, spark_merge_preopen] -> trim_for_inference
   -> run_inference (Docker) -> dvc_push_data
```
With `SKIP_SCRAPERS=1`, scrape/clean/merge/push are skipped and it runs
`trim_for_inference -> run_inference` on existing data. Live scraping
(`SKIP_SCRAPERS=0`) additionally requires the scraper dependencies
(Crawl4AI + browser, tvDatafeed) in the Airflow image.

**Training DAG** — `financial_ts_training_pipeline` (Role 2), manual/periodic:

```
finbert_inference -> sentiment_features -> feature_engineering
   -> [train_xgboost, train_lightgbm] -> dvc_push_models
```

---

## Testing

The whole-project test suite runs with `pytest`. It needs only a light set of
dependencies (`requirements-test.txt`) — heavy libraries (torch, Airflow,
Crawl4AI) are not required because FinBERT is stubbed and other imports are
`importorskip`-guarded; tests that load the DVC model artifacts skip when those
are absent.

```bash
# All tests (project env)
uv run pytest

# Or a lighter test-only env
pip install -r requirements-test.txt && pytest
# macOS: prefix with KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1
```

**CI (GitHub Actions).** `.github/workflows/ci.yml` runs this suite on every pull
request and push to `main`, acting as a merge gate (enable branch protection to
require it).

---

## Data & model versioning

All large artifacts are tracked with **DVC** (`data/*.dvc`, `models/*.dvc`) on a
shared Google Drive remote; only the small `.dvc` pointer files live in Git. Run
`dvc pull` after cloning to fetch the actual data and models. The serving
environment pins the exact library versions the artifacts were trained with
(Python 3.12, scikit-learn 1.9.0, xgboost 3.4.1) for reproducible loading.
