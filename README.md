# Financial-Time-Series-MLOps

Predict next-day market-open price direction (**Bullish / Bearish / Neutral**) for NSE F&O stocks using historical OHLCV prices and real-time news sentiment (FinBERT → PCA → XGBoost / LightGBM).

---

## Project Structure

```
Financial-Time-Series-MLOps/
│
├── shared/                              # Shared config & data contracts
│   ├── config.py                        #   Central env-based configuration
│   └── schemas/
│       └── data_contract.py             #   PriceRecord, HeadlineRecord, Prediction schemas
│
├── data/                                # DVC-tracked datasets
│   ├── stock_news/headlines.csv         #   Raw scraped headlines
│   ├── headlines.csv                    #   Cleaned headlines (DVC)
│   ├── merged_ohlc_15min.csv            #   15-min OHLCV bars (DVC)
│   ├── headlines_enriched.csv           #   FinBERT sentiment + embeddings
│   ├── news_features.csv                #   PCA-reduced sentiment features
│   ├── merged_features.csv              #   Final training features
│   ├── preopen_csv/                     #   NSE pre-open market data
│   └── stock_list.csv                   #   Stock universe (50 NSE F&O names)
│
├── models/                              # DVC-tracked model artifacts
│   ├── xgboost_model.pkl                #   XGBoost classifier
│   ├── lightgbm_model.pkl               #   LightGBM classifier
│   ├── finbert_pca.pkl                  #   PCA transformer for FinBERT embeddings
│   ├── feature_columns.pkl              #   Feature column list
│   └── symbol_encoder.pkl               #   Label encoder for stock symbols
│
├── role1_data_engineering/              # Data ingestion & orchestration
│   ├── scrapers/
│   │   ├── headline_scraper.py          #   Groww/NSE news scraper (Crawl4AI)
│   │   ├── ohlc_scraper.py              #   TradingView 15-min OHLCV scraper
│   │   ├── nse_preopen_scraper.py       #   NSE F&O pre-open data scraper
│   │   ├── finbert_inference.py         #   FinBERT sentiment inference
│   │   └── daily_scraper_orchestrator.py
│   ├── spark/
│   │   ├── clean_headlines.py           #   Deduplicate & clean headlines
│   │   ├── clean_prices.py              #   Price data cleaning
│   │   └── merge_preopen.py             #   Merge pre-open data into OHLCV
│   ├── airflow/dags/
│   │   └── data_pipeline_dag.py         #   Inference pipeline DAG
│   └── db/
│       └── init_db.py                   #   Database initialisation
│
├── role2_ml_modeling/                   # Feature engineering & model training
│   ├── features/
│   │   ├── feature_engineering.py       #   OHLCV technical indicators + merge
│   │   └── sentiment_features.py        #   PCA + 7-day sentiment aggregation
│   ├── training/
│   │   ├── train.py                     #   XGBoost / LightGBM training + MLflow
│   │   └── evaluate.py                  #   Model evaluation
│   ├── models/
│   │   ├── sentiment_model.py           #   FinBERT wrapper (PyTorch)
│   │   └── price_predictor.py           #   Sklearn prediction wrapper
│   ├── airflow/dags/
│   │   └── training_pipeline_dag.py     #   Training pipeline DAG
│   └── mlflow_utils.py                  #   MLflow helper functions
│
├── role3_mlops_devops/                  # Serving, containerisation & monitoring
│   ├── app/                             #   FastAPI application
│   │   ├── main.py                      #     Uvicorn entrypoint
│   │   ├── routes.py                    #     /predict, /health endpoints
│   │   ├── scoring.py                   #     FinBERT + XGBoost scoring
│   │   ├── pipeline.py                  #     Full inference pipeline
│   │   ├── db.py                        #     SQLAlchemy models & session
│   │   ├── metrics.py                   #     Prometheus metrics
│   │   └── schemas.py                   #     Pydantic request/response
│   ├── batch_score.py                   #   CLI batch inference script
│   ├── Dockerfile                       #   API image (FinBERT baked in)
│   ├── docker-compose.yml               #   API + Postgres + Prometheus stack
│   ├── monitoring/prometheus/           #   Prometheus scrape config
│   └── tests/
│
├── .github/workflows/ci.yml            # CI/CD: test → build Docker image
└── pyproject.toml                       # uv-managed dependencies (Python 3.12+)
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<you>/Financial-Time-Series-MLOps.git
cd Financial-Time-Series-MLOps

# Install all dependencies (Airflow, PyTorch, DVC, etc.)
uv sync
```

### 2. Create `.env`

Copy the template below into a `.env` file at the project root.
Only `GDRIVE_CREDENTIALS_DATA` is required if your DVC remote is Google Drive.

```dotenv
# ── DVC Remote (required for dvc pull / push) ────────────────────────────────
GDRIVE_CREDENTIALS_DATA=<service-account-json>   # GDrive service-account key

# ── Alerts (optional) ────────────────────────────────────────────────────────
DISCORD_WEBHOOK_URL=

# ── OHLCV Scraper (optional — default uses TradingView) ──────────────────────
# OHLC_SCRAPER_MODULE=ohlc_scraper_upstox
# UPSTOX_ACCESS_TOKEN=

# ── Database (defaults work with docker-compose) ─────────────────────────────
# DATABASE_URL=postgresql+psycopg2://mlops:mlops@localhost:5432/predictions
# POSTGRES_HOST=localhost
# POSTGRES_PORT=5432
# POSTGRES_DB=fints
# POSTGRES_USER=fints_user
# POSTGRES_PASSWORD=fints_pass

# ── MLflow (optional) ────────────────────────────────────────────────────────
# MLFLOW_TRACKING_URI=http://localhost:5000
```

### 3. Pull DVC-tracked data & models

```bash
uv run dvc pull
```

### 4. Build the inference Docker image

```bash
docker build -t fints-api:latest -f role3_mlops_devops/Dockerfile role3_mlops_devops/
```

### 5. Start Postgres (prediction storage)

```bash
docker compose -f role3_mlops_devops/docker-compose.yml up -d db
```

### 6. Run tests

```bash
uv run pytest --tb=short -q
```

---

## Inference Pipeline

DAG **`financial_ts_inference_pipeline`** orchestrates the daily inference flow:
scrape market data, clean/merge it, run FinBERT + XGBoost predictions via Docker,
and version input data with DVC.

### Task Graph

```
[scrape_headlines, scrape_ohlcv]   ← parallel
        │
   scrape_preopen
        │
 ┌──────┴──────┐
 │             │
clean_headlines  merge_preopen     ← parallel (pandas)
 │             │
 └──────┬──────┘
        │
   run_inference                   ← docker run batch_score.py
        │
   dvc_push_data                   ← dvc add + push
```

### Setup & Run

```bash
# Initialise Airflow (one-time)
export AIRFLOW_HOME=$(pwd)/airflow_home
uv run airflow db init
uv run airflow users create --username admin --password admin \
  --firstname Admin --lastname User --role Admin --email admin@example.com

# Copy DAG
mkdir -p "$AIRFLOW_HOME/dags"
cp role1_data_engineering/airflow/dags/data_pipeline_dag.py "$AIRFLOW_HOME/dags/"

# Start Airflow
AIRFLOW_HOME=$(pwd)/airflow_home uv run airflow standalone &

# Trigger (with SKIP_SCRAPERS=1 to use existing data for testing)
AIRFLOW_HOME=$(pwd)/airflow_home SKIP_SCRAPERS=1 \
  uv run airflow dags trigger financial_ts_inference_pipeline

# Monitor at http://localhost:8080  (admin / admin)
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_DOCKER_IMAGE` | `fints-api:latest` | Docker image for batch inference |
| `INFERENCE_DATABASE_URL` | `postgresql+psycopg2://mlops:mlops@db:5432/predictions` | Postgres URL for the inference container |
| `INFERENCE_DOCKER_NETWORK` | `role3_mlops_devops_default` | Docker network the inference container joins |
| `SKIP_SCRAPERS` | `0` | Set to `1` to skip live scraping and use `@once` schedule |
| `FINTS_PROJECT_ROOT` | *(auto-detected)* | Override project root path for the DAG |

### Production Timing

When `SKIP_SCRAPERS` is unset (default), the DAG runs on cron `0 3 * * 1-5`
(03:00 UTC = 08:30 IST, Mon–Fri). For finer control over pre-open data timing,
add an Airflow `TimeSensor` before `scrape_preopen` targeting 09:11 IST.

---

## Training Pipeline

DAG **`financial_ts_training_pipeline`** runs the full model training flow:
FinBERT sentiment extraction, feature engineering, parallel XGBoost / LightGBM
training with MLflow experiment tracking, and DVC push of model artifacts.

### Task Graph

```
finbert_inference          ← FinBERT sentiment on headlines
        │
sentiment_features         ← PCA + 7-day aggregation → news_features.csv
        │
feature_engineering        ← OHLCV technicals + merge → merged_features.csv
        │
 ┌──────┴──────┐
 │             │
train_xgboost  train_lightgbm   ← parallel, MLflow experiment sweeps
 │             │
 └──────┬──────┘
        │
dvc_push_models            ← dvc add + push model .pkl files
```

### Setup & Run

```bash
# Copy the training DAG into Airflow's dags folder
cp role2_ml_modeling/airflow/dags/training_pipeline_dag.py "$AIRFLOW_HOME/dags/"

# (Optional) Start MLflow tracking server
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns &

# Trigger the training pipeline
AIRFLOW_HOME=$(pwd)/airflow_home \
  uv run airflow dags trigger financial_ts_training_pipeline

# Monitor at http://localhost:8080  (admin / admin)
# MLflow UI at http://localhost:5000
```

### Model Artifacts

Training produces the following DVC-tracked files in `models/`:

| File | Description |
|------|-------------|
| `xgboost_model.pkl` | XGBoost direction classifier |
| `lightgbm_model.pkl` | LightGBM direction classifier |
| `finbert_pca.pkl` | PCA transformer for FinBERT embeddings |
| `feature_columns.pkl` | Ordered list of feature column names |
| `symbol_encoder.pkl` | LabelEncoder for stock symbols |
