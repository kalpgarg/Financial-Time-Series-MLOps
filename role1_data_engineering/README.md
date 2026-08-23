# Role 1 – Data Engineering Lead

**Focus:** Ingestion, Orchestration, and Processing

## Architecture

```
 8:30 AM IST                                         ~9:00 AM
┌──────────────┐                               ┌──────────────┐
│  Headline    │  CSV                           │  PySpark     │
│  Scraper     ├──────────┐                     │  Pipeline    │
│  (crawl4ai)  │          │                     │  clean+merge │
└──────────────┘          ├────────────────────►└──────┬───────┘
┌──────────────┐          │                            │
│  OHLCV       │  CSV     │                     ┌──────▼───────┐
│  Scraper     ├──────────┤                     │ Clean CSVs   │
│  (TV)        │          │                     │ (DVC-tracked) │
└──────────────┘          │                     └──────────────┘
┌──────────────┐          │
│  NSE PreOpen │  CSV     │
│  Scraper     ├──────────┘
└──────────────┘
```

**Airflow** orchestrates the entire sequence on weekdays (Mon–Fri).

## Components

| Directory | Description |
|-----------|-------------|
| `scrapers/` | crawl4ai headline scraper, TradingView OHLCV scraper, NSE pre-open scraper, FinBERT inference |
| `spark/` | PySpark: clean headlines, merge pre-open data into OHLCV |
| `airflow/dags/` | Airflow DAG with time-sequenced tasks |
| `tests/` | Unit tests for data cleaning and schema contracts |

## Getting Started

### 1. Run Components (dry-run, no infra needed)

```bash
# Scrape headlines → stdout
python -m role1_data_engineering.scrapers.headline_scraper --dry-run

# OHLCV scraper (TradingView) → CSV
python -m role1_data_engineering.scrapers.ohlc_scraper --dry-run

# NSE pre-open scraper → CSV
python -m role1_data_engineering.scrapers.nse_preopen_scraper --dry-run
```

### 2. Run PySpark Jobs

```bash
# Clean raw headlines → data/headlines.csv
python -m role1_data_engineering.spark.clean_headlines

# Merge pre-open bars → data/merged_ohlc_15min.csv
python -m role1_data_engineering.spark.merge_preopen
```

### 3. Run Tests

```bash
pytest role1_data_engineering/tests/ -v
```

## Deliverable

A robust pipeline that outputs clean, DVC-tracked CSV files
(`data/headlines.csv`, `data/merged_ohlc_15min.csv`) matching the
agreed-upon schema (`shared/schemas/data_contract.py`),
ready for Role 2 (ML Modeling) to consume.
