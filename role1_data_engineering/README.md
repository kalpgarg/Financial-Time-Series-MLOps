# Role 1 — Data Engineering

**Focus:** data ingestion, cleaning/processing (PySpark), and Airflow
orchestration of the daily inference pipeline.

## What this role does

1. **Scrapes** raw inputs for the 50-stock universe (`scrapers/`):
   - `headline_scraper.py` — Groww market news via Crawl4AI (headless browser).
   - `ohlc_scraper.py` — 15-minute OHLCV bars from TradingView (`tvDatafeed`).
   - `nse_preopen_scraper.py` — NSE F&O pre-open auction quotes.
   - `finbert_inference.py` — FinBERT sentiment + CLS embeddings on headlines
     (invoked by **Role 2's training DAG**, not the inference DAG).
   - `daily_scraper_orchestrator.py` + `run_scraper.sh` — a non-Airflow runner
     that executes all scrapers and syncs the output folders (for cron/CLI use).
2. **Cleans / processes** with **PySpark** (`spark/`):
   - `clean_headlines.py` — UDF text cleaning + window dedup → `data/headlines.csv`.
   - `merge_preopen.py` — maps NSE tickers to stock names (Spark join) and appends
     the synthetic 09:15 bar → `data/merged_ohlc_15min.csv`.
   - `trim_for_inference.py` — trims to the minimal window (10-day headlines,
     45-day OHLCV) → `data/inference/` for the Docker inference container.
3. **Orchestrates** the daily inference pipeline with Airflow
   (`airflow/dags/data_pipeline_dag.py`, DAG `financial_ts_inference_pipeline`):

   ```
   [scrape_headlines, scrape_ohlcv] -> scrape_preopen
      -> [spark_clean_headlines, spark_merge_preopen] -> trim_for_inference
      -> run_inference (docker run Role 3 image) -> dvc_push_data
   ```
   With `SKIP_SCRAPERS=1`, the scrape/clean/merge/push tasks become no-ops and the
   DAG runs `trim_for_inference -> run_inference` on existing DVC-pulled data.

## How to run

Run from the **repo root** (modules import `shared.config`):

```bash
# Individual scrapers (headline/ohlc/nse support --dry-run)
python -m role1_data_engineering.scrapers.headline_scraper --dry-run
python -m role1_data_engineering.scrapers.ohlc_scraper
python -m role1_data_engineering.scrapers.nse_preopen_scraper

# PySpark jobs (need a Java runtime + pyspark)
python -m role1_data_engineering.spark.clean_headlines
python -m role1_data_engineering.spark.merge_preopen

# Tests
pytest role1_data_engineering/tests/ -v
```

Install deps with `pip install -r requirements.txt`; the headline scraper also
needs `crawl4ai-setup` (installs the headless browser). The full inference DAG is
launched from the whole-system stack — see the **root README**.

## Dependencies on other roles

- **Consumes** `shared/config.py` (paths, timezone) and the
  `shared/schemas/data_contract.py` schemas; reads `data/stock_list.csv`.
- **Produces** the `data/` CSVs used by **Role 2** (training) and **Role 3**
  (inference), versioned with **DVC**.
- The inference DAG's `run_inference` task **invokes Role 3's Docker image**
  (`fints-api:latest`) on Role 3's network/database, so the Role 3 stack must be
  running for a full inference run.
