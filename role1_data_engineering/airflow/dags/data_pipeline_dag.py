"""
Airflow DAG: Orchestrates the full data pipeline on Indian market days.

Time sequence (IST, UTC+05:30):
  08:30 — Scrape headlines from Groww → CSV              (~30 min)
  08:30 — Scrape OHLCV data (TradingView / Upstox) → CSV
  09:00 — Spark: clean_headlines, clean_prices, join_data
  09:15 — Feature vectors ready in PostgreSQL for Role 2

Schedule: 0 3 * * 1-5  (03:00 UTC = 08:30 IST, weekdays only)
"""

import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# ── Project root (assumes Airflow has access to the repo) ────────────────────
PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PYTHON_BIN = sys.executable  # use the same Python interpreter as Airflow

# ── Default args ─────────────────────────────────────────────────────────────
default_args = {
    "owner": "data-engineering",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

# ── DAG definition ───────────────────────────────────────────────────────────
with DAG(
    dag_id="financial_ts_data_pipeline",
    default_args=default_args,
    description="End-to-end data pipeline: scrape → Spark → PostgreSQL",
    schedule_interval="0 3 * * 1-5",  # 03:00 UTC = 08:30 IST, Mon–Fri
    start_date=datetime(2026, 5, 30),
    catchup=False,
    tags=["data-engineering", "role1"],
) as dag:

    # ── Helper: run a module as a subprocess ─────────────────────────────────
    def _run_module(module: str, extra_args: list[str] | None = None):
        """Run a Python module via subprocess in the project root."""
        cmd = [PYTHON_BIN, "-m", module]
        if extra_args:
            cmd.extend(extra_args)
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError(f"{module} failed with exit code {result.returncode}")

    # ── Task 1: Scrape headlines → CSV ───────────────────────────────────────
    def scrape_headlines(**context):
        ts = context["execution_date"].strftime("%Y%m%d_%H%M")
        output_path = os.path.join(PROJECT_ROOT, "data", "stock_news", f"headlines_{ts}.csv")
        _run_module(
            "role1_data_engineering.scrapers.headline_scraper",
            extra_args=["--output", output_path],
        )

    task_scrape_headlines = PythonOperator(
        task_id="scrape_headlines",
        python_callable=scrape_headlines,
        execution_timeout=timedelta(minutes=35),
    )

    # ── Task 2: Scrape OHLCV data → CSV ─────────────────────────────────────
    def scrape_ohlcv(**context):
        _run_module("role1_data_engineering.scrapers.ohlc_scraper")

    task_scrape_ohlcv = PythonOperator(
        task_id="scrape_ohlcv",
        python_callable=scrape_ohlcv,
        execution_timeout=timedelta(minutes=60),
    )

    # ── Task 3: Spark — clean headlines ──────────────────────────────────────
    def spark_clean_headlines(**context):
        _run_module("role1_data_engineering.spark.clean_headlines")

    task_clean_headlines = PythonOperator(
        task_id="spark_clean_headlines",
        python_callable=spark_clean_headlines,
        execution_timeout=timedelta(minutes=10),
    )

    # ── Task 4: Spark — clean prices ─────────────────────────────────────────
    def spark_clean_prices(**context):
        _run_module("role1_data_engineering.spark.clean_prices")

    task_clean_prices = PythonOperator(
        task_id="spark_clean_prices",
        python_callable=spark_clean_prices,
        execution_timeout=timedelta(minutes=10),
    )

    # ── Task 5: Spark — join data → feature_vectors ──────────────────────────
    def spark_join_data(**context):
        _run_module("role1_data_engineering.spark.join_data")

    task_join = PythonOperator(
        task_id="spark_join_data",
        python_callable=spark_join_data,
        execution_timeout=timedelta(minutes=10),
    )

    # ── DAG dependency graph ─────────────────────────────────────────────────
    # Scrape headlines and OHLCV in parallel, then Spark pipeline
    [task_scrape_headlines, task_scrape_ohlcv] >> task_clean_headlines
    [task_scrape_headlines, task_scrape_ohlcv] >> task_clean_prices
    [task_clean_headlines, task_clean_prices] >> task_join
