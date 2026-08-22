"""
Airflow DAG: End-to-end inference pipeline on Indian market days.

Time sequence (IST, UTC+05:30):
  08:30 — Scrape headlines (Groww) + OHLCV (TradingView/Upstox) in parallel
  09:11 — TimeSensor waits, then scrape NSE F&O pre-open data
  09:12 — Spark: clean headlines → data/headlines.csv
           Spark: merge pre-open 09:15 bar → merged_ohlc_15min.csv
  09:15 — Docker: run batch_score.py (role3 inference image)
  09:20 — DVC: add + push headlines.csv & merged_ohlc_15min.csv

Schedule: 0 3 * * 1-5  (03:00 UTC = 08:30 IST, weekdays only)
"""

import os
import subprocess
import sys
from datetime import datetime, time, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.time_sensor import TimeSensor

# ── Project root (assumes Airflow has access to the repo) ────────────────────
PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PYTHON_BIN = sys.executable  # use the same Python interpreter as Airflow

# Docker image for inference (role3 API image)
INFERENCE_IMAGE = os.getenv("INFERENCE_DOCKER_IMAGE", "fints-api:latest")

# Database URL for the inference container
INFERENCE_DB_URL = os.getenv(
    "INFERENCE_DATABASE_URL",
    "postgresql+psycopg2://mlops:mlops@host.docker.internal:5432/predictions",
)

# ── Default args ─────────────────────────────────────────────────────────────
default_args = {
    "owner": "data-engineering",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


# ── Helper: run a module as a subprocess ─────────────────────────────────────
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


def _run_shell(cmd: list[str], task_name: str):
    """Run a shell command in the project root."""
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"{task_name} failed with exit code {result.returncode}")


# ── DAG definition ───────────────────────────────────────────────────────────
with DAG(
    dag_id="financial_ts_inference_pipeline",
    default_args=default_args,
    description="End-to-end inference: scrape → Spark cleanup → Docker inference → DVC push",
    schedule_interval="0 3 * * 1-5",  # 03:00 UTC = 08:30 IST, Mon–Fri
    start_date=datetime(2026, 5, 30),
    catchup=False,
    tags=["data-engineering", "inference", "role1", "role3"],
) as dag:

    # ── Task 1: Scrape headlines → CSV ───────────────────────────────────────
    def scrape_headlines(**context):
        _run_module("role1_data_engineering.scrapers.headline_scraper")

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

    # ── Task 3: Wait until 09:11 IST (03:41 UTC) ────────────────────────────
    task_wait_preopen = TimeSensor(
        task_id="wait_for_preopen_time",
        target_time=time(3, 41),  # 03:41 UTC = 09:11 IST
        poke_interval=30,
        timeout=3600,
        mode="reschedule",
    )

    # ── Task 4: Scrape NSE F&O pre-open data ────────────────────────────────
    def scrape_preopen(**context):
        _run_module("role1_data_engineering.scrapers.nse_preopen_scraper")

    task_scrape_preopen = PythonOperator(
        task_id="scrape_preopen",
        python_callable=scrape_preopen,
        execution_timeout=timedelta(minutes=5),
    )

    # ── Task 5: Spark — clean headlines → data/headlines.csv ─────────────────
    def spark_clean_headlines(**context):
        _run_module("role1_data_engineering.spark.clean_headlines")

    task_clean_headlines = PythonOperator(
        task_id="spark_clean_headlines",
        python_callable=spark_clean_headlines,
        execution_timeout=timedelta(minutes=10),
    )

    # ── Task 6: Merge pre-open into merged_ohlc_15min.csv ────────────────────
    def spark_merge_preopen(**context):
        _run_module("role1_data_engineering.spark.merge_preopen")

    task_merge_preopen = PythonOperator(
        task_id="spark_merge_preopen",
        python_callable=spark_merge_preopen,
        execution_timeout=timedelta(minutes=5),
    )

    # ── Task 7: Run inference via Docker ─────────────────────────────────────
    def run_inference(**context):
        run_id = context.get("run_id", context["logical_date"].strftime("%Y%m%d_%H%M"))
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{PROJECT_ROOT}/data:/app/data:ro",
            "-v", f"{PROJECT_ROOT}/models:/app/models:ro",
            "-e", f"DATABASE_URL={INFERENCE_DB_URL}",
            "-e", "TRANSFORMERS_OFFLINE=1",
            "-e", "HF_HUB_OFFLINE=1",
            "-e", "OMP_NUM_THREADS=1",
            INFERENCE_IMAGE,
            "python", "batch_score.py",
            "--news", "/app/data/headlines.csv",
            "--ohlcv", "/app/data/ohlc_data/merged_ohlc_15min.csv",
            "--run-id", run_id,
        ]
        _run_shell(cmd, "docker_inference")

    task_inference = PythonOperator(
        task_id="run_inference",
        python_callable=run_inference,
        execution_timeout=timedelta(minutes=15),
    )

    # ── Task 8: DVC add + push input data ────────────────────────────────────
    def dvc_push_data(**context):
        headlines_csv = os.path.join(PROJECT_ROOT, "data", "headlines.csv")
        ohlcv_csv = os.path.join(
            PROJECT_ROOT, "data", "ohlc_data", "merged_ohlc_15min.csv",
        )
        # dvc add
        _run_shell(
            [PYTHON_BIN, "-m", "dvc", "add", headlines_csv, ohlcv_csv],
            "dvc_add",
        )
        # dvc push
        _run_shell(
            [PYTHON_BIN, "-m", "dvc", "push"],
            "dvc_push",
        )

    task_dvc_push = PythonOperator(
        task_id="dvc_push_data",
        python_callable=dvc_push_data,
        execution_timeout=timedelta(minutes=10),
    )

    # ── DAG dependency graph ─────────────────────────────────────────────────
    # 1. Scrape headlines + OHLCV in parallel at 08:30 IST
    # 2. Wait until 09:11 IST, then scrape pre-open
    # 3. Spark cleanup (headlines + merge pre-open) in parallel
    # 4. Run inference via Docker
    # 5. DVC push input data
    [task_scrape_headlines, task_scrape_ohlcv] >> task_wait_preopen
    task_wait_preopen >> task_scrape_preopen
    task_scrape_preopen >> [task_clean_headlines, task_merge_preopen]
    [task_clean_headlines, task_merge_preopen] >> task_inference
    task_inference >> task_dvc_push
