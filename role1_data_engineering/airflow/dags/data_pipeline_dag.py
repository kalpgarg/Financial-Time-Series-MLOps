"""
Airflow DAG: End-to-end inference pipeline.

Task graph (all tasks run back-to-back, no time gates):
  1. Scrape headlines (Groww) + OHLCV (TradingView) in parallel
  2. Scrape NSE F&O pre-open data
  3. Spark: clean headlines → data/headlines.csv
     Spark: merge pre-open 09:15 bar → merged_ohlc_15min.csv
  4. Spark: trim to minimal date window → data/inference/
  5. Docker: run batch_score.py on trimmed data (role3 inference image)
  6. DVC: add + push headlines.csv & merged_ohlc_15min.csv

Schedule: @once (manual trigger for testing)

SKIP_SCRAPERS=1 (set in the compose stack): run inference on the existing
DVC-pulled data. The scrape, clean, merge, and DVC-push tasks become no-ops;
only trim_for_inference and run_inference do work. This makes the DAG runnable
end-to-end without live scraping or a reachable DVC remote.

Production: set SKIP_SCRAPERS=0, change schedule to '0 3 * * 1-5', and re-add a
TimeSensor before scrape_preopen (target_time=time(3,41) = 09:11 IST).
"""

import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# ── Project root ─────────────────────────────────────────────────────────────
# When the DAG lives in the repo (role1.../airflow/dags/) parents[3] is correct.
# When it's copied to airflow_home/dags/ we fall back to the env var or search
# for pyproject.toml up from cwd.
def _find_project_root() -> str:
    # 1. Explicit env var
    env = os.getenv("FINTS_PROJECT_ROOT")
    if env:
        return env
    # 2. Walk up from the DAG file looking for pyproject.toml
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / "pyproject.toml").exists():
            return str(p)
        p = p.parent
    # 3. Fallback: cwd
    return os.getcwd()


PROJECT_ROOT = _find_project_root()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PYTHON_BIN = sys.executable  # use the same Python interpreter as Airflow

# Docker image for inference (role3 API image)
INFERENCE_IMAGE = os.getenv("INFERENCE_DOCKER_IMAGE", "fints-api:latest")

# Set SKIP_SCRAPERS=1 to skip live scraping (use existing data for testing)
SKIP_SCRAPERS = os.getenv("SKIP_SCRAPERS", "0") == "1"

# Docker network where Postgres runs (from role3 docker-compose)
INFERENCE_DOCKER_NETWORK = os.getenv(
    "INFERENCE_DOCKER_NETWORK", "role3_mlops_devops_default",
)

# Database URL for the inference container (uses Docker DNS hostname "db")
INFERENCE_DB_URL = os.getenv(
    "INFERENCE_DATABASE_URL",
    "postgresql+psycopg2://mlops:mlops@db:5432/predictions",
)

# Host path of the repo, for the inference `docker run` bind mounts. When this
# DAG runs INSIDE a container (compose Airflow) talking to the host Docker
# socket, `-v` paths resolve on the HOST, not in this container -- so we need
# the host repo path (compose passes HOST_PROJECT_ROOT=${PWD}). When Airflow
# runs on the host directly, PROJECT_ROOT is already the host path.
HOST_PROJECT_ROOT = os.getenv("HOST_PROJECT_ROOT", PROJECT_ROOT)

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
    schedule_interval="@once" if SKIP_SCRAPERS else "0 3 * * 1-5",
    start_date=datetime(2026, 8, 22),
    catchup=False,
    tags=["data-engineering", "inference", "role1", "role3"],
) as dag:

    # ── Task 1: Scrape headlines → CSV ───────────────────────────────────────
    def scrape_headlines(**context):
        if SKIP_SCRAPERS:
            print("SKIP_SCRAPERS=1: skipping headline scraper")
            return
        _run_module("role1_data_engineering.scrapers.headline_scraper")

    task_scrape_headlines = PythonOperator(
        task_id="scrape_headlines",
        python_callable=scrape_headlines,
        execution_timeout=timedelta(minutes=35),
    )

    # ── Task 2: Scrape OHLCV data → CSV ─────────────────────────────────────
    def scrape_ohlcv(**context):
        if SKIP_SCRAPERS:
            print("SKIP_SCRAPERS=1: skipping OHLCV scraper")
            return
        _run_module("role1_data_engineering.scrapers.ohlc_scraper")

    task_scrape_ohlcv = PythonOperator(
        task_id="scrape_ohlcv",
        python_callable=scrape_ohlcv,
        execution_timeout=timedelta(minutes=60),
    )

    # ── Task 3: Scrape NSE F&O pre-open data ────────────────────────────────
    def scrape_preopen(**context):
        if SKIP_SCRAPERS:
            print("SKIP_SCRAPERS=1: skipping pre-open scraper")
            return
        _run_module("role1_data_engineering.scrapers.nse_preopen_scraper")

    task_scrape_preopen = PythonOperator(
        task_id="scrape_preopen",
        python_callable=scrape_preopen,
        execution_timeout=timedelta(minutes=5),
    )

    # ── Task 4: Spark — clean headlines → data/headlines.csv ─────────────────
    def spark_clean_headlines(**context):
        if SKIP_SCRAPERS:
            # No fresh raw scrape this run; cleaning would read the (absent) raw
            # data/stock_news/headlines.csv. Use the DVC-pulled clean file as-is.
            print("SKIP_SCRAPERS=1: using existing data/headlines.csv")
            return
        _run_module("role1_data_engineering.spark.clean_headlines")

    task_clean_headlines = PythonOperator(
        task_id="spark_clean_headlines",
        python_callable=spark_clean_headlines,
        execution_timeout=timedelta(minutes=10),
    )

    # ── Task 5: Merge pre-open into merged_ohlc_15min.csv ────────────────────
    def spark_merge_preopen(**context):
        if SKIP_SCRAPERS:
            # No fresh pre-open CSV this run; keep the DVC-pulled
            # data/merged_ohlc_15min.csv as-is.
            print("SKIP_SCRAPERS=1: using existing data/merged_ohlc_15min.csv")
            return
        _run_module("role1_data_engineering.spark.merge_preopen")

    task_merge_preopen = PythonOperator(
        task_id="spark_merge_preopen",
        python_callable=spark_merge_preopen,
        execution_timeout=timedelta(minutes=5),
    )

    # ── Task 6: Trim data for inference ─────────────────────────────────────
    def trim_for_inference(**context):
        _run_module("role1_data_engineering.spark.trim_for_inference")

    task_trim = PythonOperator(
        task_id="trim_for_inference",
        python_callable=trim_for_inference,
        execution_timeout=timedelta(minutes=5),
    )

    # ── Task 7: Run inference via Docker ─────────────────────────────────────
    def run_inference(**context):
        run_id = context.get("run_id", context["logical_date"].strftime("%Y%m%d_%H%M"))
        cmd = [
            "docker", "run", "--rm",
            "--network", INFERENCE_DOCKER_NETWORK,
            "-v", f"{HOST_PROJECT_ROOT}/data/inference:/app/data:ro",
            "-v", f"{HOST_PROJECT_ROOT}/models:/app/models:ro",
            "-e", f"DATABASE_URL={INFERENCE_DB_URL}",
            "-e", "TRANSFORMERS_OFFLINE=1",
            "-e", "HF_HUB_OFFLINE=1",
            "-e", "OMP_NUM_THREADS=1",
            INFERENCE_IMAGE,
            "python", "batch_score.py",
            "--news", "/app/data/headlines.csv",
            "--ohlcv", "/app/data/ohlcv_15min.csv",
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
        if SKIP_SCRAPERS:
            # No new data was produced this run, and the DVC remote (a local
            # gdrive mount) isn't reachable from inside the container.
            print("SKIP_SCRAPERS=1: skipping DVC add/push")
            return
        headlines_csv = os.path.join(PROJECT_ROOT, "data", "headlines.csv")
        ohlcv_csv = os.path.join(PROJECT_ROOT, "data", "merged_ohlc_15min.csv")
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
    # 1. Scrape headlines + OHLCV in parallel
    # 2. Scrape pre-open
    # 3. Spark cleanup (headlines + merge pre-open) in parallel
    # 4. Spark trim → data/inference/ (minimal window for Docker)
    # 5. Run inference via Docker (reads trimmed data only)
    # 6. DVC push full input data
    [task_scrape_headlines, task_scrape_ohlcv] >> task_scrape_preopen
    task_scrape_preopen >> [task_clean_headlines, task_merge_preopen]
    [task_clean_headlines, task_merge_preopen] >> task_trim
    task_trim >> task_inference
    task_inference >> task_dvc_push
