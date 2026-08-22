"""
Airflow DAG: Model training pipeline.

Task graph:
  1. finbert_inference  — FinBERT sentiment on headlines → enriched CSV + embeddings
  2. sentiment_features — PCA + 7-day aggregation → news_features.csv
  3. feature_engineering — OHLCV technicals + merge → merged_features.csv
  4. train_xgboost / train_lightgbm (parallel) — experiment sweeps with MLflow
  5. dvc_push_models — DVC add + push model artifacts

Schedule: @once (manual trigger)
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

PYTHON_BIN = sys.executable


# ── Helper: run a module as a subprocess ─────────────────────────────────────
def _run_module(module, extra_args=None):
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


def _run_shell(cmd, task_name):
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


# ── Default args ─────────────────────────────────────────────────────────────
default_args = {
    "owner": "ml-modeling",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# ── DAG definition ───────────────────────────────────────────────────────────
with DAG(
    dag_id="financial_ts_training_pipeline",
    default_args=default_args,
    description="FinBERT features → OHLCV features → XGBoost/LightGBM training → DVC push",
    schedule_interval="@once",
    start_date=datetime(2026, 8, 22),
    catchup=False,
    tags=["ml-modeling", "training", "role2"],
) as dag:

    # ── Task 1: FinBERT inference on headlines ────────────────────────────────
    def finbert_inference(**context):
        _run_module("role1_data_engineering.scrapers.finbert_inference")

    task_finbert = PythonOperator(
        task_id="finbert_inference",
        python_callable=finbert_inference,
        execution_timeout=timedelta(minutes=30),
    )

    # ── Task 2: Sentiment feature aggregation ─────────────────────────────────
    def sentiment_features(**context):
        _run_module("role2_ml_modeling.features.sentiment_features")

    task_sentiment = PythonOperator(
        task_id="sentiment_features",
        python_callable=sentiment_features,
        execution_timeout=timedelta(minutes=10),
    )

    # ── Task 3: OHLCV feature engineering + merge ─────────────────────────────
    def feature_engineering(**context):
        _run_module("role2_ml_modeling.features.feature_engineering")

    task_features = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering,
        execution_timeout=timedelta(minutes=10),
    )

    # ── Task 4a: Train XGBoost ────────────────────────────────────────────────
    def train_xgboost(**context):
        _run_module(
            "role2_ml_modeling.training.train",
            extra_args=["--model", "xgboost"],
        )

    task_train_xgb = PythonOperator(
        task_id="train_xgboost",
        python_callable=train_xgboost,
        execution_timeout=timedelta(minutes=20),
    )

    # ── Task 4b: Train LightGBM ──────────────────────────────────────────────
    def train_lightgbm(**context):
        _run_module(
            "role2_ml_modeling.training.train",
            extra_args=["--model", "lightgbm"],
        )

    task_train_lgbm = PythonOperator(
        task_id="train_lightgbm",
        python_callable=train_lightgbm,
        execution_timeout=timedelta(minutes=20),
    )

    # ── Task 5: DVC push model artifacts ──────────────────────────────────────
    def dvc_push_models(**context):
        model_files = [
            os.path.join(PROJECT_ROOT, "models", f)
            for f in [
                "xgboost_model.pkl",
                "lightgbm_model.pkl",
                "feature_columns.pkl",
                "finbert_pca.pkl",
                "symbol_encoder.pkl",
            ]
            if os.path.exists(os.path.join(PROJECT_ROOT, "models", f))
        ]
        if model_files:
            _run_shell(
                [PYTHON_BIN, "-m", "dvc", "add"] + model_files,
                "dvc_add_models",
            )
            _run_shell(
                [PYTHON_BIN, "-m", "dvc", "push"],
                "dvc_push_models",
            )

    task_dvc_push = PythonOperator(
        task_id="dvc_push_models",
        python_callable=dvc_push_models,
        execution_timeout=timedelta(minutes=10),
    )

    # ── DAG dependency graph ──────────────────────────────────────────────────
    # finbert → sentiment → features → [xgboost, lightgbm] → dvc push
    task_finbert >> task_sentiment >> task_features
    task_features >> [task_train_xgb, task_train_lgbm] >> task_dvc_push
