# syntax=docker/dockerfile:1
#
# Single multi-stage Dockerfile for the whole project (Option A).
# Each stage is an isolated, purpose-built image; docker-compose builds the
# stage it needs via `target:`. Postgres and Prometheus use official images
# and are not built here.
#
#   target: api      -> role3 FastAPI service (CPU torch + baked FinBERT)
#   target: airflow  -> Airflow (official image + pyspark/dvc/docker CLI)
#   target: mlflow   -> MLflow tracking server
#
# NOTE: the `api` stage mirrors role3_mlops_devops/Dockerfile (kept for
# role3-standalone runs). Keep the two in sync if you change either.

# =====================================================================
# Stage: api  — role3 prediction service
# =====================================================================
# Python 3.12 to match the training env (pyproject requires >=3.12) and to
# allow xgboost 3.4.x, the version the artifacts were trained/saved with.
FROM python:3.12-slim AS api

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    HF_HOME=/app/.hf_cache \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    ARTIFACTS_DIR=/app/models

WORKDIR /app

# OpenMP runtime needed by torch and xgboost on the slim base.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch (kept out of requirements so it can't be replaced by CUDA).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0

COPY role3_mlops_devops/requirements.txt .
RUN pip install -r requirements.txt \
    && (pip freeze | grep -i '^nvidia-' | cut -d= -f1 | xargs -r pip uninstall -y || true)

# Fail the build if any CUDA package remains; confirm torch + xgboost import.
RUN python -c "import sys, subprocess; \
out = subprocess.check_output(['pip', 'list'], text=True).lower(); \
nv = [l.split()[0] for l in out.splitlines() if l.startswith('nvidia-')]; \
sys.exit('CUDA packages present: ' + ', '.join(nv)) if nv else None; \
import torch, xgboost; print('torch', torch.__version__, '| xgboost', xgboost.__version__)"

# Bake FinBERT weights into the image (network needed at BUILD time only).
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c "\
from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
AutoTokenizer.from_pretrained('ProsusAI/finbert'); \
AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')"

# Application code (artifacts are mounted at /app/models at runtime, not copied).
COPY role3_mlops_devops/ /app/

RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=5 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


# =====================================================================
# Stage: airflow  — orchestrator (webserver + scheduler)
# =====================================================================
FROM apache/airflow:2.10.5-python3.11 AS airflow

USER root
# Java for PySpark; docker CLI so the inference DAG can `docker run` the api
# image via the mounted host socket; procps for task monitoring.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        default-jdk-headless docker.io procps \
    && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/default-java

USER airflow
# Deps needed to PARSE the DAGs and run the lighter tasks (Spark, DVC).
# Heavy per-task deps (torch, transformers, crawl4ai, mlflow client,
# scikit-learn/xgboost/lightgbm) are intentionally deferred — see DEPLOY.md
# "Wiring full DAG execution". Scope now: UIs up + DAGs loaded/triggerable.
RUN pip install --no-cache-dir \
    pyspark==3.5.3 \
    dvc==3.55.2 \
    requests beautifulsoup4 feedparser


# =====================================================================
# Stage: mlflow  — tracking server + UI
# =====================================================================
FROM python:3.11-slim AS mlflow

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install mlflow==2.17.2 psycopg2-binary==2.9.10

EXPOSE 5000
# Backend store + artifact destination are set by docker-compose `command`.
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
