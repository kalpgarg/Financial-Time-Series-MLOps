# syntax=docker/dockerfile:1

# Python 3.12 to match the training env (pyproject requires >=3.12) and to
# allow xgboost 3.4.x, the version the artifacts were trained/saved with.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Pin thread count: avoids CPU oversubscription and keeps inference
    # reproducible. Also sidesteps duplicate-OpenMP issues between torch and
    # xgboost.
    OMP_NUM_THREADS=1 \
    # FinBERT is baked in below; force offline so runtime never calls the Hub.
    HF_HOME=/app/.hf_cache \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    # The .pkl artifacts are NOT baked in -- they are mounted here at runtime
    # from the repo-level models/ directory (see docker-compose.yml).
    ARTIFACTS_DIR=/app/models

WORKDIR /app

# System libs: torch and xgboost both need the OpenMP runtime (libgomp1),
# which the slim base image does not ship. Without it, `import torch` fails
# with "libgomp.so.1: cannot open shared object file".
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Ensure pip trusts system CAs (some Docker VMs lack current root certs).
RUN pip install --upgrade certifi pip

# CPU-only torch, from PyTorch's CPU index. torch is deliberately absent from
# requirements.txt so the next step cannot replace this with the ~2GB CUDA
# build (which would drag in nvidia-* packages and bloat the image).
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    --trusted-host download.pytorch.org \
    --trusted-host download-r2.pytorch.org \
    torch==2.13.0

COPY requirements.txt .
# Install, then purge any CUDA packages. xgboost's Linux wheel pulls
# nvidia-nccl-cu12 (~150MB, the libnccl that broke the first build), which is
# only used for multi-GPU/distributed training -- CPU inference does not need
# it. Removing it keeps the image small and CPU-only.
RUN pip install -r requirements.txt \
    && (pip freeze | grep -i '^nvidia-' | cut -d= -f1 | xargs -r pip uninstall -y || true)

# Guard: fail the build loudly if any CUDA package remains (keeps the image
# CPU-only and small) and confirm torch and xgboost actually import. A
# regression in either is what broke the earlier builds.
RUN python -c "import sys, subprocess; \
out = subprocess.check_output(['pip', 'list'], text=True).lower(); \
nv = [l.split()[0] for l in out.splitlines() if l.startswith('nvidia-')]; \
sys.exit('CUDA packages present, expected CPU-only: ' + ', '.join(nv)) if nv else None; \
import torch, xgboost; print('torch', torch.__version__, '| xgboost', xgboost.__version__, '(CPU-only OK)')"

# Bake the FinBERT weights into the image (~440MB) so the container is fully
# offline and boots fast. Needs network at BUILD time only.
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c "\
from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
AutoTokenizer.from_pretrained('ProsusAI/finbert'); \
AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')"

# Application code. The four .pkl artifacts are NOT copied here -- they live in
# the repo-level models/ dir and are mounted at /app/models at runtime.
COPY . .

# Non-root user; own the code and the HF cache so warmup can read it.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Generous start period: first boot loads XGBoost + FinBERT before /health is ok.
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=5 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
