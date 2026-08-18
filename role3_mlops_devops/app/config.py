"""Runtime configuration, all overridable by environment variable."""

import os
from pathlib import Path
from zoneinfo import ZoneInfo

BASE_DIR = Path(__file__).resolve().parent.parent
# The model artifacts live in the repo-level models/ directory
# (github-project/models), one level above this role's folder. Override with
# ARTIFACTS_DIR to point elsewhere -- the Docker image sets it to the mount point.
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", BASE_DIR.parent / "models"))

# The four handed-over artifacts. FinBERT itself is loaded separately (see
# FINBERT_MODEL) because it comes from HuggingFace, not a local pickle.
PCA_PATH = Path(os.getenv("PCA_PATH", ARTIFACTS_DIR / "finbert_pca.pkl"))
ENCODER_PATH = Path(os.getenv("ENCODER_PATH", ARTIFACTS_DIR / "symbol_encoder.pkl"))
FEATURE_COLUMNS_PATH = Path(
    os.getenv("FEATURE_COLUMNS_PATH", ARTIFACTS_DIR / "feature_columns.pkl")
)
XGB_PATH = Path(os.getenv("XGB_PATH", ARTIFACTS_DIR / "xgboost_model.pkl"))

# Sentiment encoder. In the image this resolves to a baked-in local snapshot
# with TRANSFORMERS_OFFLINE=1 so no network call happens at runtime.
FINBERT_MODEL = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")

# There is no version string inside the artifacts, so we stamp one here. Bump
# it whenever the team hands over retrained artifacts -- it is written to every
# stored prediction and is how a row traces back to the model that produced it.
MODEL_VERSION = os.getenv("MODEL_VERSION", "finbert-pca-xgb-1.0.0")

# SQLite for local dev; the Docker stack sets a postgresql:// URL. The upsert
# and timezone handling already work on both.
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'predictions.db'}")

# Market timezone. Timestamps are stored timezone-aware in UTC and rendered in
# IST at the edges.
MARKET_TZ = ZoneInfo(os.getenv("MARKET_TZ", "Asia/Kolkata"))

SQL_ECHO = os.getenv("SQL_ECHO", "").lower() in {"1", "true", "yes"}
