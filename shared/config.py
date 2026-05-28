"""
Central configuration shared across all roles.
Each role imports the settings it needs; environment variables override defaults.
"""

import os

# ── Kafka ─────────────────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_HEADLINES = os.getenv("KAFKA_TOPIC_HEADLINES", "raw-headlines")
KAFKA_TOPIC_PRICES = os.getenv("KAFKA_TOPIC_PRICES", "raw-prices")

# ── Spark ─────────────────────────────────────────────────────────────────────
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")
SPARK_APP_NAME = os.getenv("SPARK_APP_NAME", "FinTSProcessor")

# ── Database / Storage ────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/fints")
DVC_REMOTE = os.getenv("DVC_REMOTE", "s3://fints-mlops-data")

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "financial-ts-prediction")

# ── FastAPI / Serving ─────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")

# ── Monitoring ────────────────────────────────────────────────────────────────
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "9090"))
KIBANA_URL = os.getenv("KIBANA_URL", "http://localhost:5601")

# ── Data APIs ─────────────────────────────────────────────────────────────────
NEWS_RSS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline",
    "https://www.reuters.com/rssFeed/businessNews",
]
STOCK_API_BASE = os.getenv("STOCK_API_BASE", "https://query1.finance.yahoo.com/v8/finance")
SYMBOLS = os.getenv("SYMBOLS", "^GSPC,^DJI,AAPL,MSFT,GOOG").split(",")
