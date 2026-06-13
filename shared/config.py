"""
Central configuration shared across all roles.
Each role imports the settings it needs; environment variables override defaults.
"""

import os
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# ── Kafka ─────────────────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_NEWS_FEATURES = os.getenv("KAFKA_TOPIC_NEWS_FEATURES", "news_features")
KAFKA_TOPIC_MARKET_FEATURES = os.getenv("KAFKA_TOPIC_MARKET_FEATURES", "market_features")
KAFKA_TOPIC_EXECUTION_SIGNALS = os.getenv("KAFKA_TOPIC_EXECUTION_SIGNALS", "execution_signals")
# Legacy aliases (kept for backward-compatibility)
KAFKA_TOPIC_HEADLINES = KAFKA_TOPIC_NEWS_FEATURES
KAFKA_TOPIC_PRICES = KAFKA_TOPIC_MARKET_FEATURES

# ── Spark ─────────────────────────────────────────────────────────────────────
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")
SPARK_APP_NAME = os.getenv("SPARK_APP_NAME", "FinTSProcessor")

# ── PostgreSQL ────────────────────────────────────────────────────────────────
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "fints")
POSTGRES_USER = os.getenv("POSTGRES_USER", "fints_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "fints_pass")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
)
JDBC_URL = f"jdbc:postgresql://{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# ── Data Sync ────────────────────────────────────────────────────────────
CLOUD_SYNC_DIR = os.getenv("CLOUD_SYNC_DIR", "/Users/kgarg/extras/personal_stuff/03_Resources/stock_data")

# ── Telegram Alerts ───────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

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

# ── Groww Headline Scraping ──────────────────────────────────────────────────
GROWW_NEWS_URL_TEMPLATE = "https://groww.in/stocks/{groww_name}/market-news"
STOCK_LIST_CSV_PATH = os.getenv(
    "STOCK_LIST_CSV_PATH",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "stock_list.csv"),
)
HEADLINE_FETCH_INTERVAL_MINUTES = int(os.getenv("HEADLINE_FETCH_INTERVAL_MINUTES", "30"))

# ── OHLCV Data (TradingView) ─────────────────────────────────────────────────
OHLC_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "ohlc_data",
)
OHLC_DEFAULT_BARS = int(os.getenv("OHLC_DEFAULT_BARS", "10000"))

# ── Project Timezone ──────────────────────────────────────────────────────────
# All user-facing timestamps (CSV files, logs, dry-run output) use this zone.
# Override with TZ env var if deploying outside India.
PROJECT_TZ = ZoneInfo(os.getenv("PROJECT_TZ", "Asia/Kolkata"))


def now_local() -> datetime:
    """Return the current time in the project timezone (IST by default)."""
    return datetime.now(PROJECT_TZ)


def to_local(dt: datetime) -> datetime:
    """Convert any aware datetime to the project timezone."""
    return dt.astimezone(PROJECT_TZ)
