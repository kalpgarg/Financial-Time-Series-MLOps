"""
Clean raw scraped headlines and write a deduplicated, cleaned CSV.

Reads data/stock_news/headlines.csv (raw scraper output), applies text
cleaning (lowercase, strip HTML tags/entities, collapse whitespace),
deduplicates by headline_id, and writes the result to data/headlines.csv
(the DVC-tracked file consumed by the inference pipeline).

Usage:
    python -m role1_data_engineering.spark.clean_headlines
"""

import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd

# ── Resolve project root so shared imports work ──────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("clean_headlines")

RAW_CSV = os.path.join(PROJECT_ROOT, "data", "stock_news", "headlines.csv")
CLEAN_CSV = os.path.join(PROJECT_ROOT, "data", "headlines.csv")

OUTPUT_COLUMNS = [
    "headline_id",
    "symbol",
    "published_at",
    "source",
    "headline",
    "article_url",
    "scraped_at",
]


def clean_text(text: str | None) -> str | None:
    """Lowercase, strip HTML tags/entities, and collapse whitespace."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", "", text)           # strip HTML tags
    text = re.sub(r"&[a-z]+;", " ", text)          # strip HTML entities
    text = re.sub(r"\s+", " ", text).strip()       # collapse whitespace
    return text


def main():
    if not os.path.exists(RAW_CSV):
        logger.error("Raw headlines CSV not found: %s", RAW_CSV)
        sys.exit(1)

    try:
        raw_df = pd.read_csv(RAW_CSV, encoding="utf-8-sig")
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        logger.error("Raw headlines CSV is empty or corrupt: %s", RAW_CSV)
        sys.exit(1)

    if raw_df.empty:
        logger.warning("No raw headlines to process.")
        sys.exit(0)

    logger.info("Read %d raw headlines from %s", len(raw_df), RAW_CSV)

    # Clean text fields
    raw_df["headline"] = raw_df["headline"].apply(clean_text)
    raw_df["source"] = raw_df["source"].apply(clean_text)

    # Deduplicate by headline_id (keep first occurrence)
    deduped_df = raw_df.drop_duplicates(subset=["headline_id"], keep="first")

    # Select output columns (ignore any extra columns)
    available_cols = [c for c in OUTPUT_COLUMNS if c in deduped_df.columns]
    output_df = deduped_df[available_cols].copy()

    # Write cleaned CSV
    os.makedirs(os.path.dirname(CLEAN_CSV), exist_ok=True)
    with open(CLEAN_CSV, "w", encoding="utf-8-sig", newline="") as f:
        output_df.to_csv(f, index=False, header=True)

    logger.info("Wrote %d clean headlines → %s", len(output_df), CLEAN_CSV)


if __name__ == "__main__":
    main()
