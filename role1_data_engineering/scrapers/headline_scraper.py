"""
Headline scraper: crawls stock-specific news headlines from Groww
(https://groww.in/stocks/<name>/market-news) using crawl4ai and writes
them to a timestamped CSV file under data/scraped/.

This script is invoked by Airflow at 8:30 AM IST and runs for ~30 minutes.

Usage:
    # Write to CSV (default)
    python -m role1_data_engineering.scrapers.headline_scraper

    # Dry-run – print to stdout, no file written
    python -m role1_data_engineering.scrapers.headline_scraper --dry-run
"""

import asyncio
import argparse
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from crawl4ai import AsyncWebCrawler

# ── Resolve project root so shared imports work when running as script ────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import GROWW_NEWS_URL_TEMPLATE, STOCK_LIST_CSV_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("headline_scraper")

# ── Regex to parse Groww news markdown lines ─────────────────────────────────
HEADLINE_PATTERN = re.compile(
    r"\[(?P<source>[^•\]]+?)\s*•\s*(?P<time_ago>[^a-z]*(?:min|hour|day|week|month|year)s?\s+ago)"
    r"(?P<headline>.+?)\]\((?P<url>https?://[^\)]+)\)",
    re.IGNORECASE,
)

SCRAPED_DIR = os.path.join(PROJECT_ROOT, "data", "scraped")


def _headline_id(url: str) -> str:
    """Deterministic headline ID from the article URL."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def load_stock_list(csv_path: str) -> pd.DataFrame:
    """Load stock_list.csv and return rows that have a non-empty Groww_name."""
    df = pd.read_csv(csv_path, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["Groww_name"])
    df = df[df["Groww_name"].str.strip().astype(bool)]
    logger.info("Loaded %d stocks with Groww_name from %s", len(df), csv_path)
    return df


def parse_headlines(markdown: str, stock_name: str) -> list[dict]:
    """Extract headline records from the crawled page markdown."""
    results = []
    for m in HEADLINE_PATTERN.finditer(markdown):
        source = m.group("source").strip()
        time_ago = m.group("time_ago").strip()
        headline_text = m.group("headline").strip()
        url = m.group("url").strip()

        record = {
            "headline_id": _headline_id(url),
            "symbol": stock_name,
            "published_at": time_ago,
            "source": source,
            "headline": headline_text,
            "article_url": url,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        }
        results.append(record)
    return results


async def crawl_stock_news(
    crawler: AsyncWebCrawler,
    groww_name: str,
    stock_name: str,
) -> list[dict]:
    """Crawl a single Groww market-news page and return parsed headlines."""
    url = GROWW_NEWS_URL_TEMPLATE.format(groww_name=groww_name)
    logger.info("Crawling %s for %s ...", url, stock_name)
    try:
        result = await crawler.arun(url=url)
        if not result.success:
            logger.warning("Crawl failed for %s: %s", stock_name, result.error_message)
            return []
        headlines = parse_headlines(result.markdown, stock_name)
        logger.info("  → found %d headlines for %s", len(headlines), stock_name)
        return headlines
    except Exception:
        logger.exception("Error crawling %s", stock_name)
        return []


async def scrape_all(stocks_df: pd.DataFrame) -> list[dict]:
    """Scrape headlines for every stock and return a flat list of records."""
    all_headlines: list[dict] = []
    seen_ids: set[str] = set()

    async with AsyncWebCrawler() as crawler:
        for _, row in stocks_df.iterrows():
            groww_name = row["Groww_name"].strip()
            stock_name = row["Stock_name"].strip()
            headlines = await crawl_stock_news(crawler, groww_name, stock_name)
            for h in headlines:
                if h["headline_id"] not in seen_ids:
                    seen_ids.add(h["headline_id"])
                    all_headlines.append(h)

    logger.info("Total unique headlines scraped: %d", len(all_headlines))
    return all_headlines


def save_to_csv(records: list[dict], output_path: str) -> str:
    """Write headline records to a CSV file and return the path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d headlines to %s", len(df), output_path)
    return output_path


async def main(dry_run: bool = False, output_path: str | None = None):
    stocks_df = load_stock_list(STOCK_LIST_CSV_PATH)
    if stocks_df.empty:
        logger.error("No stocks with Groww_name found in %s. Exiting.", STOCK_LIST_CSV_PATH)
        return

    headlines = await scrape_all(stocks_df)

    if dry_run:
        for h in headlines:
            print(json.dumps(h, indent=2))
        return

    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        output_path = os.path.join(SCRAPED_DIR, f"headlines_{ts}.csv")

    save_to_csv(headlines, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Groww headline scraper → CSV")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print headlines to stdout instead of writing CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output CSV path (default: data/scraped/headlines_YYYYMMDD_HHMM.csv)",
    )
    args = parser.parse_args()
    asyncio.run(main(dry_run=args.dry_run, output_path=args.output))
