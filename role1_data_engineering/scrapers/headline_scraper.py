"""
Headline scraper: crawls stock-specific news headlines from Groww
(https://groww.in/stocks/<name>/market-news) using crawl4ai and writes
them to a persistent CSV file at data/stock_news/headlines.csv.

The scraper extracts exact published-at timestamps from the <time> HTML
elements on the Groww news page. It supports incremental operation:
  - On first run (empty/missing CSV): scrolls to the bottom of each page
    to fetch all available historical news.
  - On subsequent runs: reads the CSV to find the latest timestamp and
    only appends headlines published after that timestamp.

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
from datetime import datetime
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

# ── Resolve project root so shared imports work when running as script ────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import (
    GROWW_NEWS_URL_TEMPLATE,
    STOCK_LIST_CSV_PATH,
    PROJECT_TZ,
    now_local,
    to_local,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("headline_scraper")

STOCK_NEWS_DIR = os.path.join(PROJECT_ROOT, "data", "stock_news")
HEADLINES_CSV_PATH = os.path.join(STOCK_NEWS_DIR, "headlines.csv")

CSV_COLUMNS = [
    "headline_id", "symbol", "published_at", "source",
    "headline", "article_url", "scraped_at",
]

# ── JS: scroll until all lazy-loaded news rows are rendered ──────────────────
# Used as a crawl4ai wait_for expression — blocks HTML capture until scrolling
# is complete.  The function scrolls to the bottom every 1.5 s and counts
# div[class*='stockNews_newsRow'] elements.  It resolves (returns true) once
# the count has been stable for 3 consecutive checks.
SCROLL_WAIT_FOR_JS = """() => {
    return new Promise((resolve) => {
        let prevCount = 0;
        let stableCount = 0;
        const interval = setInterval(() => {
            window.scrollTo(0, document.body.scrollHeight);
            const rows = document.querySelectorAll("div[class*='stockNews_newsRow']");
            const newCount = rows.length;
            if (newCount === prevCount) {
                stableCount++;
            } else {
                stableCount = 0;
            }
            prevCount = newCount;
            if (stableCount >= 3) {
                clearInterval(interval);
                resolve(true);
            }
        }, 1500);
    });
}"""


def _headline_id(url: str) -> str:
    """Deterministic headline ID from the article URL."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _parse_groww_time(time_tag) -> datetime | None:
    """Extract a datetime from a Groww <time> element, returned in PROJECT_TZ (IST).

    Tries, in order:
      1. The 'datetime' attribute (ISO 8601, e.g. '2026-06-07T07:25:51.000Z')
      2. The 'title' attribute   (ISO local,  e.g. '2026-06-07T12:55:51')
      3. Legacy human-readable title ('28 May 2026, 05:42 PM IST')
    """
    # 1. Prefer the datetime attribute — always ISO 8601 with tz
    dt_attr = (time_tag.get("datetime") or "").strip()
    if dt_attr:
        try:
            dt = datetime.fromisoformat(dt_attr.replace("Z", "+00:00"))
            return to_local(dt)
        except ValueError:
            pass

    # 2. Title attribute — ISO local time (IST implied)
    title_str = (time_tag.get("title") or "").strip()
    if title_str:
        try:
            dt = datetime.fromisoformat(title_str)
            dt = dt.replace(tzinfo=PROJECT_TZ)
            return dt
        except ValueError:
            pass

    # 3. Legacy human-readable format
    cleaned = title_str.replace(" IST", "")
    for fmt in (
        "%d %b %Y, %I:%M %p",
        "%d %B %Y, %I:%M %p",
        "%d %b %Y",
    ):
        try:
            dt = datetime.strptime(cleaned, fmt)
            return dt.replace(tzinfo=PROJECT_TZ)
        except ValueError:
            continue
    return None


def load_stock_list(csv_path: str) -> pd.DataFrame:
    """Load stock_list.csv and return rows that have a non-empty Groww_name."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["Groww_name"])
    df = df[df["Groww_name"].str.strip().astype(bool)]
    logger.info("Loaded %d stocks with Groww_name from %s", len(df), csv_path)
    return df


def load_existing_csv(
    csv_path: str,
) -> tuple[pd.DataFrame, dict[str, datetime]]:
    """Read the existing headlines CSV.

    Returns:
        (df, per_stock_latest_ts)

    *per_stock_latest_ts* maps each stock symbol to its most recent
    published_at datetime so the scraper can skip already-fetched news
    on a per-stock basis.
    """
    if not os.path.exists(csv_path):
        logger.info("No existing CSV at %s — first run, will fetch all history.", csv_path)
        return pd.DataFrame(columns=CSV_COLUMNS), {}

    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        logger.info("Existing CSV at %s is empty/corrupt — treating as first run.", csv_path)
        return pd.DataFrame(columns=CSV_COLUMNS), {}

    if df.empty:
        return df, {}

    # Build per-stock latest timestamp
    per_stock_latest: dict[str, datetime] = {}
    if "published_at" in df.columns and "symbol" in df.columns:
        df["_ts"] = pd.to_datetime(df["published_at"], errors="coerce")
        for symbol, group in df.dropna(subset=["_ts"]).groupby("symbol"):
            latest = group["_ts"].max()
            # Ensure timezone-aware in PROJECT_TZ
            if latest.tzinfo is None:
                latest = latest.tz_localize(PROJECT_TZ)
            per_stock_latest[symbol] = latest.to_pydatetime()
        df.drop(columns=["_ts"], inplace=True)

    logger.info(
        "Loaded %d existing headlines from %s (%d stocks with timestamps)",
        len(df), csv_path, len(per_stock_latest),
    )
    for sym, ts in sorted(per_stock_latest.items()):
        logger.info("  %s → latest %s", sym, ts.isoformat())

    return df, per_stock_latest


def parse_headlines_from_html(html: str, stock_name: str) -> list[dict]:
    """Extract headline records from the raw HTML of a Groww news page.

    Groww news cards have the structure:
      div.stockNews_newsRow  (one per article)
        div…BoxHeaderText    → source name + <time> tag
        div…BoxItemTitle     → headline text
        a[href]              → external article link (may be absent)
    """
    soup = BeautifulSoup(html, "html.parser")
    results = []

    # Each news card is a div whose class contains 'stockNews_newsRow'
    news_rows = soup.find_all("div", class_=re.compile(r"stockNews_newsRow"))
    logger.debug("Found %d news rows in HTML", len(news_rows))

    for row in news_rows:
        # ── Timestamp ────────────────────────────────────────────────────
        time_tag = row.find("time")
        if time_tag is None:
            continue

        published_dt = _parse_groww_time(time_tag)
        if published_dt is None:
            logger.debug(
                "Could not parse time: datetime=%s title=%s",
                time_tag.get("datetime"), time_tag.get("title"),
            )
            continue

        # ── Headline text ────────────────────────────────────────────────
        title_div = row.find("div", class_=re.compile(r"BoxItemTitle"))
        headline_text = title_div.get_text(" ", strip=True) if title_div else ""
        if not headline_text:
            # Fallback: longest text block in the row
            texts = [
                el.get_text(" ", strip=True)
                for el in row.find_all("div")
                if el.get_text(strip=True) and len(el.get_text(strip=True)) > 20
            ]
            headline_text = max(texts, key=len, default="")
        if not headline_text:
            continue

        # ── Source ───────────────────────────────────────────────────────
        source = ""
        header_div = row.find("div", class_=re.compile(r"BoxHeaderText"))
        if header_div:
            # Source is the text before the <time> tag
            for child in header_div.children:
                if hasattr(child, "name") and child.name == "time":
                    break
                if isinstance(child, str) and child.strip():
                    source = child.strip().rstrip(" •·")

        # ── Article URL ──────────────────────────────────────────────────
        article_link = None
        for a_tag in row.find_all("a", href=True):
            href = a_tag["href"]
            if href.startswith("http") and "groww.in" not in href:
                article_link = href
                break

        # Use the article URL for ID; if no external link, hash the headline
        if article_link:
            hid = _headline_id(article_link)
        else:
            hid = _headline_id(headline_text + published_dt.isoformat())

        record = {
            "headline_id": hid,
            "symbol": stock_name,
            "published_at": published_dt.isoformat(),
            "source": source,
            "headline": headline_text,
            "article_url": article_link or "",
            "scraped_at": now_local().isoformat(),
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
        config = CrawlerRunConfig(
            wait_for=SCROLL_WAIT_FOR_JS,
            page_timeout=120000,
        )
        result = await crawler.arun(url=url, config=config)
        if not result.success:
            logger.warning("Crawl failed for %s: %s", stock_name, result.error_message)
            return []
        headlines = parse_headlines_from_html(result.html, stock_name)
        logger.info("  → found %d headlines for %s", len(headlines), stock_name)
        return headlines
    except Exception:
        logger.exception("Error crawling %s", stock_name)
        return []


async def scrape_all(
    stocks_df: pd.DataFrame,
    per_stock_latest: dict[str, datetime],
) -> list[dict]:
    """Scrape headlines for every stock and return only new records.

    Uses *per_stock_latest* so that each stock's cutoff is independent.
    Filtering is based solely on the per-stock timestamp; headline IDs
    are not used for deduplication.
    """
    all_headlines: list[dict] = []

    async with AsyncWebCrawler() as crawler:
        for _, row in stocks_df.iterrows():
            groww_name = row["Groww_name"].strip()
            stock_name = row["Stock_name"].strip()
            cutoff = per_stock_latest.get(stock_name)
            if cutoff:
                logger.info("  cutoff for %s: %s", stock_name, cutoff.isoformat())

            headlines = await crawl_stock_news(crawler, groww_name, stock_name)
            new_for_stock = 0
            for h in headlines:
                # Skip headlines older than this stock's latest existing timestamp
                if cutoff is not None:
                    try:
                        h_ts = datetime.fromisoformat(h["published_at"])
                        if h_ts <= cutoff:
                            continue
                    except (ValueError, TypeError):
                        pass
                all_headlines.append(h)
                new_for_stock += 1
            logger.info("  → %d NEW headlines for %s", new_for_stock, stock_name)

    logger.info("Total new unique headlines scraped: %d", len(all_headlines))
    return all_headlines


def append_to_csv(records: list[dict], csv_path: str) -> str:
    """Append new headline records to the CSV. Creates the file if missing.

    Writes a UTF-8 BOM on first creation so that Excel opens the file
    with correct encoding (₹ and other non-ASCII chars display properly).
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df_new = pd.DataFrame(records, columns=CSV_COLUMNS)

    first_write = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    if first_write:
        # Write BOM + header + data
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            df_new.to_csv(f, index=False, header=True)
    else:
        df_new.to_csv(csv_path, mode="a", index=False, header=False,
                      encoding="utf-8")
    logger.info("Appended %d headlines to %s", len(df_new), csv_path)
    return csv_path


async def main(dry_run: bool = False):
    stocks_df = load_stock_list(STOCK_LIST_CSV_PATH)
    if stocks_df.empty:
        logger.error("No stocks with Groww_name found in %s. Exiting.", STOCK_LIST_CSV_PATH)
        return

    _, per_stock_latest = load_existing_csv(HEADLINES_CSV_PATH)
    headlines = await scrape_all(stocks_df, per_stock_latest)

    if not headlines:
        logger.info("No new headlines to save.")
        return

    if dry_run:
        for h in headlines:
            print(json.dumps(h, indent=2))
        return

    append_to_csv(headlines, HEADLINES_CSV_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Groww headline scraper → CSV")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print headlines to stdout instead of writing CSV",
    )
    args = parser.parse_args()
    asyncio.run(main(dry_run=args.dry_run))
