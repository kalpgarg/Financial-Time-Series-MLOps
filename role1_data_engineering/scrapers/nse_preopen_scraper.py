"""
NSE Pre-Open Market scraper: fetches the "Securities in F&O" pre-open
market data from the NSE India API and saves it as a dated CSV file
under data/preopen_csv/nse_fo_<YYYYMMDD>_preopen.csv.

The NSE API (https://www.nseindia.com/api/market-data-pre-open?key=FO)
returns JSON with per-symbol metadata matching the columns shown on
the NSE website's pre-open market page.

Usage:
    # Write to CSV (default)
    python -m role1_data_engineering.scrapers.nse_preopen_scraper

    # Dry-run – print to stdout, no file written
    python -m role1_data_engineering.scrapers.nse_preopen_scraper --dry-run
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# ── Resolve project root so shared imports work when running as script ────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import now_local

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nse_preopen_scraper")

PREOPEN_DIR = os.path.join(PROJECT_ROOT, "data", "preopen_csv")
NSE_PREOPEN_API = "https://www.nseindia.com/api/market-data-pre-open?key=FO"
NSE_BASE_URL = "https://www.nseindia.com"

# Columns matching the NSE website's pre-open page for "Securities in F&O"
CSV_COLUMNS = [
    "symbol",
    "prev_close",
    "iep",
    "change",
    "pct_change",
    "final_price",
    "final_quantity",
    "value_crores",
    "ffm_cap_crores",
    "nm_52w_high",
    "nm_52w_low",
]

# Maximum retries for NSE API (it can be flaky with cookies)
MAX_RETRIES = 3
RETRY_DELAY_SECS = 3


def _build_session() -> requests.Session:
    """Build an HTTP session with headers that NSE expects."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/market-data/pre-open-market-cm-and-emerge-market",
    })
    return session


def _warm_session(session: requests.Session) -> None:
    """Hit the NSE homepage to acquire session cookies before API calls."""
    try:
        session.get(NSE_BASE_URL, timeout=10)
    except requests.RequestException:
        logger.debug("Homepage warm-up request failed; proceeding anyway.")


def fetch_preopen_data(session: requests.Session) -> dict | None:
    """Fetch pre-open F&O data from NSE API with retry logic.

    Returns the full JSON response dict, or None on failure.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _warm_session(session)
            resp = session.get(NSE_PREOPEN_API, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                logger.info(
                    "Fetched pre-open data: %d records (advances=%s, declines=%s, timestamp=%s)",
                    len(data.get("data", [])),
                    data.get("advances"),
                    data.get("declines"),
                    data.get("timestamp"),
                )
                return data
            else:
                logger.warning(
                    "Attempt %d/%d: HTTP %d — %s",
                    attempt, MAX_RETRIES, resp.status_code, resp.text[:200],
                )
        except requests.RequestException as e:
            logger.warning("Attempt %d/%d: request error — %s", attempt, MAX_RETRIES, e)

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_SECS)

    logger.error("Failed to fetch pre-open data after %d attempts.", MAX_RETRIES)
    return None


def parse_preopen_records(raw_data: dict) -> list[dict]:
    """Extract flat records from the NSE API response.

    Each item in raw_data['data'] has a 'metadata' dict with the columns
    shown on the NSE website.
    """
    records = []
    for item in raw_data.get("data", []):
        meta = item.get("metadata", {})
        record = {
            "symbol": meta.get("symbol", ""),
            "prev_close": meta.get("previousClose"),
            "iep": meta.get("iep"),
            "change": meta.get("change"),
            "pct_change": meta.get("pChange"),
            "final_price": meta.get("lastPrice"),
            "final_quantity": meta.get("finalQuantity"),
            "value_crores": round(meta.get("totalTurnover", 0) / 1e7, 2),
            "ffm_cap_crores": round(meta.get("marketCap", 0) / 1e7, 2),
            "nm_52w_high": meta.get("yearHigh"),
            "nm_52w_low": meta.get("yearLow"),
        }
        records.append(record)

    logger.info("Parsed %d pre-open records.", len(records))
    return records


def save_preopen_csv(records: list[dict], timestamp_str: str | None = None) -> str:
    """Save records to data/preopen_csv/nse_fo_<YYYYMMDD>_preopen.csv.

    Returns the path to the saved CSV file.
    """
    os.makedirs(PREOPEN_DIR, exist_ok=True)

    # Determine date for filename
    if timestamp_str:
        # Parse NSE timestamp like "08-Jun-2026 09:07:55"
        try:
            from datetime import datetime
            dt = datetime.strptime(timestamp_str, "%d-%b-%Y %H:%M:%S")
            date_str = dt.strftime("%Y%m%d")
        except ValueError:
            date_str = now_local().strftime("%Y%m%d")
    else:
        date_str = now_local().strftime("%Y%m%d")

    filename = f"nse_fo_{date_str}_preopen.csv"
    csv_path = os.path.join(PREOPEN_DIR, filename)

    df = pd.DataFrame(records, columns=CSV_COLUMNS)
    # Sort by absolute change descending (most moved first), matching NSE default
    df = df.sort_values("pct_change", ascending=False, key=abs).reset_index(drop=True)

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        df.to_csv(f, index=False, header=True)

    logger.info("Saved %d records → %s", len(df), csv_path)
    return csv_path


def scrape_preopen(dry_run: bool = False):
    """Main entry point: fetch and save NSE F&O pre-open data."""
    session = _build_session()
    raw_data = fetch_preopen_data(session)
    if raw_data is None:
        return

    records = parse_preopen_records(raw_data)
    if not records:
        logger.warning("No records parsed from API response.")
        return

    if dry_run:
        for r in records[:10]:
            print(json.dumps(r, indent=2))
        print(f"... ({len(records)} total records)")
        return

    timestamp_str = raw_data.get("timestamp")
    csv_path = save_preopen_csv(records, timestamp_str=timestamp_str)
    logger.info("Pre-open scrape complete at %s → %s", now_local().isoformat(), csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSE F&O Pre-Open Market scraper → CSV")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print records to stdout instead of writing CSV",
    )
    args = parser.parse_args()
    scrape_preopen(dry_run=args.dry_run)
