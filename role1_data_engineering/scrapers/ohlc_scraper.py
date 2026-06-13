"""
OHLCV scraper: fetches daily and 15-minute OHLCV data for all Nifty-50
stocks from TradingView via the tvDatafeed library and writes per-stock
CSV files into data/ohlc_data/<Stock_name>_{daily,15min}_data.csv.

Incremental logic:
  - First run (no CSV): fetches the last OHLC_DEFAULT_BARS (1000) bars.
  - Subsequent runs: reads the existing CSV, determines the latest
    date/datetime, fetches only bars after that point and appends them.

After all stocks are scraped, merged files are created:
  - data/ohlc_data/merged_ohlc.csv        (daily)
  - data/ohlc_data/merged_ohlc_15min.csv   (15-minute)

Usage:
    # Write to CSV (default)
    python -m role1_data_engineering.scrapers.ohlc_scraper

    # Dry-run – print to stdout, no file written
    python -m role1_data_engineering.scrapers.ohlc_scraper --dry-run
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from tvDatafeed import TvDatafeed, Interval

# ── Resolve project root so shared imports work when running as script ────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import (
    STOCK_LIST_CSV_PATH,
    OHLC_DATA_DIR,
    OHLC_DEFAULT_BARS,
    PROJECT_TZ,
    now_local,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ohlc_scraper")

MERGED_CSV_DAILY = "merged_ohlc.csv"
MERGED_CSV_15MIN = "merged_ohlc_15min.csv"

CSV_COLUMNS_DAILY = ["symbol", "date", "open", "high", "low", "close", "volume"]
CSV_COLUMNS_15MIN = ["symbol", "datetime", "open", "high", "low", "close", "volume"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _stock_csv_path(stock_name: str, interval_label: str = "daily") -> str:
    """Return the per-stock CSV path.

    interval_label: 'daily' or '15min'
    """
    safe_name = stock_name.replace(" ", "_").replace("/", "_")
    return os.path.join(OHLC_DATA_DIR, f"{safe_name}_{interval_label}_data.csv")


def _parse_tv_name(tv_name: str) -> tuple[str, str]:
    """Parse 'NSE:HDFCBANK' into ('HDFCBANK', 'NSE')."""
    parts = tv_name.strip().split(":")
    if len(parts) == 2:
        return parts[1], parts[0]
    return tv_name, "NSE"


def load_stock_list(csv_path: str) -> pd.DataFrame:
    """Load stock_list.csv and return rows that have a non-empty TradingView_name."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["TradingView_name"])
    df = df[df["TradingView_name"].str.strip().astype(bool)]
    logger.info("Loaded %d stocks with TradingView_name from %s", len(df), csv_path)
    return df


def load_existing_stock_csv(csv_path: str) -> pd.DataFrame | None:
    """Read existing per-stock CSV. Returns None if file missing/empty."""
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return None
    if df.empty:
        return None
    return df


def get_latest_date(df: pd.DataFrame) -> pd.Timestamp | None:
    """Get the most recent date from an existing stock dataframe."""
    if df is None or "date" not in df.columns:
        return None
    dates = pd.to_datetime(df["date"], errors="coerce")
    latest = dates.max()
    if pd.isna(latest):
        return None
    return latest


def _get_latest_datetime(df: pd.DataFrame) -> pd.Timestamp | None:
    """Get the most recent datetime from an existing intraday stock dataframe."""
    if df is None or "datetime" not in df.columns:
        return None
    dts = pd.to_datetime(df["datetime"], errors="coerce")
    latest = dts.max()
    if pd.isna(latest):
        return None
    return latest


def fetch_stock_data(
    tv: TvDatafeed,
    symbol: str,
    exchange: str,
    stock_name: str,
    n_bars: int = OHLC_DEFAULT_BARS,
    interval: Interval = Interval.in_daily,
) -> pd.DataFrame | None:
    """Fetch OHLCV data from TradingView for a single stock.

    Returns a cleaned DataFrame with columns matching the appropriate
    CSV_COLUMNS list, or None.
    """
    try:
        df = tv.get_hist(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            n_bars=n_bars,
        )
    except Exception:
        logger.exception("TradingView fetch failed for %s (%s:%s)", stock_name, exchange, symbol)
        return None

    if df is None or df.empty:
        logger.warning("No data returned for %s (%s:%s)", stock_name, exchange, symbol)
        return None

    # The returned DF has a datetime index (UTC) and columns: symbol, open, high, low, close, volume
    df = df.reset_index()
    # Convert UTC datetime to IST
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_convert(PROJECT_TZ)

    is_intraday = interval != Interval.in_daily

    if is_intraday:
        # Keep full datetime string for intraday data
        df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        csv_cols = CSV_COLUMNS_15MIN
    else:
        df["date"] = df["datetime"].dt.date.astype(str)
        csv_cols = CSV_COLUMNS_DAILY

    # Use our Stock_name as the symbol (not the TradingView symbol like NSE:HDFCBANK)
    df["symbol"] = stock_name
    df = df[csv_cols].copy()
    # Round floats to 2 decimal places
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].round(2)
    df["volume"] = df["volume"].astype(int)

    return df


def save_stock_csv(df: pd.DataFrame, csv_path: str, first_write: bool = False):
    """Save stock data to CSV with UTF-8 BOM on first write."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if first_write:
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            df.to_csv(f, index=False, header=True)
    else:
        df.to_csv(csv_path, mode="a", index=False, header=False, encoding="utf-8")


def scrape_stock(
    tv: TvDatafeed,
    stock_name: str,
    tv_name: str,
    interval: Interval = Interval.in_daily,
) -> tuple[str, int]:
    """Scrape a single stock: fetch new data, append to its CSV.

    Returns (stock_name, new_rows_count).
    """
    is_intraday = interval != Interval.in_daily
    interval_label = "15min" if is_intraday else "daily"
    time_col = "datetime" if is_intraday else "date"

    symbol, exchange = _parse_tv_name(tv_name)
    csv_path = _stock_csv_path(stock_name, interval_label)
    existing_df = load_existing_stock_csv(csv_path)
    latest_date = get_latest_date(existing_df) if not is_intraday else _get_latest_datetime(existing_df)

    if latest_date is not None:
        logger.info("  %s [%s]: existing data up to %s", stock_name, interval_label, str(latest_date))
        if latest_date.tzinfo is None:
            latest_date = latest_date.tz_localize(PROJECT_TZ)
        gap_seconds = (pd.Timestamp.now(tz=PROJECT_TZ) - latest_date).total_seconds()
        if is_intraday:
            # Each bar = 15 min; add a buffer of 20 extra bars
            n_bars = max(int(gap_seconds / 900) + 20, 10)
        else:
            days_gap = int(gap_seconds / 86400) + 5
            n_bars = max(days_gap, 10)
    else:
        logger.info("  %s [%s]: no existing data — fetching %d bars", stock_name, interval_label, OHLC_DEFAULT_BARS)
        n_bars = OHLC_DEFAULT_BARS

    fetched_df = fetch_stock_data(tv, symbol, exchange, stock_name, n_bars=n_bars, interval=interval)
    if fetched_df is None:
        return stock_name, 0

    # Filter to only new rows strictly after the latest existing point
    if latest_date is not None:
        latest_str = latest_date.strftime("%Y-%m-%d %H:%M:%S") if is_intraday else latest_date.strftime("%Y-%m-%d")
        fetched_df = fetched_df[fetched_df[time_col] > latest_str]

    if fetched_df.empty:
        logger.info("  %s [%s]: 0 new rows", stock_name, interval_label)
        return stock_name, 0

    # Drop duplicates within the fetched batch
    fetched_df = fetched_df.drop_duplicates(subset=[time_col], keep="last")

    first_write = existing_df is None
    save_stock_csv(fetched_df, csv_path, first_write=first_write)
    logger.info("  %s [%s]: %d new rows appended → %s", stock_name, interval_label, len(fetched_df), csv_path)
    return stock_name, len(fetched_df)


def build_merged_csv(interval_label: str = "daily"):
    """Concatenate all per-stock CSVs into a merged file.

    interval_label: 'daily' or '15min'
    """
    suffix = f"_{interval_label}_data.csv"
    merged_name = MERGED_CSV_DAILY if interval_label == "daily" else MERGED_CSV_15MIN
    sort_col = "date" if interval_label == "daily" else "datetime"
    merged_path = os.path.join(OHLC_DATA_DIR, merged_name)
    all_frames = []

    for fname in sorted(os.listdir(OHLC_DATA_DIR)):
        if fname == merged_name or not fname.endswith(suffix):
            continue
        fpath = os.path.join(OHLC_DATA_DIR, fname)
        try:
            df = pd.read_csv(fpath, encoding="utf-8-sig")
            if not df.empty:
                all_frames.append(df)
        except Exception:
            logger.warning("Skipping corrupt file: %s", fpath)

    if not all_frames:
        logger.warning("No per-stock %s CSVs found — skipping merged CSV.", interval_label)
        return

    merged = pd.concat(all_frames, ignore_index=True)
    merged = merged.sort_values(["symbol", sort_col]).reset_index(drop=True)

    with open(merged_path, "w", encoding="utf-8-sig", newline="") as f:
        merged.to_csv(f, index=False, header=True)
    logger.info(
        "Merged %s CSV: %d rows across %d stocks → %s",
        interval_label, len(merged), merged["symbol"].nunique(), merged_path,
    )


def _run_interval_scrape(
    tv: TvDatafeed,
    stocks_df: pd.DataFrame,
    interval: Interval,
) -> tuple[int, list[tuple[str, int]]]:
    """Scrape all stocks for a given interval. Returns (total_new, results)."""
    label = "15min" if interval != Interval.in_daily else "daily"
    logger.info("── Starting %s scrape ──", label)

    total_new = 0
    results: list[tuple[str, int]] = []

    for _, row in stocks_df.iterrows():
        stock_name = row["Stock_name"].strip()
        tv_name = row["TradingView_name"].strip()

        try:
            name, count = scrape_stock(tv, stock_name, tv_name, interval=interval)
            results.append((name, count))
            total_new += count
        except Exception:
            logger.exception("Failed to scrape %s [%s]", stock_name, label)
            results.append((stock_name, 0))

        # Small delay to avoid rate-limiting
        time.sleep(1)

    return total_new, results


def scrape_all(dry_run: bool = False):
    """Main entry point: scrape all stocks (daily + 15min), then build merged CSVs."""
    stocks_df = load_stock_list(STOCK_LIST_CSV_PATH)
    if stocks_df.empty:
        logger.error("No stocks with TradingView_name found in %s. Exiting.", STOCK_LIST_CSV_PATH)
        return

    # Initialize TradingView connection
    tv_uname = os.getenv("TV_UNAME")
    tv_pass = os.getenv("TV_PASSWD")
    if tv_uname and tv_pass:
        logger.info("Logging into TradingView as %s", tv_uname)
        tv = TvDatafeed(tv_uname, tv_pass)
    else:
        logger.info("No TV_UNAME/TV_PASSWD set — using anonymous access (data may be limited)")
        tv = TvDatafeed()

    # Daily scrape
    # daily_total, daily_results = _run_interval_scrape(tv, stocks_df, Interval.in_daily)

    # 15-minute scrape
    min15_total, min15_results = _run_interval_scrape(tv, stocks_df, Interval.in_15_minute)

    # Summary
    logger.info("=" * 60)
    logger.info("OHLCV scrape complete at %s", now_local().isoformat())
    # logger.info("Daily — total new rows: %d", daily_total)
    # for name, count in daily_results:
    #     if count > 0:
    #         logger.info("  %s [daily]: +%d rows", name, count)
    logger.info("15min — total new rows: %d", min15_total)
    for name, count in min15_results:
        if count > 0:
            logger.info("  %s [15min]: +%d rows", name, count)
    logger.info("=" * 60)

    if dry_run:
        logger.info("Dry-run mode — skipping merged CSV build.")
        return

    # Build merged CSVs for both intervals
    build_merged_csv("daily")
    build_merged_csv("15min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TradingView OHLCV scraper → CSV")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch data but skip building merged CSV",
    )
    args = parser.parse_args()
    scrape_all(dry_run=args.dry_run)
