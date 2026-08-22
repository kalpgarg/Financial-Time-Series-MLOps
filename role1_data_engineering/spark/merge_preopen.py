"""
Merge pre-open data into the 15-minute OHLCV merged CSV.

Reads the day's NSE pre-open CSV (data/preopen_csv/nse_fo_<YYYYMMDD>_preopen.csv),
maps NSE ticker symbols to Stock_name values via stock_list.csv, and appends
a synthetic 09:15 bar to data/ohlc_data/merged_ohlc_15min.csv.

Mapping:
  open = close = high = low = final_price
  volume = final_quantity
  datetime = <today> 09:15:00

Usage:
    python -m role1_data_engineering.spark.merge_preopen
    python -m role1_data_engineering.spark.merge_preopen --date 20260619
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# ── Resolve project root so shared imports work when running as script ────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import STOCK_LIST_CSV_PATH, now_local

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("merge_preopen")

PREOPEN_DIR = os.path.join(PROJECT_ROOT, "data", "preopen_csv")
MERGED_15MIN_CSV = os.path.join(PROJECT_ROOT, "data", "merged_ohlc_15min.csv")


def _build_symbol_map(stock_list_path: str) -> dict[str, str]:
    """Build a mapping from NSE ticker symbol → Stock_name.

    stock_list.csv has columns like Stock_name, TradingView_name, Groww_name, etc.
    The NSE pre-open scraper uses the raw NSE symbol (e.g. RELIANCE, HDFCBANK).
    TradingView_name is formatted as 'NSE:SYMBOL', so we extract the symbol part
    and map it to Stock_name.
    """
    df = pd.read_csv(stock_list_path)
    df.columns = [c.strip() for c in df.columns]

    symbol_map: dict[str, str] = {}

    # Primary mapping: Symbol column (stock_list.csv uses "Symbol" for NSE ticker)
    if "Symbol" in df.columns:
        for _, row in df.iterrows():
            nse_sym = str(row.get("Symbol", "")).strip()
            stock_name = str(row.get("Stock_name", "")).strip()
            if nse_sym and stock_name:
                symbol_map[nse_sym] = stock_name

    # Fallback: NSE_symbol column if it exists
    if not symbol_map and "NSE_symbol" in df.columns:
        for _, row in df.iterrows():
            nse_sym = str(row.get("NSE_symbol", "")).strip()
            stock_name = str(row.get("Stock_name", "")).strip()
            if nse_sym and stock_name:
                symbol_map[nse_sym] = stock_name

    # Fallback: extract from TradingView_name (format "NSE:SYMBOL")
    if not symbol_map and "TradingView_name" in df.columns:
        for _, row in df.iterrows():
            tv_name = str(row.get("TradingView_name", "")).strip()
            stock_name = str(row.get("Stock_name", "")).strip()
            if ":" in tv_name and stock_name:
                nse_sym = tv_name.split(":")[-1]
                symbol_map[nse_sym] = stock_name

    logger.info("Built symbol map: %d NSE symbols → Stock_name", len(symbol_map))
    return symbol_map


def merge_preopen(date_str: str | None = None) -> int:
    """Read the pre-open CSV for the given date and append 09:15 bars.

    Args:
        date_str: Date in YYYYMMDD format. Defaults to today (IST).

    Returns:
        Number of rows appended.
    """
    if date_str is None:
        date_str = now_local().strftime("%Y%m%d")

    preopen_filename = f"nse_fo_{date_str}_preopen.csv"
    preopen_path = os.path.join(PREOPEN_DIR, preopen_filename)

    if not os.path.exists(preopen_path):
        logger.error("Pre-open CSV not found: %s", preopen_path)
        return 0

    preopen_df = pd.read_csv(preopen_path, encoding="utf-8-sig")
    if preopen_df.empty:
        logger.warning("Pre-open CSV is empty: %s", preopen_path)
        return 0

    logger.info("Read %d pre-open records from %s", len(preopen_df), preopen_path)

    # Build symbol mapping
    symbol_map = _build_symbol_map(STOCK_LIST_CSV_PATH)

    # Map pre-open symbols to Stock_name and create OHLCV rows
    datetime_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} 09:15:00"
    rows = []

    for _, row in preopen_df.iterrows():
        nse_symbol = str(row.get("symbol", "")).strip()
        stock_name = symbol_map.get(nse_symbol)

        if stock_name is None:
            logger.debug("No Stock_name mapping for NSE symbol: %s", nse_symbol)
            continue

        final_price = row.get("final_price")
        final_quantity = row.get("final_quantity", 0)

        if pd.isna(final_price) or final_price is None:
            logger.warning("Skipping %s: no final_price", nse_symbol)
            continue

        rows.append({
            "symbol": stock_name,
            "datetime": datetime_str,
            "open": round(float(final_price), 2),
            "high": round(float(final_price), 2),
            "low": round(float(final_price), 2),
            "close": round(float(final_price), 2),
            "volume": int(final_quantity) if not pd.isna(final_quantity) else 0,
        })

    if not rows:
        logger.warning("No symbols matched between pre-open and stock_list.")
        return 0

    new_df = pd.DataFrame(rows)
    logger.info("Created %d OHLCV rows from pre-open data for %s", len(new_df), datetime_str)

    # Append to merged_ohlc_15min.csv
    os.makedirs(os.path.dirname(MERGED_15MIN_CSV), exist_ok=True)

    if not os.path.exists(MERGED_15MIN_CSV) or os.path.getsize(MERGED_15MIN_CSV) == 0:
        with open(MERGED_15MIN_CSV, "w", encoding="utf-8-sig", newline="") as f:
            new_df.to_csv(f, index=False, header=True)
        logger.info("Created new merged CSV with %d rows → %s", len(new_df), MERGED_15MIN_CSV)
    else:
        # Read existing to check for duplicates on (symbol, datetime)
        existing_df = pd.read_csv(MERGED_15MIN_CSV, encoding="utf-8-sig")
        mask = existing_df["datetime"] == datetime_str
        if mask.any():
            logger.info(
                "Removing %d existing rows for datetime=%s before appending",
                mask.sum(), datetime_str,
            )
            existing_df = existing_df[~mask]
            # Rewrite the full file
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined = combined.sort_values(["symbol", "datetime"]).reset_index(drop=True)
            with open(MERGED_15MIN_CSV, "w", encoding="utf-8-sig", newline="") as f:
                combined.to_csv(f, index=False, header=True)
        else:
            new_df.to_csv(MERGED_15MIN_CSV, mode="a", index=False, header=False, encoding="utf-8")

        logger.info("Appended %d rows to %s", len(new_df), MERGED_15MIN_CSV)

    return len(new_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge NSE pre-open data into merged_ohlc_15min.csv"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date in YYYYMMDD format (default: today IST)",
    )
    args = parser.parse_args()
    count = merge_preopen(date_str=args.date)
    if count == 0:
        sys.exit(1)
