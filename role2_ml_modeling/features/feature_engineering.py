"""
Feature engineering pipeline.

Reads 15-min OHLCV bars and sentiment features to produce a training-ready
feature matrix with technical indicators, next-day 9:15 features, and a
3-class target (negative / neutral / positive return).

Usage:
    python -m role2_ml_modeling.features.feature_engineering \
        --ohlcv data/merged_ohlc_15min.csv \
        --sentiment data/news_features.csv \
        --output data/merged_features.csv
"""

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ── Daily OHLCV aggregation ──────────────────────────────────────────────────


def build_daily_ohlcv(ohlcv_15min_df):
    """Aggregate 15-min bars into daily OHLCV.

    Args:
        ohlcv_15min_df: DataFrame with columns: symbol, datetime, open,
            high, low, close, volume.

    Returns:
        Daily OHLCV DataFrame with columns: symbol, date, day_open,
        day_high, day_low, day_close, day_volume.
    """
    df = ohlcv_15min_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["symbol", "datetime"])
    df["date"] = df["datetime"].dt.normalize()

    daily = (
        df.groupby(["symbol", "date"])
        .agg(
            day_open=("open", "first"),
            day_high=("high", "max"),
            day_low=("low", "min"),
            day_close=("close", "last"),
            day_volume=("volume", "sum"),
        )
        .reset_index()
    )
    return daily.sort_values(["symbol", "date"])


# ── Technical indicators ─────────────────────────────────────────────────────


def add_technical_features(daily_df):
    """Add technical indicators to daily OHLCV DataFrame.

    Adds: daily_return, open_close_pct, high_low_pct, return lags (1-3),
    rolling returns (7/14/30d), volatility (7/14/30d), price MAs (7/14/30),
    price vs MA ratios, volume MAs (7/14/30d), volume_ratio.

    Args:
        daily_df: Daily OHLCV DataFrame (modified in-place).

    Returns:
        Same DataFrame with added columns.
    """
    g = daily_df.groupby("symbol")

    # Daily return
    daily_df["daily_return"] = g["day_close"].pct_change()

    # Intra-day movement
    daily_df["open_close_pct"] = (
        (daily_df["day_close"] - daily_df["day_open"]) / daily_df["day_open"]
    )

    # Daily range
    daily_df["high_low_pct"] = (
        (daily_df["day_high"] - daily_df["day_low"]) / daily_df["day_low"]
    )

    # Lag features
    for lag in [1, 2, 3]:
        daily_df[f"return_lag_{lag}"] = g["daily_return"].shift(lag)

    # Rolling returns
    for w in [7, 14, 30]:
        daily_df[f"return_{w}d"] = g["daily_return"].transform(
            lambda x, _w=w: x.rolling(_w).mean()
        )

    # Volatility
    for w in [7, 14, 30]:
        daily_df[f"volatility_{w}d"] = g["daily_return"].transform(
            lambda x, _w=w: x.rolling(_w).std()
        )

    # Moving averages
    for w in [7, 14, 30]:
        daily_df[f"price_ma_{w}"] = g["day_close"].transform(
            lambda x, _w=w: x.rolling(_w).mean()
        )

    # Price vs moving averages
    for w in [7, 14, 30]:
        daily_df[f"price_vs_ma{w}"] = (
            (daily_df["day_close"] - daily_df[f"price_ma_{w}"])
            / daily_df[f"price_ma_{w}"]
        )

    # Volume features
    for w in [7, 14, 30]:
        daily_df[f"volume_{w}d"] = g["day_volume"].transform(
            lambda x, _w=w: x.rolling(_w).mean()
        )

    daily_df["volume_ratio"] = daily_df["day_volume"] / daily_df["volume_7d"]

    return daily_df


# ── Next-day 9:15 features ───────────────────────────────────────────────────


def add_next_day_features(daily_df, ohlcv_15min_df):
    """Add next-day 9:15 open/close features.

    Extracts the first 15-min candle (9:15 bar) for each stock-date,
    shifts it back by 1 day to attach D+1's opening data to D's row.

    Args:
        daily_df: Daily OHLCV DataFrame (modified in-place).
        ohlcv_15min_df: Raw 15-min OHLCV DataFrame.

    Returns:
        Same DataFrame with added columns: open_915, close_915,
        gap_from_prev_close, first15_return, first15_direction.
    """
    intra = ohlcv_15min_df.copy()
    intra["datetime"] = pd.to_datetime(intra["datetime"])
    intra = intra.sort_values(["symbol", "datetime"])
    intra["date"] = intra["datetime"].dt.normalize()

    # First candle of each day (9:15 bar)
    open_915 = (
        intra.groupby(["symbol", "date"]).first().reset_index()
    )[["symbol", "date", "open", "close"]]
    open_915 = open_915.rename(
        columns={"open": "open_915", "close": "close_915"}
    )

    # Shift D+1's 9:15 data back to D's row (single shift — bug fixed)
    open_915["date"] = open_915["date"] - pd.Timedelta(days=1)

    daily_df = daily_df.merge(open_915, on=["symbol", "date"], how="left")

    # Overnight gap
    daily_df["gap_from_prev_close"] = (
        (daily_df["open_915"] - daily_df["day_close"]) / daily_df["day_close"]
    )

    # First 15-min return
    daily_df["first15_return"] = (
        (daily_df["close_915"] - daily_df["open_915"]) / daily_df["open_915"]
    )

    # First 15-min direction
    daily_df["first15_direction"] = np.sign(daily_df["first15_return"])

    return daily_df


# ── Target creation ──────────────────────────────────────────────────────────


def create_target(daily_df, threshold=0.003):
    """Create the 3-class prediction target.

    Target return = (D+1 close - D+1 open_915) / open_915
    Classes: 0 (negative, < -threshold), 1 (neutral), 2 (positive, > +threshold)

    Args:
        daily_df: Daily OHLCV DataFrame with open_915 column.
        threshold: Return threshold for class boundaries.

    Returns:
        Same DataFrame with target_return and target columns.
    """
    # Next-day close per symbol. Using groupby().shift() instead of
    # groupby().apply(...) keeps this correct across pandas versions
    # (the apply form raises under pandas 3.x when assigned to one column).
    next_day_close = daily_df.groupby("symbol")["day_close"].shift(-1)
    daily_df["target_return"] = (
        (next_day_close - daily_df["open_915"]) / daily_df["open_915"]
    )

    daily_df["target"] = np.select(
        [
            daily_df["target_return"] < -threshold,
            daily_df["target_return"] > threshold,
        ],
        [0, 2],
        default=1,
    )

    return daily_df


# ── Merge and prepare ────────────────────────────────────────────────────────


def merge_and_prepare(daily_df, sentiment_df):
    """Merge OHLCV features with sentiment features and prepare for training.

    Args:
        daily_df: Daily OHLCV DataFrame with technical features and target.
        sentiment_df: Sentiment features DataFrame from sentiment_features.py.

    Returns:
        Tuple of (X, y, merged_df, label_encoder, feature_columns).
    """
    # Normalize date columns
    sentiment_df["date"] = (
        pd.to_datetime(sentiment_df["date"], utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )
    daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.normalize()

    # Inner merge on symbol + date
    merged_df = sentiment_df.merge(
        daily_df, on=["symbol", "date"], how="inner"
    )
    merged_df = merged_df.sort_values(["date", "symbol"])

    # Encode symbols
    le = LabelEncoder()
    merged_df["symbol_encoded"] = le.fit_transform(merged_df["symbol"])

    # Handle infinities
    merged_df = merged_df.replace([np.inf, -np.inf], np.nan)

    # Feature matrix
    drop_cols = ["symbol", "date", "target", "target_return"]
    X = merged_df.drop(columns=drop_cols)
    feature_columns = X.columns.tolist()

    y = merged_df["target"]

    return X, y, merged_df, le, feature_columns


def main():
    parser = argparse.ArgumentParser(description="Feature engineering pipeline")
    parser.add_argument(
        "--ohlcv",
        default="data/merged_ohlc_15min.csv",
        help="15-min OHLCV CSV path",
    )
    parser.add_argument(
        "--sentiment",
        default="data/news_features.csv",
        help="Sentiment features CSV path",
    )
    parser.add_argument(
        "--output",
        default="data/merged_features.csv",
        help="Output merged features CSV",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]

    # Build daily OHLCV
    ohlcv_path = project_root / args.ohlcv
    print(f"Reading 15-min OHLCV from {ohlcv_path}")
    ohlcv_15min = pd.read_csv(ohlcv_path)

    daily = build_daily_ohlcv(ohlcv_15min)
    daily = add_technical_features(daily)
    daily = add_next_day_features(daily, ohlcv_15min)
    daily = create_target(daily)
    print(f"Daily OHLCV shape: {daily.shape}")

    # Load sentiment features
    sentiment_path = project_root / args.sentiment
    print(f"Reading sentiment features from {sentiment_path}")
    sentiment_df = pd.read_csv(sentiment_path)

    # Merge and prepare
    X, y, merged_df, le, feature_columns = merge_and_prepare(daily, sentiment_df)
    print(f"Merged features shape: {X.shape}, target shape: {y.shape}")

    # Save artifacts
    output_path = project_root / args.output
    os.makedirs(output_path.parent, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    print(f"Merged features saved to {output_path}")

    models_dir = project_root / "models"
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(feature_columns, models_dir / "feature_columns.pkl")
    print(f"Feature columns saved ({len(feature_columns)} features)")

    joblib.dump(le, models_dir / "symbol_encoder.pkl")
    print("Symbol encoder saved")


if __name__ == "__main__":
    main()
