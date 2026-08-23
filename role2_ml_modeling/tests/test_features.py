"""Tests for role2 feature engineering (pandas-based; no Spark/torch)."""

import pandas as pd
import pytest

from role2_ml_modeling.features import feature_engineering as fe


def test_build_daily_ohlcv_aggregates_bars():
    bars = pd.DataFrame(
        [
            {"symbol": "A", "datetime": "2026-01-05 09:15", "open": 100, "high": 105, "low": 99, "close": 102, "volume": 10},
            {"symbol": "A", "datetime": "2026-01-05 09:30", "open": 102, "high": 108, "low": 101, "close": 107, "volume": 20},
            {"symbol": "A", "datetime": "2026-01-05 09:45", "open": 107, "high": 110, "low": 106, "close": 109, "volume": 30},
        ]
    )
    daily = fe.build_daily_ohlcv(bars).reset_index(drop=True)
    row = daily.iloc[0]
    assert row["day_open"] == 100          # first bar's open
    assert row["day_high"] == 110          # max
    assert row["day_low"] == 99            # min
    assert row["day_close"] == 109         # last bar's close
    assert row["day_volume"] == 60         # sum


def test_add_next_day_features_uses_single_shift():
    """Row D must receive D+1's 09:15 bar (the corrected single-day shift)."""
    d1 = pd.Timestamp("2026-01-05")
    d2 = pd.Timestamp("2026-01-06")
    daily = pd.DataFrame(
        {"symbol": ["A", "A"], "date": [d1, d2], "day_close": [101.0, 205.0]}
    )
    bars = pd.DataFrame(
        [
            {"symbol": "A", "datetime": "2026-01-05 09:15", "open": 100.0, "high": 101.0, "low": 99.0, "close": 101.0, "volume": 5},
            {"symbol": "A", "datetime": "2026-01-06 09:15", "open": 200.0, "high": 206.0, "low": 199.0, "close": 205.0, "volume": 7},
        ]
    )
    out = fe.add_next_day_features(daily, bars).set_index("date")

    # D1 gets D2's 09:15 bar
    assert out.loc[d1, "open_915"] == 200.0
    assert out.loc[d1, "close_915"] == 205.0
    assert out.loc[d1, "first15_return"] == pytest.approx((205.0 - 200.0) / 200.0)
    # D2's "next day" doesn't exist -> NaN (this is the latest-date limitation)
    assert pd.isna(out.loc[d2, "open_915"])


def test_create_target_three_classes():
    daily = pd.DataFrame(
        {
            "symbol": ["A", "A", "A", "A"],
            "day_close": [100.0, 100.2, 90.0, 200.0],
            "open_915": [100.0, 100.0, 100.0, 100.0],
        }
    )
    out = fe.create_target(daily, threshold=0.003)
    # target_return[D] = (day_close[D+1] - open_915[D]) / open_915[D]
    #   D0: (100.2-100)/100 = +0.002  -> within band -> neutral (1)
    #   D1: (90-100)/100    = -0.10   -> negative (0)
    #   D2: (200-100)/100   = +1.00   -> positive (2)
    #   D3: next day is NaN           -> default neutral (1)
    assert list(out["target"]) == [1, 0, 2, 1]
