"""Tests for shared/config.py (paths, timezone helpers)."""

from datetime import datetime, timezone

import pytest

# shared.config imports python-dotenv at module load.
pytest.importorskip("dotenv")

from shared import config  # noqa: E402


def test_project_timezone_is_ist():
    assert str(config.PROJECT_TZ) == "Asia/Kolkata"


def test_now_local_is_timezone_aware_ist():
    now = config.now_local()
    assert now.tzinfo is not None
    assert str(now.tzinfo) == "Asia/Kolkata"


def test_to_local_converts_utc_to_ist():
    utc_midnight = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    ist = config.to_local(utc_midnight)
    # IST is UTC+5:30
    assert (ist.hour, ist.minute) == (5, 30)


def test_stock_list_path_points_at_data_dir():
    assert config.STOCK_LIST_CSV_PATH.replace("\\", "/").endswith("data/stock_list.csv")


def test_api_port_is_int():
    assert isinstance(config.API_PORT, int)
