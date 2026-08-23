"""Tests for role2 model factories and evaluation metrics."""

import pytest

from role2_ml_modeling.training.evaluate import compute_fold_metrics


def test_compute_fold_metrics_perfect_prediction():
    m = compute_fold_metrics([0, 1, 2, 1], [0, 1, 2, 1])
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0


def test_compute_fold_metrics_partial():
    m = compute_fold_metrics([0, 1, 2, 1], [0, 1, 2, 0])  # 1 of 4 wrong
    assert m["accuracy"] == pytest.approx(0.75)
    assert 0.0 <= m["macro_f1"] <= 1.0


def test_create_xgboost_is_3class():
    # price_predictor imports lightgbm at module top.
    pytest.importorskip("lightgbm")
    from role2_ml_modeling.models.price_predictor import (
        XGBOOST_PARAM_SETS,
        create_xgboost,
    )
    from xgboost import XGBClassifier

    model = create_xgboost(XGBOOST_PARAM_SETS[0])
    assert isinstance(model, XGBClassifier)
    params = model.get_params()
    assert params["n_estimators"] == 300
    assert params["max_depth"] == 4
    assert params["objective"] == "multi:softprob"


def test_create_lightgbm_is_3class():
    pytest.importorskip("lightgbm")
    from lightgbm import LGBMClassifier

    from role2_ml_modeling.models.price_predictor import (
        LGBM_PARAM_SETS,
        create_lightgbm,
    )

    model = create_lightgbm(LGBM_PARAM_SETS[0])
    assert isinstance(model, LGBMClassifier)
    assert model.get_params()["objective"] == "multiclass"
