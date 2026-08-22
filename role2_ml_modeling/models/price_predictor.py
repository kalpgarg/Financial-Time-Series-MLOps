"""
Price direction predictor model factories.

Provides XGBoost and LightGBM classifier constructors with pre-defined
parameter sets for experiment sweeps. Both predict 3-class next-day
market direction: 0 (negative) / 1 (neutral) / 2 (positive).
"""

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# ── XGBoost ──────────────────────────────────────────────────────────────────


def create_xgboost(params):
    """Create a configured XGBClassifier for 3-class prediction.

    Args:
        params: Dict of XGBoost hyperparameters (n_estimators, max_depth, etc.).

    Returns:
        Configured XGBClassifier instance.
    """
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        eval_metric="mlogloss",
        **params,
    )


XGBOOST_PARAM_SETS = [
    {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    {
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
]


# ── LightGBM ─────────────────────────────────────────────────────────────────


def create_lightgbm(params):
    """Create a configured LGBMClassifier for 3-class prediction.

    Args:
        params: Dict of LightGBM hyperparameters (n_estimators, max_depth, etc.).

    Returns:
        Configured LGBMClassifier instance.
    """
    return LGBMClassifier(
        objective="multiclass",
        num_class=3,
        random_state=42,
        verbosity=-1,
        **params,
    )


LGBM_PARAM_SETS = [
    {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    {
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.05,
        "num_leaves": 20,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
]
