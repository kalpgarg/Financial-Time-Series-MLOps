"""
Model evaluation utilities.

Computes fold-level classification metrics and builds prediction
DataFrames for experiment tracking.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_fold_metrics(y_true, y_pred):
    """Compute classification metrics for a single CV fold.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dict with 'accuracy' and 'macro_f1' keys.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def build_fold_predictions(
    merged_df, test_idx, y_true, y_pred, y_prob, fold, experiment_number, model_name
):
    """Build a predictions DataFrame for a single CV fold.

    Args:
        merged_df: Full merged DataFrame (for metadata columns).
        test_idx: Test set row indices.
        y_true: Ground truth labels (array-like).
        y_pred: Predicted labels (array-like).
        y_prob: Predicted probabilities (ndarray [N, 3]).
        fold: Fold number (1-indexed).
        experiment_number: Experiment number (1-indexed).
        model_name: Model identifier string (e.g. "XGBoost", "LightGBM").

    Returns:
        DataFrame with columns: symbol, date, target_return, Actual,
        Predicted, Prob_Negative, Prob_Neutral, Prob_Positive, Confidence,
        Correct, Fold, Experiment, Model.
    """
    fold_predictions = merged_df.iloc[test_idx][
        ["symbol", "date", "target_return"]
    ].copy()

    fold_predictions["Actual"] = np.asarray(y_true)
    fold_predictions["Predicted"] = np.asarray(y_pred)
    fold_predictions["Prob_Negative"] = y_prob[:, 0]
    fold_predictions["Prob_Neutral"] = y_prob[:, 1]
    fold_predictions["Prob_Positive"] = y_prob[:, 2]
    fold_predictions["Confidence"] = y_prob.max(axis=1)
    fold_predictions["Correct"] = (
        fold_predictions["Actual"] == fold_predictions["Predicted"]
    )
    fold_predictions["Fold"] = fold
    fold_predictions["Experiment"] = experiment_number
    fold_predictions["Model"] = model_name

    return fold_predictions
