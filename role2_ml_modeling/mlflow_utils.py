"""
MLflow helper functions: experiment setup, parameter/metric logging,
and artifact tracking for training experiments.
"""

import mlflow


def setup_experiment(name):
    """Set (or create) an MLflow experiment by name.

    Args:
        name: Experiment name string.

    Returns:
        MLflow Experiment object.
    """
    return mlflow.set_experiment(name)


def log_pipeline_params(
    finbert_model="ProsusAI/finbert",
    pca_components=50,
    news_window_days=7,
    n_splits=5,
    model_type=None,
):
    """Log common pipeline parameters to the active MLflow run.

    Args:
        finbert_model: FinBERT model identifier.
        pca_components: Number of PCA components used.
        news_window_days: Sentiment aggregation window.
        n_splits: Number of TimeSeriesSplit folds.
        model_type: Model type string (e.g. "XGBoost", "LightGBM").
    """
    mlflow.log_param("finbert_model", finbert_model)
    mlflow.log_param("pca_components", pca_components)
    mlflow.log_param("news_window_days", news_window_days)
    mlflow.log_param("time_series_splits", n_splits)
    if model_type:
        mlflow.log_param("model_type", model_type)


def log_fold_metrics(fold, metrics):
    """Log fold-level metrics to the active MLflow run.

    Args:
        fold: Fold number (1-indexed).
        metrics: Dict with metric names as keys, values as floats.
    """
    for name, value in metrics.items():
        mlflow.log_metric(f"fold_{fold}_{name}", value)


def log_experiment_summary(avg_accuracy, avg_f1, prediction_file=None):
    """Log experiment-level summary metrics and optional prediction artifact.

    Args:
        avg_accuracy: Average accuracy across folds.
        avg_f1: Average macro F1 across folds.
        prediction_file: Optional path to predictions CSV to log as artifact.
    """
    mlflow.log_metric("average_accuracy", avg_accuracy)
    mlflow.log_metric("average_macro_f1", avg_f1)
    if prediction_file:
        mlflow.log_artifact(prediction_file)
