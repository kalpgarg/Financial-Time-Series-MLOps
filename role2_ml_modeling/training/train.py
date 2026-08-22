"""
Main training script.

Runs experiment sweeps over XGBoost and/or LightGBM parameter sets using
TimeSeriesSplit cross-validation, logs everything to MLflow, and saves
the best model as a .pkl artifact.

Usage:
    python -m role2_ml_modeling.training.train                 # both models
    python -m role2_ml_modeling.training.train --model xgboost # XGBoost only
    python -m role2_ml_modeling.training.train --model lightgbm
"""

import argparse
import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from role2_ml_modeling.mlflow_utils import (
    log_experiment_summary,
    log_fold_metrics,
    log_pipeline_params,
    setup_experiment,
)
from role2_ml_modeling.models.price_predictor import (
    LGBM_PARAM_SETS,
    XGBOOST_PARAM_SETS,
    create_lightgbm,
    create_xgboost,
)
from role2_ml_modeling.training.evaluate import (
    build_fold_predictions,
    compute_fold_metrics,
)


def run_experiment(
    model_factory,
    param_sets,
    X,
    y,
    merged_df,
    experiment_name,
    model_name,
    n_splits=5,
    output_dir=None,
):
    """Run a hyperparameter sweep with TimeSeriesSplit cross-validation.

    Args:
        model_factory: Callable(params) → sklearn-compatible classifier.
        param_sets: List of parameter dicts to sweep.
        X: Feature DataFrame.
        y: Target Series.
        merged_df: Full merged DataFrame (for building predictions).
        experiment_name: MLflow experiment name.
        model_name: Human-readable model name (e.g. "XGBoost").
        n_splits: Number of TimeSeriesSplit folds.
        output_dir: Directory to save prediction CSVs (optional).

    Returns:
        Tuple of (best_params, best_f1, best_predictions_df).
    """
    setup_experiment(experiment_name)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    best_f1 = -1
    best_accuracy = -1
    best_params = None
    best_predictions_df = None

    for experiment_number, params in enumerate(param_sets, start=1):
        accuracy_scores = []
        f1_scores = []
        all_predictions = []

        print(f"\n{'=' * 44}")
        print(f"{model_name} Experiment {experiment_number}")
        print(params)
        print("=" * 44)

        with mlflow.start_run(
            run_name=f"{model_name.lower()}_experiment_{experiment_number}"
        ):
            log_pipeline_params(model_type=model_name, n_splits=n_splits)
            mlflow.log_params(params)

            for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]

                model = model_factory(params)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)

                metrics = compute_fold_metrics(y_test, y_pred)
                accuracy_scores.append(metrics["accuracy"])
                f1_scores.append(metrics["macro_f1"])

                log_fold_metrics(fold, metrics)

                print(f"Fold {fold}")
                print(f"  Accuracy : {metrics['accuracy']:.4f}")
                print(f"  Macro F1 : {metrics['macro_f1']:.4f}")
                print("-" * 40)

                fold_preds = build_fold_predictions(
                    merged_df, test_idx, y_test.values, y_pred, y_prob,
                    fold, experiment_number, model_name,
                )
                all_predictions.append(fold_preds)

            predictions_df = pd.concat(all_predictions, ignore_index=True)

            avg_accuracy = np.mean(accuracy_scores)
            avg_f1 = np.mean(f1_scores)

            print(f"\nAverage Accuracy: {avg_accuracy:.4f}")
            print(f"Average Macro F1: {avg_f1:.4f}")

            # Save predictions CSV
            prediction_file = None
            if output_dir:
                prediction_file = os.path.join(
                    output_dir,
                    f"{model_name.lower()}_predictions_experiment_{experiment_number}.csv",
                )
                predictions_df.to_csv(prediction_file, index=False)

            log_experiment_summary(avg_accuracy, avg_f1, prediction_file)

            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_accuracy = avg_accuracy
                best_params = params.copy()
                best_predictions_df = predictions_df.copy()

    print(f"\nBest {model_name} — Accuracy: {best_accuracy:.4f}, F1: {best_f1:.4f}")
    print(f"Best params: {best_params}")

    return best_params, best_f1, best_predictions_df


def train_final_model(model_factory, best_params, X, y, save_path):
    """Train the final model on all data and save as .pkl.

    Args:
        model_factory: Callable(params) → sklearn-compatible classifier.
        best_params: Best hyperparameters from sweep.
        X: Full feature DataFrame.
        y: Full target Series.
        save_path: Path to save the .pkl model file.

    Returns:
        Trained model instance.
    """
    model = model_factory(best_params)
    model.fit(X, y)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"Final model saved to {save_path}")

    return model


def main(model_type=None):
    """End-to-end training orchestrator.

    Args:
        model_type: "xgboost", "lightgbm", or None (both).
    """
    project_root = Path(__file__).resolve().parents[2]

    # Load merged features
    features_path = project_root / "data" / "merged_features.csv"
    print(f"Loading features from {features_path}")
    merged_df = pd.read_csv(features_path)
    merged_df = merged_df.sort_values(["date", "symbol"])

    # Handle infinities
    merged_df = merged_df.replace([np.inf, -np.inf], np.nan)

    # Load feature columns
    feature_columns = joblib.load(project_root / "models" / "feature_columns.pkl")

    X = merged_df[feature_columns]
    y = merged_df["target"]

    output_dir = str(project_root / "data")
    models_dir = project_root / "models"

    run_xgb = model_type in (None, "xgboost")
    run_lgbm = model_type in (None, "lightgbm")

    # XGBoost experiments
    if run_xgb:
        xgb_best_params, _xgb_best_f1, _ = run_experiment(
            model_factory=create_xgboost,
            param_sets=XGBOOST_PARAM_SETS,
            X=X, y=y,
            merged_df=merged_df,
            experiment_name="finbert_xgboost_experiments",
            model_name="XGBoost",
            output_dir=output_dir,
        )
        train_final_model(
            create_xgboost, xgb_best_params, X, y,
            str(models_dir / "xgboost_model.pkl"),
        )

    # LightGBM experiments
    if run_lgbm:
        lgbm_best_params, _lgbm_best_f1, _ = run_experiment(
            model_factory=create_lightgbm,
            param_sets=LGBM_PARAM_SETS,
            X=X, y=y,
            merged_df=merged_df,
            experiment_name="lightgbm_experiments",
            model_name="LightGBM",
            output_dir=output_dir,
        )
        train_final_model(
            create_lightgbm, lgbm_best_params, X, y,
            str(models_dir / "lightgbm_model.pkl"),
        )

    print("\nTraining complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train price direction models")
    parser.add_argument(
        "--model",
        choices=["xgboost", "lightgbm", "both"],
        default="both",
        help="Which model(s) to train",
    )
    args = parser.parse_args()
    main(model_type=None if args.model == "both" else args.model)
