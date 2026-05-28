"""
Main training script.
  1. Pulls data via DVC (dvc pull)
  2. Runs feature engineering
  3. Trains sentiment model (PyTorch) + price predictor (sklearn)
  4. Logs hyperparameters, metrics (accuracy, F1), and artifacts to MLflow
  5. Registers the best model to the MLflow Model Registry

Usage:
    python -m role2_ml_modeling.training.train
"""

# TODO: Implement end-to-end training pipeline
# Use shared.config for MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
