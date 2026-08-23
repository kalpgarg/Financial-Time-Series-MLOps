# Role 2 — ML & Modeling

**Focus:** feature engineering, model training (XGBoost + LightGBM), and
experiment tracking with MLflow.

## What this role does

1. **Feature engineering** (`features/`):
   - `sentiment_features.py` — reduces FinBERT embeddings with **PCA** (768→50,
     saved as `models/finbert_pca.pkl`) and aggregates sentiment over a 7-day
     exponential-decay window → `data/news_features.csv`.
   - `feature_engineering.py` — builds OHLCV technical indicators (returns, lags,
     rolling volatility/MA, volume, the 09:15 features), creates the 3-class
     target, and merges news + price features → `data/merged_features.csv` (the
     95-feature training matrix).
2. **Model training** (`training/`, `models/`):
   - `models/price_predictor.py` — XGBoost / LightGBM factories + parameter sets.
   - `training/train.py` — TimeSeriesSplit cross-validation sweeps for both
     models, logs to **MLflow**, and saves the best `.pkl` artifacts.
   - `training/evaluate.py` — fold metrics (accuracy, macro-F1).
   - `mlflow_utils.py` — MLflow experiment/metric logging helpers.
   - `models/stock_prediction.py` — the original end-to-end reference script
     (kept for reference; not used by the modular pipeline).
3. **Orchestrates** training with Airflow
   (`airflow/dags/training_pipeline_dag.py`, DAG `financial_ts_training_pipeline`):

   ```
   finbert_inference -> sentiment_features -> feature_engineering
      -> [train_xgboost, train_lightgbm] -> dvc_push_models
   ```

## How to run

Run from the **repo root**, with an MLflow server reachable at
`MLFLOW_TRACKING_URI`. Execute the stages in order (each consumes the previous
one's output):

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python -m role1_data_engineering.scrapers.finbert_inference     # FinBERT enrichment
python -m role2_ml_modeling.features.sentiment_features         # -> news_features.csv
python -m role2_ml_modeling.features.feature_engineering        # -> merged_features.csv
python -m role2_ml_modeling.training.train --model both         # trains + logs to MLflow

# Tests
pytest role2_ml_modeling/tests/ -v
```

Install deps with `pip install -r requirements.txt`. On macOS, prefix training
with `KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1` (torch + xgboost/lightgbm
OpenMP). The full flow is also runnable via the training DAG in the whole-system
stack — see the **root README**.

## Dependencies on other roles

- **Consumes** Role 1's outputs: `data/headlines.csv` and
  `data/merged_ohlc_15min.csv`, and calls Role 1's `finbert_inference` as the
  first training step.
- **Produces** the model artifacts in `models/` (`xgboost_model.pkl`,
  `lightgbm_model.pkl`, `finbert_pca.pkl`, `feature_columns.pkl`,
  `symbol_encoder.pkl`), versioned with **DVC** and served by **Role 3**.
- `tests/test_artifacts.py` guards the `.pkl` interface contract (95 features,
  3 classes, 50 symbols, 768→50 PCA) that Role 3 depends on.
