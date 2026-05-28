# Role 2 – ML & Modeling Lead

**Focus:** Feature Engineering, Model Training, and Experiment Tracking

## Responsibilities

| Component | Description |
|-----------|-------------|
| `features/` | Feature engineering pipeline (TF-IDF, technical indicators, etc.) |
| `models/` | PyTorch sentiment model + Scikit-learn price direction predictor |
| `training/` | End-to-end train script with MLflow logging and model registration |
| `dvc/` | DVC pipeline config for data versioning |
| `notebooks/` | Exploratory data analysis and prototyping |

## Getting Started

```bash
pip install -r requirements.txt
# Start with the Day-1 fake CSV in data/day1_sample/
python -m role2_ml_modeling.training.train
```

## Deliverable

A robust Python script that:
1. Pulls data via DVC
2. Trains sentiment + price models
3. Logs everything to MLflow
4. Registers the best production model
