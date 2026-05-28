# Financial-Time-Series-MLOps

Predict next-day market-open price direction (**high / low / flat**) from historical prices and real-time news sentiment.

---

## Project Structure

```
Financial-Time-Series-MLOps/
│
├── shared/                          # Shared contracts & config (ALL roles)
│   ├── config.py                    #   Central env-based configuration
│   └── schemas/
│       └── data_contract.py         #   PriceRecord, HeadlineRecord, PredictionRequest/Response
│
├── data/
│   └── day1_sample/                 # Day-1 fake CSVs (Role 2 starts here)
│       ├── sample_prices.csv
│       └── sample_headlines.csv
│
├── models/
│   ├── create_dummy_model.py        # Generates dummy model.pkl (Role 3 starts here)
│   └── model.pkl                    # Model artifact (gitignored when real)
│
├── role1_data_engineering/          # ── ROLE 1: Data Engineering Lead ──
│   ├── kafka/
│   │   ├── producers/               #   headline_producer.py, price_producer.py
│   │   └── consumers/               #   headline_consumer.py, price_consumer.py
│   ├── spark/                       #   clean_headlines.py, clean_prices.py, join_data.py
│   ├── airflow/
│   │   ├── dags/                    #   data_pipeline_dag.py
│   │   └── plugins/
│   ├── tests/
│   ├── requirements.txt
│   └── README.md
│
├── role2_ml_modeling/               # ── ROLE 2: ML & Modeling Lead ──
│   ├── features/                    #   feature_engineering.py
│   ├── models/                      #   sentiment_model.py (PyTorch), price_predictor.py (sklearn)
│   ├── training/                    #   train.py, evaluate.py
│   ├── dvc/                         #   dvc.yaml, DVC pipeline config
│   ├── mlflow_utils.py              #   MLflow helper functions
│   ├── notebooks/                   #   EDA & prototyping
│   ├── tests/
│   ├── requirements.txt
│   └── README.md
│
├── role3_mlops_devops/              # ── ROLE 3: MLOps & DevOps Lead ──
│   ├── api/                         #   app.py (FastAPI), model_loader.py
│   ├── docker/                      #   Dockerfile, docker-compose.yml
│   ├── ci_cd/
│   │   └── github_actions/          #   ci.yml (GitHub Actions workflow)
│   ├── monitoring/
│   │   ├── prometheus/              #   prometheus.yml, alert_rules.yml
│   │   ├── kibana/dashboards/       #   Kibana saved objects
│   │   └── drift_detector.py        #   Data drift monitoring
│   ├── tests/
│   ├── requirements.txt
│   └── README.md
│
└── .github/
    └── workflows/
        └── ci.yml                   # Symlink / copy of ci_cd workflow
```

---

## Roles & Responsibilities

### Role 1 – Data Engineering Lead
**Focus:** Ingestion, Orchestration, Processing

- **Kafka Streamer** – Producers & consumers for live headlines + prices
- **Spark Processor** – PySpark jobs to clean text, handle nulls, join data
- **Airflow Orchestrator** – DAG scheduling the full pipeline
- **Deliverable:** Clean data matching `shared/schemas/data_contract.py` → DB / DVC

### Role 2 – ML & Modeling Lead
**Focus:** Feature Engineering, Model Training, Experiment Tracking

- **Model Development** – PyTorch (sentiment) + Scikit-learn (price direction)
- **MLflow Integration** – Log params, metrics (accuracy, F1), artifacts
- **DVC** – Version training datasets as Role 1 updates them
- **Deliverable:** Training script → DVC pull → train → MLflow log → register best model

### Role 3 – MLOps & DevOps Lead
**Focus:** API Serving, Containerization, CI/CD, Monitoring

- **FastAPI & Docker** – `/predict` endpoint, containerized
- **CI/CD** – GitHub Actions: test → build → deploy
- **Monitoring** – Prometheus (latency, errors), Kibana (logs), drift detection
- **Deliverable:** Deployment infra — swap `model.pkl` and the pipeline auto-deploys

---

## Quick Start

```bash
# 1. Create the dummy model for Role 3
python models/create_dummy_model.py

# 2. Install role-specific dependencies
pip install -r role1_data_engineering/requirements.txt
pip install -r role2_ml_modeling/requirements.txt
pip install -r role3_mlops_devops/requirements.txt

# 3. Run tests
pytest
```
