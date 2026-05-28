# Role 1 – Data Engineering Lead

**Focus:** Ingestion, Orchestration, and Processing

## Responsibilities

| Component | Description |
|-----------|-------------|
| `kafka/producers/` | Kafka producers fetching live headlines (RSS) and stock prices (API) |
| `kafka/consumers/` | Kafka consumers landing raw data to staging |
| `spark/` | PySpark jobs to clean text, handle missing values, and join datasets |
| `airflow/dags/` | Airflow DAG that schedules the full pipeline |

## Getting Started

```bash
pip install -r requirements.txt
```

## Deliverable

A robust pipeline that outputs clean data matching the agreed-upon schema
(`shared/schemas/data_contract.py`) to the database or DVC storage.
