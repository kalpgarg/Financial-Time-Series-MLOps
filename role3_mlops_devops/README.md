# Role 3 – MLOps & DevOps Lead

**Focus:** API Serving, Containerization, CI/CD, and Monitoring

## Responsibilities

| Component | Description |
|-----------|-------------|
| `api/` | FastAPI app wrapping the model in a `/predict` endpoint |
| `docker/` | Dockerfile + docker-compose (API, Prometheus, Kibana, Elasticsearch) |
| `ci_cd/` | GitHub Actions workflow for test → build → deploy |
| `monitoring/` | Prometheus config, alert rules, Kibana dashboards, drift detection |

## Getting Started

```bash
pip install -r requirements.txt
# Start with the Day-1 dummy model.pkl
uvicorn role3_mlops_devops.api.app:app --reload
```

## Deliverable

The deployment infrastructure. When Role 2 finishes the real PyTorch model,
simply swap `models/model.pkl` for the real one and the pipeline
automatically deploys it.
