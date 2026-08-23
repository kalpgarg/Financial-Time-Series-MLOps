# Whole-Project Deployment

One `docker compose up` brings up the entire stack from a single multi-stage
[Dockerfile](Dockerfile) + [docker-compose.yml](docker-compose.yml). Postgres
and Prometheus use official images; everything else is built from the one
Dockerfile via `target:`.

## Services & UIs

| Service | Image / build target | URL | Notes |
|---|---|---|---|
| **Airflow** (webserver + scheduler) | `airflow` | http://localhost:8080 | login **admin / admin** |
| **MLflow** tracking + UI | `mlflow` | http://localhost:5000 | Postgres backend + artifact volume |
| role3 **API** | `api` | http://localhost:8000/docs | Deployment + `/metrics` |
| **Prometheus** | official | http://localhost:9090 | scrapes API `/metrics` |
| **Postgres** | official | localhost:5432 | DBs: `predictions`, `airflow`, `mlflow` |

## Run it

```bash
cd github-project
docker compose up --build          # first build ~5–8 min (CPU torch + FinBERT + Airflow/Spark)
# ... then open the UIs above ...
docker compose down                # stop   (add -v to wipe DB + MLflow artifacts)
```

> **Run one stack at a time.** This reuses ports (5432/8000/9090) with the
> role3-only compose in `role3_mlops_devops/`.

### Prerequisites
- **Docker Desktop** running.
- **`models/` populated** (via `dvc pull`) — the API mounts it read-only and
  returns 503 on `/predict` until the artifacts are present.

## MLflow

The server uses the shared Postgres (`mlflow` DB) as the backend store and a
named volume (`mlflow-artifacts`) for artifacts, with `--serve-artifacts` so
clients upload through the server. Both survive `docker compose down` (unless
you pass `-v`).

**Importing earlier training runs** (from teammates): with the stack up, point
the MLflow client at this server and import — e.g. copy their `mlruns/` and use
`mlflow` import tooling, or re-log. Because the backend is Postgres + a volume,
imported runs persist and show in the UI for comparing metrics across
retrainings.

Future retraining logs here automatically: the Airflow tasks run with
`MLFLOW_TRACKING_URI=http://mlflow:5000`.

## Airflow

DAGs are loaded from the repo (mounted at `/opt/project`, which is
`DAGS_FOLDER`). [.airflowignore](.airflowignore) excludes non-DAG modules so
parsing doesn't import heavy libs. Two DAGs appear in the UI:

- `financial_ts_inference_pipeline` (role1) — scrape → Spark clean/merge/trim → docker inference → dvc push
- `financial_ts_training_pipeline` (role2) — FinBERT → features → train XGBoost/LightGBM → dvc push

`SKIP_SCRAPERS=1` is set so runs use existing data instead of live scraping.
Trigger a DAG from the UI and watch task status/logs — that's the run history
and results view you wanted.

## Wiring full DAG execution (deferred — scope was "UIs up + DAGs loaded")

The stack starts and DAGs are triggerable now. To make **every task run
end-to-end** inside containers, three things remain:

1. **Add the heavy per-task deps to the `airflow` stage** of the Dockerfile
   (kept out for now to keep the image buildable and conflict-free):
   - `finbert_inference` → `torch`, `transformers`
   - training → `scikit-learn`, `xgboost`, `lightgbm`, `mlflow` (client)
   - live scraping → `crawl4ai` (+ Playwright browsers)
2. **`run_inference` volume paths (Docker-out-of-Docker).** The task does
   `docker run -v /opt/project/...` through the mounted host socket, so those
   bind paths resolve on the **host**, not inside the Airflow container. Pass
   the host repo path (e.g. via an env var the DAG reads) or switch the task to
   call `batch_score.py` in-process. Until then the inference task's mounts
   won't line up.
3. **Raw scraper inputs.** With `SKIP_SCRAPERS=1`, the Spark tasks expect the
   raw scraper outputs (`data/stock_news/…`, `data/ohlc_data/…`,
   `data/preopen_csv/…`) to already exist. Provide them or run the scrapers.

## Architecture note

One Dockerfile, three build targets (`api` / `airflow` / `mlflow`), each an
isolated image — so Airflow's tight dependency pins never collide with the
API's `torch`/`pandas==3.0.5`. The `api` target mirrors
`role3_mlops_devops/Dockerfile`; keep them in sync.
