# Role 1 – Data Engineering Lead

**Focus:** Ingestion, Orchestration, and Processing

## Architecture

```
 8:30 AM IST                    9:00 AM             9:07 AM            ~9:10 AM
┌──────────────┐          ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Headline    │  CSV     │  Headline    │   │  Price       │   │  Spark       │
│  Scraper     ├─────────►│  Producer    │   │  Producer    │   │  Pipeline    │
│  (crawl4ai)  │          │  (→ Kafka)   │   │  (Upstox→K)  │   │  clean+join  │
└──────────────┘          └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
                                 │                  │                  │
                          ┌──────▼───────┐   ┌──────▼───────┐   ┌──────▼───────┐
                          │ news_features│   │market_features│  │feature_vectors│
                          │  (Kafka)     │   │  (Kafka)     │   │ (PostgreSQL) │
                          └──────┬───────┘   └──────┬───────┘   └──────────────┘
                                 │                  │
                          ┌──────▼───────┐   ┌──────▼───────┐
                          │  Headline    │   │  Price       │
                          │  Consumer    │   │  Consumer    │
                          │  (→ PG)      │   │  (→ PG)      │
                          └──────────────┘   └──────────────┘
```

**Airflow** orchestrates the entire sequence on weekdays (Mon–Fri).

## Components

| Directory | Description |
|-----------|-------------|
| `scrapers/` | crawl4ai headline scraper → timestamped CSV |
| `kafka/producers/` | CSV → Kafka (`news_features`), Upstox API → Kafka (`market_features`) |
| `kafka/consumers/` | Kafka → PostgreSQL staging tables (`raw_headlines`, `raw_prices`) |
| `spark/` | PySpark: clean headlines, clean prices, join → `feature_vectors` |
| `airflow/dags/` | Airflow DAG with time-sequenced tasks + KafkaSensor |
| `db/` | PostgreSQL schema initialisation (`init_db.py`) |
| `tests/` | Unit tests for producers, consumers, and Spark logic |

## Getting Started

### 1. Start Infrastructure (Docker)

```bash
cd role3_mlops_devops/docker
docker-compose up -d zookeeper kafka kafka-init local_postgres
```

> **Local PostgreSQL:** If you already have PostgreSQL running locally
> (e.g. via Homebrew), you can skip the Docker PostgreSQL service.
> The init script will auto-create the `fints_user` role and `fints`
> database for you.

### 2. Initialise Database

```bash
# Run from the project root
python -m role1_data_engineering.db.init_db
```

This command automatically:
1. Connects to the default `postgres` database (tries `fints_user`, your OS user, then `postgres`)
2. Creates the `fints_user` role and `fints` database if they don't exist
3. Creates all pipeline tables (`raw_headlines`, `raw_prices`, `clean_headlines`, `clean_prices`, `feature_vectors`, `execution_signals`)

The script is **idempotent** — safe to re-run at any time.

### 3. Run Components (dry-run, no infra needed)

```bash
# Scrape headlines → stdout
python -m role1_data_engineering.scrapers.headline_scraper --dry-run

# Price producer → stdout (uses sample data)
python -m role1_data_engineering.kafka.producers.price_producer --dry-run

# Headline producer → stdout (requires scraped CSV)
python -m role1_data_engineering.kafka.producers.headline_producer \
    --csv-path data/scraped/headlines_YYYYMMDD_HHMM.csv --dry-run
```

### 4. Run Tests

```bash
pytest role1_data_engineering/tests/ -v
```

## Docker Services

| Service | Internal Host | External (Host) |
|---------|--------------|-----------------|
| Kafka | `kafka:29092` | `localhost:9092` |
| Zookeeper | `zookeeper:2181` | `localhost:2181` |
| PostgreSQL | `local_postgres:5432` | `localhost:5432` |
| Kafka UI | — | `localhost:8085` |
| pgAdmin | — | `localhost:5050` |

**DB credentials:** `fints_user` / `fints_pass` / database `fints`

### Admin UIs

- **Kafka UI** — [http://localhost:8085](http://localhost:8085)
  Browse topics (`news_features`, `market_features`, `execution_signals`), inspect messages, monitor consumer groups and lag.

- **pgAdmin** — [http://localhost:5050](http://localhost:5050)
  Login: `admin@quant.com` / `admin`.
  To connect to the database, add a server with host `local_postgres`, port `5432`, user `fints_user`, password `fints_pass`, database `fints`.

## Deliverable

A robust pipeline that outputs clean, joined feature vectors matching the
agreed-upon schema (`shared/schemas/data_contract.py`) to PostgreSQL,
ready for Role 2 (ML Modeling) to consume.
