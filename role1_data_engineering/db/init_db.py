"""
PostgreSQL schema initialisation for the Financial Time-Series pipeline.

Creates all staging, clean, and output tables if they do not already exist.

Usage:
    python -m role1_data_engineering.db.init_db
"""

import argparse
import logging
import sys
from pathlib import Path

import psycopg2

# ── Resolve project root so shared imports work when running as script ────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import POSTGRES_DB, POSTGRES_HOST, POSTGRES_PASSWORD, POSTGRES_PORT, POSTGRES_USER

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("init_db")

# ── DDL Statements ───────────────────────────────────────────────────────────

TABLES_SQL = """
-- Staging tables (written by Kafka consumers)
CREATE TABLE IF NOT EXISTS raw_headlines (
    id              SERIAL PRIMARY KEY,
    headline_id     VARCHAR(64) UNIQUE NOT NULL,
    symbol          VARCHAR(128) NOT NULL,
    published_at    VARCHAR(128),
    source          VARCHAR(256),
    headline        TEXT NOT NULL,
    article_url     TEXT,
    scraped_at      TIMESTAMPTZ DEFAULT NOW(),
    inserted_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS raw_prices (
    id              SERIAL PRIMARY KEY,
    symbol          VARCHAR(128) NOT NULL,
    date            DATE NOT NULL,
    open            DOUBLE PRECISION,
    high            DOUBLE PRECISION,
    low             DOUBLE PRECISION,
    close           DOUBLE PRECISION,
    volume          BIGINT,
    adjusted_close  DOUBLE PRECISION,
    inserted_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (symbol, date)
);

-- Clean tables (written by Spark jobs)
CREATE TABLE IF NOT EXISTS clean_headlines (
    id              SERIAL PRIMARY KEY,
    headline_id     VARCHAR(64) UNIQUE NOT NULL,
    symbol          VARCHAR(128) NOT NULL,
    published_at    VARCHAR(128),
    source          VARCHAR(256),
    headline        TEXT NOT NULL,
    article_url     TEXT,
    scraped_at      TIMESTAMPTZ,
    cleaned_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS clean_prices (
    id              SERIAL PRIMARY KEY,
    symbol          VARCHAR(128) NOT NULL,
    date            DATE NOT NULL,
    open            DOUBLE PRECISION,
    high            DOUBLE PRECISION,
    low             DOUBLE PRECISION,
    close           DOUBLE PRECISION,
    volume          BIGINT,
    adjusted_close  DOUBLE PRECISION,
    cleaned_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (symbol, date)
);

-- Joined feature vectors (written by Spark join job)
CREATE TABLE IF NOT EXISTS feature_vectors (
    id              SERIAL PRIMARY KEY,
    symbol          VARCHAR(128) NOT NULL,
    date            DATE NOT NULL,
    open            DOUBLE PRECISION,
    high            DOUBLE PRECISION,
    low             DOUBLE PRECISION,
    close           DOUBLE PRECISION,
    volume          BIGINT,
    adjusted_close  DOUBLE PRECISION,
    headline_count  INTEGER DEFAULT 0,
    headlines_json  TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (symbol, date)
);

-- Execution signals (output for Role 2 / broker API)
CREATE TABLE IF NOT EXISTS execution_signals (
    id              SERIAL PRIMARY KEY,
    symbol          VARCHAR(128) NOT NULL,
    date            DATE NOT NULL,
    signal          VARCHAR(16) NOT NULL,
    confidence      DOUBLE PRECISION,
    model_version   VARCHAR(256),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
"""


def _get_admin_connection():
    """Connect to the default 'postgres' database to bootstrap role/db.

    Tries the configured user first; if that role doesn't exist, falls
    back to the current OS user (typical for local macOS/Homebrew installs)
    and finally tries 'postgres'.
    """
    import getpass

    candidates = [POSTGRES_USER, getpass.getuser(), "postgres"]
    for user in candidates:
        try:
            conn = psycopg2.connect(
                host=POSTGRES_HOST,
                port=POSTGRES_PORT,
                dbname="postgres",
                user=user,
            )
            conn.autocommit = True
            logger.info("Admin connection established as user '%s'.", user)
            return conn
        except psycopg2.OperationalError:
            continue
    raise RuntimeError(
        "Cannot connect to PostgreSQL as any candidate user. "
        "Ensure PostgreSQL is running and accessible on "
        f"{POSTGRES_HOST}:{POSTGRES_PORT}."
    )


def _bootstrap_role_and_db():
    """Create the application role and database if they do not exist."""
    conn = _get_admin_connection()
    try:
        with conn.cursor() as cur:
            # Create role if missing
            cur.execute(
                "SELECT 1 FROM pg_roles WHERE rolname = %s;", (POSTGRES_USER,)
            )
            if cur.fetchone() is None:
                cur.execute(
                    f"CREATE ROLE {POSTGRES_USER} WITH LOGIN PASSWORD %s;",
                    (POSTGRES_PASSWORD,),
                )
                logger.info("Created PostgreSQL role '%s'.", POSTGRES_USER)
            else:
                logger.info("Role '%s' already exists.", POSTGRES_USER)

            # Create database if missing
            cur.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s;", (POSTGRES_DB,)
            )
            if cur.fetchone() is None:
                cur.execute(
                    f"CREATE DATABASE {POSTGRES_DB} OWNER {POSTGRES_USER};"
                )
                logger.info("Created database '%s'.", POSTGRES_DB)
            else:
                logger.info("Database '%s' already exists.", POSTGRES_DB)

            # Ensure the role owns the database
            cur.execute(
                f"GRANT ALL PRIVILEGES ON DATABASE {POSTGRES_DB} TO {POSTGRES_USER};"
            )
    finally:
        conn.close()


def get_connection():
    """Return a psycopg2 connection to the configured PostgreSQL database."""
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


def init_tables():
    """Bootstrap role/db if needed, then create all pipeline tables."""
    _bootstrap_role_and_db()

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(TABLES_SQL)
        conn.commit()
        logger.info("All tables created / verified successfully.")
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialise PostgreSQL schema")
    parser.add_argument("--dry-run", action="store_true", help="Print SQL without executing")
    args = parser.parse_args()

    if args.dry_run:
        print(TABLES_SQL)
    else:
        init_tables()
