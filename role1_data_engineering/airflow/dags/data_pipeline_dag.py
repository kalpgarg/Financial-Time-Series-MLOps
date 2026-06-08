"""
Airflow DAG: Orchestrates the full data pipeline on Indian market days.

Time sequence (IST, UTC+05:30):
  08:30 — Scrape headlines from Groww → CSV              (~30 min)
  09:00 — Push scraped CSV to Kafka news_features topic
  09:07 — Fetch pre-market prices → Kafka market_features topic
  09:07 — KafkaSensor waits for market_features data
  09:08 — Spark: clean_headlines, clean_prices, join_data
  09:15 — Feature vectors ready in PostgreSQL for Role 2

Schedule: 0 3 * * 1-5  (03:00 UTC = 08:30 IST, weekdays only)
"""

import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.base import BaseSensorOperator
from airflow.utils.decorators import apply_defaults

# ── Project root (assumes Airflow has access to the repo) ────────────────────
PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PYTHON_BIN = sys.executable  # use the same Python interpreter as Airflow

# ── Default args ─────────────────────────────────────────────────────────────
default_args = {
    "owner": "data-engineering",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

# ── DAG definition ───────────────────────────────────────────────────────────
with DAG(
    dag_id="financial_ts_data_pipeline",
    default_args=default_args,
    description="End-to-end data pipeline: scrape → Kafka → Spark → PostgreSQL",
    schedule_interval="0 3 * * 1-5",  # 03:00 UTC = 08:30 IST, Mon–Fri
    start_date=datetime(2026, 5, 30),
    catchup=False,
    tags=["data-engineering", "role1"],
) as dag:

    # ── Helper: run a module as a subprocess ─────────────────────────────────
    def _run_module(module: str, extra_args: list[str] | None = None):
        """Run a Python module via subprocess in the project root."""
        cmd = [PYTHON_BIN, "-m", module]
        if extra_args:
            cmd.extend(extra_args)
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            raise RuntimeError(f"{module} failed with exit code {result.returncode}")

    # ── Task 1: Scrape headlines → CSV (8:30 AM IST) ────────────────────────
    def scrape_headlines(**context):
        ts = context["execution_date"].strftime("%Y%m%d_%H%M")
        output_path = os.path.join(PROJECT_ROOT, "data", "stock_news", f"headlines_{ts}.csv")
        _run_module(
            "role1_data_engineering.scrapers.headline_scraper",
            extra_args=["--output", output_path],
        )
        # Push the output path to XCom for the next task
        context["ti"].xcom_push(key="headlines_csv_path", value=output_path)

    task_scrape = PythonOperator(
        task_id="scrape_headlines",
        python_callable=scrape_headlines,
        execution_timeout=timedelta(minutes=35),
    )

    # ── Task 2: Headline CSV → Kafka news_features (9:00 AM IST) ────────────
    def produce_headlines(**context):
        csv_path = context["ti"].xcom_pull(
            task_ids="scrape_headlines", key="headlines_csv_path"
        )
        _run_module(
            "role1_data_engineering.kafka.producers.headline_producer",
            extra_args=["--csv-path", csv_path],
        )

    task_produce_headlines = PythonOperator(
        task_id="produce_headlines",
        python_callable=produce_headlines,
    )

    # ── Task 3: Fetch prices → Kafka market_features (9:07 AM IST) ──────────
    def produce_prices(**context):
        _run_module("role1_data_engineering.kafka.producers.price_producer")

    task_produce_prices = PythonOperator(
        task_id="produce_prices",
        python_callable=produce_prices,
    )

    # ── Task 4: Kafka sensor — wait for market_features data ─────────────────
    class KafkaTopicSensor(BaseSensorOperator):
        """Simple sensor that checks if a Kafka topic has new messages.

        Uses kafka-python to peek at topic offsets. Pokes every 30 seconds.
        """

        @apply_defaults
        def __init__(self, topic: str, bootstrap_servers: str, **kwargs):
            super().__init__(**kwargs)
            self.topic = topic
            self.bootstrap_servers = bootstrap_servers

        def poke(self, context):
            try:
                from kafka import KafkaConsumer

                consumer = KafkaConsumer(
                    self.topic,
                    bootstrap_servers=self.bootstrap_servers,
                    auto_offset_reset="latest",
                    consumer_timeout_ms=5_000,
                    group_id=None,  # do not commit offsets
                )
                # Check if any messages are available
                partitions = consumer.partitions_for_topic(self.topic)
                if partitions:
                    consumer.close()
                    return True
                consumer.close()
            except Exception as e:
                self.log.warning("KafkaTopicSensor poke error: %s", e)
            return False

    from shared.config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC_MARKET_FEATURES

    task_wait_prices = KafkaTopicSensor(
        task_id="wait_for_market_data",
        topic=KAFKA_TOPIC_MARKET_FEATURES,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        poke_interval=30,
        timeout=600,  # 10 minutes max wait
        mode="poke",
    )

    # ── Task 5: Consume both topics → PostgreSQL ────────────────────────────
    def consume_headlines(**context):
        _run_module(
            "role1_data_engineering.kafka.consumers.headline_consumer",
            extra_args=["--max-messages", "10000"],
        )

    task_consume_headlines = PythonOperator(
        task_id="consume_headlines",
        python_callable=consume_headlines,
    )

    def consume_prices(**context):
        _run_module(
            "role1_data_engineering.kafka.consumers.price_consumer",
            extra_args=["--max-messages", "10000"],
        )

    task_consume_prices = PythonOperator(
        task_id="consume_prices",
        python_callable=consume_prices,
    )

    # ── Task 6: Spark — clean headlines ──────────────────────────────────────
    def spark_clean_headlines(**context):
        _run_module("role1_data_engineering.spark.clean_headlines")

    task_clean_headlines = PythonOperator(
        task_id="spark_clean_headlines",
        python_callable=spark_clean_headlines,
        execution_timeout=timedelta(minutes=10),
    )

    # ── Task 7: Spark — clean prices ─────────────────────────────────────────
    def spark_clean_prices(**context):
        _run_module("role1_data_engineering.spark.clean_prices")

    task_clean_prices = PythonOperator(
        task_id="spark_clean_prices",
        python_callable=spark_clean_prices,
        execution_timeout=timedelta(minutes=10),
    )

    # ── Task 8: Spark — join data → feature_vectors ──────────────────────────
    def spark_join_data(**context):
        _run_module("role1_data_engineering.spark.join_data")

    task_join = PythonOperator(
        task_id="spark_join_data",
        python_callable=spark_join_data,
        execution_timeout=timedelta(minutes=10),
    )

    # ── DAG dependency graph ─────────────────────────────────────────────────
    # 8:30 scrape → 9:00 produce headlines → consume headlines
    task_scrape >> task_produce_headlines >> task_consume_headlines

    # 9:07 produce prices → sensor → consume prices
    task_produce_prices >> task_wait_prices >> task_consume_prices

    # Both consumers complete → Spark pipeline
    [task_consume_headlines, task_consume_prices] >> task_clean_headlines
    [task_consume_headlines, task_consume_prices] >> task_clean_prices
    [task_clean_headlines, task_clean_prices] >> task_join
