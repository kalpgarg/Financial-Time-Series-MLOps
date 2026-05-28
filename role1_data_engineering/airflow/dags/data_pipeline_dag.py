"""
Airflow DAG: Orchestrates the full data pipeline.

Schedule:
  - Trigger Kafka producers (headlines + prices)
  - Wait for data landing
  - Run Spark clean_headlines → clean_prices → join_data
  - Push final dataset to DVC remote

Runs daily before market open.
"""

# TODO: Define DAG with BashOperator / SparkSubmitOperator tasks
