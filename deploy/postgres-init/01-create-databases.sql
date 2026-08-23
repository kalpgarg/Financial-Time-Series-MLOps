-- Runs once on first Postgres startup (empty data dir).
-- The default database `predictions` (role3 API) is created from
-- POSTGRES_DB; here we add the Airflow metadata DB and the MLflow backend DB,
-- both owned by the same `mlops` user.
CREATE DATABASE airflow OWNER mlops;
CREATE DATABASE mlflow OWNER mlops;
