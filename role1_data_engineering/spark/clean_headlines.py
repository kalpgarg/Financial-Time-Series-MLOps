"""
PySpark job: reads raw headlines from staging, cleans text
(lowercase, remove HTML, deduplicate), and writes clean parquet
conforming to HeadlineRecord schema.
"""

# TODO: SparkSession init, read staging, text cleaning UDFs, write clean output
