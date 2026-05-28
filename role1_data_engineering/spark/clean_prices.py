"""
PySpark job: reads raw price data from staging, handles missing values
(forward-fill, interpolation), and writes clean parquet conforming to
PriceRecord schema.
"""

# TODO: SparkSession init, read staging, handle nulls/gaps, write clean output
