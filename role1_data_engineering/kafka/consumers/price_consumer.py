"""
Kafka consumer: reads raw price data from KAFKA_TOPIC_PRICES and writes
them to a staging area (DB table or raw parquet files) for Spark to process.
"""

# TODO: Implement KafkaConsumer that deserialises PriceRecord-shaped JSON
