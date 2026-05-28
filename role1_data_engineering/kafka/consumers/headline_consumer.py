"""
Kafka consumer: reads raw headlines from KAFKA_TOPIC_HEADLINES and writes
them to a staging area (DB table or raw parquet files) for Spark to process.
"""

# TODO: Implement KafkaConsumer that deserialises HeadlineRecord-shaped JSON
