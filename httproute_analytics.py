"""
HTTPRoute Analytics Parser

Batch processing job for Envoy Gateway HTTPRoute access logs.
Runs hourly, reads from ClickHouse, aggregates metrics, and writes to Apache Iceberg tables via Nessie catalog.
"""

import os
import sys
import logging
from datetime import datetime, timedelta

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    avg,
    min as spark_min,
    max as spark_max,
    sum as spark_sum,
    window,
    lit,
    current_timestamp,
)
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    LongType,
    DoubleType,
    TimestampType,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("HTTPRouteAnalytics")

# Environment configuration with defaults
NESSIE_URI = os.getenv(
    "NESSIE_URI", "http://nessie.analytics.svc.cluster.local:19120/api/v2"
)
WAREHOUSE_PATH = os.getenv("WAREHOUSE_PATH", "s3a://iceberg-warehouse")
S3_ENDPOINT = "http://candlekeep.lab.daviestechlabs.io:80"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# ClickHouse configuration
CLICKHOUSE_JDBC_URL = os.getenv(
    "CLICKHOUSE_JDBC_URL",
    "jdbc:clickhouse://clickhouse.analytics.svc.cluster.local:8123/logs",
)
CLICKHOUSE_TABLE = os.getenv("CLICKHOUSE_TABLE", "envoy_access_logs")
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "analytics")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "")

# Output table names
ROUTE_METRICS_TABLE = "nessie.httproute_analytics.route_metrics"
STATUS_METRICS_TABLE = "nessie.httproute_analytics.status_metrics"

# Window duration for aggregation
WINDOW_DURATION = "5 minutes"


def get_input_schema() -> StructType:
    """Define the expected input schema for envoy access logs."""
    return StructType([
        StructField("timestamp", TimestampType(), nullable=False),
        StructField("http_route", StringType(), nullable=True),
        StructField("hostname", StringType(), nullable=True),
        StructField("method", StringType(), nullable=True),
        StructField("path", StringType(), nullable=True),
        StructField("status_code", IntegerType(), nullable=True),
        StructField("response_time_ms", DoubleType(), nullable=True),
        StructField("bytes_sent", LongType(), nullable=True),
        StructField("bytes_received", LongType(), nullable=True),
        StructField("user_agent", StringType(), nullable=True),
        StructField("remote_addr", StringType(), nullable=True),
    ])


def create_sample_data(spark: SparkSession):
    """Create sample data for demonstration when ClickHouse is unavailable."""
    logger.info("Creating sample data for demonstration")
    
    now = datetime.now()
    sample_data = [
        (now - timedelta(minutes=5), "api-gateway", "api.example.com", "GET", "/api/v1/users", 200, 45.5, 1024, 256, "Mozilla/5.0", "10.0.0.1"),
        (now - timedelta(minutes=10), "api-gateway", "api.example.com", "POST", "/api/v1/users", 201, 120.3, 512, 2048, "curl/7.68.0", "10.0.0.2"),
        (now - timedelta(minutes=15), "api-gateway", "api.example.com", "GET", "/api/v1/products", 200, 30.2, 4096, 128, "Mozilla/5.0", "10.0.0.3"),
        (now - timedelta(minutes=20), "web-frontend", "www.example.com", "GET", "/", 200, 15.8, 8192, 64, "Mozilla/5.0", "10.0.0.4"),
        (now - timedelta(minutes=25), "web-frontend", "www.example.com", "GET", "/static/app.js", 304, 5.2, 0, 32, "Mozilla/5.0", "10.0.0.4"),
        (now - timedelta(minutes=30), "api-gateway", "api.example.com", "GET", "/api/v1/users/1", 404, 12.1, 256, 64, "curl/7.68.0", "10.0.0.5"),
        (now - timedelta(minutes=35), "api-gateway", "api.example.com", "POST", "/api/v1/login", 401, 8.5, 128, 512, "Mozilla/5.0", "10.0.0.6"),
        (now - timedelta(minutes=40), "api-gateway", "api.example.com", "GET", "/api/v1/health", 200, 2.1, 64, 32, "kube-probe/1.25", "10.0.0.100"),
        (now - timedelta(minutes=45), "web-frontend", "www.example.com", "POST", "/contact", 500, 250.5, 512, 1024, "Mozilla/5.0", "10.0.0.7"),
        (now - timedelta(minutes=50), "api-gateway", "api.example.com", "DELETE", "/api/v1/users/2", 204, 35.7, 0, 64, "curl/7.68.0", "10.0.0.8"),
    ]
    
    return spark.createDataFrame(sample_data, schema=get_input_schema())


def read_from_clickhouse(spark: SparkSession):
    """
    Read the last 1 hour of data from ClickHouse.
    
    Returns a DataFrame with envoy access logs from the past hour.
    Falls back to sample data if ClickHouse is unavailable.
    """
    # Calculate time window for the last hour
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)
    
    # Format timestamps for ClickHouse query
    start_ts = start_time.strftime("%Y-%m-%d %H:%M:%S")
    end_ts = end_time.strftime("%Y-%m-%d %H:%M:%S")
    
    query = f"""
        SELECT 
            timestamp,
            http_route,
            hostname,
            method,
            path,
            status_code,
            response_time_ms,
            bytes_sent,
            bytes_received,
            user_agent,
            remote_addr
        FROM {CLICKHOUSE_TABLE}
        WHERE timestamp >= '{start_ts}' AND timestamp < '{end_ts}'
    """
    
    logger.info(f"Reading data from ClickHouse for time range: {start_ts} to {end_ts}")
    
    try:
        df = (
            spark.read
            .format("jdbc")
            .option("url", CLICKHOUSE_JDBC_URL)
            .option("query", query)
            .option("user", CLICKHOUSE_USER)
            .option("password", CLICKHOUSE_PASSWORD)
            .option("driver", "com.clickhouse.jdbc.ClickHouseDriver")
            .load()
        )
        
        record_count = df.count()
        logger.info(f"Successfully read {record_count} records from ClickHouse")
        
        if record_count == 0:
            logger.warning("No records found in ClickHouse for the specified time range")
        
        return df
        
    except Exception as e:
        logger.warning(f"Failed to read from ClickHouse: {e}")
        logger.warning("Falling back to sample data for demonstration")
        return create_sample_data(spark)


def compute_route_metrics(df):
    """
    Compute route-level metrics aggregated by 5-minute windows.
    
    Groups by: 5-minute time window, http_route, hostname
    Metrics: request_count, avg/min/max response_time_ms, total_bytes_sent, total_bytes_received
    """
    logger.info("Computing route metrics")
    
    route_metrics = (
        df
        .groupBy(
            window(col("timestamp"), WINDOW_DURATION),
            col("http_route"),
            col("hostname"),
        )
        .agg(
            count("*").alias("request_count"),
            avg("response_time_ms").alias("avg_response_time_ms"),
            spark_min("response_time_ms").alias("min_response_time_ms"),
            spark_max("response_time_ms").alias("max_response_time_ms"),
            spark_sum("bytes_sent").alias("total_bytes_sent"),
            spark_sum("bytes_received").alias("total_bytes_received"),
        )
        .withColumn("window_start", col("window.start"))
        .withColumn("window_end", col("window.end"))
        .drop("window")
        .withColumn("processed_at", current_timestamp())
    )
    
    return route_metrics


def compute_status_metrics(df):
    """
    Compute status code metrics aggregated by 5-minute windows.
    
    Groups by: 5-minute time window, http_route, status_code
    Metrics: count per status code
    """
    logger.info("Computing status metrics")
    
    status_metrics = (
        df
        .groupBy(
            window(col("timestamp"), WINDOW_DURATION),
            col("http_route"),
            col("status_code"),
        )
        .agg(
            count("*").alias("count"),
        )
        .withColumn("window_start", col("window.start"))
        .withColumn("window_end", col("window.end"))
        .drop("window")
        .withColumn("processed_at", current_timestamp())
    )
    
    return status_metrics


def ensure_namespace_exists(spark: SparkSession, namespace: str):
    """Ensure the Iceberg namespace exists, create if it doesn't."""
    try:
        spark.sql(f"CREATE NAMESPACE IF NOT EXISTS {namespace}")
        logger.info(f"Namespace '{namespace}' is ready")
    except Exception as e:
        logger.warning(f"Could not create namespace '{namespace}': {e}")


def write_to_iceberg(df, table_name: str):
    """
    Write DataFrame to Iceberg table using Nessie catalog.
    
    Uses append mode to preserve historical data.
    Falls back to create if the table does not yet exist.
    Configures Iceberg format-version 2 and zstd compression.
    """
    logger.info(f"Writing to Iceberg table: {table_name}")
    
    try:
        (
            df.writeTo(table_name)
            .tableProperty("format-version", "2")
            .tableProperty("write.parquet.compression-codec", "zstd")
            .append()
        )
        logger.info(f"Successfully appended data to {table_name}")
        
    except Exception as e:
        error_msg = str(e).lower()
        if "table_or_view_not_found" in error_msg or (
            "table" in error_msg and "not_found" in error_msg
        ):
            logger.info(f"Table {table_name} does not exist, creating")
            (
                df.writeTo(table_name)
                .tableProperty("format-version", "2")
                .tableProperty("write.parquet.compression-codec", "zstd")
                .create()
            )
            logger.info(f"Successfully created {table_name}")
        else:
            raise


def main():
    """
    Main entry point for the HTTPRoute Analytics job.
    
    Orchestrates the ETL pipeline:
    1. Initialize Spark session
    2. Read data from ClickHouse
    3. Compute aggregations
    4. Write to Iceberg tables
    """
    exit_code = 0
    spark = None
    
    try:
        logger.info("Starting HTTPRoute Analytics Parser")
        logger.info(f"Nessie URI: {NESSIE_URI}")
        logger.info(f"Warehouse Path: {WAREHOUSE_PATH}")
        logger.info(f"S3 Endpoint: {S3_ENDPOINT}")
        
        # Create Spark session - configuration is passed externally via SparkOperator
        spark = (
            SparkSession.builder
            .appName("HTTPRoute Analytics Parser")
            .getOrCreate()
        )
        
        logger.info(f"Spark session created: {spark.sparkContext.applicationId}")
        logger.info(f"Spark version: {spark.version}")
        
        # Ensure namespace exists
        ensure_namespace_exists(spark, "nessie.httproute_analytics")
        
        # Read input data from ClickHouse
        input_df = read_from_clickhouse(spark)
        
        if input_df.isEmpty():
            logger.warning("No data to process, exiting")
            return
        
        # Cache input DataFrame as it's used multiple times
        input_df.cache()
        logger.info(f"Input data cached, total records: {input_df.count()}")
        
        # Compute aggregations
        route_metrics_df = compute_route_metrics(input_df)
        status_metrics_df = compute_status_metrics(input_df)
        
        # Log aggregation results
        route_count = route_metrics_df.count()
        status_count = status_metrics_df.count()
        logger.info(f"Route metrics aggregated: {route_count} records")
        logger.info(f"Status metrics aggregated: {status_count} records")
        
        # Write to Iceberg tables
        write_to_iceberg(route_metrics_df, ROUTE_METRICS_TABLE)
        write_to_iceberg(status_metrics_df, STATUS_METRICS_TABLE)
        
        # Unpersist cached data
        input_df.unpersist()
        
        logger.info("HTTPRoute Analytics Parser completed successfully")
        
    except Exception as e:
        logger.error(f"HTTPRoute Analytics Parser failed: {e}", exc_info=True)
        exit_code = 1
        
    finally:
        if spark is not None:
            spark.stop()
            logger.info("Spark session stopped")
    
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
