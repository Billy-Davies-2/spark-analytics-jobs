"""
Unit tests for HTTPRoute Analytics Parser.

Run with: pytest test_httproute_analytics.py -v
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    LongType,
    DoubleType,
    TimestampType,
)

import httproute_analytics as ha


@pytest.fixture(scope="module")
def spark():
    """Create a SparkSession for testing."""
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("HTTPRouteAnalytics-Test")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.default.parallelism", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield spark
    spark.stop()


@pytest.fixture
def sample_access_logs(spark):
    """Create sample access log data for testing."""
    now = datetime.now()
    
    data = [
        # Window 1: First 5-minute window
        (now - timedelta(minutes=2), "api-gateway", "api.example.com", "GET", "/api/v1/users", 200, 45.5, 1024, 256, "Mozilla/5.0", "10.0.0.1"),
        (now - timedelta(minutes=3), "api-gateway", "api.example.com", "GET", "/api/v1/users/1", 200, 55.0, 512, 128, "Mozilla/5.0", "10.0.0.2"),
        (now - timedelta(minutes=4), "api-gateway", "api.example.com", "POST", "/api/v1/users", 201, 120.3, 256, 2048, "curl/7.68.0", "10.0.0.3"),
        (now - timedelta(minutes=4), "api-gateway", "api.example.com", "GET", "/api/v1/users", 500, 250.0, 128, 64, "Mozilla/5.0", "10.0.0.4"),
        
        # Window 2: Second 5-minute window
        (now - timedelta(minutes=7), "api-gateway", "api.example.com", "GET", "/api/v1/products", 200, 30.2, 4096, 128, "Mozilla/5.0", "10.0.0.5"),
        (now - timedelta(minutes=8), "api-gateway", "api.example.com", "GET", "/api/v1/products/1", 404, 12.1, 256, 64, "curl/7.68.0", "10.0.0.6"),
        
        # Different route and hostname
        (now - timedelta(minutes=3), "web-frontend", "www.example.com", "GET", "/", 200, 15.8, 8192, 64, "Mozilla/5.0", "10.0.0.7"),
        (now - timedelta(minutes=4), "web-frontend", "www.example.com", "GET", "/static/app.js", 304, 5.2, 0, 32, "Mozilla/5.0", "10.0.0.8"),
    ]
    
    return spark.createDataFrame(data, schema=ha.get_input_schema())


class TestInputSchema:
    """Tests for input schema definition."""
    
    def test_schema_has_correct_fields(self):
        """Verify the schema has all required fields."""
        schema = ha.get_input_schema()
        field_names = [f.name for f in schema.fields]
        
        expected_fields = [
            "timestamp",
            "http_route",
            "hostname",
            "method",
            "path",
            "status_code",
            "response_time_ms",
            "bytes_sent",
            "bytes_received",
            "user_agent",
            "remote_addr",
        ]
        
        assert field_names == expected_fields
    
    def test_schema_field_types(self):
        """Verify the schema has correct field types."""
        schema = ha.get_input_schema()
        fields = {f.name: f.dataType for f in schema.fields}
        
        assert isinstance(fields["timestamp"], TimestampType)
        assert isinstance(fields["http_route"], StringType)
        assert isinstance(fields["hostname"], StringType)
        assert isinstance(fields["method"], StringType)
        assert isinstance(fields["path"], StringType)
        assert isinstance(fields["status_code"], IntegerType)
        assert isinstance(fields["response_time_ms"], DoubleType)
        assert isinstance(fields["bytes_sent"], LongType)
        assert isinstance(fields["bytes_received"], LongType)
        assert isinstance(fields["user_agent"], StringType)
        assert isinstance(fields["remote_addr"], StringType)


class TestSampleData:
    """Tests for sample data generation."""
    
    def test_create_sample_data_returns_dataframe(self, spark):
        """Verify sample data returns a valid DataFrame."""
        df = ha.create_sample_data(spark)
        assert df is not None
        assert df.count() > 0
    
    def test_create_sample_data_has_correct_schema(self, spark):
        """Verify sample data matches expected schema."""
        df = ha.create_sample_data(spark)
        expected_schema = ha.get_input_schema()
        
        assert df.schema == expected_schema
    
    def test_create_sample_data_has_valid_values(self, spark):
        """Verify sample data contains valid values."""
        df = ha.create_sample_data(spark)
        
        # Check that status codes are valid HTTP codes
        status_codes = [row.status_code for row in df.select("status_code").distinct().collect()]
        for code in status_codes:
            assert 100 <= code <= 599
        
        # Check that response times are positive
        min_response_time = df.agg({"response_time_ms": "min"}).collect()[0][0]
        assert min_response_time >= 0
        
        # Check that bytes are non-negative
        min_bytes_sent = df.agg({"bytes_sent": "min"}).collect()[0][0]
        min_bytes_received = df.agg({"bytes_received": "min"}).collect()[0][0]
        assert min_bytes_sent >= 0
        assert min_bytes_received >= 0


class TestRouteMetrics:
    """Tests for route metrics aggregation."""
    
    def test_compute_route_metrics_groups_correctly(self, spark, sample_access_logs):
        """Verify route metrics groups by window, http_route, and hostname."""
        result = ha.compute_route_metrics(sample_access_logs)
        
        # Should have window_start, window_end, http_route, hostname columns
        column_names = result.columns
        assert "window_start" in column_names
        assert "window_end" in column_names
        assert "http_route" in column_names
        assert "hostname" in column_names
    
    def test_compute_route_metrics_has_all_metrics(self, spark, sample_access_logs):
        """Verify route metrics contains all required metric columns."""
        result = ha.compute_route_metrics(sample_access_logs)
        
        required_metrics = [
            "request_count",
            "avg_response_time_ms",
            "min_response_time_ms",
            "max_response_time_ms",
            "total_bytes_sent",
            "total_bytes_received",
            "processed_at",
        ]
        
        for metric in required_metrics:
            assert metric in result.columns, f"Missing metric: {metric}"
    
    def test_compute_route_metrics_request_count(self, spark, sample_access_logs):
        """Verify request count is calculated correctly."""
        result = ha.compute_route_metrics(sample_access_logs)
        
        # Total requests should equal input record count
        total_requests = result.agg({"request_count": "sum"}).collect()[0][0]
        input_count = sample_access_logs.count()
        
        assert total_requests == input_count
    
    def test_compute_route_metrics_response_time_stats(self, spark, sample_access_logs):
        """Verify response time statistics are calculated correctly."""
        result = ha.compute_route_metrics(sample_access_logs)
        
        # For each row, min <= avg <= max
        for row in result.collect():
            assert row.min_response_time_ms <= row.avg_response_time_ms
            assert row.avg_response_time_ms <= row.max_response_time_ms
    
    def test_compute_route_metrics_bytes_totals(self, spark, sample_access_logs):
        """Verify bytes totals are calculated correctly."""
        result = ha.compute_route_metrics(sample_access_logs)
        
        # Sum of bytes in result should equal sum in input
        result_bytes_sent = result.agg({"total_bytes_sent": "sum"}).collect()[0][0]
        input_bytes_sent = sample_access_logs.agg({"bytes_sent": "sum"}).collect()[0][0]
        
        assert result_bytes_sent == input_bytes_sent
        
        result_bytes_received = result.agg({"total_bytes_received": "sum"}).collect()[0][0]
        input_bytes_received = sample_access_logs.agg({"bytes_received": "sum"}).collect()[0][0]
        
        assert result_bytes_received == input_bytes_received


class TestStatusMetrics:
    """Tests for status code metrics aggregation."""
    
    def test_compute_status_metrics_groups_correctly(self, spark, sample_access_logs):
        """Verify status metrics groups by window, http_route, and status_code."""
        result = ha.compute_status_metrics(sample_access_logs)
        
        column_names = result.columns
        assert "window_start" in column_names
        assert "window_end" in column_names
        assert "http_route" in column_names
        assert "status_code" in column_names
    
    def test_compute_status_metrics_has_count(self, spark, sample_access_logs):
        """Verify status metrics contains count and processed_at columns."""
        result = ha.compute_status_metrics(sample_access_logs)
        
        assert "count" in result.columns
        assert "processed_at" in result.columns
    
    def test_compute_status_metrics_count_sum(self, spark, sample_access_logs):
        """Verify total count equals input record count."""
        result = ha.compute_status_metrics(sample_access_logs)
        
        total_count = result.agg({"count": "sum"}).collect()[0][0]
        input_count = sample_access_logs.count()
        
        assert total_count == input_count
    
    def test_compute_status_metrics_distinct_status_codes(self, spark, sample_access_logs):
        """Verify all distinct status codes from input appear in output."""
        result = ha.compute_status_metrics(sample_access_logs)
        
        input_status_codes = set(
            row.status_code for row in sample_access_logs.select("status_code").distinct().collect()
        )
        output_status_codes = set(
            row.status_code for row in result.select("status_code").distinct().collect()
        )
        
        assert input_status_codes == output_status_codes


class TestClickHouseIntegration:
    """Tests for ClickHouse integration (mocked)."""
    
    def test_read_from_clickhouse_fallback_on_error(self, spark):
        """Verify fallback to sample data when ClickHouse is unavailable."""
        with patch.object(spark.read, 'format') as mock_format:
            # Simulate ClickHouse connection failure
            mock_format.return_value.option.return_value.option.return_value.option.return_value.option.return_value.option.return_value.load.side_effect = Exception("Connection refused")
            
            # Should fall back to sample data without raising exception
            result = ha.read_from_clickhouse(spark)
            
            assert result is not None
            assert result.count() > 0


class TestIcebergIntegration:
    """Tests for Iceberg write operations (mocked)."""
    
    def test_write_to_iceberg_append_mode(self, spark, sample_access_logs):
        """Verify write uses append mode when table exists."""
        route_metrics = ha.compute_route_metrics(sample_access_logs)
        
        with patch.object(route_metrics, 'writeTo') as mock_write:
            mock_writer = MagicMock()
            mock_write.return_value = mock_writer
            mock_writer.tableProperty.return_value = mock_writer
            
            ha.write_to_iceberg(route_metrics, "test.table")
            
            mock_writer.append.assert_called_once()
    
    def test_write_to_iceberg_creates_table_if_not_exists(self, spark, sample_access_logs):
        """Verify table is created when it doesn't exist."""
        route_metrics = ha.compute_route_metrics(sample_access_logs)
        
        with patch.object(route_metrics, 'writeTo') as mock_write:
            mock_writer = MagicMock()
            mock_write.return_value = mock_writer
            mock_writer.tableProperty.return_value = mock_writer
            # First call (append) raises "table not found", then create succeeds
            mock_writer.append.side_effect = Exception("Table test.table not found")
            
            ha.write_to_iceberg(route_metrics, "test.table")
            
            mock_writer.create.assert_called_once()


class TestEnvironmentConfiguration:
    """Tests for environment configuration."""
    
    def test_default_nessie_uri(self):
        """Verify default Nessie URI is set correctly."""
        with patch.dict('os.environ', {}, clear=True):
            # Re-import to get default value
            import importlib
            importlib.reload(ha)
            
            assert "nessie" in ha.NESSIE_URI.lower()
            assert "19120" in ha.NESSIE_URI
            assert "/api/v2" in ha.NESSIE_URI
    
    def test_default_warehouse_path(self):
        """Verify default warehouse path is set correctly."""
        with patch.dict('os.environ', {}, clear=True):
            import importlib
            importlib.reload(ha)
            
            assert "s3a://" in ha.WAREHOUSE_PATH
            assert "iceberg" in ha.WAREHOUSE_PATH.lower()
    
    def test_clickhouse_config(self):
        """Verify ClickHouse configuration is correct."""
        assert "clickhouse" in ha.CLICKHOUSE_JDBC_URL.lower()
        assert "8123" in ha.CLICKHOUSE_JDBC_URL
        assert ha.CLICKHOUSE_TABLE == "envoy_access_logs"
        assert ha.CLICKHOUSE_USER == "analytics"


class TestWindowAggregation:
    """Tests for 5-minute window aggregation."""
    
    def test_window_duration_is_5_minutes(self, spark, sample_access_logs):
        """Verify window duration is 5 minutes."""
        result = ha.compute_route_metrics(sample_access_logs)
        
        for row in result.collect():
            window_duration = (row.window_end - row.window_start).total_seconds()
            assert window_duration == 300  # 5 minutes = 300 seconds
    
    def test_records_in_same_window_are_grouped(self, spark):
        """Verify records within same 5-minute window are grouped together."""
        now = datetime.now()
        # Create data where all records are in the same 5-minute window
        data = [
            (now, "route-a", "host.example.com", "GET", "/path", 200, 10.0, 100, 50, "agent", "10.0.0.1"),
            (now - timedelta(seconds=30), "route-a", "host.example.com", "GET", "/path", 200, 20.0, 200, 100, "agent", "10.0.0.2"),
            (now - timedelta(seconds=60), "route-a", "host.example.com", "GET", "/path", 200, 30.0, 300, 150, "agent", "10.0.0.3"),
        ]
        
        df = spark.createDataFrame(data, schema=ha.get_input_schema())
        result = ha.compute_route_metrics(df)
        
        # Should result in exactly 1 row (all in same window + route + hostname)
        assert result.count() == 1
        
        row = result.first()
        assert row.request_count == 3
        assert row.avg_response_time_ms == 20.0  # (10 + 20 + 30) / 3
        assert row.min_response_time_ms == 10.0
        assert row.max_response_time_ms == 30.0
        assert row.total_bytes_sent == 600  # 100 + 200 + 300
        assert row.total_bytes_received == 300  # 50 + 100 + 150


class TestMainFunction:
    """Tests for main function orchestration."""
    
    def test_main_completes_successfully(self, spark):
        """Verify main completes without error on success."""
        with patch.object(ha, 'read_from_clickhouse') as mock_read, \
             patch.object(ha, 'write_to_iceberg') as mock_write, \
             patch('httproute_analytics.SparkSession') as mock_spark_class:
            
            mock_spark = MagicMock()
            mock_spark_class.builder.appName.return_value.getOrCreate.return_value = mock_spark
            mock_spark.version = "3.5.3"
            mock_spark.sparkContext.applicationId = "test-app-id"
            mock_spark.sql = MagicMock()
            
            # Return sample data
            sample_df = ha.create_sample_data(spark)
            mock_read.return_value = sample_df
            
            # main() should complete without raising an exception
            # On success, it doesn't call sys.exit()
            ha.main()
            
            # Verify the pipeline was executed
            assert mock_read.called
            assert mock_write.call_count == 2  # route_metrics and status_metrics
    
    def test_main_handles_empty_data(self, spark):
        """Verify main handles empty input data gracefully."""
        with patch.object(ha, 'read_from_clickhouse') as mock_read, \
             patch('httproute_analytics.SparkSession') as mock_spark_class:
            
            mock_spark = MagicMock()
            mock_spark_class.builder.appName.return_value.getOrCreate.return_value = mock_spark
            mock_spark.version = "3.5.3"
            mock_spark.sparkContext.applicationId = "test-app-id"
            mock_spark.sql = MagicMock()
            
            # Return empty DataFrame
            empty_df = MagicMock()
            empty_df.isEmpty.return_value = True
            mock_read.return_value = empty_df
            
            # Should not raise exception
            ha.main()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
