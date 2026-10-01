"""
Unit tests for MetricsCollector with SLA buckets.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from app.monitoring.metrics_collector import (
    MetricsCollector,
    MetricsConfig,
    SLA_BUCKETS,
    ENDPOINT_BUCKETS,
    get_metrics_collector,
)


class TestMetricsConfig:
    """Test MetricsConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = MetricsConfig()
        
        assert config.sla_buckets == SLA_BUCKETS
        assert config.endpoint_buckets == ENDPOINT_BUCKETS
        assert config.collect_interval_seconds == 15
        assert config.require_auth is True
        assert "metrics:read" in config.allowed_scopes
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = MetricsConfig(
            sla_buckets=[0.01, 0.05, 0.1, 0.5, 1.0],
            require_auth=False,
            allowed_scopes=["custom:scope"]
        )
        
        assert config.sla_buckets == [0.01, 0.05, 0.1, 0.5, 1.0]
        assert config.require_auth is False
        assert config.allowed_scopes == ["custom:scope"]


class TestMetricsCollector:
    """Test MetricsCollector functionality."""
    
    @pytest.fixture
    def metrics_collector(self):
        """Create metrics collector for testing."""
        config = MetricsConfig()
        return MetricsCollector(config)
    
    def test_init_metrics(self, metrics_collector):
        """Test that all metrics are initialized."""
        # Check that key metrics exist
        assert hasattr(metrics_collector, 'requests_total')
        assert hasattr(metrics_collector, 'request_latency')
        assert hasattr(metrics_collector, 'requests_in_progress')
        assert hasattr(metrics_collector, 'recommendations_total')
        assert hasattr(metrics_collector, 'recommendation_latency')
        assert hasattr(metrics_collector, 'cache_hits')
        assert hasattr(metrics_collector, 'cache_misses')
        assert hasattr(metrics_collector, 'errors_total')
        assert hasattr(metrics_collector, 'auth_failures')
        assert hasattr(metrics_collector, 'authz_denials')
        assert hasattr(metrics_collector, 'rate_limit_exceeded')
    
    def test_record_request(self, metrics_collector):
        """Test recording HTTP request metrics."""
        metrics_collector.record_request(
            method="GET",
            endpoint="/recommend",
            status_code=200,
            duration=0.045,
            request_id="req_123"
        )
        
        # Metrics should be recorded without errors
        # (Prometheus client doesn't expose direct getters for testing)
    
    def test_start_end_request(self, metrics_collector):
        """Test request in-progress tracking."""
        metrics_collector.start_request("POST", "/recommend")
        metrics_collector.end_request("POST", "/recommend")
        # Should not raise errors
    
    def test_record_recommendation(self, metrics_collector):
        """Test recording recommendation metrics."""
        metrics_collector.record_recommendation(
            user_id="user_123",
            num_recommendations=10,
            latency_ms=45.2,
            cache_hit=True,
            experiment_id="exp_001"
        )
        # Should not raise errors
    
    def test_record_feedback(self, metrics_collector):
        """Test recording feedback metrics."""
        metrics_collector.record_feedback(
            user_id="user_123",
            item_id="item_001",
            interaction_type="click",
            rating=4.5
        )
        # Should not raise errors
    
    def test_record_feature_operation(self, metrics_collector):
        """Test recording feature store operations."""
        metrics_collector.record_feature_operation(
            operation="read",
            entity_type="user",
            latency_ms=5.0,
            endpoint="/user/user_123/features"
        )
        # Should not raise errors
    
    def test_record_error(self, metrics_collector):
        """Test recording error metrics."""
        metrics_collector.record_error(
            endpoint="/recommend",
            error_type="ValueError",
            status_code=500,
            user_id="user_123"
        )
        # Should not raise errors
    
    def test_record_validation_error(self, metrics_collector):
        """Test recording validation error."""
        metrics_collector.record_validation_error(
            endpoint="/recommend",
            field="user_id"
        )
    
    def test_record_auth_failure(self, metrics_collector):
        """Test recording auth failure."""
        metrics_collector.record_auth_failure(
            reason="invalid_token",
            endpoint="/recommend"
        )
    
    def test_record_authz_denial(self, metrics_collector):
        """Test recording authorization denial."""
        metrics_collector.record_authz_denial(
            endpoint="/admin",
            required_scope="rec:admin"
        )
    
    def test_record_rate_limit_exceeded(self, metrics_collector):
        """Test recording rate limit exceeded."""
        metrics_collector.record_rate_limit_exceeded(
            scope="user",
            endpoint="/recommend"
        )
    
    def test_set_circuit_breaker_state(self, metrics_collector):
        """Test setting circuit breaker state."""
        metrics_collector.set_circuit_breaker_state("/recommend", "open")
        metrics_collector.set_circuit_breaker_state("/recommend", "half-open")
        metrics_collector.set_circuit_breaker_state("/recommend", "closed")
    
    def test_set_active_connections(self, metrics_collector):
        """Test setting active connections."""
        metrics_collector.set_active_connections(100)
    
    def test_set_queue_depth(self, metrics_collector):
        """Test setting queue depth."""
        metrics_collector.set_queue_depth("background_tasks", 500)
    
    def test_record_index_build(self, metrics_collector):
        """Test recording index build metrics."""
        metrics_collector.record_index_build(
            index_type="scann",
            duration_seconds=45.2,
            size_bytes=1024 * 1024 * 100,
            num_items=1000000
        )
    
    def test_record_index_query(self, metrics_collector):
        """Test recording index query latency."""
        metrics_collector.record_index_query(
            index_type="faiss",
            latency_ms=2.5
        )
    
    def test_set_model_version(self, metrics_collector):
        """Test setting model version."""
        metrics_collector.set_model_version("recommendation_model", "2.1.0")
    
    def test_set_active_users(self, metrics_collector):
        """Test setting active users."""
        metrics_collector.set_active_users(5000)
    
    def test_set_active_sessions(self, metrics_collector):
        """Test setting active sessions."""
        metrics_collector.set_active_sessions(10000)
    
    def test_record_experiment_assignment(self, metrics_collector):
        """Test recording experiment assignment."""
        metrics_collector.record_experiment_assignment(
            experiment_id="exp_001",
            variant="treatment"
        )
    
    def test_update_slo_metrics(self, metrics_collector):
        """Test updating SLO metrics."""
        latencies = [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
        metrics_collector.update_slo_metrics(
            endpoint="/recommend",
            latencies=latencies,
            error_count=2,
            total_count=100
        )
    
    def test_get_metrics(self, metrics_collector):
        """Test getting metrics in Prometheus format."""
        output = metrics_collector.get_metrics()
        
        assert isinstance(output, bytes)
        assert len(output) > 0
        # Should contain metric names
        output_str = output.decode('utf-8')
        assert 'rec_engine_requests_total' in output_str
    
    def test_get_content_type(self, metrics_collector):
        """Test getting content type."""
        content_type = metrics_collector.get_content_type()
        assert content_type == "text/plain; version=0.0.4; charset=utf-8"
    
    def test_get_current_metrics(self, metrics_collector):
        """Test getting current metrics as dict."""
        result = metrics_collector.get_current_metrics()
        
        assert isinstance(result, dict)
        assert "timestamp" in result
        assert "registry_metrics" in result
    
    def test_normalize_endpoint(self, metrics_collector):
        """Test endpoint normalization."""
        assert metrics_collector._normalize_endpoint("/user/12345/features") == "/user/{user_id}/features"
        assert metrics_collector._normalize_endpoint("/item/abc-123/feature") == "/item/{item_id}/feature"
        assert metrics_collector._normalize_endpoint("/api/v1/users/12345678-1234-5678-1234-567812345678") == "/api/v1/users/{uuid}"
        assert metrics_collector._normalize_endpoint("/recommend") == "/recommend"
    
    def test_global_getter(self):
        """Test global getter function."""
        import app.monitoring.metrics_collector as mc_module
        mc_module._metrics_collector = None
        
        collector = get_metrics_collector()
        assert collector is not None
        
        collector2 = get_metrics_collector()
        assert collector is collector2


class TestSLABuckets:
    """Test SLA bucket configurations."""
    
    def test_sla_buckets_defined(self):
        """Test that SLA buckets are properly defined."""
        assert len(SLA_BUCKETS) > 0
        assert SLA_BUCKETS[0] == 0.001  # 1ms
        assert 0.1 in SLA_BUCKETS  # 100ms SLO target
        assert 1.0 in SLA_BUCKETS  # 1s
        assert max(SLA_BUCKETS) >= 60.0  # Up to 60s
    
    def test_endpoint_buckets_defined(self):
        """Test that endpoint-specific buckets exist."""
        assert "/recommend" in ENDPOINT_BUCKETS
        assert "/feedback" in ENDPOINT_BUCKETS
        assert "/health" in ENDPOINT_BUCKETS
        assert "default" in ENDPOINT_BUCKETS
        
        # Recommend endpoint should have tighter buckets around 100ms
        recommend_buckets = ENDPOINT_BUCKETS["/recommend"]
        assert 0.05 in recommend_buckets  # 50ms
        assert 0.1 in recommend_buckets   # 100ms SLO
        assert 0.125 in recommend_buckets # 125ms
        assert 0.15 in recommend_buckets  # 150ms


if __name__ == "__main__":
    pytest.main([__file__, "-v"])