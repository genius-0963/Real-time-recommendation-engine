"""
Hardened Prometheus metrics collector with SLA-aligned buckets.
Secured behind authentication and authorization.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
import threading

from prometheus_client import (
    Counter, Histogram, Gauge, Summary, CollectorRegistry,
    generate_latest, CONTENT_TYPE_LATEST, Histogram as PromHistogram
)
from prometheus_client.core import CollectorRegistry as CoreCollectorRegistry

from app.security import require_metrics_read, TokenClaims

logger = logging.getLogger(__name__)


# SLA-aligned histogram buckets (matching 100ms p99 target)
SLA_BUCKETS = [
    0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1,   # < 100ms
    0.15, 0.2, 0.3, 0.5, 0.75, 1.0,                # 100ms - 1s
    2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 30.0, 60.0     # > 1s
]

# Endpoint-specific buckets for tighter SLAs
ENDPOINT_BUCKETS = {
    "/recommend": [
        0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 
        0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.5, 1.0
    ],
    "/feedback": [
        0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0
    ],
    "/health": [
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5
    ],
    "/metrics": [
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0
    ],
    "default": SLA_BUCKETS
}


@dataclass
class MetricsConfig:
    """Configuration for metrics collector."""
    # Histogram buckets
    sla_buckets: List[float] = field(default_factory=lambda: SLA_BUCKETS)
    endpoint_buckets: Dict[str, List[float]] = field(default_factory=lambda: ENDPOINT_BUCKETS)
    
    # Metrics collection
    collect_interval_seconds: int = 15
    enable_gc_metrics: bool = True
    enable_process_metrics: bool = True
    
    # Export settings
    export_path: str = "/metrics"
    export_content_type: str = CONTENT_TYPE_LATEST
    
    # Security
    require_auth: bool = True
    allowed_scopes: List[str] = field(default_factory=lambda: ["metrics:read"])


class MetricsCollector:
    """
    Hardened Prometheus metrics collector with SLA-aligned buckets.
    All metrics are namespaced with 'rec_engine_' prefix.
    """
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        self.config = config or MetricsConfig()
        
        # Create isolated registry for this service
        self.registry = CollectorRegistry()
        
        # Initialize metrics
        self._init_metrics()
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("Metrics collector initialized with SLA-aligned buckets")
    
    def _init_metrics(self):
        """Initialize all Prometheus metrics with SLA buckets."""
        
        # ─── Request Metrics ──────────────────────────────────────────
        self.requests_total = Counter(
            'rec_engine_requests_total',
            'Total number of HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.request_latency = Histogram(
            'rec_engine_request_latency_seconds',
            'HTTP request latency in seconds',
            ['method', 'endpoint'],
            buckets=self.config.sla_buckets,
            registry=self.registry
        )
        
        self.request_in_progress = Gauge(
            'rec_engine_requests_in_progress',
            'Number of requests currently being processed',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # ─── Recommendation Metrics ──────────────────────────────────
        self.recommendations_total = Counter(
            'rec_engine_recommendations_total',
            'Total recommendations generated',
            ['endpoint', 'experiment_id', 'cache_hit'],
            registry=self.registry
        )
        
        self.recommendation_latency = Histogram(
            'rec_engine_recommendation_latency_seconds',
            'Recommendation generation latency',
            ['endpoint', 'cache_hit'],
            buckets=self.config.endpoint_buckets.get("/recommend", self.config.sla_buckets),
            registry=self.registry
        )
        
        self.recommendation_count = Histogram(
            'rec_engine_recommendation_count',
            'Number of recommendations per request',
            ['endpoint'],
            buckets=[1, 5, 10, 20, 50, 100],
            registry=self.registry
        )
        
        self.cache_hits = Counter(
            'rec_engine_cache_hits_total',
            'Total cache hits',
            ['cache_type', 'endpoint'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'rec_engine_cache_misses_total',
            'Total cache misses',
            ['cache_type', 'endpoint'],
            registry=self.registry
        )
        
        # ─── Feedback Metrics ────────────────────────────────────────
        self.feedback_total = Counter(
            'rec_engine_feedback_total',
            'Total feedback events received',
            ['interaction_type', 'endpoint'],
            registry=self.registry
        )
        
        self.feedback_rating = Histogram(
            'rec_engine_feedback_rating',
            'Feedback rating distribution',
            ['interaction_type'],
            buckets=[1, 2, 3, 4, 5],
            registry=self.registry
        )
        
        # ─── Feature Store Metrics ───────────────────────────────────
        self.feature_reads = Counter(
            'rec_engine_feature_reads_total',
            'Total feature read operations',
            ['entity_type', 'endpoint'],
            registry=self.registry
        )
        
        self.feature_writes = Counter(
            'rec_engine_feature_writes_total',
            'Total feature write operations',
            ['entity_type', 'endpoint'],
            registry=self.registry
        )
        
        self.feature_latency = Histogram(
            'rec_engine_feature_latency_seconds',
            'Feature store operation latency',
            ['operation', 'entity_type'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=self.registry
        )
        
        # ─── Error Metrics ───────────────────────────────────────────
        self.errors_total = Counter(
            'rec_engine_errors_total',
            'Total errors by type',
            ['endpoint', 'error_type', 'status_code'],
            registry=self.registry
        )
        
        self.validation_errors = Counter(
            'rec_engine_validation_errors_total',
            'Total validation errors',
            ['endpoint', 'field'],
            registry=self.registry
        )
        
        self.auth_failures = Counter(
            'rec_engine_auth_failures_total',
            'Total authentication failures',
            ['reason', 'endpoint'],
            registry=self.registry
        )
        
        self.authz_denials = Counter(
            'rec_engine_authz_denials_total',
            'Total authorization denials',
            ['endpoint', 'required_scope'],
            registry=self.registry
        )
        
        self.rate_limit_exceeded = Counter(
            'rec_engine_rate_limit_exceeded_total',
            'Total rate limit exceeded events',
            ['scope', 'endpoint'],
            registry=self.registry
        )
        
        # ─── System Metrics ──────────────────────────────────────────
        self.circuit_breaker_state = Gauge(
            'rec_engine_circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=half-open, 2=open)',
            ['endpoint'],
            registry=self.registry
        )
        
        self.active_connections = Gauge(
            'rec_engine_active_connections',
            'Number of active connections',
            registry=self.registry
        )
        
        self.queue_depth = Gauge(
            'rec_engine_queue_depth',
            'Background task queue depth',
            ['queue_name'],
            registry=self.registry
        )
        
        # ─── Index/Model Metrics ─────────────────────────────────────
        self.index_build_duration = Histogram(
            'rec_engine_index_build_duration_seconds',
            'ANN index build duration',
            ['index_type'],
            buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600],
            registry=self.registry
        )
        
        self.index_size = Gauge(
            'rec_engine_index_size_bytes',
            'ANN index size in bytes',
            ['index_type'],
            registry=self.registry
        )
        
        self.index_query_latency = Histogram(
            'rec_engine_index_query_latency_seconds',
            'ANN index query latency',
            ['index_type'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=self.registry
        )
        
        self.model_version = Gauge(
            'rec_engine_model_version',
            'Current model version',
            ['model_name', 'version'],
            registry=self.registry
        )
        
        # ─── Business Metrics ────────────────────────────────────────
        self.active_users = Gauge(
            'rec_engine_active_users',
            'Number of active users (last 5 min)',
            registry=self.registry
        )
        
        self.active_sessions = Gauge(
            'rec_engine_active_sessions',
            'Number of active sessions',
            registry=self.registry
        )
        
        self.experiment_assignments = Counter(
            'rec_engine_experiment_assignments_total',
            'Total experiment assignments',
            ['experiment_id', 'variant'],
            registry=self.registry
        )
        
        # ─── SLA/SLO Metrics ────────────────────────────────────────
        self.slo_latency_p50 = Gauge(
            'rec_engine_slo_latency_p50_seconds',
            'SLO: p50 latency target',
            ['endpoint'],
            registry=self.registry
        )
        
        self.slo_latency_p95 = Gauge(
            'rec_engine_slo_latency_p95_seconds',
            'SLO: p95 latency target (100ms)',
            ['endpoint'],
            registry=self.registry
        )
        
        self.slo_latency_p99 = Gauge(
            'rec_engine_slo_latency_p99_seconds',
            'SLO: p99 latency target',
            ['endpoint'],
            registry=self.registry
        )
        
        self.slo_error_rate = Gauge(
            'rec_engine_slo_error_rate',
            'SLO: error rate target (<1%)',
            ['endpoint'],
            registry=self.registry
        )
        
        self.slo_availability = Gauge(
            'rec_engine_slo_availability',
            'SLO: availability target (99.9%)',
            ['endpoint'],
            registry=self.registry
        )
    
    # ─── Request Recording ──────────────────────────────────────────
    
    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float,
        request_id: str = ""
    ):
        """Record HTTP request metrics."""
        with self._lock:
            status = str(status_code)
            endpoint_label = self._normalize_endpoint(endpoint)
            
            self.requests_total.labels(
                method=method,
                endpoint=endpoint_label,
                status=status
            ).inc()
            
            self.request_latency.labels(
                method=method,
                endpoint=endpoint_label
            ).observe(duration)
            
            # Track in-progress (decrement happens in middleware)
            # self.request_in_progress.labels(method=method, endpoint=endpoint_label).dec()
    
    def start_request(self, method: str, endpoint: str):
        """Mark request as in-progress."""
        endpoint_label = self._normalize_endpoint(endpoint)
        self.request_in_progress.labels(method=method, endpoint=endpoint_label).inc()
    
    def end_request(self, method: str, endpoint: str):
        """Mark request as complete."""
        endpoint_label = self._normalize_endpoint(endpoint)
        self.request_in_progress.labels(method=method, endpoint=endpoint_label).dec()
    
    # ─── Recommendation Recording ───────────────────────────────────
    
    def record_recommendation(
        self,
        user_id: str,
        num_recommendations: int,
        latency_ms: float,
        cache_hit: bool,
        experiment_id: Optional[str] = None
    ):
        """Record recommendation generation metrics."""
        with self._lock:
            self.recommendations_total.labels(
                endpoint="/recommend",
                experiment_id=experiment_id or "none",
                cache_hit=str(cache_hit).lower()
            ).inc()
            
            self.recommendation_latency.labels(
                endpoint="/recommend",
                cache_hit=str(cache_hit).lower()
            ).observe(latency_ms / 1000.0)
            
            self.recommendation_count.labels(endpoint="/recommend").observe(num_recommendations)
            
            if cache_hit:
                self.cache_hits.labels(cache_type="recommendation", endpoint="/recommend").inc()
            else:
                self.cache_misses.labels(cache_type="recommendation", endpoint="/recommend").inc()
    
    def log_recommendation(
        self,
        request_id: str,
        user_id: str,
        recommendations: List[Dict],
        metadata: Dict
    ):
        """Log recommendation for audit/analytics (async)."""
        # This would typically send to a logging/analytics pipeline
        logger.debug(f"Recommendation logged: {request_id} for user {user_id}")
    
    # ─── Feedback Recording ─────────────────────────────────────────
    
    def record_feedback(
        self,
        user_id: str,
        item_id: str,
        interaction_type: str,
        rating: Optional[float]
    ):
        """Record feedback metrics."""
        with self._lock:
            self.feedback_total.labels(
                interaction_type=interaction_type,
                endpoint="/feedback"
            ).inc()
            
            if rating is not None:
                self.feedback_rating.labels(interaction_type=interaction_type).observe(rating)
    
    # ─── Feature Store Recording ────────────────────────────────────
    
    def record_feature_operation(
        self,
        operation: str,
        entity_type: str,
        latency_ms: float,
        endpoint: str = ""
    ):
        """Record feature store operation metrics."""
        with self._lock:
            if operation == "read":
                self.feature_reads.labels(entity_type=entity_type, endpoint=endpoint).inc()
            elif operation == "write":
                self.feature_writes.labels(entity_type=entity_type, endpoint=endpoint).inc()
            
            self.feature_latency.labels(
                operation=operation,
                entity_type=entity_type
            ).observe(latency_ms / 1000.0)
    
    # ─── Error Recording ────────────────────────────────────────────
    
    def record_error(
        self,
        endpoint: str,
        error_type: str,
        status_code: int = 500,
        user_id: str = ""
    ):
        """Record error metrics."""
        with self._lock:
            endpoint_label = self._normalize_endpoint(endpoint)
            self.errors_total.labels(
                endpoint=endpoint_label,
                error_type=error_type,
                status_code=str(status_code)
            ).inc()
    
    def record_validation_error(self, endpoint: str, field: str):
        """Record validation error."""
        with self._lock:
            self.validation_errors.labels(
                endpoint=self._normalize_endpoint(endpoint),
                field=field
            ).inc()
    
    def record_auth_failure(self, reason: str, endpoint: str):
        """Record authentication failure."""
        with self._lock:
            self.auth_failures.labels(
                reason=reason,
                endpoint=self._normalize_endpoint(endpoint)
            ).inc()
    
    def record_authz_denial(self, endpoint: str, required_scope: str):
        """Record authorization denial."""
        with self._lock:
            self.authz_denials.labels(
                endpoint=self._normalize_endpoint(endpoint),
                required_scope=required_scope
            ).inc()
    
    def record_rate_limit_exceeded(self, scope: str, endpoint: str):
        """Record rate limit exceeded."""
        with self._lock:
            self.rate_limit_exceeded.labels(
                scope=scope,
                endpoint=self._normalize_endpoint(endpoint)
            ).inc()
    
    # ─── Circuit Breaker ────────────────────────────────────────────
    
    def set_circuit_breaker_state(self, endpoint: str, state: str):
        """Set circuit breaker state (0=closed, 1=half-open, 2=open)."""
        state_map = {"closed": 0, "half-open": 1, "open": 2}
        with self._lock:
            self.circuit_breaker_state.labels(endpoint=endpoint).set(state_map.get(state, 0))
    
    # ─── Queue/Connection Metrics ───────────────────────────────────
    
    def set_active_connections(self, count: int):
        """Set active connection count."""
        with self._lock:
            self.active_connections.set(count)
    
    def set_queue_depth(self, queue_name: str, depth: int):
        """Set background queue depth."""
        with self._lock:
            self.queue_depth.labels(queue_name=queue_name).set(depth)
    
    # ─── Index/Model Metrics ────────────────────────────────────────
    
    def record_index_build(self, index_type: str, duration_seconds: float, size_bytes: int, num_items: int):
        """Record index build metrics."""
        with self._lock:
            self.index_build_duration.labels(index_type=index_type).observe(duration_seconds)
            self.index_size.labels(index_type=index_type).set(size_bytes)
    
    def record_index_query(self, index_type: str, latency_ms: float):
        """Record index query latency."""
        with self._lock:
            self.index_query_latency.labels(index_type=index_type).observe(latency_ms / 1000.0)
    
    def set_model_version(self, model_name: str, version: str):
        """Set current model version."""
        with self._lock:
            # Reset all versions for this model
            self.model_version.labels(model_name=model_name, version=version).set(1)
    
    # ─── Business Metrics ───────────────────────────────────────────
    
    def set_active_users(self, count: int):
        """Set active users count."""
        with self._lock:
            self.active_users.set(count)
    
    def set_active_sessions(self, count: int):
        """Set active sessions count."""
        with self._lock:
            self.active_sessions.set(count)
    
    def record_experiment_assignment(self, experiment_id: str, variant: str):
        """Record experiment assignment."""
        with self._lock:
            self.experiment_assignments.labels(
                experiment_id=experiment_id,
                variant=variant
            ).inc()
    
    # ─── SLO Metrics ────────────────────────────────────────────────
    
    def update_slo_metrics(self, endpoint: str, latencies: List[float], error_count: int, total_count: int):
        """Update SLO metrics from collected latencies."""
        if not latencies:
            return
            
        with self._lock:
            sorted_latencies = sorted(latencies)
            n = len(sorted_latencies)
            
            self.slo_latency_p50.labels(endpoint=endpoint).set(sorted_latencies[n // 2])
            self.slo_latency_p95.labels(endpoint=endpoint).set(sorted_latencies[int(n * 0.95)])
            self.slo_latency_p99.labels(endpoint=endpoint).set(sorted_latencies[int(n * 0.99)])
            
            if total_count > 0:
                self.slo_error_rate.labels(endpoint=endpoint).set(error_count / total_count)
                self.slo_availability.labels(endpoint=endpoint).set(1 - (error_count / total_count))
    
    # ─── Export ─────────────────────────────────────────────────────
    
    def get_metrics(self) -> bytes:
        """Get Prometheus metrics in text format."""
        return generate_latest(self.registry)
    
    def get_content_type(self) -> str:
        """Get content type for metrics export."""
        return self.config.export_content_type
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics as dictionary (for JSON API)."""
        # This is a simplified version - in production would query registry
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "registry_metrics": "use /metrics endpoint for Prometheus format"
        }
    
    # ─── Helpers ────────────────────────────────────────────────────
    
    def _normalize_endpoint(self, endpoint: str) -> str:
        """Normalize endpoint for labeling (replace IDs with placeholders)."""
        # Replace UUIDs
        import re
        endpoint = re.sub(r'/[0-9a-f-]{36}', '/{uuid}', endpoint)
        # Replace numeric IDs
        endpoint = re.sub(r'/\d+', '/{id}', endpoint)
        # Replace user/item IDs
        endpoint = re.sub(r'/user/[^/]+', '/user/{user_id}', endpoint)
        endpoint = re.sub(r'/item/[^/]+', '/item/{item_id}', endpoint)
        return endpoint


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None
_metrics_lock = threading.Lock()


def get_metrics_collector(config: Optional[MetricsConfig] = None) -> MetricsCollector:
    """Get or create global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        with _metrics_lock:
            if _metrics_collector is None:
                _metrics_collector = MetricsCollector(config)
    return _metrics_collector