"""
Shared pytest fixtures for the Real-Time Recommendation Engine test suite.
"""

import sys
import os
import json
import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone

# Ensure the project root is on sys.path so that imports like 'app.config' resolve
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Simple value fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_user_id() -> str:
    """A sample user identifier."""
    return "user_42"


@pytest.fixture
def sample_item_id() -> str:
    """A sample item identifier."""
    return "item_99"


# ---------------------------------------------------------------------------
# Embedding / ID fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """100 items × 128-dim embeddings (float32, deterministic)."""
    rng = np.random.RandomState(42)
    return rng.randn(100, 128).astype(np.float32)


@pytest.fixture
def sample_item_ids() -> list:
    """List of 100 item IDs that correspond to *sample_embeddings*."""
    return [f"item_{i}" for i in range(100)]


# ---------------------------------------------------------------------------
# Mock external clients
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_redis() -> MagicMock:
    """A MagicMock that stands in for a Redis client."""
    client = MagicMock()
    # Make common Redis methods return sensible defaults
    client.ping.return_value = True
    client.get.return_value = None
    client.mget.return_value = []
    client.set.return_value = True
    client.setex.return_value = True
    client.delete.return_value = 1
    client.keys.return_value = []
    client.hgetall.return_value = {}
    client.hget.return_value = None
    client.hset.return_value = 1
    client.expire.return_value = True
    client.pipeline.return_value = MagicMock()
    client.pipeline.return_value.execute.return_value = []
    client.info.return_value = {
        "connected_clients": 1,
        "used_memory_human": "1M",
        "total_commands_processed": 100,
        "keyspace_hits": 50,
        "keyspace_misses": 10,
    }
    return client


@pytest.fixture
def mock_postgres() -> MagicMock:
    """A MagicMock that stands in for a PostgreSQL connection / session."""
    conn = MagicMock()
    conn.execute.return_value = MagicMock()
    conn.commit.return_value = None
    conn.close.return_value = None
    # Provide a mock cursor-like interface as well
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    cursor.fetchone.return_value = None
    conn.cursor.return_value = cursor
    return conn


# ---------------------------------------------------------------------------
# FastAPI TestClient
# ---------------------------------------------------------------------------

@pytest.fixture
def app_client():
    """
    FastAPI TestClient that mocks out all heavy service-layer dependencies
    so the app can start without Redis / Kafka / model weights.
    """
    # Patch heavy dependencies *before* importing the app module so the
    # lifespan handler does not try to connect to real services.
    with patch("app.api.main.RedisCache") as MockRedisCache, \
         patch("app.api.main.FeatureService") as MockFeatureService, \
         patch("app.api.main.RecommendationService") as MockRecService, \
         patch("app.api.main.MetricsCollector") as MockMetrics, \
         patch("app.api.main.RateLimiter") as MockRateLimiter, \
         patch("app.api.main.ABTestManager") as MockABTest:

        # Configure async mocks for the services that are awaited during lifespan
        mock_rec_service = AsyncMock()
        mock_rec_service.initialize = AsyncMock()
        mock_rec_service.cleanup = AsyncMock()
        mock_rec_service.health_check = AsyncMock(return_value=True)
        mock_rec_service.get_recommendations = AsyncMock(return_value=[
            {
                "item_id": "item_1",
                "score": 0.95,
                "explanation": "Similar to your recent views",
                "metadata": {},
                "model_version": "2.0.0",
                "cache_hit": False,
            }
        ])
        mock_rec_service.record_feedback = AsyncMock()
        mock_rec_service.process_feedback = AsyncMock()
        MockRecService.return_value = mock_rec_service

        mock_feature_service = AsyncMock()
        mock_feature_service.health_check = AsyncMock(return_value=True)
        mock_feature_service.get_user_features = AsyncMock(return_value=[])
        mock_feature_service.get_all_user_features = AsyncMock(return_value=[])
        mock_feature_service.get_item_features = AsyncMock(return_value=[])
        mock_feature_service.get_all_item_features = AsyncMock(return_value=[])
        MockFeatureService.return_value = mock_feature_service

        mock_cache = AsyncMock()
        mock_cache.health_check = AsyncMock(return_value=True)
        mock_cache.close = AsyncMock()
        MockRedisCache.return_value = mock_cache

        mock_metrics_instance = AsyncMock()
        mock_metrics_instance.record_request = AsyncMock()
        mock_metrics_instance.record_recommendation = AsyncMock()
        mock_metrics_instance.record_feedback = AsyncMock()
        mock_metrics_instance.record_error = AsyncMock()
        mock_metrics_instance.log_recommendation = AsyncMock()
        mock_metrics_instance.get_current_metrics = AsyncMock(return_value={
            "total_requests": 100,
            "avg_latency_ms": 25.0,
        })
        MockMetrics.return_value = mock_metrics_instance

        mock_rate_limiter = MagicMock()
        mock_rate_limiter.is_allowed.return_value = True
        mock_rate_limiter.get_retry_after.return_value = 60
        MockRateLimiter.return_value = mock_rate_limiter

        mock_ab_test = AsyncMock()
        mock_ab_test.get_experiment_config = AsyncMock(return_value=None)
        mock_ab_test.get_active_experiments = AsyncMock(return_value=[])
        mock_ab_test.assign_user = AsyncMock(return_value={"variant": "control"})
        MockABTest.return_value = mock_ab_test

        # Now import the app – the patches are already active
        from app.api.main import app  # noqa: E402
        from httpx import ASGITransport, AsyncClient
        from starlette.testclient import TestClient

        # Use Starlette's synchronous TestClient which handles the lifespan
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client


# ---------------------------------------------------------------------------
# Request / event payload fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_recommendation_request() -> dict:
    """A valid POST body for ``/recommend``."""
    return {
        "user_id": "user_42",
        "num_recommendations": 5,
        "context": {"device": "mobile", "time_of_day": "evening"},
    }


@pytest.fixture
def sample_event() -> dict:
    """A sample user-interaction event payload."""
    return {
        "user_id": "user_42",
        "item_id": "item_99",
        "event_type": "click",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
