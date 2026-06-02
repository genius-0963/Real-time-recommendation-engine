"""
Integration tests for the feature pipeline (online ↔ offline sync).

Marked with ``@pytest.mark.integration``; skipped unless external services
(Redis, PostgreSQL) are available.
"""

import os
import pytest
from unittest.mock import MagicMock, patch


def services_available() -> bool:
    """Return True only when the operator has confirmed services are running."""
    return os.getenv("INTEGRATION_TESTS", "0") == "1"


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not services_available(), reason="Requires running services"),
]


class TestFeatureSync:
    """Test synchronisation between the online (Redis) and offline (Postgres) stores."""

    @patch("feature_store.online_store.redis.Redis")
    @patch("feature_store.offline_store.create_engine")
    @patch("feature_store.offline_store.sessionmaker")
    def test_feature_sync(self, mock_sessionmaker, mock_create_engine, MockRedis):
        """
        Write features to the offline store, then verify that the online
        store can be populated with the same data (simulated sync).
        """
        from feature_store.online_store import OnlineFeatureStore
        from feature_store.offline_store import OfflineFeatureStore, FeatureDefinition
        from app.config import RedisConfig, DatabaseConfig

        # --- Setup online store (mocked Redis) ---
        mock_redis_client = MagicMock()
        MockRedis.return_value = mock_redis_client
        mock_redis_client.setex.return_value = True
        mock_redis_client.set.return_value = True
        mock_redis_client.hget.return_value = None
        mock_redis_client.hset.return_value = 1
        mock_redis_client.expire.return_value = True

        online_store = OnlineFeatureStore.__new__(OnlineFeatureStore)
        online_store.config = RedisConfig()
        online_store.redis_client = mock_redis_client
        online_store.async_redis_client = None
        online_store.FEATURE_PREFIX = "feature:"
        online_store.ENTITY_PREFIX = "entity:"
        online_store.VERSION_PREFIX = "version:"
        online_store.METADATA_PREFIX = "meta:"

        # --- Setup offline store (mocked Postgres) ---
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__ = MagicMock()
        mock_engine.connect.return_value.__exit__ = MagicMock()

        mock_session = MagicMock()
        mock_sessionmaker.return_value = MagicMock(return_value=mock_session)
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        offline_store = OfflineFeatureStore.__new__(OfflineFeatureStore)
        offline_store.config = DatabaseConfig()
        offline_store.engine = mock_engine
        offline_store.SessionLocal = mock_sessionmaker.return_value
        offline_store.feature_definitions = {
            "user_score": FeatureDefinition(
                name="user_score", entity_type="user",
                data_type="numerical", description="Engagement score"
            )
        }

        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.update.return_value = 0

        # --- Simulate sync: write offline, then write online ---
        features = {"user_score": 0.85}

        offline_ok = offline_store.write_features("user", "u1", features)
        assert offline_ok is True

        online_ok = online_store.set_feature("user", "u1", "user_score", 0.85)
        assert online_ok is True

        # Verify the online store attempted to persist
        mock_redis_client.setex.assert_called()
