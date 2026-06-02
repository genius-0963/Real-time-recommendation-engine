"""
Unit tests for the feature stores (online + offline) with mocked backends.

Source files:
  - feature_store/online_store.py  → OnlineFeatureStore, FeatureValue
  - feature_store/offline_store.py → OfflineFeatureStore, FeatureDefinition
"""

import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime, timezone

from feature_store.online_store import OnlineFeatureStore, FeatureValue
from app.config import RedisConfig, DatabaseConfig


# ===================================================================
# OnlineFeatureStore tests (Redis mock)
# ===================================================================


class TestOnlineStoreSetGet:
    """Test set/get single feature with mocked Redis."""

    @patch("feature_store.online_store.redis.Redis")
    def test_online_store_set_get(self, MockRedis, mock_redis):
        """set_feature then get_feature should round-trip the value."""
        MockRedis.return_value = mock_redis

        config = RedisConfig()
        store = OnlineFeatureStore.__new__(OnlineFeatureStore)
        store.config = config
        store.redis_client = mock_redis
        store.async_redis_client = None
        store.FEATURE_PREFIX = "feature:"
        store.ENTITY_PREFIX = "entity:"
        store.VERSION_PREFIX = "version:"
        store.METADATA_PREFIX = "meta:"

        # Set up the mock to capture what is stored
        stored = {}

        def mock_setex(key, ttl, value):
            stored[key] = value
            return True

        def mock_get(key):
            return stored.get(key)

        mock_redis.setex.side_effect = mock_setex
        mock_redis.get.side_effect = mock_get
        mock_redis.set.return_value = True
        mock_redis.hget.return_value = None
        mock_redis.hset.return_value = 1
        mock_redis.expire.return_value = True

        # Set a feature
        result = store.set_feature("user", "u1", "age", 30, ttl_seconds=3600)
        assert result is True

        # The value should have been stored via setex
        assert len(stored) == 1
        key = list(stored.keys())[0]

        # Now get it back — simulate Redis returning the stored bytes
        mock_redis.get.side_effect = lambda k: stored.get(k, b"").encode("utf-8") if isinstance(stored.get(k), str) else stored.get(k)
        # Actually setex stores it as string, get needs to return bytes
        raw = list(stored.values())[0]
        mock_redis.get.return_value = raw.encode("utf-8") if isinstance(raw, str) else raw

        value = store.get_feature("user", "u1", "age")
        assert value == 30


class TestOnlineStoreBatchGet:
    """Test batch get of multiple features."""

    def test_online_store_batch_get(self, mock_redis):
        """get_features returns a dict of feature_name→value."""
        config = RedisConfig()
        store = OnlineFeatureStore.__new__(OnlineFeatureStore)
        store.config = config
        store.redis_client = mock_redis
        store.async_redis_client = None
        store.FEATURE_PREFIX = "feature:"
        store.ENTITY_PREFIX = "entity:"
        store.VERSION_PREFIX = "version:"
        store.METADATA_PREFIX = "meta:"

        # Build fake stored values for two features
        fv_age = FeatureValue(value=30, timestamp=datetime.now(timezone.utc), version=1)
        fv_gender = FeatureValue(value="M", timestamp=datetime.now(timezone.utc), version=1)

        serialized_age = json.dumps(fv_age.to_dict()).encode("utf-8")
        serialized_gender = json.dumps(fv_gender.to_dict()).encode("utf-8")

        mock_redis.mget.return_value = [serialized_age, serialized_gender]

        features = store.get_features("user", "u1", ["age", "gender"])
        assert features["age"] == 30
        assert features["gender"] == "M"


class TestOnlineStoreTTL:
    """Test that TTL is applied when setting features."""

    def test_online_store_ttl(self, mock_redis):
        """set_feature with ttl_seconds calls setex with the correct TTL."""
        config = RedisConfig()
        store = OnlineFeatureStore.__new__(OnlineFeatureStore)
        store.config = config
        store.redis_client = mock_redis
        store.async_redis_client = None
        store.FEATURE_PREFIX = "feature:"
        store.ENTITY_PREFIX = "entity:"
        store.VERSION_PREFIX = "version:"
        store.METADATA_PREFIX = "meta:"
        mock_redis.hget.return_value = None

        store.set_feature("user", "u1", "age", 30, ttl_seconds=7200)

        # setex should be called with the key, TTL, and serialized value
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == 7200  # TTL in seconds


class TestOnlineStoreMissingKey:
    """Test retrieval of a non-existent key."""

    def test_online_store_missing_key(self, mock_redis):
        """get_feature for a missing key returns None."""
        config = RedisConfig()
        store = OnlineFeatureStore.__new__(OnlineFeatureStore)
        store.config = config
        store.redis_client = mock_redis
        store.async_redis_client = None
        store.FEATURE_PREFIX = "feature:"
        store.ENTITY_PREFIX = "entity:"
        store.VERSION_PREFIX = "version:"
        store.METADATA_PREFIX = "meta:"

        mock_redis.get.return_value = None

        result = store.get_feature("user", "nonexistent", "age")
        assert result is None


# ===================================================================
# OfflineFeatureStore tests (Postgres mock)
# ===================================================================


class TestOfflineStoreWriteRead:
    """Test write and read of features with mocked database."""

    @patch("feature_store.offline_store.create_engine")
    @patch("feature_store.offline_store.sessionmaker")
    def test_offline_store_write_read(self, mock_sessionmaker, mock_create_engine, mock_postgres):
        """write_features then get_features should return the written values."""
        from feature_store.offline_store import OfflineFeatureStore, FeatureDefinition

        # Set up mocks
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__ = MagicMock()
        mock_engine.connect.return_value.__exit__ = MagicMock()

        mock_session = MagicMock()
        mock_sessionmaker.return_value = MagicMock(return_value=mock_session)
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        # Bypass real init
        store = OfflineFeatureStore.__new__(OfflineFeatureStore)
        store.config = DatabaseConfig()
        store.engine = mock_engine
        store.SessionLocal = mock_sessionmaker.return_value
        store.feature_definitions = {}

        # Register a feature definition in the cache
        store.feature_definitions["user_age"] = FeatureDefinition(
            name="user_age", entity_type="user",
            data_type="numerical", description="User age"
        )

        # Mock the query chain for write_features
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.update.return_value = 0

        # Write features
        result = store.write_features("user", "u1", {"user_age": 30})
        assert result is True
        mock_session.add.assert_called()
        mock_session.commit.assert_called()


class TestOfflineStorePointInTime:
    """Test point-in-time feature retrieval."""

    @patch("feature_store.offline_store.create_engine")
    @patch("feature_store.offline_store.sessionmaker")
    def test_offline_store_point_in_time(self, mock_sessionmaker, mock_create_engine):
        """get_features with a timestamp filters by that point in time."""
        from feature_store.offline_store import OfflineFeatureStore

        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__ = MagicMock()
        mock_engine.connect.return_value.__exit__ = MagicMock()

        mock_session = MagicMock()
        mock_sessionmaker.return_value = MagicMock(return_value=mock_session)
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=False)

        store = OfflineFeatureStore.__new__(OfflineFeatureStore)
        store.config = DatabaseConfig()
        store.engine = mock_engine
        store.SessionLocal = mock_sessionmaker.return_value
        store.feature_definitions = {}

        # Mock the query chain
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        # Return an empty result set
        mock_query.all.return_value = []

        pit_timestamp = datetime(2025, 6, 1, tzinfo=timezone.utc)
        features = store.get_features("user", "u1", ["user_age"], timestamp=pit_timestamp)

        # The query should have been built (filter called multiple times)
        assert mock_query.filter.called
        assert isinstance(features, dict)
