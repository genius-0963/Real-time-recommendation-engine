"""
Unit tests for configuration loading.

Source: app/config.py
Classes tested:
  - Config
  - ModelConfig
  - RedisConfig
  - KafkaConfig
  - DatabaseConfig
  - APIConfig
  - MonitoringConfig
  - FeatureStoreConfig
  - ExperimentConfig
  - TrainingConfig
"""

import os
import pytest
from unittest.mock import patch

from app.config import (
    Config,
    ModelConfig,
    RedisConfig,
    KafkaConfig,
    DatabaseConfig,
    APIConfig,
    MonitoringConfig,
    FeatureStoreConfig,
    ExperimentConfig,
    TrainingConfig,
)


# ===================================================================
# Default construction
# ===================================================================


class TestDefaultConfig:
    """Test that Config() creates with sensible defaults."""

    def test_default_config(self):
        """Config() should succeed and populate all sub-configs."""
        config = Config()

        assert config.environment in ("development", "production", "staging", "test")
        assert isinstance(config.kafka, KafkaConfig)
        assert isinstance(config.redis, RedisConfig)
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.api, APIConfig)
        assert isinstance(config.monitoring, MonitoringConfig)
        assert isinstance(config.feature_store, FeatureStoreConfig)
        assert isinstance(config.experiments, ExperimentConfig)

    def test_default_redis_config(self):
        """RedisConfig defaults are reasonable."""
        rc = RedisConfig()
        assert rc.host == "localhost"
        assert rc.port == 6379
        assert rc.db == 0

    def test_default_database_config(self):
        """DatabaseConfig defaults are reasonable."""
        dc = DatabaseConfig()
        assert dc.host == "localhost"
        assert dc.port == 5432
        assert dc.database == "rec_engine"

    def test_default_kafka_config(self):
        """KafkaConfig defaults are reasonable."""
        kc = KafkaConfig()
        assert kc.bootstrap_servers == ["localhost:9092"]
        assert kc.compression_type == "snappy"


# ===================================================================
# Config.from_env()
# ===================================================================


class TestConfigFromEnv:
    """Test that environment variables override defaults."""

    @patch.dict(os.environ, {
        "REDIS_HOST": "redis-prod.example.com",
        "REDIS_PORT": "6380",
        "KAFKA_BOOTSTRAP_SERVERS": "broker1:9092,broker2:9092",
    }, clear=False)
    def test_config_from_env(self):
        """Config.from_env() picks up REDIS_HOST, REDIS_PORT, KAFKA_BOOTSTRAP_SERVERS."""
        config = Config.from_env()

        assert config.redis.host == "redis-prod.example.com"
        assert config.redis.port == 6380
        assert config.kafka.bootstrap_servers == ["broker1:9092", "broker2:9092"]

    @patch.dict(os.environ, {
        "DATABASE_URL": "postgresql://admin:secret@db.example.com:5433/mydb",
    }, clear=False)
    def test_config_from_env_database_url(self):
        """Config.from_env() parses DATABASE_URL into individual fields."""
        config = Config.from_env()

        assert config.database.username == "admin"
        assert config.database.password == "secret"
        assert config.database.host == "db.example.com"
        assert config.database.port == 5433
        assert config.database.database == "mydb"


# ===================================================================
# ModelConfig defaults
# ===================================================================


class TestModelConfigDefaults:
    """Test ModelConfig has the documented defaults."""

    def test_model_config_defaults(self):
        """embedding_dim=128, hidden_layers=[512, 256, 128]."""
        mc = ModelConfig()
        assert mc.embedding_dim == 128
        assert mc.hidden_layers == [512, 256, 128]
        assert mc.dropout_rate == 0.2
        assert mc.learning_rate == 0.001
        assert mc.batch_size == 2048
        assert mc.num_epochs == 100
        assert mc.num_negatives == 5
        assert mc.index_type == "scann"
