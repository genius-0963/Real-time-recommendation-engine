"""
Unit tests for the Kafka streaming layer with mocked Kafka dependencies.

Source files:
  - streaming/kafka_consumer.py → ConsumerMetrics, EventDeserializer, DeadLetterQueue, KafkaEventConsumer
  - streaming/kafka_producer.py → KafkaEventProducer, EventSerializer, UserEvent, FeatureUpdateEvent
"""

import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timezone

from streaming.kafka_consumer import ConsumerMetrics, EventDeserializer, DeadLetterQueue, KafkaEventConsumer
from streaming.kafka_producer import UserEvent, FeatureUpdateEvent, EventGenerator


# ===================================================================
# EventDeserializer — JSON fallback
# ===================================================================


class TestEventDeserializerJsonFallback:
    """Test that the deserializer falls back to JSON when Avro is unavailable."""

    @patch("streaming.kafka_consumer.SchemaRegistryClient")
    def test_event_deserializer_json_fallback(self, MockSRC):
        """When no Avro deserializer is cached, raw JSON bytes are decoded."""
        deserializer = EventDeserializer.__new__(EventDeserializer)
        deserializer.schema_registry_client = MockSRC.return_value
        deserializer.deserializers = {}  # empty → no Avro available

        payload = {"event_id": "e1", "user_id": "u1", "event_type": "click"}
        raw_bytes = json.dumps(payload).encode("utf-8")

        # Use an event_type not in the schemas dict so Avro path is skipped
        result = deserializer.deserialize(raw_bytes, "unknown_type")

        assert result == payload
        assert result["user_id"] == "u1"


# ===================================================================
# DeadLetterQueue
# ===================================================================


class TestDeadLetterQueue:
    """Test that DLQ forwards failed messages to the producer."""

    def test_dead_letter_queue(self):
        """send_to_dlq calls produce_feature_update on the underlying producer."""
        mock_producer = MagicMock()
        dlq_topic = "rec-engine.dlq"
        dlq = DeadLetterQueue(producer=mock_producer, topic=dlq_topic)

        original_message = {"event_id": "e1", "user_id": "u1"}
        error = ValueError("bad data")

        dlq.send_to_dlq(
            original_message=original_message,
            error=error,
            topic="user-events",
            partition=0,
            offset=42,
        )

        mock_producer.produce_feature_update.assert_called_once()
        call_args = mock_producer.produce_feature_update.call_args
        # The second positional (or keyword) argument should be the DLQ topic
        assert call_args[0][1] == dlq_topic or call_args[1].get("topic") == dlq_topic or call_args.args[1] == dlq_topic


# ===================================================================
# ConsumerMetrics
# ===================================================================


class TestConsumerMetrics:
    """Test the ConsumerMetrics dataclass computed properties."""

    def test_consumer_metrics_avg_processing_time(self):
        """avg_processing_time = total_time / processed_count."""
        m = ConsumerMetrics(
            messages_consumed=100,
            messages_processed=80,
            messages_failed=20,
            processing_time_total=8.0,  # seconds
        )
        assert m.avg_processing_time == pytest.approx(0.1)

    def test_consumer_metrics_avg_processing_time_zero(self):
        """avg_processing_time is 0 when nothing has been processed."""
        m = ConsumerMetrics()
        assert m.avg_processing_time == 0.0

    def test_consumer_metrics_success_rate(self):
        """success_rate = processed / consumed."""
        m = ConsumerMetrics(
            messages_consumed=200,
            messages_processed=190,
            messages_failed=10,
        )
        assert m.success_rate == pytest.approx(190 / 200)

    def test_consumer_metrics_success_rate_zero(self):
        """success_rate is 0 when nothing consumed."""
        m = ConsumerMetrics()
        assert m.success_rate == 0.0


# ===================================================================
# Consumer handler registration
# ===================================================================


class TestConsumerRegisterHandler:
    """Test that registering handlers populates the handler dict."""

    @patch("streaming.kafka_consumer.Consumer")
    @patch("streaming.kafka_consumer.SchemaRegistryClient")
    @patch("streaming.kafka_consumer.signal.signal")
    def test_consumer_register_handler(self, mock_signal, mock_src, MockConsumer):
        """register_handler adds the callable to processing_handlers."""
        from app.config import KafkaConfig

        config = KafkaConfig()
        topics = ["test-topic"]

        consumer = KafkaEventConsumer(config, topics, group_id="test-group")

        handler_fn = MagicMock()
        consumer.register_handler("user_event", handler_fn)

        assert "user_event" in consumer.processing_handlers
        assert consumer.processing_handlers["user_event"] is handler_fn
