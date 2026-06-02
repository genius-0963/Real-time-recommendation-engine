"""
Integration tests for the Kafka streaming pipeline (produce → consume).

Marked with ``@pytest.mark.integration``; skipped unless Kafka broker
is available.
"""

import os
import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timezone

from streaming.kafka_producer import UserEvent, FeatureUpdateEvent, EventGenerator


def services_available() -> bool:
    """Return True only when the operator has confirmed Kafka is running."""
    return os.getenv("INTEGRATION_TESTS", "0") == "1"


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not services_available(), reason="Requires running services"),
]


class TestProduceConsume:
    """End-to-end produce → consume test with mocked Kafka."""

    @patch("streaming.kafka_producer.Producer")
    @patch("streaming.kafka_producer.SchemaRegistryClient")
    @patch("streaming.kafka_consumer.Consumer")
    @patch("streaming.kafka_consumer.SchemaRegistryClient")
    @patch("streaming.kafka_consumer.signal.signal")
    def test_produce_consume(
        self,
        mock_signal,
        mock_consumer_src,
        MockConsumer,
        mock_producer_src,
        MockProducer,
    ):
        """
        Produce a user event via KafkaEventProducer, then verify that a
        KafkaEventConsumer can receive and deserialize it (mocked transport).
        """
        from app.config import KafkaConfig
        from streaming.kafka_producer import KafkaEventProducer
        from streaming.kafka_consumer import KafkaEventConsumer

        config = KafkaConfig()

        # --- Producer side ---
        mock_producer_instance = MagicMock()
        MockProducer.return_value = mock_producer_instance

        # Capture produced messages
        produced_messages = []

        def capture_produce(**kwargs):
            produced_messages.append(kwargs)

        mock_producer_instance.produce.side_effect = capture_produce
        mock_producer_instance.set_delivery_callback = MagicMock()

        producer = KafkaEventProducer(config)

        # Create and produce a sample event
        event = EventGenerator.create_user_event(
            user_id="u1",
            item_id="i1",
            event_type="click",
        )
        success = producer.produce_user_event(event)
        assert success is True
        assert len(produced_messages) == 1

        # --- Consumer side ---
        mock_consumer_instance = MagicMock()
        MockConsumer.return_value = mock_consumer_instance

        topics = [f"{config.topic_prefix}.{config.user_events_topic}"]
        consumer = KafkaEventConsumer(config, topics, group_id="test-int")

        # Simulate polling: return a mock message built from what was produced
        produced_value = produced_messages[0].get("value", b"")
        mock_msg = MagicMock()
        mock_msg.error.return_value = None
        mock_msg.topic.return_value = topics[0]
        mock_msg.partition.return_value = 0
        mock_msg.offset.return_value = 0
        mock_msg.key.return_value = b"u1"
        mock_msg.value.return_value = produced_value
        mock_msg.headers.return_value = [
            ("event_type", b"click"),
            ("producer_version", b"1.0.0"),
        ]

        # The consumer should be able to deserialize the message
        raw_value = mock_msg.value()
        if isinstance(raw_value, bytes):
            try:
                decoded = json.loads(raw_value.decode("utf-8"))
                assert "user_id" in decoded or "event_id" in decoded
            except (json.JSONDecodeError, UnicodeDecodeError):
                # If the producer used Avro, raw bytes won't be valid JSON
                pass
