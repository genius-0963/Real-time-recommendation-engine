"""
Kafka consumer for real-time event processing.
Supports exactly-once semantics, backpressure handling, and dead-letter queues.
"""

import json
import time
import logging
import asyncio
from typing import Dict, List, Optional, Callable, Any, Union
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import signal
import sys

from confluent_kafka import Consumer, KafkaException, KafkaError, TopicPartition
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroDeserializer
from confluent_kafka.serialization import SerializationContext, MessageField

from app.config import KafkaConfig
from streaming.kafka_producer import KafkaEventProducer, UserEvent, FeatureUpdateEvent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConsumerMetrics:
    """Consumer performance metrics."""
    messages_consumed: int = 0
    messages_processed: int = 0
    messages_failed: int = 0
    processing_time_total: float = 0.0
    lag_max: int = 0
    last_commit_time: float = 0.0
    
    @property
    def avg_processing_time(self) -> float:
        return (self.processing_time_total / self.messages_processed 
                if self.messages_processed > 0 else 0.0)
    
    @property
    def success_rate(self) -> float:
        total = self.messages_consumed
        return (self.messages_processed / total if total > 0 else 0.0)


class EventDeserializer:
    """Event deserialization with schema validation."""
    
    def __init__(self, schema_registry_url: str):
        self.schema_registry_client = SchemaRegistryClient({'url': schema_registry_url})
        self.deserializers = {}
    
    def _get_avro_schema(self, event_type: str) -> str:
        """Get Avro schema for event type."""
        schemas = {
            'user_event': '''
            {
                "type": "record",
                "name": "UserEvent",
                "fields": [
                    {"name": "event_id", "type": "string"},
                    {"name": "user_id", "type": "string"},
                    {"name": "session_id", "type": "string"},
                    {"name": "event_type", "type": "string"},
                    {"name": "item_id", "type": "string"},
                    {"name": "item_category", "type": "string"},
                    {"name": "timestamp", "type": "string"},
                    {"name": "properties", "type": {"type": "map", "values": "string"}},
                    {"name": "device_info", "type": {"type": "map", "values": "string"}},
                    {"name": "location", "type": {"type": "map", "values": "string"}},
                    {"name": "referrer", "type": ["null", "string"], "default": null}
                ]
            }
            ''',
            'feature_update': '''
            {
                "type": "record",
                "name": "FeatureUpdateEvent",
                "fields": [
                    {"name": "event_id", "type": "string"},
                    {"name": "entity_type", "type": "string"},
                    {"name": "entity_id", "type": "string"},
                    {"name": "features", "type": {"type": "map", "values": "string"}},
                    {"name": "timestamp", "type": "string"},
                    {"name": "version", "type": "int"}
                ]
            }
            '''
        }
        return schemas.get(event_type)
    
    def deserialize(self, message_value: bytes, event_type: str) -> Dict[str, Any]:
        """Deserialize message to dictionary."""
        if event_type not in self.deserializers:
            schema_str = self._get_avro_schema(event_type)
            if schema_str:
                self.deserializers[event_type] = AvroDeserializer(
                    schema_registry_client=self.schema_registry_client,
                    schema_str=schema_str
                )
        
        if event_type in self.deserializers:
            return self.deserializers[event_type](
                message_value,
                SerializationContext(event_type, MessageField.VALUE)
            )
        else:
            # Fallback to JSON deserialization
            return json.loads(message_value.decode('utf-8'))


class DeadLetterQueue:
    """Dead letter queue for failed messages."""
    
    def __init__(self, producer: KafkaEventProducer, topic: str):
        self.producer = producer
        self.topic = topic
    
    def send_to_dlq(self, original_message: Dict[str, Any], 
                    error: Exception, topic: str, partition: int, offset: int):
        """Send failed message to dead letter queue."""
        dlq_message = {
            'original_message': original_message,
            'error': str(error),
            'error_type': type(error).__name__,
            'original_topic': topic,
            'original_partition': partition,
            'original_offset': offset,
            'failed_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Create a feature update event for DLQ
        dlq_event = FeatureUpdateEvent(
            event_id=f"dlq_{original_message.get('event_id', 'unknown')}",
            entity_type="dlq_message",
            entity_id=f"{topic}_{partition}_{offset}",
            features=dlq_message,
            timestamp=datetime.now(timezone.utc),
            version=1
        )
        
        self.producer.produce_feature_update(dlq_event, self.topic)


class KafkaEventConsumer:
    """High-performance Kafka consumer for real-time event processing."""
    
    def __init__(self, config: KafkaConfig, topics: List[str],
                 group_id: Optional[str] = None):
        self.config = config
        self.topics = topics
        self.group_id = group_id or config.consumer_group
        self.consumer = None
        self.deserializer = EventDeserializer(config.schema_registry_url)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Processing state
        self.running = False
        self.metrics = ConsumerMetrics()
        self.processing_handlers = {}
        
        # Dead letter queue
        self.dlq_producer = None
        self.dlq_topic = f"{config.topic_prefix}.dlq"
        
        # Backpressure control
        self.max_poll_records = config.max_poll_records
        self.processing_semaphore = asyncio.Semaphore(100)  # Max concurrent processing
        
        # Initialize consumer
        self._initialize_consumer()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _initialize_consumer(self):
        """Initialize Kafka consumer with configuration."""
        consumer_config = {
            'bootstrap.servers': ','.join(self.config.bootstrap_servers),
            'group.id': self.group_id,
            'client.id': f'rec-engine-consumer-{self.group_id}',
            'auto.offset.reset': self.config.auto_offset_reset,
            'enable.auto.commit': self.config.enable_auto_commit,
            'auto.commit.interval.ms': 1000 if self.config.enable_auto_commit else 0,
            'session.timeout.ms': self.config.session_timeout_ms,
            'heartbeat.interval.ms': self.config.heartbeat_interval_ms,
            'max.poll.records': self.max_poll_records,
            'max.poll.interval.ms': 300000,  # 5 minutes
            'fetch.min.bytes': 1,
            'fetch.max.wait.ms': 500,
            'enable.partition.eof': False,
            'isolation.level': 'read_committed',  # Exactly-once semantics
        }
        
        # Add security configuration if needed
        if self.config.security_protocol != "PLAINTEXT":
            consumer_config.update({
                'security.protocol': self.config.security_protocol,
                'sasl.mechanisms': self.config.sasl_mechanism,
                'sasl.username': self.config.sasl_username,
                'sasl.password': self.config.sasl_password
            })
        
        self.consumer = Consumer(consumer_config)
        
        # Subscribe to topics
        self.consumer.subscribe(self.topics)
        
        logger.info(f"Kafka consumer initialized for topics: {self.topics}")
    
    def register_handler(self, event_type: str, handler: Callable):
        """Register event handler for specific event type."""
        self.processing_handlers[event_type] = handler
        logger.info(f"Registered handler for event type: {event_type}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    async def _process_message(self, message) -> bool:
        """Process a single message."""
        start_time = time.time()
        
        try:
            # Extract message metadata
            topic = message.topic()
            partition = message.partition()
            offset = message.offset()
            key = message.key().decode('utf-8') if message.key() else None
            
            # Determine event type from topic or headers
            event_type = self._extract_event_type(topic, message.headers())
            
            # Deserialize message
            if event_type:
                event_data = self.deserializer.deserialize(message.value(), event_type)
            else:
                # Fallback to JSON
                event_data = json.loads(message.value().decode('utf-8'))
                event_type = event_data.get('event_type', 'unknown')
            
            # Process message
            if event_type in self.processing_handlers:
                handler = self.processing_handlers[event_type]
                if asyncio.iscoroutinefunction(handler):
                    await handler(event_data, topic, partition, offset)
                else:
                    await asyncio.get_event_loop().run_in_executor(
                        self.executor, handler, event_data, topic, partition, offset
                    )
            else:
                logger.warning(f"No handler registered for event type: {event_type}")
            
            self.metrics.messages_processed += 1
            
            return True
            
        except Exception as e:
            self.metrics.messages_failed += 1
            logger.error(f"Failed to process message: {str(e)}")
            
            # Send to dead letter queue
            if self.dlq_producer:
                try:
                    original_message = {
                        'key': message.key().decode('utf-8') if message.key() else None,
                        'value': message.value().decode('utf-8') if message.value() else None,
                        'headers': dict(message.headers()) if message.headers() else {}
                    }
                    self.dlq.send_to_dlq(original_message, e, topic, partition, offset)
                except Exception as dlq_error:
                    logger.error(f"Failed to send message to DLQ: {dlq_error}")
            
            return False
        
        finally:
            processing_time = time.time() - start_time
            self.metrics.processing_time_total += processing_time
    
    def _extract_event_type(self, topic: str, headers: Optional[List] = None) -> Optional[str]:
        """Extract event type from topic or headers."""
        # Extract from topic name
        if 'user-events' in topic:
            return 'user_event'
        elif 'feature-updates' in topic:
            return 'feature_update'
        elif 'interaction-events' in topic:
            return 'user_event'
        
        # Extract from headers
        if headers:
            for header_key, header_value in headers:
                if header_key == 'event_type':
                    return header_value.decode('utf-8')
        
        return None
    
    async def consume_messages(self, timeout: float = 1.0):
        """Consume and process messages."""
        while self.running:
            try:
                # Poll for messages
                msg = self.consumer.poll(timeout)
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logger.error(f"Consumer error: {msg.error()}")
                        continue
                
                self.metrics.messages_consumed += 1
                
                # Process message with backpressure control
                async with self.processing_semaphore:
                    await self._process_message(msg)
                
                # Commit offsets manually (exactly-once semantics)
                if not self.config.enable_auto_commit:
                    self.consumer.commit(asynchronous=False)
                    self.metrics.last_commit_time = time.time()
                
            except KafkaException as e:
                logger.error(f"Kafka exception: {e}")
                await asyncio.sleep(1)
            
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                await asyncio.sleep(1)
    
    def get_consumer_lag(self) -> Dict[int, int]:
        """Get consumer lag per partition."""
        try:
            # Get current assignments
            assignments = self.consumer.assignment()
            lag_info = {}
            
            for partition in assignments:
                # Get current position
                current_pos = self.consumer.position(partition)
                
                # Get high watermark (latest offset)
                _, high_watermark = self.consumer.get_watermark_offsets(partition)
                
                # Calculate lag
                lag = high_watermark - current_pos
                lag_info[partition.partition] = lag
                
                # Update max lag metric
                if lag > self.metrics.lag_max:
                    self.metrics.lag_max = lag
            
            return lag_info
            
        except Exception as e:
            logger.error(f"Failed to get consumer lag: {e}")
            return {}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get consumer metrics."""
        lag_info = self.get_consumer_lag()
        
        return {
            'messages_consumed': self.metrics.messages_consumed,
            'messages_processed': self.metrics.messages_processed,
            'messages_failed': self.metrics.messages_failed,
            'success_rate': self.metrics.success_rate,
            'avg_processing_time_ms': self.metrics.avg_processing_time * 1000,
            'lag_info': lag_info,
            'max_lag': self.metrics.lag_max,
            'last_commit_time': self.metrics.last_commit_time
        }
    
    def start_consuming(self):
        """Start consuming messages."""
        self.running = True
        logger.info("Starting message consumption")
        
        # Run async event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self.consume_messages())
        except KeyboardInterrupt:
            logger.info("Consumption interrupted by user")
        finally:
            loop.close()
            self.close()
    
    def close(self):
        """Close consumer and cleanup resources."""
        self.running = False
        
        try:
            # Final commit
            if not self.config.enable_auto_commit:
                self.consumer.commit(asynchronous=False)
            
            # Close consumer
            self.consumer.close()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            logger.info("Kafka consumer closed")
            
        except Exception as e:
            logger.error(f"Error closing consumer: {e}")


class EventProcessor:
    """Example event processor with handlers."""
    
    def __init__(self, feature_store_client=None):
        self.feature_store = feature_store_client
    
    async def handle_user_event(self, event_data: Dict[str, Any], 
                               topic: str, partition: int, offset: int):
        """Handle user interaction events."""
        logger.info(f"Processing user event: {event_data.get('event_type')} for user {event_data.get('user_id')}")
        
        # Update user features in real-time
        if self.feature_store:
            user_id = event_data.get('user_id')
            item_id = event_data.get('item_id')
            event_type = event_data.get('event_type')
            
            # Update user interaction history
            await self._update_user_interaction_history(user_id, item_id, event_type)
            
            # Update user preferences
            await self._update_user_preferences(user_id, event_data)
    
    async def handle_feature_update(self, event_data: Dict[str, Any],
                                   topic: str, partition: int, offset: int):
        """Handle feature update events."""
        logger.info(f"Processing feature update for {event_data.get('entity_type')} {event_data.get('entity_id')}")
        
        if self.feature_store:
            entity_type = event_data.get('entity_type')
            entity_id = event_data.get('entity_id')
            features = event_data.get('features')
            
            # Update features in store
            await self._update_features(entity_type, entity_id, features)
    
    async def _update_user_interaction_history(self, user_id: str, item_id: str, event_type: str):
        """Update user interaction history."""
        # Implementation would depend on feature store
        pass
    
    async def _update_user_preferences(self, user_id: str, event_data: Dict[str, Any]):
        """Update user preferences based on event."""
        # Implementation would depend on feature store
        pass
    
    async def _update_features(self, entity_type: str, entity_id: str, features: Dict[str, Any]):
        """Update features in feature store."""
        # Implementation would depend on feature store
        pass


# Example usage
def main():
    """Example usage of Kafka consumer."""
    from app.config import Config
    
    # Load configuration
    config = Config()
    
    # Initialize consumer
    topics = [
        f"{config.kafka.topic_prefix}.{config.kafka.user_events_topic}",
        f"{config.kafka.topic_prefix}.{config.kafka.feature_updates_topic}"
    ]
    
    consumer = KafkaEventConsumer(config.kafka, topics)
    
    # Initialize event processor
    processor = EventProcessor()
    
    # Register handlers
    consumer.register_handler('user_event', processor.handle_user_event)
    consumer.register_handler('feature_update', processor.handle_feature_update)
    
    # Start consuming
    consumer.start_consuming()


if __name__ == "__main__":
    main()
