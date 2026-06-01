"""
Kafka producer for real-time user interaction events.
Supports schema validation, batching, and exactly-once semantics.
"""

import json
import time
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import uuid

from confluent_kafka import Producer, KafkaException, KafkaError
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer
from confluent_kafka.serialization import SerializationContext, MessageField

from app.config import KafkaConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UserEvent:
    """User interaction event schema."""
    event_id: str
    user_id: str
    session_id: str
    event_type: str  # 'view', 'click', 'purchase', 'like', 'share'
    item_id: str
    item_category: str
    timestamp: datetime
    properties: Dict[str, Any]  # Additional event properties
    device_info: Dict[str, str]
    location: Dict[str, str]
    referrer: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary with timestamp as ISO string."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class FeatureUpdateEvent:
    """Feature update event schema."""
    event_id: str
    entity_type: str  # 'user' or 'item'
    entity_id: str
    features: Dict[str, Any]
    timestamp: datetime
    version: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary with timestamp as ISO string."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


class EventSerializer:
    """Event serialization with schema validation."""
    
    def __init__(self, schema_registry_url: str):
        self.schema_registry_client = SchemaRegistryClient({'url': schema_registry_url})
        self.serializers = {}
        
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
    
    def serialize(self, event: Union[UserEvent, FeatureUpdateEvent], 
                  event_type: str) -> bytes:
        """Serialize event to bytes."""
        if event_type not in self.serializers:
            schema_str = self._get_avro_schema(event_type)
            if schema_str:
                self.serializers[event_type] = AvroSerializer(
                    schema_registry_client=self.schema_registry_client,
                    schema_str=schema_str
                )
        
        if event_type in self.serializers:
            return self.serializers[event_type](
                event.to_dict(),
                SerializationContext(event_type, MessageField.VALUE)
            )
        else:
            # Fallback to JSON serialization
            return json.dumps(event.to_dict()).encode('utf-8')


class KafkaEventProducer:
    """High-performance Kafka producer for real-time events."""
    
    def __init__(self, config: KafkaConfig):
        self.config = config
        self.producer = None
        self.serializer = EventSerializer(config.schema_registry_url)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Metrics
        self.produced_count = 0
        self.error_count = 0
        self.last_flush_time = time.time()
        
        # Initialize producer
        self._initialize_producer()
    
    def _initialize_producer(self):
        """Initialize Kafka producer with configuration."""
        producer_config = {
            'bootstrap.servers': ','.join(self.config.bootstrap_servers),
            'client.id': f'rec-engine-producer-{uuid.uuid4().hex[:8]}',
            'acks': 'all',  # Wait for all replicas for durability
            'retries': 3,
            'retry.backoff.ms': 100,
            'batch.size': 16384,  # 16KB batches
            'linger.ms': 5,  # Wait up to 5ms for batching
            'compression.type': self.config.compression_type,
            'max.in.flight.requests.per.connection': 5,
            'enable.idempotence': True,  # Exactly-once semantics
            'message.send.max.retries': 3,
            'queue.buffering.max.messages': 100000,
            'queue.buffering.max.kbytes': 10240,  # 10MB
            'socket.send.buffer.bytes': 102400,
            'socket.receive.buffer.bytes': 102400,
        }
        
        # Add security configuration if needed
        if self.config.security_protocol != "PLAINTEXT":
            producer_config.update({
                'security.protocol': self.config.security_protocol,
                'sasl.mechanisms': self.config.sasl_mechanism,
                'sasl.username': self.config.sasl_username,
                'sasl.password': self.config.sasl_password
            })
        
        self.producer = Producer(producer_config)
        
        # Set up delivery callback
        self.producer.set_delivery_callback(self._delivery_callback)
        
        logger.info("Kafka producer initialized")
    
    def _delivery_callback(self, err, msg):
        """Callback for message delivery reports."""
        if err is not None:
            self.error_count += 1
            logger.error(f"Message delivery failed: {err}")
        else:
            self.produced_count += 1
            if self.produced_count % 1000 == 0:
                logger.info(f"Produced {self.produced_count} messages")
    
    def produce_user_event(self, event: UserEvent, 
                          topic: Optional[str] = None) -> bool:
        """Produce user interaction event."""
        try:
            topic_name = topic or f"{self.config.topic_prefix}.{self.config.user_events_topic}"
            
            # Serialize event
            value = self.serializer.serialize(event, 'user_event')
            
            # Create message key for partitioning (user_id for consistent ordering)
            key = event.user_id.encode('utf-8')
            
            # Produce message
            self.producer.produce(
                topic=topic_name,
                key=key,
                value=value,
                headers={
                    'event_type': event.event_type.encode('utf-8'),
                    'producer_version': '1.0.0'.encode('utf-8')
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to produce user event: {str(e)}")
            return False
    
    def produce_feature_update(self, event: FeatureUpdateEvent,
                              topic: Optional[str] = None) -> bool:
        """Produce feature update event."""
        try:
            topic_name = topic or f"{self.config.topic_prefix}.{self.config.feature_updates_topic}"
            
            # Serialize event
            value = self.serializer.serialize(event, 'feature_update')
            
            # Create message key
            key = f"{event.entity_type}:{event.entity_id}".encode('utf-8')
            
            # Produce message
            self.producer.produce(
                topic=topic_name,
                key=key,
                value=value,
                headers={
                    'entity_type': event.entity_type.encode('utf-8'),
                    'version': str(event.version).encode('utf-8')
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to produce feature update event: {str(e)}")
            return False
    
    def produce_batch(self, events: List[Union[UserEvent, FeatureUpdateEvent]],
                     topic: Optional[str] = None) -> int:
        """Produce a batch of events."""
        successful_count = 0
        
        for event in events:
            if isinstance(event, UserEvent):
                if self.produce_user_event(event, topic):
                    successful_count += 1
            elif isinstance(event, FeatureUpdateEvent):
                if self.produce_feature_update(event, topic):
                    successful_count += 1
        
        return successful_count
    
    def flush(self, timeout: float = 10.0) -> int:
        """Flush pending messages."""
        try:
            remaining = self.producer.flush(timeout)
            if remaining > 0:
                logger.warning(f"{remaining} messages still pending after flush")
            return remaining
        except KafkaException as e:
            logger.error(f"Flush failed: {e}")
            return -1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get producer metrics."""
        return {
            'produced_count': self.produced_count,
            'error_count': self.error_count,
            'success_rate': (self.produced_count / (self.produced_count + self.error_count) 
                           if (self.produced_count + self.error_count) > 0 else 0),
            'last_flush_time': self.last_flush_time
        }
    
    def close(self):
        """Close producer and cleanup resources."""
        try:
            self.flush(timeout=30)
            self.executor.shutdown(wait=True)
            logger.info("Kafka producer closed")
        except Exception as e:
            logger.error(f"Error closing producer: {e}")


class AsyncKafkaProducer:
    """Async wrapper for Kafka producer."""
    
    def __init__(self, config: KafkaConfig):
        self.producer = KafkaEventProducer(config)
        self.loop = asyncio.get_event_loop()
    
    async def produce_user_event_async(self, event: UserEvent) -> bool:
        """Async produce user event."""
        return await self.loop.run_in_executor(
            self.producer.executor,
            self.producer.produce_user_event,
            event
        )
    
    async def produce_feature_update_async(self, event: FeatureUpdateEvent) -> bool:
        """Async produce feature update event."""
        return await self.loop.run_in_executor(
            self.producer.executor,
            self.producer.produce_feature_update,
            event
        )
    
    async def produce_batch_async(self, events: List[Union[UserEvent, FeatureUpdateEvent]]) -> int:
        """Async produce batch of events."""
        return await self.loop.run_in_executor(
            self.producer.executor,
            self.producer.produce_batch,
            events
        )
    
    async def flush_async(self, timeout: float = 10.0) -> int:
        """Async flush pending messages."""
        return await self.loop.run_in_executor(
            self.producer.executor,
            self.producer.flush,
            timeout
        )
    
    async def close_async(self):
        """Async close producer."""
        await self.loop.run_in_executor(
            self.producer.executor,
            self.producer.close
        )


class EventGenerator:
    """Utility class for generating sample events."""
    
    @staticmethod
    def create_user_event(user_id: str, item_id: str, event_type: str,
                         session_id: Optional[str] = None,
                         properties: Optional[Dict[str, Any]] = None) -> UserEvent:
        """Create a user event."""
        return UserEvent(
            event_id=str(uuid.uuid4()),
            user_id=user_id,
            session_id=session_id or str(uuid.uuid4()),
            event_type=event_type,
            item_id=item_id,
            item_category="electronics",  # Would be looked up in production
            timestamp=datetime.now(timezone.utc),
            properties=properties or {},
            device_info={
                "type": "web",
                "os": "iOS",
                "version": "16.0"
            },
            location={
                "country": "US",
                "city": "San Francisco"
            }
        )
    
    @staticmethod
    def create_feature_update_event(entity_type: str, entity_id: str,
                                   features: Dict[str, Any],
                                   version: int = 1) -> FeatureUpdateEvent:
        """Create a feature update event."""
        return FeatureUpdateEvent(
            event_id=str(uuid.uuid4()),
            entity_type=entity_type,
            entity_id=entity_id,
            features=features,
            timestamp=datetime.now(timezone.utc),
            version=version
        )


# Example usage and testing
def main():
    """Example usage of Kafka producer."""
    from app.config import Config
    
    # Load configuration
    config = Config()
    
    # Initialize producer
    producer = KafkaEventProducer(config.kafka)
    
    try:
        # Create sample events
        events = []
        for i in range(10):
            event = EventGenerator.create_user_event(
                user_id=f"user_{i}",
                item_id=f"item_{i % 100}",
                event_type="click",
                properties={"duration": 120, "scroll_depth": 0.8}
            )
            events.append(event)
        
        # Produce events
        successful = producer.produce_batch(events)
        logger.info(f"Successfully produced {successful} events")
        
        # Flush messages
        remaining = producer.flush()
        logger.info(f"Remaining messages after flush: {remaining}")
        
        # Print metrics
        metrics = producer.get_metrics()
        logger.info(f"Producer metrics: {metrics}")
        
    finally:
        producer.close()


if __name__ == "__main__":
    main()
