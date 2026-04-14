"""
Schema registry management for Kafka event schemas.
Supports Avro schema evolution, validation, and compatibility checking.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime

from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer, AvroDeserializer
from confluent_kafka.schema_registry.error import SchemaRegistryError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SchemaInfo:
    """Schema information metadata."""
    name: str
    schema_type: str  # 'AVRO', 'JSON', 'PROTOBUF'
    schema_str: str
    version: int
    id: int
    compatibility_level: str = 'BACKWARD'
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class SchemaManager:
    """Manages Kafka schema registry operations."""
    
    def __init__(self, schema_registry_url: str):
        self.schema_registry_url = schema_registry_url
        self.client = SchemaRegistryClient({'url': schema_registry_url})
        self.schemas = {}  # Local cache of schemas
        
    def register_schema(self, subject: str, schema_str: str, 
                       schema_type: str = 'AVRO') -> int:
        """Register a new schema."""
        try:
            schema_id = self.client.register_schema(
                subject=subject,
                schema=schema_str,
                schema_type=schema_type
            )
            
            logger.info(f"Registered schema '{subject}' with ID: {schema_id}")
            
            # Cache schema locally
            self.schemas[subject] = SchemaInfo(
                name=subject,
                schema_type=schema_type,
                schema_str=schema_str,
                version=1,  # Will be updated when we get version info
                id=schema_id,
                created_at=datetime.now()
            )
            
            return schema_id
            
        except SchemaRegistryError as e:
            logger.error(f"Failed to register schema '{subject}': {e}")
            raise
    
    def get_schema(self, subject: str, version: Optional[int] = None) -> Optional[SchemaInfo]:
        """Get schema information."""
        try:
            if version:
                schema = self.client.get_version(subject, version)
            else:
                schema = self.client.get_latest_version(subject)
            
            schema_info = SchemaInfo(
                name=subject,
                schema_type=schema.schema.schema_type,
                schema_str=schema.schema.schema_str,
                version=schema.version,
                id=schema.schema_id,
                compatibility_level=schema.compatibility
            )
            
            # Cache locally
            self.schemas[subject] = schema_info
            
            return schema_info
            
        except SchemaRegistryError as e:
            logger.error(f"Failed to get schema '{subject}': {e}")
            return None
    
    def list_subjects(self) -> List[str]:
        """List all registered subjects."""
        try:
            return self.client.list_subjects()
        except SchemaRegistryError as e:
            logger.error(f"Failed to list subjects: {e}")
            return []
    
    def delete_subject(self, subject: str) -> bool:
        """Delete a subject and all its versions."""
        try:
            self.client.delete_subject(subject)
            
            # Remove from local cache
            if subject in self.schemas:
                del self.schemas[subject]
            
            logger.info(f"Deleted subject '{subject}'")
            return True
            
        except SchemaRegistryError as e:
            logger.error(f"Failed to delete subject '{subject}': {e}")
            return False
    
    def test_compatibility(self, subject: str, schema_str: str, 
                         version: Optional[int] = None) -> bool:
        """Test schema compatibility."""
        try:
            is_compatible = self.client.test_compatibility(
                subject=subject,
                schema=schema_str,
                version=version
            )
            
            logger.info(f"Schema compatibility for '{subject}': {is_compatible}")
            return is_compatible
            
        except SchemaRegistryError as e:
            logger.error(f"Failed to test compatibility for '{subject}': {e}")
            return False
    
    def update_compatibility(self, subject: str, compatibility_level: str) -> bool:
        """Update compatibility level for a subject."""
        try:
            self.client.update_compatibility(
                subject=subject,
                compatibility_level=compatibility_level
            )
            
            logger.info(f"Updated compatibility for '{subject}' to {compatibility_level}")
            return True
            
        except SchemaRegistryError as e:
            logger.error(f"Failed to update compatibility for '{subject}': {e}")
            return False
    
    def get_all_versions(self, subject: str) -> List[int]:
        """Get all versions of a subject."""
        try:
            return self.client.get_versions(subject)
        except SchemaRegistryError as e:
            logger.error(f"Failed to get versions for '{subject}': {e}")
            return []


class EventSchemaRegistry:
    """Pre-defined schemas for recommendation engine events."""
    
    # User Event Schema
    USER_EVENT_SCHEMA = '''
    {
        "type": "record",
        "name": "UserEvent",
        "namespace": "com.netflix.recengine",
        "doc": "User interaction event schema",
        "fields": [
            {
                "name": "event_id",
                "type": "string",
                "doc": "Unique identifier for the event"
            },
            {
                "name": "user_id",
                "type": "string",
                "doc": "User identifier"
            },
            {
                "name": "session_id",
                "type": "string",
                "doc": "Session identifier"
            },
            {
                "name": "event_type",
                "type": {
                    "type": "enum",
                    "name": "EventType",
                    "symbols": ["view", "click", "purchase", "like", "share", "add_to_cart", "remove_from_cart"]
                },
                "doc": "Type of user interaction"
            },
            {
                "name": "item_id",
                "type": "string",
                "doc": "Item identifier"
            },
            {
                "name": "item_category",
                "type": "string",
                "doc": "Item category"
            },
            {
                "name": "timestamp",
                "type": {
                    "type": "long",
                    "logicalType": "timestamp-micros"
                },
                "doc": "Event timestamp in microseconds"
            },
            {
                "name": "properties",
                "type": {
                    "type": "map",
                    "values": "string"
                },
                "doc": "Additional event properties"
            },
            {
                "name": "device_info",
                "type": {
                    "type": "record",
                    "name": "DeviceInfo",
                    "fields": [
                        {"name": "type", "type": "string", "doc": "Device type (web, mobile, tablet)"},
                        {"name": "os", "type": "string", "doc": "Operating system"},
                        {"name": "version", "type": "string", "doc": "App/browser version"},
                        {"name": "screen_resolution", "type": ["null", "string"], "default": null}
                    ]
                },
                "doc": "Device information"
            },
            {
                "name": "location",
                "type": {
                    "type": "record",
                    "name": "Location",
                    "fields": [
                        {"name": "country", "type": "string", "doc": "Country code"},
                        {"name": "region", "type": ["null", "string"], "default": null, "doc": "Region/state"},
                        {"name": "city", "type": ["null", "string"], "default": null, "doc": "City"},
                        {"name": "timezone", "type": ["null", "string"], "default": null, "doc": "Timezone"}
                    ]
                },
                "doc": "Location information"
            },
            {
                "name": "referrer",
                "type": ["null", "string"],
                "default": null,
                "doc": "Referrer URL or source"
            },
            {
                "name": "value",
                "type": ["null", "double"],
                "default": null,
                "doc": "Event value (e.g., purchase amount)"
            }
        ]
    }
    '''
    
    # Feature Update Schema
    FEATURE_UPDATE_SCHEMA = '''
    {
        "type": "record",
        "name": "FeatureUpdateEvent",
        "namespace": "com.netflix.recengine",
        "doc": "Feature update event schema",
        "fields": [
            {
                "name": "event_id",
                "type": "string",
                "doc": "Unique identifier for the event"
            },
            {
                "name": "entity_type",
                "type": {
                    "type": "enum",
                    "name": "EntityType",
                    "symbols": ["user", "item", "category", "session"]
                },
                "doc": "Type of entity being updated"
            },
            {
                "name": "entity_id",
                "type": "string",
                "doc": "Entity identifier"
            },
            {
                "name": "features",
                "type": {
                    "type": "map",
                    "values": "string"
                },
                "doc": "Feature name-value pairs"
            },
            {
                "name": "timestamp",
                "type": {
                    "type": "long",
                    "logicalType": "timestamp-micros"
                },
                "doc": "Update timestamp in microseconds"
            },
            {
                "name": "version",
                "type": "int",
                "doc": "Feature version"
            },
            {
                "name": "update_type",
                "type": {
                    "type": "enum",
                    "name": "UpdateType",
                    "symbols": ["incremental", "full", "delete"]
                },
                "default": "incremental",
                "doc": "Type of update"
            },
            {
                "name": "ttl_seconds",
                "type": ["null", "int"],
                "default": null,
                "doc": "Time-to-live in seconds"
            }
        ]
    }
    '''
    
    # Model Update Schema
    MODEL_UPDATE_SCHEMA = '''
    {
        "type": "record",
        "name": "ModelUpdateEvent",
        "namespace": "com.netflix.recengine",
        "doc": "Model update event schema",
        "fields": [
            {
                "name": "event_id",
                "type": "string",
                "doc": "Unique identifier for the event"
            },
            {
                "name": "model_name",
                "type": "string",
                "doc": "Model name"
            },
            {
                "name": "model_version",
                "type": "string",
                "doc": "Model version"
            },
            {
                "name": "update_type",
                "type": {
                    "type": "enum",
                    "name": "ModelUpdateType",
                    "symbols": ["full_retrain", "incremental", "parameters_only", "architecture_change"]
                },
                "doc": "Type of model update"
            },
            {
                "name": "model_metadata",
                "type": {
                    "type": "map",
                    "values": "string"
                },
                "doc": "Model metadata"
            },
            {
                "name": "performance_metrics",
                "type": {
                    "type": "map",
                    "values": "double"
                },
                "doc": "Performance metrics"
            },
            {
                "name": "timestamp",
                "type": {
                    "type": "long",
                    "logicalType": "timestamp-micros"
                },
                "doc": "Update timestamp in microseconds"
            },
            {
                "name": "deployment_status",
                "type": {
                    "type": "enum",
                    "name": "DeploymentStatus",
                    "symbols": ["pending", "deploying", "active", "failed", "rolled_back"]
                },
                "doc": "Deployment status"
            }
        ]
    }
    '''
    
    # Recommendation Request Schema
    RECOMMENDATION_REQUEST_SCHEMA = '''
    {
        "type": "record",
        "name": "RecommendationRequest",
        "namespace": "com.netflix.recengine",
        "doc": "Recommendation request schema",
        "fields": [
            {
                "name": "request_id",
                "type": "string",
                "doc": "Unique request identifier"
            },
            {
                "name": "user_id",
                "type": "string",
                "doc": "User identifier"
            },
            {
                "name": "session_id",
                "type": "string",
                "doc": "Session identifier"
            },
            {
                "name": "request_context",
                "type": {
                    "type": "record",
                    "name": "RequestContext",
                    "fields": [
                        {"name": "page_type", "type": "string", "doc": "Page type"},
                        {"name": "num_recommendations", "type": "int", "default": 10, "doc": "Number of recommendations requested"},
                        {"name": "filters", "type": {"type": "map", "values": "string"}, "doc": "Request filters"},
                        {"name": "candidate_pool", "type": ["null", {"type": "array", "items": "string"}], "default": null, "doc": "Specific candidate items"}
                    ]
                },
                "doc": "Request context"
            },
            {
                "name": "timestamp",
                "type": {
                    "type": "long",
                    "logicalType": "timestamp-micros"
                },
                "doc": "Request timestamp in microseconds"
            },
            {
                "name": "ab_test_info",
                "type": ["null", {
                    "type": "record",
                    "name": "ABTestInfo",
                    "fields": [
                        {"name": "experiment_id", "type": "string"},
                        {"name": "variant", "type": "string"},
                        {"name": "traffic_split", "type": "double"}
                    ]
                }],
                "default": null,
                "doc": "A/B test information"
            }
        ]
    }
    '''
    
    # Recommendation Response Schema
    RECOMMENDATION_RESPONSE_SCHEMA = '''
    {
        "type": "record",
        "name": "RecommendationResponse",
        "namespace": "com.netflix.recengine",
        "doc": "Recommendation response schema",
        "fields": [
            {
                "name": "request_id",
                "type": "string",
                "doc": "Original request identifier"
            },
            {
                "name": "recommendations",
                "type": {
                    "type": "array",
                    "items": {
                        "type": "record",
                        "name": "RecommendationItem",
                        "fields": [
                            {"name": "item_id", "type": "string", "doc": "Item identifier"},
                            {"name": "score", "type": "double", "doc": "Recommendation score"},
                            {"name": "rank", "type": "int", "doc": "Recommendation rank"},
                            {"name": "explanation", "type": ["null", "string"], "default": null, "doc": "Explanation for recommendation"},
                            {"name": "model_version", "type": "string", "doc": "Model version used"}
                        ]
                    }
                },
                "doc": "List of recommended items"
            },
            {
                "name": "response_metadata",
                "type": {
                    "type": "record",
                    "name": "ResponseMetadata",
                    "fields": [
                        {"name": "model_name", "type": "string", "doc": "Model name"},
                        {"name": "model_version", "type": "string", "doc": "Model version"},
                        {"name": "latency_ms", "type": "double", "doc": "Response latency in milliseconds"},
                        {"name": "cache_hit", "type": "boolean", "doc": "Whether response was from cache"},
                        {"name": "candidate_pool_size", "type": "int", "doc": "Size of candidate pool"}
                    ]
                },
                "doc": "Response metadata"
            },
            {
                "name": "timestamp",
                "type": {
                    "type": "long",
                    "logicalType": "timestamp-micros"
                },
                "doc": "Response timestamp in microseconds"
            }
        ]
    }
    '''


class SchemaInitializer:
    """Initialize schemas in schema registry."""
    
    def __init__(self, schema_manager: SchemaManager):
        self.schema_manager = schema_manager
    
    def initialize_all_schemas(self, topic_prefix: str = "rec-engine"):
        """Initialize all required schemas."""
        schemas_to_register = [
            (f"{topic_prefix}-user-events-value", EventSchemaRegistry.USER_EVENT_SCHEMA),
            (f"{topic_prefix}-feature-updates-value", EventSchemaRegistry.FEATURE_UPDATE_SCHEMA),
            (f"{topic_prefix}-model-updates-value", EventSchemaRegistry.MODEL_UPDATE_SCHEMA),
            (f"{topic_prefix}-recommendation-requests-value", EventSchemaRegistry.RECOMMENDATION_REQUEST_SCHEMA),
            (f"{topic_prefix}-recommendation-responses-value", EventSchemaRegistry.RECOMMENDATION_RESPONSE_SCHEMA),
        ]
        
        registered_schemas = []
        
        for subject, schema_str in schemas_to_register:
            try:
                schema_id = self.schema_manager.register_schema(subject, schema_str)
                registered_schemas.append((subject, schema_id))
                
                # Set compatibility level
                self.schema_manager.update_compatibility(subject, 'BACKWARD')
                
                logger.info(f"Successfully registered schema: {subject} (ID: {schema_id})")
                
            except Exception as e:
                logger.error(f"Failed to register schema {subject}: {e}")
        
        return registered_schemas
    
    def verify_schemas(self, topic_prefix: str = "rec-engine") -> Dict[str, bool]:
        """Verify that all schemas are properly registered."""
        expected_subjects = [
            f"{topic_prefix}-user-events-value",
            f"{topic_prefix}-feature-updates-value",
            f"{topic_prefix}-model-updates-value",
            f"{topic_prefix}-recommendation-requests-value",
            f"{topic_prefix}-recommendation-responses-value",
        ]
        
        verification_results = {}
        
        for subject in expected_subjects:
            schema_info = self.schema_manager.get_schema(subject)
            verification_results[subject] = schema_info is not None
            
            if schema_info:
                logger.info(f"✓ Schema verified: {subject} (v{schema_info.version})")
            else:
                logger.error(f"✗ Schema missing: {subject}")
        
        return verification_results


# Example usage
def main():
    """Example usage of schema registry."""
    from app.config import Config
    
    # Load configuration
    config = Config()
    
    # Initialize schema manager
    schema_manager = SchemaManager(config.kafka.schema_registry_url)
    
    # Initialize schemas
    initializer = SchemaInitializer(schema_manager)
    
    # Register all schemas
    registered = initializer.initialize_all_schemas()
    logger.info(f"Registered {len(registered)} schemas")
    
    # Verify schemas
    verification = initializer.verify_schemas()
    logger.info(f"Schema verification: {verification}")
    
    # List all subjects
    subjects = schema_manager.list_subjects()
    logger.info(f"All subjects: {subjects}")


if __name__ == "__main__":
    main()
