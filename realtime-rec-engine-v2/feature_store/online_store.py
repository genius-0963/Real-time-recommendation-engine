"""
Online feature store using Redis for real-time feature serving.
Supports TTL management, versioning, and high-throughput operations.
"""

import json
import logging
import time
import pickle
import gzip
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
import asyncio
import aioredis
from redis.cluster import RedisCluster
import redis
import numpy as np
import pandas as pd

from app.config import RedisConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FeatureValue:
    """Feature value with metadata."""
    value: Any
    timestamp: datetime
    version: int
    ttl_seconds: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'value': self._serialize_value(self.value),
            'timestamp': self.timestamp.isoformat(),
            'version': self.version,
            'ttl_seconds': self.ttl_seconds
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureValue':
        """Create from dictionary."""
        return cls(
            value=cls._deserialize_value(data['value']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            version=data['version'],
            ttl_seconds=data.get('ttl_seconds')
        )
    
    @staticmethod
    def _serialize_value(value: Any) -> str:
        """Serialize value for storage."""
        if isinstance(value, (dict, list, str, int, float, bool)):
            return json.dumps(value)
        elif isinstance(value, np.ndarray):
            return pickle.dumps(value).hex()
        else:
            return str(value)
    
    @staticmethod
    def _deserialize_value(serialized: str) -> Any:
        """Deserialize value from storage."""
        try:
            # Try JSON first
            return json.loads(serialized)
        except (json.JSONDecodeError, ValueError):
            try:
                # Try pickle
                return pickle.loads(bytes.fromhex(serialized))
            except (ValueError, pickle.UnpicklingError):
                # Return as string
                return serialized


class OnlineFeatureStore:
    """Redis-based online feature store."""
    
    def __init__(self, config: RedisConfig):
        self.config = config
        self.redis_client = None
        self.async_redis_client = None
        
        # Key prefixes
        self.FEATURE_PREFIX = "feature:"
        self.ENTITY_PREFIX = "entity:"
        self.VERSION_PREFIX = "version:"
        self.METADATA_PREFIX = "meta:"
        
        # Initialize connections
        self._initialize_connections()
    
    def _initialize_connections(self):
        """Initialize Redis connections."""
        if self.config.cluster_enabled:
            # Redis Cluster configuration
            self.redis_client = RedisCluster(
                startup_nodes=self.config.cluster_nodes,
                decode_responses=False,
                skip_full_coverage_check=True,
                max_connections=self.config.max_connections
            )
        else:
            # Single Redis instance
            self.redis_client = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                decode_responses=False,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                retry_on_timeout=self.config.retry_on_timeout,
                health_check_interval=self.config.health_check_interval
            )
        
        logger.info("Redis connection initialized")
    
    async def _initialize_async_connection(self):
        """Initialize async Redis connection."""
        if self.config.cluster_enabled:
            # Async cluster connection (requires aioredis-cluster)
            # For now, use single node for async
            self.async_redis_client = await aioredis.from_url(
                f"redis://{self.config.host}:{self.config.port}/{self.config.db}",
                password=self.config.password,
                max_connections=self.config.max_connections
            )
        else:
            self.async_redis_client = await aioredis.from_url(
                f"redis://{self.config.host}:{self.config.port}/{self.config.db}",
                password=self.config.password,
                max_connections=self.config.max_connections
            )
        
        logger.info("Async Redis connection initialized")
    
    def _make_key(self, entity_type: str, entity_id: str, feature_name: str) -> str:
        """Create Redis key for feature."""
        return f"{self.FEATURE_PREFIX}{self.ENTITY_PREFIX}{entity_type}:{entity_id}:{feature_name}"
    
    def _make_entity_key(self, entity_type: str, entity_id: str) -> str:
        """Create Redis key for all features of an entity."""
        return f"{self.FEATURE_PREFIX}{self.ENTITY_PREFIX}{entity_type}:{entity_id}"
    
    def _make_version_key(self, entity_type: str, entity_id: str, feature_name: str) -> str:
        """Create Redis key for feature version tracking."""
        return f"{self.VERSION_PREFIX}{self.ENTITY_PREFIX}{entity_type}:{entity_id}:{feature_name}"
    
    def _make_metadata_key(self, entity_type: str, entity_id: str) -> str:
        """Create Redis key for entity metadata."""
        return f"{self.METADATA_PREFIX}{self.ENTITY_PREFIX}{entity_type}:{entity_id}"
    
    def set_feature(self, entity_type: str, entity_id: str, feature_name: str,
                   value: Any, ttl_seconds: Optional[int] = None,
                   version: Optional[int] = None) -> bool:
        """Set a single feature value."""
        try:
            # Create feature value
            feature_value = FeatureValue(
                value=value,
                timestamp=datetime.now(timezone.utc),
                version=version or int(time.time()),
                ttl_seconds=ttl_seconds or self._get_default_ttl(entity_type)
            )
            
            # Store feature
            key = self._make_key(entity_type, entity_id, feature_name)
            serialized_value = json.dumps(feature_value.to_dict())
            
            # Set with TTL
            if feature_value.ttl_seconds:
                self.redis_client.setex(key, feature_value.ttl_seconds, serialized_value)
            else:
                self.redis_client.set(key, serialized_value)
            
            # Update version tracking
            version_key = self._make_version_key(entity_type, entity_id, feature_name)
            self.redis_client.set(version_key, feature_value.version)
            
            # Update entity metadata
            self._update_entity_metadata(entity_type, entity_id, feature_name)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set feature {entity_type}:{entity_id}:{feature_name}: {e}")
            return False
    
    def set_features(self, entity_type: str, entity_id: str, 
                    features: Dict[str, Any], ttl_seconds: Optional[int] = None,
                    version: Optional[int] = None) -> bool:
        """Set multiple features for an entity."""
        try:
            pipe = self.redis_client.pipeline()
            current_time = datetime.now(timezone.utc)
            feature_version = version or int(time.time())
            default_ttl = ttl_seconds or self._get_default_ttl(entity_type)
            
            for feature_name, value in features.items():
                # Create feature value
                feature_value = FeatureValue(
                    value=value,
                    timestamp=current_time,
                    version=feature_version,
                    ttl_seconds=default_ttl
                )
                
                # Store feature
                key = self._make_key(entity_type, entity_id, feature_name)
                serialized_value = json.dumps(feature_value.to_dict())
                
                if feature_value.ttl_seconds:
                    pipe.setex(key, feature_value.ttl_seconds, serialized_value)
                else:
                    pipe.set(key, serialized_value)
                
                # Update version tracking
                version_key = self._make_version_key(entity_type, entity_id, feature_name)
                pipe.set(version_key, feature_version)
            
            # Execute pipeline
            pipe.execute()
            
            # Update entity metadata
            self._update_entity_metadata(entity_type, entity_id, list(features.keys()))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set features for {entity_type}:{entity_id}: {e}")
            return False
    
    def get_feature(self, entity_type: str, entity_id: str, 
                   feature_name: str) -> Optional[Any]:
        """Get a single feature value."""
        try:
            key = self._make_key(entity_type, entity_id, feature_name)
            serialized_value = self.redis_client.get(key)
            
            if serialized_value is None:
                return None
            
            # Deserialize feature value
            data = json.loads(serialized_value.decode('utf-8'))
            feature_value = FeatureValue.from_dict(data)
            
            return feature_value.value
            
        except Exception as e:
            logger.error(f"Failed to get feature {entity_type}:{entity_id}:{feature_name}: {e}")
            return None
    
    def get_features(self, entity_type: str, entity_id: str,
                    feature_names: List[str]) -> Dict[str, Any]:
        """Get multiple features for an entity."""
        try:
            # Create keys for batch get
            keys = [self._make_key(entity_type, entity_id, name) for name in feature_names]
            
            # Batch get
            serialized_values = self.redis_client.mget(keys)
            
            # Deserialize results
            features = {}
            for i, (feature_name, serialized_value) in enumerate(zip(feature_names, serialized_values)):
                if serialized_value is not None:
                    try:
                        data = json.loads(serialized_value.decode('utf-8'))
                        feature_value = FeatureValue.from_dict(data)
                        features[feature_name] = feature_value.value
                    except Exception as e:
                        logger.warning(f"Failed to deserialize feature {feature_name}: {e}")
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to get features for {entity_type}:{entity_id}: {e}")
            return {}
    
    def get_all_features(self, entity_type: str, entity_id: str) -> Dict[str, Any]:
        """Get all features for an entity."""
        try:
            # Get all feature keys for entity
            pattern = self._make_entity_key(entity_type, entity_id) + ":*"
            keys = self.redis_client.keys(pattern)
            
            if not keys:
                return {}
            
            # Batch get all features
            serialized_values = self.redis_client.mget(keys)
            
            # Deserialize results
            features = {}
            for key, serialized_value in zip(keys, serialized_values):
                if serialized_value is not None:
                    try:
                        # Extract feature name from key
                        feature_name = key.decode('utf-8').split(':')[-1]
                        
                        data = json.loads(serialized_value.decode('utf-8'))
                        feature_value = FeatureValue.from_dict(data)
                        features[feature_name] = feature_value.value
                    except Exception as e:
                        logger.warning(f"Failed to deserialize feature from key {key}: {e}")
            
            return features
            
        except Exception as e:
            logger.error(f"Failed to get all features for {entity_type}:{entity_id}: {e}")
            return {}
    
    def delete_feature(self, entity_type: str, entity_id: str, 
                      feature_name: str) -> bool:
        """Delete a single feature."""
        try:
            # Delete feature
            key = self._make_key(entity_type, entity_id, feature_name)
            self.redis_client.delete(key)
            
            # Delete version tracking
            version_key = self._make_version_key(entity_type, entity_id, feature_name)
            self.redis_client.delete(version_key)
            
            # Update entity metadata
            self._remove_from_entity_metadata(entity_type, entity_id, feature_name)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete feature {entity_type}:{entity_id}:{feature_name}: {e}")
            return False
    
    def delete_entity(self, entity_type: str, entity_id: str) -> bool:
        """Delete all features for an entity."""
        try:
            # Get all feature keys for entity
            pattern = self._make_entity_key(entity_type, entity_id) + ":*"
            keys = self.redis_client.keys(pattern)
            
            if keys:
                # Delete all features
                self.redis_client.delete(*keys)
            
            # Delete metadata
            metadata_key = self._make_metadata_key(entity_type, entity_id)
            self.redis_client.delete(metadata_key)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete entity {entity_type}:{entity_id}: {e}")
            return False
    
    def get_feature_version(self, entity_type: str, entity_id: str, 
                           feature_name: str) -> Optional[int]:
        """Get version of a feature."""
        try:
            version_key = self._make_version_key(entity_type, entity_id, feature_name)
            version = self.redis_client.get(version_key)
            
            return int(version.decode('utf-8')) if version else None
            
        except Exception as e:
            logger.error(f"Failed to get feature version {entity_type}:{entity_id}:{feature_name}: {e}")
            return None
    
    def get_entity_metadata(self, entity_type: str, entity_id: str) -> Dict[str, Any]:
        """Get metadata for an entity."""
        try:
            metadata_key = self._make_metadata_key(entity_type, entity_id)
            metadata = self.redis_client.hgetall(metadata_key)
            
            if metadata:
                return {k.decode('utf-8'): v.decode('utf-8') for k, v in metadata.items()}
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get entity metadata {entity_type}:{entity_id}: {e}")
            return {}
    
    def _get_default_ttl(self, entity_type: str) -> int:
        """Get default TTL for entity type."""
        ttl_map = {
            'user': self.config.user_features_ttl,
            'item': self.config.item_features_ttl,
            'session': 1800,  # 30 minutes
            'recommendation': self.config.recommendation_cache_ttl,
            'model': self.config.model_cache_ttl
        }
        return ttl_map.get(entity_type, 3600)  # Default 1 hour
    
    def _update_entity_metadata(self, entity_type: str, entity_id: str, 
                               feature_names: Union[str, List[str]]):
        """Update entity metadata."""
        try:
            metadata_key = self._make_metadata_key(entity_type, entity_id)
            
            if isinstance(feature_names, str):
                feature_names = [feature_names]
            
            # Update last updated timestamp
            self.redis_client.hset(
                metadata_key,
                'last_updated',
                datetime.now(timezone.utc).isoformat()
            )
            
            # Update feature list
            existing_features = self.redis_client.hget(metadata_key, 'features')
            if existing_features:
                features = set(json.loads(existing_features.decode('utf-8')))
                features.update(feature_names)
            else:
                features = set(feature_names)
            
            self.redis_client.hset(
                metadata_key,
                'features',
                json.dumps(list(features))
            )
            
            # Set TTL for metadata
            self.redis_client.expire(metadata_key, self._get_default_ttl(entity_type))
            
        except Exception as e:
            logger.error(f"Failed to update entity metadata {entity_type}:{entity_id}: {e}")
    
    def _remove_from_entity_metadata(self, entity_type: str, entity_id: str, 
                                    feature_name: str):
        """Remove feature from entity metadata."""
        try:
            metadata_key = self._make_metadata_key(entity_type, entity_id)
            existing_features = self.redis_client.hget(metadata_key, 'features')
            
            if existing_features:
                features = set(json.loads(existing_features.decode('utf-8')))
                features.discard(feature_name)
                
                self.redis_client.hset(
                    metadata_key,
                    'features',
                    json.dumps(list(features))
                )
                
        except Exception as e:
            logger.error(f"Failed to remove from entity metadata {entity_type}:{entity_id}:{feature_name}: {e}")
    
    async def set_feature_async(self, entity_type: str, entity_id: str, feature_name: str,
                               value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Async version of set_feature."""
        if not self.async_redis_client:
            await self._initialize_async_connection()
        
        try:
            feature_value = FeatureValue(
                value=value,
                timestamp=datetime.now(timezone.utc),
                version=int(time.time()),
                ttl_seconds=ttl_seconds or self._get_default_ttl(entity_type)
            )
            
            key = self._make_key(entity_type, entity_id, feature_name)
            serialized_value = json.dumps(feature_value.to_dict())
            
            if feature_value.ttl_seconds:
                await self.async_redis_client.setex(key, feature_value.ttl_seconds, serialized_value)
            else:
                await self.async_redis_client.set(key, serialized_value)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to set feature async {entity_type}:{entity_id}:{feature_name}: {e}")
            return False
    
    async def get_feature_async(self, entity_type: str, entity_id: str, 
                              feature_name: str) -> Optional[Any]:
        """Async version of get_feature."""
        if not self.async_redis_client:
            await self._initialize_async_connection()
        
        try:
            key = self._make_key(entity_type, entity_id, feature_name)
            serialized_value = await self.async_redis_client.get(key)
            
            if serialized_value is None:
                return None
            
            data = json.loads(serialized_value)
            feature_value = FeatureValue.from_dict(data)
            
            return feature_value.value
            
        except Exception as e:
            logger.error(f"Failed to get feature async {entity_type}:{entity_id}:{feature_name}: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis statistics."""
        try:
            info = self.redis_client.info()
            
            return {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory_human', '0B'),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'hit_rate': (info.get('keyspace_hits', 0) / 
                           max(info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0), 1))
            }
            
        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}")
            return {}
    
    def health_check(self) -> bool:
        """Check Redis health."""
        try:
            self.redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    def close(self):
        """Close Redis connections."""
        if self.redis_client:
            self.redis_client.close()
        
        if self.async_redis_client:
            asyncio.create_task(self.async_redis_client.close())
        
        logger.info("Redis connections closed")


# Example usage
def main():
    """Example usage of online feature store."""
    from app.config import Config
    
    # Load configuration
    config = Config()
    
    # Initialize feature store
    feature_store = OnlineFeatureStore(config.redis)
    
    try:
        # Set some features
        user_features = {
            'age': 30,
            'gender': 'M',
            'membership_days': 365,
            'last_login': datetime.now(timezone.utc).isoformat(),
            'preferences': ['action', 'comedy', 'drama']
        }
        
        success = feature_store.set_features('user', 'user_123', user_features)
        logger.info(f"Set user features: {success}")
        
        # Get features
        retrieved_features = feature_store.get_features('user', 'user_123', ['age', 'gender', 'preferences'])
        logger.info(f"Retrieved features: {retrieved_features}")
        
        # Get all features
        all_features = feature_store.get_all_features('user', 'user_123')
        logger.info(f"All features: {all_features}")
        
        # Get stats
        stats = feature_store.get_stats()
        logger.info(f"Redis stats: {stats}")
        
        # Health check
        is_healthy = feature_store.health_check()
        logger.info(f"Redis healthy: {is_healthy}")
        
    finally:
        feature_store.close()


if __name__ == "__main__":
    main()
