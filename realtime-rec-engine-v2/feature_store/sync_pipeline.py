"""
Synchronization pipeline between online and offline feature stores.
Handles bidirectional sync, conflict resolution, and data consistency.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import pandas as pd

from feature_store.online_store import OnlineFeatureStore, FeatureValue
from feature_store.offline_store import OfflineFeatureStore, FeatureDefinition
from app.config import FeatureStoreConfig, RedisConfig, DatabaseConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SyncConfig:
    """Configuration for sync pipeline."""
    sync_interval_seconds: int = 300  # 5 minutes
    batch_size: int = 1000
    max_workers: int = 4
    enable_bidirectional_sync: bool = True
    conflict_resolution: str = 'timestamp'  # 'timestamp', 'online_wins', 'offline_wins'
    enable_point_in_time_consistency: bool = True
    sync_history_retention_days: int = 7


@dataclass
class SyncMetrics:
    """Metrics for sync operations."""
    entities_processed: int = 0
    features_synced: int = 0
    conflicts_resolved: int = 0
    errors: int = 0
    last_sync_time: Optional[datetime] = None
    sync_duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'entities_processed': self.entities_processed,
            'features_synced': self.features_synced,
            'conflicts_resolved': self.conflicts_resolved,
            'errors': self.errors,
            'last_sync_time': self.last_sync_time.isoformat() if self.last_sync_time else None,
            'sync_duration_seconds': self.sync_duration_seconds
        }


class ConflictResolver:
    """Handles conflict resolution between online and offline stores."""
    
    def __init__(self, strategy: str = 'timestamp'):
        self.strategy = strategy
    
    def resolve_conflict(self, online_value: FeatureValue, offline_value: Dict[str, Any]) -> FeatureValue:
        """Resolve conflict between online and offline values."""
        if self.strategy == 'online_wins':
            return online_value
        
        elif self.strategy == 'offline_wins':
            # Convert offline to FeatureValue format
            return FeatureValue(
                value=offline_value.get('value'),
                timestamp=datetime.fromisoformat(offline_value['timestamp']),
                version=offline_value.get('version', 1)
            )
        
        elif self.strategy == 'timestamp':
            # Use the most recent timestamp
            offline_timestamp = datetime.fromisoformat(offline_value['timestamp'])
            if online_value.timestamp > offline_timestamp:
                return online_value
            else:
                return FeatureValue(
                    value=offline_value.get('value'),
                    timestamp=offline_timestamp,
                    version=offline_value.get('version', 1)
                )
        
        else:
            raise ValueError(f"Unknown conflict resolution strategy: {self.strategy}")


class FeatureSyncPipeline:
    """Pipeline for synchronizing features between online and offline stores."""
    
    def __init__(self, online_store: OnlineFeatureStore, offline_store: OfflineFeatureStore,
                 config: SyncConfig):
        self.online_store = online_store
        self.offline_store = offline_store
        self.config = config
        
        # Conflict resolver
        self.conflict_resolver = ConflictResolver(config.conflict_resolution)
        
        # Sync state
        self.is_running = False
        self.last_sync_timestamps = {}  # entity_type -> timestamp
        self.sync_metrics = SyncMetrics()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        
        # Track synced entities
        self.synced_entities = set()
        
    async def start_sync_loop(self):
        """Start the continuous sync loop."""
        self.is_running = True
        logger.info("Starting feature sync loop")
        
        while self.is_running:
            try:
                start_time = time.time()
                
                # Perform sync
                metrics = await self.sync_all_entities()
                
                # Update metrics
                self.sync_metrics = metrics
                self.sync_metrics.last_sync_time = datetime.now(timezone.utc)
                self.sync_metrics.sync_duration_seconds = time.time() - start_time
                
                logger.info(f"Sync completed: {metrics.to_dict()}")
                
                # Wait for next sync
                await asyncio.sleep(self.config.sync_interval_seconds)
                
            except Exception as e:
                logger.error(f"Sync loop error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def sync_all_entities(self) -> SyncMetrics:
        """Sync all entities between stores."""
        metrics = SyncMetrics()
        
        try:
            # Get all entity types from offline store
            entity_types = self._get_entity_types()
            
            for entity_type in entity_types:
                entity_metrics = await self.sync_entity_type(entity_type)
                
                # Aggregate metrics
                metrics.entities_processed += entity_metrics.entities_processed
                metrics.features_synced += entity_metrics.features_synced
                metrics.conflicts_resolved += entity_metrics.conflicts_resolved
                metrics.errors += entity_metrics.errors
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to sync all entities: {e}")
            metrics.errors += 1
            return metrics
    
    async def sync_entity_type(self, entity_type: str) -> SyncMetrics:
        """Sync all entities of a specific type."""
        metrics = SyncMetrics()
        
        try:
            # Get entity IDs from offline store
            entity_ids = self._get_entity_ids(entity_type)
            
            logger.info(f"Syncing {len(entity_ids)} entities of type {entity_type}")
            
            # Process entities in batches
            for i in range(0, len(entity_ids), self.config.batch_size):
                batch_ids = entity_ids[i:i + self.config.batch_size]
                
                batch_metrics = await self.sync_entity_batch(entity_type, batch_ids)
                
                # Aggregate metrics
                metrics.entities_processed += batch_metrics.entities_processed
                metrics.features_synced += batch_metrics.features_synced
                metrics.conflicts_resolved += batch_metrics.conflicts_resolved
                metrics.errors += batch_metrics.errors
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to sync entity type {entity_type}: {e}")
            metrics.errors += 1
            return metrics
    
    async def sync_entity_batch(self, entity_type: str, entity_ids: List[str]) -> SyncMetrics:
        """Sync a batch of entities."""
        metrics = SyncMetrics()
        
        # Create tasks for parallel processing
        tasks = []
        for entity_id in entity_ids:
            task = asyncio.create_task(
                self.sync_single_entity(entity_type, entity_id)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Entity sync error: {result}")
                metrics.errors += 1
            else:
                entity_metrics = result
                metrics.entities_processed += entity_metrics.entities_processed
                metrics.features_synced += entity_metrics.features_synced
                metrics.conflicts_resolved += entity_metrics.conflicts_resolved
                metrics.errors += entity_metrics.errors
        
        return metrics
    
    async def sync_single_entity(self, entity_type: str, entity_id: str) -> SyncMetrics:
        """Sync a single entity between stores."""
        metrics = SyncMetrics()
        
        try:
            # Get features from online store
            online_features = self.online_store.get_all_features(entity_type, entity_id)
            
            # Get features from offline store
            offline_features = self._get_offline_features(entity_type, entity_id)
            
            # Find all feature names
            all_feature_names = set(online_features.keys()) | set(offline_features.keys())
            
            # Sync each feature
            for feature_name in all_feature_names:
                try:
                    feature_synced = await self.sync_feature(
                        entity_type, entity_id, feature_name,
                        online_features.get(feature_name),
                        offline_features.get(feature_name)
                    )
                    
                    if feature_synced:
                        metrics.features_synced += 1
                        
                except Exception as e:
                    logger.error(f"Failed to sync feature {feature_name}: {e}")
                    metrics.errors += 1
            
            metrics.entities_processed += 1
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to sync entity {entity_type}:{entity_id}: {e}")
            metrics.errors += 1
            return metrics
    
    async def sync_feature(self, entity_type: str, entity_id: str, feature_name: str,
                          online_value: Optional[Any], offline_value: Optional[Dict[str, Any]]) -> bool:
        """Sync a single feature between stores."""
        needs_sync = False
        final_value = None
        
        # Determine if sync is needed and resolve conflicts
        if online_value is not None and offline_value is None:
            # Feature exists only online
            needs_sync = True
            final_value = online_value
            
        elif online_value is None and offline_value is not None:
            # Feature exists only offline
            needs_sync = True
            final_value = offline_value['value']
            
        elif online_value is not None and offline_value is not None:
            # Feature exists in both - check for conflicts
            offline_timestamp = datetime.fromisoformat(offline_value['timestamp'])
            
            # Get online feature metadata
            online_metadata = self.online_store.get_entity_metadata(entity_type, entity_id)
            online_timestamp = datetime.fromisoformat(
                online_metadata.get('last_updated', datetime.now(timezone.utc).isoformat())
            )
            
            # Check if values differ
            if online_value != offline_value['value']:
                # Conflict detected
                resolved_value = self._resolve_feature_conflict(
                    entity_type, entity_id, feature_name,
                    online_value, offline_value, online_timestamp, offline_timestamp
                )
                final_value = resolved_value.value
                needs_sync = True
                
        # Perform sync if needed
        if needs_sync and final_value is not None:
            if self.config.enable_bidirectional_sync:
                # Sync to both stores
                await self._sync_to_online_store(entity_type, entity_id, feature_name, final_value)
                await self._sync_to_offline_store(entity_type, entity_id, feature_name, final_value)
            else:
                # Default sync to offline store
                await self._sync_to_offline_store(entity_type, entity_id, feature_name, final_value)
            
            return True
        
        return False
    
    def _resolve_feature_conflict(self, entity_type: str, entity_id: str, feature_name: str,
                                  online_value: Any, offline_value: Dict[str, Any],
                                  online_timestamp: datetime, offline_timestamp: datetime) -> FeatureValue:
        """Resolve feature conflict using configured strategy."""
        # Create online FeatureValue
        online_feature_value = FeatureValue(
            value=online_value,
            timestamp=online_timestamp,
            version=int(online_timestamp.timestamp())
        )
        
        # Resolve conflict
        resolved_value = self.conflict_resolver.resolve_conflict(online_feature_value, offline_value)
        
        logger.info(f"Resolved conflict for {entity_type}:{entity_id}:{feature_name} "
                   f"using {self.config.conflict_resolution} strategy")
        
        return resolved_value
    
    async def _sync_to_online_store(self, entity_type: str, entity_id: str, 
                                   feature_name: str, value: Any):
        """Sync feature to online store."""
        try:
            success = self.online_store.set_feature(entity_type, entity_id, feature_name, value)
            if not success:
                raise Exception("Failed to set feature in online store")
        except Exception as e:
            logger.error(f"Failed to sync to online store: {e}")
            raise
    
    async def _sync_to_offline_store(self, entity_type: str, entity_id: str,
                                    feature_name: str, value: Any):
        """Sync feature to offline store."""
        try:
            success = self.offline_store.write_features(
                entity_type, entity_id, {feature_name: value}
            )
            if not success:
                raise Exception("Failed to write feature to offline store")
        except Exception as e:
            logger.error(f"Failed to sync to offline store: {e}")
            raise
    
    def _get_entity_types(self) -> List[str]:
        """Get all entity types from offline store."""
        try:
            # This would typically query the offline store for distinct entity types
            # For now, return common types
            return ['user', 'item', 'session']
        except Exception as e:
            logger.error(f"Failed to get entity types: {e}")
            return []
    
    def _get_entity_ids(self, entity_type: str, limit: int = 10000) -> List[str]:
        """Get entity IDs for a specific type."""
        try:
            # This would typically query the offline store for entity IDs
            # For now, return sample IDs
            return [f"{entity_type}_{i}" for i in range(1000)]
        except Exception as e:
            logger.error(f"Failed to get entity IDs for {entity_type}: {e}")
            return []
    
    def _get_offline_features(self, entity_type: str, entity_id: str) -> Dict[str, Dict[str, Any]]:
        """Get features from offline store with metadata."""
        try:
            # Get all feature names for this entity type
            feature_definitions = self.offline_store.feature_definitions
            feature_names = [name for name, def_ in feature_definitions.items() 
                           if def_.entity_type == entity_type]
            
            if not feature_names:
                return {}
            
            # Get features from offline store
            features = self.offline_store.get_features(entity_type, entity_id, feature_names)
            
            # Convert to expected format with metadata
            result = {}
            for feature_name, value in features.items():
                result[feature_name] = {
                    'value': value,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'version': 1
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get offline features for {entity_type}:{entity_id}: {e}")
            return {}
    
    async def force_sync_entity(self, entity_type: str, entity_id: str) -> SyncMetrics:
        """Force sync a specific entity."""
        logger.info(f"Force syncing entity {entity_type}:{entity_id}")
        return await self.sync_single_entity(entity_type, entity_id)
    
    async def force_sync_feature(self, entity_type: str, entity_id: str, feature_name: str) -> bool:
        """Force sync a specific feature."""
        logger.info(f"Force syncing feature {entity_type}:{entity_id}:{feature_name}")
        
        try:
            # Get values from both stores
            online_value = self.online_store.get_feature(entity_type, entity_id, feature_name)
            offline_features = self._get_offline_features(entity_type, entity_id)
            offline_value = offline_features.get(feature_name)
            
            # Sync feature
            synced = await self.sync_feature(
                entity_type, entity_id, feature_name,
                online_value, offline_value
            )
            
            return synced
            
        except Exception as e:
            logger.error(f"Failed to force sync feature: {e}")
            return False
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get current sync status and metrics."""
        return {
            'is_running': self.is_running,
            'config': {
                'sync_interval_seconds': self.config.sync_interval_seconds,
                'batch_size': self.config.batch_size,
                'conflict_resolution': self.config.conflict_resolution,
                'bidirectional_sync': self.config.enable_bidirectional_sync
            },
            'metrics': self.sync_metrics.to_dict(),
            'last_sync_timestamps': self.last_sync_timestamps
        }
    
    def stop_sync(self):
        """Stop the sync loop."""
        self.is_running = False
        logger.info("Stopping feature sync loop")
    
    def cleanup(self):
        """Cleanup resources."""
        self.stop_sync()
        self.executor.shutdown(wait=True)
        logger.info("Sync pipeline cleanup completed")


class FeatureSyncManager:
    """Manager for the feature sync pipeline."""
    
    def __init__(self, online_store: OnlineFeatureStore, offline_store: OfflineFeatureStore,
                 config: FeatureStoreConfig):
        self.online_store = online_store
        self.offline_store = offline_store
        
        # Create sync config
        sync_config = SyncConfig(
            sync_interval_seconds=config.sync_interval,
            batch_size=config.batch_size,
            enable_bidirectional_sync=True,
            conflict_resolution='timestamp'
        )
        
        # Initialize sync pipeline
        self.sync_pipeline = FeatureSyncPipeline(online_store, offline_store, sync_config)
        
        # Background task
        self.sync_task = None
    
    async def start(self):
        """Start the sync manager."""
        logger.info("Starting feature sync manager")
        
        # Start sync loop in background
        self.sync_task = asyncio.create_task(self.sync_pipeline.start_sync_loop())
    
    async def stop(self):
        """Stop the sync manager."""
        logger.info("Stopping feature sync manager")
        
        # Stop sync pipeline
        self.sync_pipeline.stop_sync()
        
        # Cancel background task
        if self.sync_task:
            self.sync_task.cancel()
            try:
                await self.sync_task
            except asyncio.CancelledError:
                pass
        
        # Cleanup
        self.sync_pipeline.cleanup()
    
    async def force_sync_entity(self, entity_type: str, entity_id: str) -> Dict[str, Any]:
        """Force sync a specific entity."""
        metrics = await self.sync_pipeline.force_sync_entity(entity_type, entity_id)
        return metrics.to_dict()
    
    async def force_sync_feature(self, entity_type: str, entity_id: str, 
                                feature_name: str) -> bool:
        """Force sync a specific feature."""
        return await self.sync_pipeline.force_sync_feature(entity_type, entity_id, feature_name)
    
    def get_status(self) -> Dict[str, Any]:
        """Get sync manager status."""
        return self.sync_pipeline.get_sync_status()


# Example usage
async def main():
    """Example usage of feature sync pipeline."""
    from app.config import Config
    
    # Load configuration
    config = Config()
    
    # Initialize stores
    online_store = OnlineFeatureStore(config.redis)
    offline_store = OfflineFeatureStore(config.database)
    
    # Initialize sync manager
    sync_manager = FeatureSyncManager(online_store, offline_store, config.feature_store)
    
    try:
        # Start sync manager
        await sync_manager.start()
        
        # Run for some time
        await asyncio.sleep(60)
        
        # Get status
        status = sync_manager.get_status()
        logger.info(f"Sync status: {status}")
        
        # Force sync specific entity
        entity_metrics = await sync_manager.force_sync_entity('user', 'user_123')
        logger.info(f"Entity sync metrics: {entity_metrics}")
        
    finally:
        # Stop sync manager
        await sync_manager.stop()
        
        # Close stores
        online_store.close()
        offline_store.close()


if __name__ == "__main__":
    asyncio.run(main())
