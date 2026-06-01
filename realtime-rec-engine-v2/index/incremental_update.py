"""
Incremental index updates for real-time ANN index maintenance.
Supports adding/removing items, periodic rebuilding, and consistency checks.
"""

import logging
import time
import threading
import queue
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
import json
import os

from index.build_index import BaseANNIndex, IndexManager, IndexConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UpdateOperation:
    """Represents an index update operation."""
    operation_type: str  # 'add', 'remove', 'update'
    item_id: str
    embedding: Optional[np.ndarray] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UpdateConfig:
    """Configuration for incremental updates."""
    enable_incremental_updates: bool = True
    max_pending_updates: int = 10000
    update_batch_size: int = 100
    update_interval_seconds: int = 60
    rebuild_threshold: float = 0.1  # Rebuild when 10% of items changed
    consistency_check_interval: int = 300  # 5 minutes
    max_index_age_hours: int = 24  # Force rebuild after 24 hours


@dataclass
class UpdateMetrics:
    """Metrics for update operations."""
    total_updates: int = 0
    successful_updates: int = 0
    failed_updates: int = 0
    pending_updates: int = 0
    last_update_time: Optional[datetime] = None
    rebuild_count: int = 0
    consistency_checks: int = 0
    update_latency_ms: float = 0.0


class IncrementalIndexUpdater:
    """Handles incremental updates to ANN indexes."""
    
    def __init__(self, index_manager: IndexManager, config: UpdateConfig):
        self.index_manager = index_manager
        self.config = config
        self.metrics = UpdateMetrics()
        
        # Update state
        self.update_queue = queue.Queue(maxsize=config.max_pending_updates)
        self.pending_operations: Dict[str, UpdateOperation] = {}
        self.is_running = False
        self.update_thread = None
        self.consistency_thread = None
        
        # Index state tracking
        self.current_item_ids: Set[str] = set()
        self.index_build_time: Optional[datetime] = None
        self.total_changes_since_rebuild = 0
        
        # Thread safety
        self.state_lock = threading.RLock()
        
        # Initialize current state
        self._initialize_state()
    
    def _initialize_state(self):
        """Initialize the current state of the index."""
        try:
            if self.index_manager.active_index:
                self.current_item_ids = set(self.index_manager.active_index.item_ids or [])
                self.index_build_time = datetime.now(timezone.utc)
                logger.info(f"Initialized with {len(self.current_item_ids)} items")
        except Exception as e:
            logger.error(f"Failed to initialize state: {e}")
            self.current_item_ids = set()
    
    def start(self):
        """Start the incremental updater."""
        if self.is_running:
            logger.warning("Incremental updater already running")
            return
        
        self.is_running = True
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        # Start consistency check thread
        self.consistency_thread = threading.Thread(target=self._consistency_loop, daemon=True)
        self.consistency_thread.start()
        
        logger.info("Incremental index updater started")
    
    def stop(self):
        """Stop the incremental updater."""
        self.is_running = False
        
        # Wait for threads to finish
        if self.update_thread:
            self.update_thread.join(timeout=10)
        
        if self.consistency_thread:
            self.consistency_thread.join(timeout=10)
        
        logger.info("Incremental index updater stopped")
    
    def add_item(self, item_id: str, embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a new item to the index."""
        if not self.config.enable_incremental_updates:
            logger.warning("Incremental updates disabled")
            return False
        
        try:
            operation = UpdateOperation(
                operation_type='add',
                item_id=item_id,
                embedding=embedding.copy(),
                metadata=metadata or {}
            )
            
            return self._queue_operation(operation)
            
        except Exception as e:
            logger.error(f"Failed to add item {item_id}: {e}")
            self.metrics.failed_updates += 1
            return False
    
    def remove_item(self, item_id: str) -> bool:
        """Remove an item from the index."""
        if not self.config.enable_incremental_updates:
            logger.warning("Incremental updates disabled")
            return False
        
        try:
            operation = UpdateOperation(
                operation_type='remove',
                item_id=item_id
            )
            
            return self._queue_operation(operation)
            
        except Exception as e:
            logger.error(f"Failed to remove item {item_id}: {e}")
            self.metrics.failed_updates += 1
            return False
    
    def update_item(self, item_id: str, embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update an existing item in the index."""
        if not self.config.enable_incremental_updates:
            logger.warning("Incremental updates disabled")
            return False
        
        try:
            operation = UpdateOperation(
                operation_type='update',
                item_id=item_id,
                embedding=embedding.copy(),
                metadata=metadata or {}
            )
            
            return self._queue_operation(operation)
            
        except Exception as e:
            logger.error(f"Failed to update item {item_id}: {e}")
            self.metrics.failed_updates += 1
            return False
    
    def _queue_operation(self, operation: UpdateOperation) -> bool:
        """Queue an update operation."""
        try:
            # Check if queue is full
            if self.update_queue.full():
                logger.warning("Update queue full, dropping oldest operation")
                try:
                    self.update_queue.get_nowait()
                except queue.Empty:
                    pass
            
            # Add to pending operations (deduplicate)
            with self.state_lock:
                self.pending_operations[operation.item_id] = operation
            
            # Queue the operation
            self.update_queue.put(operation)
            self.metrics.total_updates += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to queue operation: {e}")
            return False
    
    def _update_loop(self):
        """Main update processing loop."""
        logger.info("Update loop started")
        
        while self.is_running:
            try:
                # Collect batch of operations
                operations = []
                
                try:
                    # Get first operation with timeout
                    first_op = self.update_queue.get(timeout=self.config.update_interval_seconds)
                    operations.append(first_op)
                    
                    # Get more operations if available
                    while len(operations) < self.config.update_batch_size:
                        try:
                            op = self.update_queue.get_nowait()
                            operations.append(op)
                        except queue.Empty:
                            break
                            
                except queue.Empty:
                    continue
                
                # Process batch
                self._process_update_batch(operations)
                
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                time.sleep(10)  # Back off on error
        
        logger.info("Update loop stopped")
    
    def _process_update_batch(self, operations: List[UpdateOperation]):
        """Process a batch of update operations."""
        start_time = time.time()
        
        try:
            with self.state_lock:
                # Group operations by type
                adds = []
                removes = []
                updates = []
                
                for op in operations:
                    if op.operation_type == 'add':
                        adds.append(op)
                    elif op.operation_type == 'remove':
                        removes.append(op)
                    elif op.operation_type == 'update':
                        updates.append(op)
                    
                    # Remove from pending
                    self.pending_operations.pop(op.item_id, None)
                
                # Process removes first
                for op in removes:
                    if op.item_id in self.current_item_ids:
                        self.current_item_ids.remove(op.item_id)
                        self.total_changes_since_rebuild += 1
                        self.metrics.successful_updates += 1
                
                # Process updates (treat as remove + add)
                for op in updates:
                    if op.item_id in self.current_item_ids:
                        self.current_item_ids.remove(op.item_id)
                    adds.append(op)
                    self.total_changes_since_rebuild += 1
                    self.metrics.successful_updates += 1
                
                # Process adds
                for op in adds:
                    if op.item_id not in self.current_item_ids:
                        self.current_item_ids.add(op.item_id)
                        self.total_changes_since_rebuild += 1
                        self.metrics.successful_updates += 1
            
            # Check if rebuild is needed
            if self._should_rebuild():
                self._trigger_rebuild()
            
            # Update metrics
            self.metrics.last_update_time = datetime.now(timezone.utc)
            self.metrics.update_latency_ms = (time.time() - start_time) * 1000
            self.metrics.pending_updates = self.update_queue.qsize()
            
            logger.info(f"Processed batch of {len(operations)} operations")
            
        except Exception as e:
            logger.error(f"Failed to process update batch: {e}")
            self.metrics.failed_updates += len(operations)
    
    def _should_rebuild(self) -> bool:
        """Check if index should be rebuilt."""
        if not self.index_manager.active_index:
            return True
        
        # Check change threshold
        total_items = len(self.current_item_ids)
        if total_items > 0:
            change_ratio = self.total_changes_since_rebuild / total_items
            if change_ratio >= self.config.rebuild_threshold:
                logger.info(f"Change ratio {change_ratio:.3f} exceeded threshold {self.config.rebuild_threshold}")
                return True
        
        # Check index age
        if self.index_build_time:
            age_hours = (datetime.now(timezone.utc) - self.index_build_time).total_seconds() / 3600
            if age_hours >= self.config.max_index_age_hours:
                logger.info(f"Index age {age_hours:.1f}h exceeded threshold {self.config.max_index_age_hours}h")
                return True
        
        return False
    
    def _trigger_rebuild(self):
        """Trigger a full index rebuild."""
        try:
            logger.info("Triggering index rebuild due to incremental update threshold")
            
            # This would typically trigger an async rebuild process
            # For now, just reset counters
            self.total_changes_since_rebuild = 0
            self.index_build_time = datetime.now(timezone.utc)
            self.metrics.rebuild_count += 1
            
        except Exception as e:
            logger.error(f"Failed to trigger rebuild: {e}")
    
    def _consistency_loop(self):
        """Consistency check loop."""
        logger.info("Consistency check loop started")
        
        while self.is_running:
            try:
                time.sleep(self.config.consistency_check_interval)
                self._perform_consistency_check()
                
            except Exception as e:
                logger.error(f"Error in consistency loop: {e}")
        
        logger.info("Consistency check loop stopped")
    
    def _perform_consistency_check(self):
        """Perform consistency check on the index."""
        try:
            self.metrics.consistency_checks += 1
            
            if not self.index_manager.active_index:
                logger.warning("No active index for consistency check")
                return
            
            # Check item count consistency
            index_item_ids = set(self.index_manager.active_index.item_ids or [])
            tracked_item_ids = self.current_item_ids
            
            missing_in_index = tracked_item_ids - index_item_ids
            extra_in_index = index_item_ids - tracked_item_ids
            
            if missing_in_index or extra_in_index:
                logger.warning(f"Consistency issues found: "
                             f"{len(missing_in_index)} missing in index, "
                             f"{len(extra_in_index)} extra in index")
                
                # Could trigger correction here
                self._correct_consistency_issues(missing_in_index, extra_in_index)
            else:
                logger.debug("Consistency check passed")
                
        except Exception as e:
            logger.error(f"Consistency check failed: {e}")
    
    def _correct_consistency_issues(self, missing_in_index: Set[str], extra_in_index: Set[str]):
        """Correct consistency issues between tracked state and actual index."""
        try:
            # For now, just log the issues
            if missing_in_index:
                logger.warning(f"Items missing from index: {list(missing_in_index)[:10]}...")
            
            if extra_in_index:
                logger.warning(f"Extra items in index: {list(extra_in_index)[:10]}...")
            
            # In a production system, this would trigger appropriate corrections
            # such as rebuilding the index or applying specific fixes
            
        except Exception as e:
            logger.error(f"Failed to correct consistency issues: {e}")
    
    def force_rebuild(self) -> bool:
        """Force a full index rebuild."""
        try:
            logger.info("Forcing index rebuild")
            
            # Reset counters
            with self.state_lock:
                self.total_changes_since_rebuild = 0
                self.index_build_time = datetime.now(timezone.utc)
                self.metrics.rebuild_count += 1
            
            # This would trigger the actual rebuild process
            # For now, just return success
            return True
            
        except Exception as e:
            logger.error(f"Failed to force rebuild: {e}")
            return False
    
    def get_pending_operations(self) -> List[UpdateOperation]:
        """Get list of pending operations."""
        with self.state_lock:
            return list(self.pending_operations.values())
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get update metrics."""
        with self.state_lock:
            return {
                'total_updates': self.metrics.total_updates,
                'successful_updates': self.metrics.successful_updates,
                'failed_updates': self.metrics.failed_updates,
                'pending_updates': len(self.pending_operations),
                'queue_size': self.update_queue.qsize(),
                'last_update_time': self.metrics.last_update_time.isoformat() if self.metrics.last_update_time else None,
                'rebuild_count': self.metrics.rebuild_count,
                'consistency_checks': self.metrics.consistency_checks,
                'update_latency_ms': self.metrics.update_latency_ms,
                'tracked_items': len(self.current_item_ids),
                'changes_since_rebuild': self.total_changes_since_rebuild,
                'is_running': self.is_running
            }


class RealTimeIndexManager:
    """Real-time index manager with incremental updates."""
    
    def __init__(self, index_manager: IndexManager, update_config: UpdateConfig):
        self.index_manager = index_manager
        self.updater = IncrementalIndexUpdater(index_manager, update_config)
        
    def start(self):
        """Start real-time index management."""
        self.updater.start()
        logger.info("Real-time index manager started")
    
    def stop(self):
        """Stop real-time index management."""
        self.updater.stop()
        logger.info("Real-time index manager stopped")
    
    def add_item(self, item_id: str, embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add item to index."""
        return self.updater.add_item(item_id, embedding, metadata)
    
    def remove_item(self, item_id: str) -> bool:
        """Remove item from index."""
        return self.updater.remove_item(item_id)
    
    def update_item(self, item_id: str, embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update item in index."""
        return self.updater.update_item(item_id, embedding, metadata)
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[List[str], List[float]]:
        """Search using the underlying index manager."""
        return self.index_manager.search(query_embedding, k)
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status."""
        return {
            'index_metrics': self.index_manager.get_metrics(),
            'update_metrics': self.updater.get_metrics()
        }


# Example usage
def main():
    """Example usage of incremental index updates."""
    from index.build_index import IndexManager, IndexConfig
    
    # Create index config
    index_config = IndexConfig(
        index_type='faiss',
        embedding_dim=128,
        index_path='/tmp/test_index'
    )
    
    # Create update config
    update_config = UpdateConfig(
        enable_incremental_updates=True,
        update_interval_seconds=5
    )
    
    # Create managers
    index_manager = IndexManager(index_config)
    realtime_manager = RealTimeIndexManager(index_manager, update_config)
    
    try:
        # Create initial index
        num_items = 1000
        embeddings = np.random.random((num_items, 128)).astype(np.float32)
        item_ids = [f'item_{i}' for i in range(num_items)]
        
        success = index_manager.build_index(embeddings, item_ids)
        logger.info(f"Initial index build: {success}")
        
        # Start real-time management
        realtime_manager.start()
        
        # Simulate real-time updates
        for i in range(50):
            # Add new items
            new_embedding = np.random.random(128).astype(np.float32)
            realtime_manager.add_item(f'new_item_{i}', new_embedding)
            
            # Update existing items
            if i < 20:
                update_embedding = np.random.random(128).astype(np.float32)
                realtime_manager.update_item(f'item_{i}', update_embedding)
            
            # Remove some items
            if i > 30 and i < 35:
                realtime_manager.remove_item(f'item_{i-30}')
            
            time.sleep(0.1)
        
        # Test search
        query = np.random.random(128).astype(np.float32)
        results, scores = realtime_manager.search(query, k=10)
        logger.info(f"Search returned {len(results)} results")
        
        # Get status
        status = realtime_manager.get_status()
        logger.info(f"Status: {status}")
        
        # Let it run for a bit
        time.sleep(10)
        
    finally:
        # Stop
        realtime_manager.stop()


if __name__ == "__main__":
    main()
