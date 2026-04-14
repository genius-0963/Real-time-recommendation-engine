"""
Stream processor for real-time feature engineering and model scoring.
Handles windowing, aggregations, and real-time transformations.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import numpy as np
import pandas as pd

from streaming.kafka_consumer import KafkaEventConsumer, EventProcessor
from streaming.kafka_producer import KafkaEventProducer, EventGenerator
from app.config import KafkaConfig, RedisConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WindowConfig:
    """Configuration for time windows."""
    window_size: timedelta  # Size of the window
    slide_size: timedelta    # How often the window slides
    max_size: int = 10000    # Maximum number of items in window


@dataclass
class AggregationConfig:
    """Configuration for feature aggregations."""
    functions: List[str] = field(default_factory=lambda: ['count', 'sum', 'mean', 'std'])
    time_windows: List[WindowConfig] = field(default_factory=lambda: [
        WindowConfig(timedelta(minutes=5), timedelta(minutes=1)),
        WindowConfig(timedelta(hours=1), timedelta(minutes=5)),
        WindowConfig(timedelta(days=1), timedelta(hours=1))
    ])


class TimeWindow:
    """Sliding time window for event aggregation."""
    
    def __init__(self, window_size: timedelta, max_size: int = 10000):
        self.window_size = window_size
        self.max_size = max_size
        self.events = deque()
        self.last_cleanup = time.time()
    
    def add_event(self, event: Dict[str, Any], timestamp: Optional[datetime] = None):
        """Add event to window."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        self.events.append((timestamp, event))
        
        # Cleanup old events periodically
        current_time = time.time()
        if current_time - self.last_cleanup > 10:  # Cleanup every 10 seconds
            self._cleanup_old_events()
            self.last_cleanup = current_time
    
    def _cleanup_old_events(self):
        """Remove events outside the window."""
        cutoff_time = datetime.now(timezone.utc) - self.window_size
        
        while self.events and self.events[0][0] < cutoff_time:
            self.events.popleft()
        
        # Enforce max size
        while len(self.events) > self.max_size:
            self.events.popleft()
    
    def get_events(self) -> List[Dict[str, Any]]:
        """Get all events in current window."""
        self._cleanup_old_events()
        return [event for _, event in self.events]
    
    def get_count(self) -> int:
        """Get number of events in window."""
        self._cleanup_old_events()
        return len(self.events)
    
    def is_empty(self) -> bool:
        """Check if window is empty."""
        return len(self.events) == 0


class FeatureAggregator:
    """Real-time feature aggregation engine."""
    
    def __init__(self, config: AggregationConfig):
        self.config = config
        self.windows = {}  # entity_id -> window_type -> TimeWindow
        self.aggregations = {}  # Cache for computed aggregations
        
    def _get_window(self, entity_id: str, window_config: WindowConfig) -> TimeWindow:
        """Get or create window for entity."""
        window_key = f"{entity_id}_{window_config.window_size.total_seconds()}"
        
        if window_key not in self.windows:
            self.windows[window_key] = TimeWindow(
                window_config.window_size,
                window_config.max_size
            )
        
        return self.windows[window_key]
    
    def add_event(self, entity_id: str, event: Dict[str, Any], 
                  timestamp: Optional[datetime] = None):
        """Add event for aggregation."""
        for window_config in self.config.time_windows:
            window = self._get_window(entity_id, window_config)
            window.add_event(event, timestamp)
    
    def compute_aggregations(self, entity_id: str, 
                           feature_name: str) -> Dict[str, float]:
        """Compute aggregations for entity and feature."""
        results = {}
        
        for window_config in self.config.time_windows:
            window = self._get_window(entity_id, window_config)
            events = window.get_events()
            
            if not events:
                continue
            
            # Extract feature values
            values = []
            for event in events:
                if feature_name in event:
                    try:
                        value = float(event[feature_name])
                        values.append(value)
                    except (ValueError, TypeError):
                        continue
            
            if not values:
                continue
            
            window_key = f"{window_config.window_size.total_seconds()}s"
            
            # Compute aggregations
            for func in self.config.functions:
                if func == 'count':
                    results[f"{feature_name}_{func}_{window_key}"] = len(values)
                elif func == 'sum':
                    results[f"{feature_name}_{func}_{window_key}"] = sum(values)
                elif func == 'mean':
                    results[f"{feature_name}_{func}_{window_key}"] = np.mean(values)
                elif func == 'std':
                    results[f"{feature_name}_{func}_{window_key}"] = np.std(values)
                elif func == 'min':
                    results[f"{feature_name}_{func}_{window_key}"] = min(values)
                elif func == 'max':
                    results[f"{feature_name}_{func}_{window_key}"] = max(values)
                elif func == 'median':
                    results[f"{feature_name}_{func}_{window_key}"] = np.median(values)
        
        return results
    
    def get_all_aggregations(self, entity_id: str) -> Dict[str, float]:
        """Get all aggregations for entity."""
        all_results = {}
        
        # Get all feature names from events
        feature_names = set()
        for window in self.windows.values():
            for _, event in window.get_events():
                feature_names.update(event.keys())
        
        # Compute aggregations for each feature
        for feature_name in feature_names:
            if feature_name not in ['user_id', 'item_id', 'timestamp', 'event_type']:
                aggregations = self.compute_aggregations(entity_id, feature_name)
                all_results.update(aggregations)
        
        return all_results


class RealTimeFeatureProcessor:
    """Real-time feature processing and scoring engine."""
    
    def __init__(self, kafka_config: KafkaConfig, redis_config: RedisConfig):
        self.kafka_config = kafka_config
        self.redis_config = redis_config
        
        # Initialize components
        self.producer = KafkaEventProducer(kafka_config)
        self.aggregator = FeatureAggregator(AggregationConfig())
        
        # Feature processing state
        self.user_sessions = {}  # user_id -> session_data
        self.item_popularity = defaultdict(int)
        self.category_popularity = defaultdict(int)
        
        # Scoring cache
        self.score_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
    async def process_user_event(self, event_data: Dict[str, Any],
                                topic: str, partition: int, offset: int):
        """Process user interaction events."""
        user_id = event_data.get('user_id')
        item_id = event_data.get('item_id')
        event_type = event_data.get('event_type')
        timestamp = datetime.fromisoformat(event_data.get('timestamp').replace('Z', '+00:00'))
        
        logger.info(f"Processing {event_type} event for user {user_id}, item {item_id}")
        
        # Update session data
        await self._update_user_session(user_id, event_data, timestamp)
        
        # Update popularity counters
        self._update_popularity_counters(item_id, event_data)
        
        # Add to aggregation windows
        self.aggregator.add_event(user_id, event_data, timestamp)
        
        # Compute real-time features
        user_features = await self._compute_user_features(user_id, timestamp)
        
        # Trigger model scoring if needed
        if event_type in ['view', 'click']:
            await self._trigger_scoring(user_id, user_features)
        
        # Produce feature update event
        await self._produce_feature_update('user', user_id, user_features)
    
    async def _update_user_session(self, user_id: str, event_data: Dict[str, Any], 
                                   timestamp: datetime):
        """Update user session information."""
        session_id = event_data.get('session_id')
        
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                'session_id': session_id,
                'start_time': timestamp,
                'last_activity': timestamp,
                'events': [],
                'items_viewed': set(),
                'categories_viewed': set(),
                'total_events': 0
            }
        
        session = self.user_sessions[user_id]
        
        # Update session if new session
        if session['session_id'] != session_id:
            session['session_id'] = session_id
            session['start_time'] = timestamp
            session['events'] = []
            session['items_viewed'] = set()
            session['categories_viewed'] = set()
        
        # Update session data
        session['last_activity'] = timestamp
        session['total_events'] += 1
        session['events'].append(event_data)
        
        # Track items and categories
        if 'item_id' in event_data:
            session['items_viewed'].add(event_data['item_id'])
        if 'item_category' in event_data:
            session['categories_viewed'].add(event_data['item_category'])
    
    def _update_popularity_counters(self, item_id: str, event_data: Dict[str, Any]):
        """Update popularity counters for items and categories."""
        event_type = event_data.get('event_type')
        weight = self._get_event_weight(event_type)
        
        self.item_popularity[item_id] += weight
        
        if 'item_category' in event_data:
            category = event_data['item_category']
            self.category_popularity[category] += weight
    
    def _get_event_weight(self, event_type: str) -> float:
        """Get weight for different event types."""
        weights = {
            'view': 1.0,
            'click': 2.0,
            'like': 3.0,
            'share': 4.0,
            'purchase': 10.0,
            'add_to_cart': 5.0
        }
        return weights.get(event_type, 1.0)
    
    async def _compute_user_features(self, user_id: str, 
                                   timestamp: datetime) -> Dict[str, Any]:
        """Compute real-time user features."""
        features = {}
        
        # Session features
        if user_id in self.user_sessions:
            session = self.user_sessions[user_id]
            session_duration = (timestamp - session['start_time']).total_seconds()
            
            features.update({
                'session_duration_seconds': session_duration,
                'session_events_count': session['total_events'],
                'unique_items_viewed': len(session['items_viewed']),
                'unique_categories_viewed': len(session['categories_viewed']),
                'events_per_minute': session['total_events'] / max(session_duration / 60, 1)
            })
        
        # Aggregated features
        aggregations = self.aggregator.get_all_aggregations(user_id)
        features.update(aggregations)
        
        # Time-based features
        features.update({
            'hour_of_day': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'is_weekend': timestamp.weekday() >= 5,
            'is_night': timestamp.hour < 6 or timestamp.hour > 22
        })
        
        return features
    
    async def _trigger_scoring(self, user_id: str, user_features: Dict[str, Any]):
        """Trigger real-time model scoring."""
        # Check cache first
        cache_key = f"score_{user_id}"
        current_time = time.time()
        
        if (cache_key in self.score_cache and 
            current_time - self.score_cache[cache_key]['timestamp'] < self.cache_ttl):
            return
        
        # Get candidate items (simplified - in production would use ANN)
        candidate_items = self._get_candidate_items(user_id)
        
        # Score items (placeholder for actual model inference)
        scored_items = []
        for item_id in candidate_items[:10]:  # Top 10 candidates
            score = self._compute_item_score(user_id, item_id, user_features)
            scored_items.append({
                'item_id': item_id,
                'score': score,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        
        # Sort by score
        scored_items.sort(key=lambda x: x['score'], reverse=True)
        
        # Cache results
        self.score_cache[cache_key] = {
            'scored_items': scored_items,
            'timestamp': current_time
        }
        
        # Produce scoring results
        await self._produce_scoring_results(user_id, scored_items)
    
    def _get_candidate_items(self, user_id: str) -> List[str]:
        """Get candidate items for scoring."""
        # Simplified candidate generation
        # In production, would use ANN retrieval, collaborative filtering, etc.
        
        candidates = []
        
        # Popular items
        popular_items = sorted(self.item_popularity.items(), 
                             key=lambda x: x[1], reverse=True)[:100]
        candidates.extend([item_id for item_id, _ in popular_items])
        
        # Items from user's preferred categories
        if user_id in self.user_sessions:
            session = self.user_sessions[user_id]
            for category in session['categories_viewed']:
                # Get popular items in this category
                category_items = [item_id for item_id in self.item_popularity.keys() 
                               if self._get_item_category(item_id) == category]
                candidates.extend(category_items[:20])
        
        return list(set(candidates))
    
    def _get_item_category(self, item_id: str) -> str:
        """Get category for item (placeholder)."""
        # In production, would look up from item catalog
        return "electronics"
    
    def _compute_item_score(self, user_id: str, item_id: str, 
                          user_features: Dict[str, Any]) -> float:
        """Compute score for item (placeholder for actual model)."""
        # Simplified scoring - in production would use trained model
        base_score = 0.5
        
        # Popularity boost
        popularity_score = min(self.item_popularity.get(item_id, 0) / 1000, 1.0)
        
        # Category preference boost
        category_boost = 0.0
        if user_id in self.user_sessions:
            session = self.user_sessions[user_id]
            item_category = self._get_item_category(item_id)
            if item_category in session['categories_viewed']:
                category_boost = 0.2
        
        # Time decay for recent interactions
        time_decay = 1.0
        
        final_score = base_score + (0.3 * popularity_score) + category_boost
        return min(final_score, 1.0)
    
    async def _produce_feature_update(self, entity_type: str, entity_id: str, 
                                    features: Dict[str, Any]):
        """Produce feature update event."""
        feature_update = EventGenerator.create_feature_update_event(
            entity_type=entity_type,
            entity_id=entity_id,
            features={k: str(v) for k, v in features.items()},  # Convert to strings for Avro
            version=int(time.time())
        )
        
        self.producer.produce_feature_update(feature_update)
    
    async def _produce_scoring_results(self, user_id: str, scored_items: List[Dict[str, Any]]):
        """Produce scoring results event."""
        scoring_event = {
            'event_id': f"scoring_{user_id}_{int(time.time())}",
            'user_id': user_id,
            'scored_items': scored_items,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'model_version': 'v1.0'
        }
        
        # Convert to feature update for now
        feature_update = EventGenerator.create_feature_update_event(
            entity_type='scoring_result',
            entity_id=user_id,
            features={k: str(v) for k, v in scoring_event.items()},
            version=int(time.time())
        )
        
        self.producer.produce_feature_update(feature_update)
    
    def get_processing_metrics(self) -> Dict[str, Any]:
        """Get processing metrics."""
        return {
            'active_sessions': len(self.user_sessions),
            'cached_scores': len(self.score_cache),
            'tracked_items': len(self.item_popularity),
            'tracked_categories': len(self.category_popularity),
            'aggregation_windows': len(self.aggregator.windows)
        }


class StreamProcessorManager:
    """Manager for stream processing pipeline."""
    
    def __init__(self, kafka_config: KafkaConfig, redis_config: RedisConfig):
        self.kafka_config = kafka_config
        self.redis_config = redis_config
        
        # Initialize components
        self.consumer = None
        self.feature_processor = None
        
    async def start(self):
        """Start the stream processing pipeline."""
        logger.info("Starting stream processing pipeline")
        
        # Initialize feature processor
        self.feature_processor = RealTimeFeatureProcessor(
            self.kafka_config, 
            self.redis_config
        )
        
        # Initialize consumer
        topics = [
            f"{self.kafka_config.topic_prefix}.{self.kafka_config.user_events_topic}",
            f"{self.kafka_config.topic_prefix}.{self.kafka_config.interaction_events_topic}"
        ]
        
        self.consumer = KafkaEventConsumer(self.kafka_config, topics)
        
        # Register handlers
        self.consumer.register_handler('user_event', self.feature_processor.process_user_event)
        
        # Start consuming
        self.consumer.start_consuming()
    
    async def stop(self):
        """Stop the stream processing pipeline."""
        logger.info("Stopping stream processing pipeline")
        
        if self.consumer:
            self.consumer.close()
        
        if self.feature_processor and self.feature_processor.producer:
            self.feature_processor.producer.close()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics."""
        metrics = {}
        
        if self.consumer:
            metrics['consumer'] = self.consumer.get_metrics()
        
        if self.feature_processor:
            metrics['feature_processor'] = self.feature_processor.get_processing_metrics()
        
        return metrics


# Example usage
async def main():
    """Example usage of stream processor."""
    from app.config import Config
    
    # Load configuration
    config = Config()
    
    # Initialize manager
    manager = StreamProcessorManager(config.kafka, config.redis)
    
    try:
        # Start processing
        await manager.start()
    except KeyboardInterrupt:
        logger.info("Stream processing interrupted")
    finally:
        # Cleanup
        await manager.stop()


if __name__ == "__main__":
    asyncio.run(main())
