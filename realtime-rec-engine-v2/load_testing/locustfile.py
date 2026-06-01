"""
Production-grade load testing for Netflix/Meta scale recommendation engine
Simulates 1M+ concurrent users with realistic user behavior patterns
"""

import random
import time
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from locust import HttpUser, task, between, events
from locust.exception import RescheduleTask
import numpy as np
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
TARGET_QPS = 10000000  # 10M QPS target
MAX_CONCURRENT_USERS = 1000000
TEST_DURATION = 3600  # 1 hour
WARMUP_DURATION = 300  # 5 minutes

# Test data
USER_IDS = [f"user_{i:08d}" for i in range(10000000)]
ITEM_IDS = [f"item_{i:08d}" for i in range(1000000)]
CATEGORIES = ["electronics", "clothing", "books", "home", "sports", "beauty", "toys", "food"]
REGIONS = ["us-west-2", "us-east-1", "eu-west-1", "ap-southeast-1"]

# User behavior patterns
@dataclass
class UserBehavior:
    """User behavior pattern configuration."""
    request_rate: float  # Requests per second
    session_duration: int  # Session duration in seconds
    recommendations_per_session: int
    feedback_probability: float
    feature_lookup_probability: float
    cache_hit_probability: float
    network_latency_ms: float
    device_type: str  # mobile, desktop, tablet

# Behavior profiles
BEHAVIOR_PROFILES = {
    "power_user": UserBehavior(
        request_rate=10.0,
        session_duration=1800,
        recommendations_per_session=50,
        feedback_probability=0.3,
        feature_lookup_probability=0.2,
        cache_hit_probability=0.8,
        network_latency_ms=50,
        device_type="desktop"
    ),
    "casual_user": UserBehavior(
        request_rate=2.0,
        session_duration=600,
        recommendations_per_session=10,
        feedback_probability=0.1,
        feature_lookup_probability=0.05,
        cache_hit_probability=0.6,
        network_latency_ms=100,
        device_type="mobile"
    ),
    "api_client": UserBehavior(
        request_rate=50.0,
        session_duration=3600,
        recommendations_per_session=500,
        feedback_probability=0.5,
        feature_lookup_probability=0.3,
        cache_hit_probability=0.9,
        network_latency_ms=20,
        device_type="desktop"
    )
}

class RecommendationEngineUser(HttpUser):
    """Simulated user for load testing the recommendation engine."""
    
    wait_time = between(0.1, 2.0)  # Random wait between requests
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # User configuration
        self.user_id = random.choice(USER_IDS)
        self.session_id = str(uuid.uuid4())
        self.behavior_profile = random.choice(list(BEHAVIOR_PROFILES.values()))
        self.region = random.choice(REGIONS)
        self.device_type = self.behavior_profile.device_type
        
        # Session state
        self.session_start = time.time()
        self.recommendations_count = 0
        self.feedback_count = 0
        self.feature_lookups = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Performance tracking
        self.request_times = []
        self.error_count = 0
        
        # User preferences (for realistic requests)
        self.user_preferences = self._generate_user_preferences()
        
        logger.info(f"User {self.user_id} started session with {self.behavior_profile.device_type} profile")
    
    def _generate_user_preferences(self) -> Dict[str, Any]:
        """Generate realistic user preferences."""
        return {
            "categories": random.sample(CATEGORIES, k=random.randint(1, 3)),
            "price_range": {
                "min": random.uniform(10, 100),
                "max": random.uniform(100, 1000)
            },
            "brand_preferences": random.sample(["nike", "apple", "samsung", "sony", "lg"], k=random.randint(0, 3)),
            "recent_items": random.sample(ITEM_IDS, k=random.randint(5, 20))
        }
    
    def _generate_context(self) -> Dict[str, Any]:
        """Generate request context."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device_type": self.device_type,
            "region": self.region,
            "session_id": self.session_id,
            "user_agent": f"RecEngine-LoadTest/1.0 ({self.device_type})",
            "ip_address": f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}",
            "page": random.choice(["home", "product", "category", "search", "cart"]),
            "referrer": random.choice(["google", "direct", "social", "email"])
        }
    
    def _generate_filters(self) -> Dict[str, Any]:
        """Generate realistic filters."""
        filters = {}
        
        # Category filter
        if random.random() < 0.7:
            filters["categories"] = [random.choice(self.user_preferences["categories"])]
        
        # Price filter
        if random.random() < 0.5:
            price_range = self.user_preferences["price_range"]
            filters["price_range"] = {
                "min": price_range["min"],
                "max": price_range["max"]
            }
        
        # Brand filter
        if random.random() < 0.3 and self.user_preferences["brand_preferences"]:
            filters["brands"] = random.sample(self.user_preferences["brand_preferences"], k=1)
        
        # Availability filter
        if random.random() < 0.8:
            filters["in_stock"] = True
        
        return filters
    
    def _generate_candidate_items(self) -> Optional[List[str]]:
        """Generate candidate items for recommendation."""
        # 30% chance to provide candidate items
        if random.random() < 0.3:
            # Mix of recent items and random items
            recent_items = self.user_preferences["recent_items"]
            random_items = random.sample(ITEM_IDS, k=min(50, len(ITEM_IDS)))
            candidates = list(set(recent_items + random_items))
            return random.sample(candidates, k=min(100, len(candidates)))
        return None
    
    def _calculate_session_duration(self) -> float:
        """Calculate realistic session duration."""
        base_duration = self.behavior_profile.session_duration
        # Add some randomness
        variation = random.uniform(0.8, 1.2)
        return base_duration * variation
    
    def _should_continue_session(self) -> bool:
        """Check if session should continue."""
        session_duration = time.time() - self.session_start
        max_duration = self._calculate_session_duration()
        
        return session_duration < max_duration
    
    def _record_request_time(self, start_time: float):
        """Record request time for performance tracking."""
        request_time = (time.time() - start_time) * 1000  # Convert to ms
        self.request_times.append(request_time)
        
        # Keep only last 100 request times
        if len(self.request_times) > 100:
            self.request_times = self.request_times[-100:]
    
    def _handle_error(self, error):
        """Handle request errors."""
        self.error_count += 1
        logger.error(f"User {self.user_id} encountered error: {error}")
        
        # Reschedule task on certain errors
        if "timeout" in str(error).lower() or "connection" in str(error).lower():
            raise RescheduleTask()
    
    @task(weight=70)  # 70% of requests are recommendations
    def get_recommendations(self):
        """Get recommendations for the user."""
        if not self._should_continue_session():
            return
        
        start_time = time.time()
        
        try:
            # Prepare request
            num_recommendations = random.randint(5, 50)
            context = self._generate_context()
            filters = self._generate_filters()
            candidates = self._generate_candidate_items()
            
            # Make request
            response = self.client.post(
                "/recommend",
                json={
                    "user_id": self.user_id,
                    "session_id": self.session_id,
                    "num_recommendations": num_recommendations,
                    "candidate_items": candidates,
                    "filters": filters,
                    "context": context
                },
                headers={
                    "Content-Type": "application/json",
                    "X-User-ID": self.user_id,
                    "X-Session-ID": self.session_id,
                    "X-Device-Type": self.device_type,
                    "X-Region": self.region
                },
                timeout=10.0
            )
            
            # Record metrics
            self._record_request_time(start_time)
            self.recommendations_count += 1
            
            # Check for cache hit (simulated based on response headers)
            if "X-Cache-Hit" in response.headers:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
            
            # Process recommendations
            if response.status_code == 200:
                recommendations = response.json()
                
                # Simulate user interaction with recommendations
                if recommendations.get("recommendations"):
                    self._simulate_interaction(recommendations["recommendations"])
            
            # Simulate network latency
            time.sleep(self.behavior_profile.network_latency_ms / 1000)
            
        except Exception as e:
            self._handle_error(e)
    
    def _simulate_interaction(self, recommendations: List[Dict[str, Any]]):
        """Simulate user interaction with recommendations."""
        if not recommendations:
            return
        
        # Random interaction probability
        if random.random() < 0.1:  # 10% chance to interact
            recommendation = random.choice(recommendations)
            item_id = recommendation.get("item_id")
            
            if item_id and random.random() < self.behavior_profile.feedback_probability:
                # Send feedback
                self._send_feedback(item_id)
    
    def _send_feedback(self, item_id: str):
        """Send feedback for an item."""
        try:
            interaction_type = random.choice(["click", "view", "purchase", "like", "dislike"])
            rating = None
            
            if interaction_type in ["like", "dislike"]:
                rating = random.uniform(1, 5)
            
            response = self.client.post(
                "/feedback",
                json={
                    "user_id": self.user_id,
                    "item_id": item_id,
                    "interaction_type": interaction_type,
                    "rating": rating,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metadata": {
                        "session_id": self.session_id,
                        "device_type": self.device_type,
                        "region": self.region
                    }
                },
                headers={
                    "Content-Type": "application/json",
                    "X-User-ID": self.user_id
                },
                timeout=5.0
            )
            
            self.feedback_count += 1
            
        except Exception as e:
            logger.error(f"Failed to send feedback: {e}")
    
    @task(weight=15)  # 15% of requests are feature lookups
    def get_user_features(self):
        """Get user features."""
        if not self._should_continue_session():
            return
        
        if random.random() > self.behavior_profile.feature_lookup_probability:
            return
        
        start_time = time.time()
        
        try:
            feature_names = random.sample([
                "user_age", "user_gender", "user_location", "user_preferences",
                "purchase_history", "browse_history", "search_history",
                "engagement_score", "loyalty_tier", "last_active"
            ], k=random.randint(1, 5))
            
            response = self.client.get(
                f"/user/{self.user_id}/features",
                params={"feature_names": ",".join(feature_names)},
                headers={
                    "X-User-ID": self.user_id,
                    "X-Session-ID": self.session_id
                },
                timeout=5.0
            )
            
            self._record_request_time(start_time)
            self.feature_lookups += 1
            
        except Exception as e:
            self._handle_error(e)
    
    @task(weight=10)  # 10% of requests are item features
    def get_item_features(self):
        """Get item features."""
        if not self._should_continue_session():
            return
        
        start_time = time.time()
        
        try:
            item_id = random.choice(ITEM_IDS)
            feature_names = random.sample([
                "item_category", "item_price", "item_brand", "item_rating",
                "item_availability", "item_popularity", "item_description",
                "item_images", "item_specs", "item_reviews"
            ], k=random.randint(1, 5))
            
            response = self.client.get(
                f"/item/{item_id}/features",
                params={"feature_names": ",".join(feature_names)},
                timeout=5.0
            )
            
            self._record_request_time(start_time)
            
        except Exception as e:
            self._handle_error(e)
    
    @task(weight=5)  # 5% of requests are health checks
    def health_check(self):
        """Health check endpoint."""
        try:
            response = self.client.get("/health", timeout=5.0)
            
            if response.status_code != 200:
                logger.warning(f"Health check failed: {response.status_code}")
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
    
    def on_stop(self):
        """Called when the user stops."""
        session_duration = time.time() - self.session_start
        
        # Log session summary
        logger.info(f"User {self.user_id} session completed:")
        logger.info(f"  Duration: {session_duration:.2f}s")
        logger.info(f"  Recommendations: {self.recommendations_count}")
        logger.info(f"  Feedback: {self.feedback_count}")
        logger.info(f"  Feature lookups: {self.feature_lookups}")
        logger.info(f"  Cache hits: {self.cache_hits}")
        logger.info(f"  Cache misses: {self.cache_misses}")
        logger.info(f"  Errors: {self.error_count}")
        
        if self.request_times:
            avg_time = np.mean(self.request_times)
            p95_time = np.percentile(self.request_times, 95)
            logger.info(f"  Avg response time: {avg_time:.2f}ms")
            logger.info(f"  P95 response time: {p95_time:.2f}ms")


# Performance monitoring
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Track request metrics."""
    if exception:
        logger.error(f"Request failed: {name} - {exception}")
    else:
        # Log slow requests
        if response_time > 1000:  # > 1 second
            logger.warning(f"Slow request: {name} - {response_time:.2f}ms")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when the test starts."""
    logger.info("Load test starting...")
    logger.info(f"Target QPS: {TARGET_QPS}")
    logger.info(f"Max concurrent users: {MAX_CONCURRENT_USERS}")
    logger.info(f"Test duration: {TEST_DURATION}s")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when the test stops."""
    logger.info("Load test completed")
    
    # Print summary statistics
    stats = environment.stats
    
    logger.info(f"Total requests: {stats.total_requests}")
    logger.info(f"Total failures: {stats.total_failures}")
    logger.info(f"Failure rate: {stats.fail_ratio:.2%}")
    
    if stats.total_requests > 0:
        logger.info(f"Average response time: {stats.avg_response_time:.2f}ms")
        logger.info(f"P95 response time: {stats.get_response_time_percentile(0.95):.2f}ms")
        logger.info(f"P99 response time: {stats.get_response_time_percentile(0.99):.2f}ms")
        logger.info(f"Requests per second: {stats.total_requests / (stats.last_request_timestamp - stats.start_time):.2f}")


# Custom load shape for realistic traffic patterns
class LoadShape:
    """Custom load shape for realistic traffic patterns."""
    
    @staticmethod
    def get_users(time_elapsed):
        """Calculate number of users at given time."""
        # Warmup phase
        if time_elapsed < WARMUP_DURATION:
            # Gradual ramp-up
            return int(MAX_CONCURRENT_USERS * (time_elapsed / WARMUP_DURATION))
        
        # Main test phase
        elif time_elapsed < TEST_DURATION - WARMUP_DURATION:
            # Maintain full load with some variation
            variation = 0.8 + 0.4 * np.sin(2 * np.pi * time_elapsed / 300)  # 5-minute cycles
            return int(MAX_CONCURRENT_USERS * variation)
        
        # Cool-down phase
        else:
            # Gradual ramp-down
            remaining_time = TEST_DURATION - time_elapsed
            return int(MAX_CONCURRENT_USERS * (remaining_time / WARMUP_DURATION))


# User classes for different behavior patterns
class PowerUser(RecommendationEngineUser):
    """Power user with high request rate."""
    
    wait_time = between(0.05, 0.5)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.behavior_profile = BEHAVIOR_PROFILES["power_user"]


class CasualUser(RecommendationEngineUser):
    """Casual user with normal request rate."""
    
    wait_time = between(0.5, 3.0)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.behavior_profile = BEHAVIOR_PROFILES["casual_user"]


class APIClient(RecommendationEngineUser):
    """API client with very high request rate."""
    
    wait_time = between(0.01, 0.1)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.behavior_profile = BEHAVIOR_PROFILES["api_client"]


# Stress testing scenarios
class StressTestUser(RecommendationEngineUser):
    """Stress test user with extreme behavior."""
    
    wait_time = between(0.001, 0.01)  # Very high frequency
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override with extreme behavior
        self.behavior_profile = UserBehavior(
            request_rate=100.0,
            session_duration=7200,
            recommendations_per_session=1000,
            feedback_probability=0.8,
            feature_lookup_probability=0.5,
            cache_hit_probability=0.95,
            network_latency_ms=10,
            device_type="desktop"
        )


# Configuration for different test scenarios
class TestConfig:
    """Test configuration for different scenarios."""
    
    @staticmethod
    def baseline_test():
        """Baseline load test."""
        return {
            "user_classes": [PowerUser, CasualUser, APIClient],
            "users": {
                PowerUser: 1000,
                CasualUser: 3000,
                APIClient: 100
            },
            "spawn_rate": 100,
            "host": "https://api.rec-engine.company.com"
        }
    
    @staticmethod
    def stress_test():
        """Stress test with maximum load."""
        return {
            "user_classes": [StressTestUser],
            "users": {
                StressTestUser: 10000
            },
            "spawn_rate": 1000,
            "host": "https://api.rec-engine.company.com"
        }
    
    @staticmethod
    def endurance_test():
        """Endurance test for long-running stability."""
        return {
            "user_classes": [PowerUser, CasualUser],
            "users": {
                PowerUser: 5000,
                CasualUser: 15000
            },
            "spawn_rate": 500,
            "host": "https://api.rec-engine.company.com",
            "run_time": "4h"  # 4 hours
        }
    
    @staticmethod
    def spike_test():
        """Spike test for sudden traffic bursts."""
        return {
            "user_classes": [APIClient, StressTestUser],
            "users": {
                APIClient: 5000,
                StressTestUser: 5000
            },
            "spawn_rate": 2000,
            "host": "https://api.rec-engine.company.com"
        }


# Example usage
if __name__ == "__main__":
    # This would be used with the Locust CLI
    print("Load testing script for Recommendation Engine")
    print("Usage: locust -f locustfile.py --host=https://api.rec-engine.company.com")
    print("Available test scenarios:")
    print("  - Baseline test: Normal traffic patterns")
    print("  - Stress test: Maximum load")
    print("  - Endurance test: Long-running stability")
    print("  - Spike test: Sudden traffic bursts")
