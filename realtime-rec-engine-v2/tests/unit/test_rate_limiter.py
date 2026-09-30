"""
Unit tests for rate limiter module.
"""

import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

import redis.asyncio as redis

from app.monitoring.rate_limiter import (
    RateLimitConfig,
    RateLimitScope,
    RateLimitResult,
    TokenBucketRateLimiter,
    MultiTierRateLimiter,
    create_rate_limiter,
    CircuitBreakerConfig,
    CircuitBreaker,
)


class TestRateLimitConfig:
    """Test rate limit configuration."""
    
    def test_default_config(self):
        """Default config should have expected values."""
        config = RateLimitConfig()
        assert config.requests == 1000
        assert config.window_seconds == 60
        assert config.burst_allowance == 0.1
        assert "rec-engine-admin" in config.role_multipliers
        assert config.role_multipliers["rec-engine-admin"] == 10.0
    
    def test_custom_config(self):
        """Custom config should accept overrides."""
        config = RateLimitConfig(
            requests=100,
            window_seconds=30,
            role_multipliers={"admin": 5.0}
        )
        assert config.requests == 100
        assert config.window_seconds == 30
        assert config.role_multipliers["admin"] == 5.0


class TestRateLimitResult:
    """Test rate limit result."""
    
    def test_result_creation(self):
        """Result should store all fields."""
        result = RateLimitResult(
            allowed=True,
            remaining=500,
            limit=1000,
            scope="user",
            key="ratelimit:user:user_123"
        )
        assert result.allowed is True
        assert result.remaining == 500
        assert result.limit == 1000
        assert result.scope == "user"
    
    def test_result_with_retry_after(self):
        """Result with retry_after should store it."""
        result = RateLimitResult(
            allowed=False,
            remaining=0,
            retry_after=30,
            limit=100,
            scope="endpoint",
            key="ratelimit:endpoint:recommend"
        )
        assert result.allowed is False
        assert result.retry_after == 30


class TestTokenBucketRateLimiter:
    """Test token bucket rate limiter."""
    
    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        mock = AsyncMock(spec=redis.Redis)
        mock.script_load = AsyncMock(return_value="sha123")
        mock.evalsha = AsyncMock(return_value=[1, 500, 0, 1000, int(time.time() * 1000) + 60000])
        return mock
    
    @pytest.fixture
    def rate_limiter(self, mock_redis):
        """Create rate limiter with mock Redis."""
        config = RateLimitConfig(requests=1000, window_seconds=60)
        return TokenBucketRateLimiter(mock_redis, config, scope=RateLimitScope.USER)
    
    @pytest.mark.asyncio
    async def test_check_limit_allowed(self, rate_limiter, mock_redis):
        """Check limit should return allowed result."""
        result = await rate_limiter.check_limit("user_123", roles=["rec-engine-user"])
        
        assert result.allowed is True
        assert result.remaining >= 0
        assert result.limit > 0
        mock_redis.evalsha.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_check_limit_with_roles(self, rate_limiter, mock_redis):
        """Check limit should apply role multipliers."""
        # Admin role should get 10x multiplier
        result = await rate_limiter.check_limit("user_123", roles=["rec-engine-admin"])
        assert result.limit >= 1000  # 1000 * 10 = 10000
    
    @pytest.mark.asyncio
    async def test_check_limit_with_endpoint(self, rate_limiter, mock_redis):
        """Check limit should include endpoint in key."""
        result = await rate_limiter.check_limit("user_123", endpoint="/recommend")
        assert "recommend" in result.key
    
    @pytest.mark.asyncio
    async def test_get_current_usage(self, rate_limiter, mock_redis):
        """Get current usage should return token info."""
        mock_redis.hmget = AsyncMock(return_value=[500.0, str(int(time.time() * 1000))])
        
        usage = await rate_limiter.get_current_usage("user_123")
        assert "tokens_available" in usage
    
    @pytest.mark.asyncio
    async def test_reset_limit(self, rate_limiter, mock_redis):
        """Reset limit should delete key."""
        result = await rate_limiter.reset_limit("user_123")
        assert result is True
        mock_redis.delete.assert_called_once()


class TestMultiTierRateLimiter:
    """Test multi-tier rate limiter."""
    
    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        mock = AsyncMock(spec=redis.Redis)
        mock.script_load = AsyncMock(return_value="sha123")
        mock.evalsha = AsyncMock(return_value=[1, 500, 0, 1000, int(time.time() * 1000) + 60000])
        return mock
    
    @pytest.fixture
    def multi_limiter(self, mock_redis):
        """Create multi-tier rate limiter."""
        return MultiTierRateLimiter(mock_redis)
    
    @pytest.mark.asyncio
    async def test_check_all_limits_allowed(self, multi_limiter):
        """All limits should pass for authenticated user."""
        allowed, most_restrictive, all_results = await multi_limiter.check_all_limits(
            user_id="user_123",
            ip="192.168.1.1",
            roles=["rec-engine-user"],
            endpoint="/recommend"
        )
        
        assert allowed is True
        assert "global" in all_results
        assert "user" in all_results
        assert "endpoint" in all_results
        assert "ip" in all_results
    
    @pytest.mark.asyncio
    async def test_check_all_limits_anonymous(self, multi_limiter):
        """Anonymous user should still get IP limit."""
        allowed, most_restrictive, all_results = await multi_limiter.check_all_limits(
            user_id=None,
            ip="192.168.1.1",
            roles=["anonymous"],
            endpoint="/recommend"
        )
        
        assert allowed is True
        assert "global" in all_results
        assert "user" not in all_results  # No user limit for anonymous
        assert "ip" in all_results
    
    @pytest.mark.asyncio
    async def test_get_headers(self, multi_limiter):
        """Get headers should return rate limit headers."""
        result = RateLimitResult(
            allowed=True,
            remaining=500,
            limit=1000,
            reset_at=int(time.time() * 1000) + 60000,
            scope="user",
            key="ratelimit:user:user_123"
        )
        
        headers = await multi_limiter.get_headers(result)
        assert "X-RateLimit-Limit" in headers
        assert "X-RateLimit-Remaining" in headers
        assert "X-RateLimit-Scope" in headers
        assert "X-RateLimit-Reset" in headers


class TestCircuitBreaker:
    """Test circuit breaker."""
    
    @pytest.fixture
    def breaker(self):
        """Create circuit breaker."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=1.0
        )
        return CircuitBreaker(config)
    
    def test_initial_state_closed(self, breaker):
        """Initial state should be closed."""
        allowed, retry = breaker.is_allowed("/test")
        assert allowed is True
        assert retry is None
    
    def test_opens_after_failures(self, breaker):
        """Should open after failure threshold."""
        for _ in range(3):
            breaker.record_failure("/test", 500)
        
        allowed, retry = breaker.is_allowed("/test")
        assert allowed is False
        assert retry is not None
        assert retry > 0
    
    def test_client_errors_dont_count(self, breaker):
        """4xx errors should not count as failures."""
        for _ in range(5):
            breaker.record_failure("/test", 400)
        
        allowed, retry = breaker.is_allowed("/test")
        assert allowed is True  # Still closed
    
    def test_half_open_after_timeout(self, breaker):
        """Should go half-open after timeout."""
        for _ in range(3):
            breaker.record_failure("/test", 500)
        
        # Immediately should be open
        allowed, _ = breaker.is_allowed("/test")
        assert allowed is False
        
        # Wait for timeout
        time.sleep(1.1)
        
        allowed, _ = breaker.is_allowed("/test")
        assert allowed is True  # Half-open allows
    
    def test_closes_after_successes_in_half_open(self, breaker):
        """Should close after successes in half-open."""
        for _ in range(3):
            breaker.record_failure("/test", 500)
        
        time.sleep(1.1)  # Wait for half-open
        
        breaker.record_success("/test")
        breaker.record_success("/test")
        
        state = breaker.get_status("/test")
        assert state["state"] == "closed"
    
    def test_reopens_on_failure_in_half_open(self, breaker):
        """Should reopen on failure in half-open."""
        for _ in range(3):
            breaker.record_failure("/test", 500)
        
        time.sleep(1.1)  # Wait for half-open
        
        breaker.record_failure("/test", 500)  # Failure in half-open
        
        state = breaker.get_status("/test")
        assert state["state"] == "open"
    
    def test_reset(self, breaker):
        """Reset should clear state."""
        for _ in range(3):
            breaker.record_failure("/test", 500)
        
        breaker.reset("/test")
        
        state = breaker.get_status("/test")
        assert state["state"] == "closed"
        assert state["failures"] == 0


class TestCreateRateLimiter:
    """Test factory function."""
    
    def test_create_rate_limiter(self):
        """Factory should create MultiTierRateLimiter."""
        mock_redis = AsyncMock(spec=redis.Redis)
        mock_redis.script_load = AsyncMock(return_value="sha123")
        
        limiter = create_rate_limiter(mock_redis)
        assert isinstance(limiter, MultiTierRateLimiter)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])