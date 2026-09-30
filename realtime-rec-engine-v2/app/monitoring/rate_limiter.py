"""
Distributed rate limiting using Redis-backed token bucket algorithm.
Supports per-user, per-IP, and per-endpoint limits with sliding window.
"""

import time
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import redis.asyncio as redis
from redis.asyncio import Redis

from app.config import Config
from app.security.config import get_security_config

logger = logging.getLogger(__name__)


class RateLimitScope(Enum):
    """Rate limit scope."""
    USER = "user"
    IP = "ip"
    ENDPOINT = "endpoint"
    GLOBAL = "global"


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests: int = 1000          # Max requests
    window_seconds: int = 60      # Time window
    burst_allowance: float = 0.1  # Allow 10% burst
    
    # Per-role overrides (multipliers)
    role_multipliers: Dict[str, float] = None
    
    def __post_init__(self):
        if self.role_multipliers is None:
            self.role_multipliers = {
                "rec-engine-admin": 10.0,
                "rec-engine-user": 1.0,
                "data-scientist": 2.0,
                "anonymous": 0.1
            }


@dataclass
class RateLimitResult:
    """Rate limit check result."""
    allowed: bool
    remaining: int
    retry_after: Optional[int] = None
    limit: int = 0
    reset_at: Optional[int] = None
    scope: str = ""
    key: str = ""


class TokenBucketRateLimiter:
    """
    Redis-backed token bucket rate limiter with sliding window.
    
    Uses Lua script for atomic operations.
    """
    
    # Lua script for atomic token bucket check-and-consume
    TOKEN_BUCKET_SCRIPT = """
    local key = KEYS[1]
    local capacity = tonumber(ARGV[1])
    local refill_rate = tonumber(ARGV[2])  -- tokens per second
    local requested = tonumber(ARGV[3])
    local now = tonumber(ARGV[4])
    local window = tonumber(ARGV[5])
    
    local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
    local tokens = tonumber(bucket[1])
    local last_refill = tonumber(bucket[2])
    
    if tokens == nil then
        tokens = capacity
        last_refill = now
    end
    
    -- Refill tokens based on elapsed time
    local elapsed = now - last_refill
    local new_tokens = math.min(capacity, tokens + elapsed * refill_rate)
    
    local allowed = 0
    local remaining = 0
    local retry_after = 0
    
    if new_tokens >= requested then
        allowed = 1
        new_tokens = new_tokens - requested
        remaining = math.floor(new_tokens)
    else
        -- Calculate time until enough tokens
        local needed = requested - new_tokens
        retry_after = math.ceil(needed / refill_rate)
        remaining = math.floor(new_tokens)
    end
    
    -- Save with expiry (window + buffer)
    redis.call('HMSET', key, 'tokens', new_tokens, 'last_refill', now)
    redis.call('EXPIRE', key, window + 10)
    
    return {allowed, remaining, retry_after, capacity, now + window}
    """
    
    # Lua script for sliding window log (more precise)
    SLIDING_WINDOW_SCRIPT = """
    local key = KEYS[1]
    local limit = tonumber(ARGV[1])
    local window = tonumber(ARGV[2])
    local now = tonumber(ARGV[3])
    local requested = tonumber(ARGV[4])
    
    local window_start = now - window
    
    -- Remove expired entries
    redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)
    
    -- Count current requests
    local current = redis.call('ZCARD', key)
    
    local allowed = 0
    local remaining = 0
    local retry_after = 0
    
    if current + requested <= limit then
        allowed = 1
        -- Add new entries
        for i = 1, requested do
            redis.call('ZADD', key, now, now .. ':' .. math.random(1000000))
        end
        remaining = limit - current - requested
    else
        -- Get oldest entry to calculate retry time
        local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
        if #oldest > 0 then
            retry_after = math.ceil(tonumber(oldest[2]) + window - now)
        else
            retry_after = window
        end
        remaining = math.max(0, limit - current)
    end
    
    redis.call('EXPIRE', key, window + 10)
    
    return {allowed, remaining, retry_after, limit, now + window}
    """
    
    def __init__(
        self,
        redis_client: Redis,
        config: RateLimitConfig,
        scope: RateLimitScope = RateLimitScope.USER,
        algorithm: str = "token_bucket"  # or "sliding_window"
    ):
        self.redis = redis_client
        self.config = config
        self.scope = scope
        self.algorithm = algorithm
        
        # Register Lua scripts
        if algorithm == "token_bucket":
            self._script = self.redis.script_load(self.TOKEN_BUCKET_SCRIPT)
        else:
            self._script = self.redis.script_load(self.SLIDING_WINDOW_SCRIPT)
    
    def _get_role_multiplier(self, roles: list[str]) -> float:
        """Get rate limit multiplier for user roles."""
        max_mult = 1.0
        for role in roles:
            mult = self.config.role_multipliers.get(role, 1.0)
            max_mult = max(max_mult, mult)
        return max_mult
    
    def _build_key(self, identifier: str, endpoint: Optional[str] = None) -> str:
        """Build Redis key for rate limiting."""
        parts = ["ratelimit", self.scope.value, identifier]
        if endpoint:
            # Sanitize endpoint for key
            safe_endpoint = endpoint.replace("/", "_").replace("-", "_")
            parts.append(safe_endpoint.strip("_"))
        return ":".join(parts)
    
    async def check_limit(
        self,
        identifier: str,
        roles: list[str] = None,
        endpoint: Optional[str] = None,
        requested: int = 1
    ) -> RateLimitResult:
        """
        Check and consume rate limit.
        
        Args:
            identifier: User ID or IP address
            roles: User roles for multiplier
            endpoint: Optional endpoint for per-endpoint limits
            requested: Number of tokens to consume
            
        Returns:
            RateLimitResult with limit info
        """
        roles = roles or []
        multiplier = self._get_role_multiplier(roles)
        
        effective_limit = int(self.config.requests * multiplier)
        effective_burst = int(effective_limit * (1 + self.config.burst_allowance))
        
        key = self._build_key(identifier, endpoint)
        now = int(time.time() * 1000)  # milliseconds
        
        try:
            if self.algorithm == "token_bucket":
                # Token bucket: capacity = burst, refill_rate = limit/window
                result = await self.redis.evalsha(
                    self._script,
                    1,  # number of keys
                    key,
                    effective_burst,                    # capacity
                    effective_limit / self.config.window_seconds,  # refill rate per second
                    requested,
                    now,
                    self.config.window_seconds * 1000  # window in ms
                )
            else:
                # Sliding window log
                result = await self.redis.evalsha(
                    self._script,
                    1,
                    key,
                    effective_limit,
                    self.config.window_seconds * 1000,
                    now,
                    requested
                )
            
            allowed = bool(result[0])
            remaining = int(result[1])
            retry_after = int(result[2]) if result[2] > 0 else None
            limit = int(result[3])
            reset_at = int(result[4])
            
            return RateLimitResult(
                allowed=allowed,
                remaining=max(0, remaining),
                retry_after=retry_after,
                limit=limit,
                reset_at=reset_at,
                scope=self.scope.value,
                key=key
            )
            
        except redis.exceptions.NoScriptError:
            # Script was flushed, reload
            logger.warning("Rate limit script flushed, reloading")
            if self.algorithm == "token_bucket":
                self._script = self.redis.script_load(self.TOKEN_BUCKET_SCRIPT)
            else:
                self._script = self.redis.script_load(self.SLIDING_WINDOW_SCRIPT)
            return await self.check_limit(identifier, roles, endpoint, requested)
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Fail open - allow request but log error
            return RateLimitResult(
                allowed=True,
                remaining=effective_limit,
                limit=effective_limit,
                scope=self.scope.value,
                key=key
            )
    
    async def get_current_usage(self, identifier: str, endpoint: Optional[str] = None) -> Dict[str, Any]:
        """Get current usage without consuming tokens."""
        key = self._build_key(identifier, endpoint)
        try:
            if self.algorithm == "token_bucket":
                bucket = await self.redis.hmget(key, 'tokens', 'last_refill')
                tokens = float(bucket[0]) if bucket[0] else self.config.requests * self._get_role_multiplier([])
                return {"tokens_available": tokens, "key": key}
            else:
                now = int(time.time() * 1000)
                window_start = now - self.config.window_seconds * 1000
                await self.redis.zremrangebyscore(key, '-inf', window_start)
                current = await self.redis.zcard(key)
                return {"current_requests": current, "key": key}
        except Exception as e:
            logger.error(f"Failed to get current usage: {e}")
            return {"error": str(e)}
    
    async def reset_limit(self, identifier: str, endpoint: Optional[str] = None) -> bool:
        """Reset rate limit for identifier (admin operation)."""
        key = self._build_key(identifier, endpoint)
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Failed to reset rate limit: {e}")
            return False


class MultiTierRateLimiter:
    """
    Multi-tier rate limiter applying multiple limits simultaneously.
    Checks: global -> per-user -> per-endpoint -> per-IP
    """
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        
        # Global limits (most restrictive)
        self.global_limiter = TokenBucketRateLimiter(
            redis_client,
            RateLimitConfig(requests=100000, window_seconds=60),
            scope=RateLimitScope.GLOBAL,
            algorithm="sliding_window"
        )
        
        # Per-user limits
        self.user_limiter = TokenBucketRateLimiter(
            redis_client,
            RateLimitConfig(
                requests=1000,
                window_seconds=60,
                role_multipliers={
                    "rec-engine-admin": 10.0,
                    "rec-engine-user": 1.0,
                    "data-scientist": 2.0,
                    "anonymous": 0.1
                }
            ),
            scope=RateLimitScope.USER,
            algorithm="token_bucket"
        )
        
        # Per-endpoint limits
        self.endpoint_limiter = TokenBucketRateLimiter(
            redis_client,
            RateLimitConfig(
                requests=5000,
                window_seconds=60,
                role_multipliers={
                    "rec-engine-admin": 5.0,
                    "rec-engine-user": 1.0,
                    "data-scientist": 2.0,
                    "anonymous": 0.2
                }
            ),
            scope=RateLimitScope.ENDPOINT,
            algorithm="token_bucket"
        )
        
        # Per-IP limits (for anonymous)
        self.ip_limiter = TokenBucketRateLimiter(
            redis_client,
            RateLimitConfig(
                requests=100,
                window_seconds=60,
                role_multipliers={"anonymous": 1.0}
            ),
            scope=RateLimitScope.IP,
            algorithm="sliding_window"
        )
    
    async def check_all_limits(
        self,
        user_id: Optional[str] = None,
        ip: Optional[str] = None,
        roles: list[str] = None,
        endpoint: Optional[str] = None,
        requested: int = 1
    ) -> Tuple[bool, RateLimitResult, Dict[str, RateLimitResult]]:
        """
        Check all applicable rate limits.
        
        Returns:
            (allowed, most_restrictive_result, all_results)
        """
        roles = roles or []
        results = {}
        
        # 1. Global limit (always checked)
        global_result = await self.global_limiter.check_limit("global", roles, None, requested)
        results["global"] = global_result
        if not global_result.allowed:
            return False, global_result, results
        
        # 2. Per-user limit (if authenticated)
        if user_id:
            user_result = await self.user_limiter.check_limit(user_id, roles, endpoint, requested)
            results["user"] = user_result
            if not user_result.allowed:
                return False, user_result, results
        
        # 3. Per-endpoint limit (if endpoint specified)
        if endpoint:
            ep_result = await self.endpoint_limiter.check_limit("endpoint", roles, endpoint, requested)
            results["endpoint"] = ep_result
            if not ep_result.allowed:
                return False, ep_result, results
        
        # 4. Per-IP limit (for anonymous or additional protection)
        if ip:
            ip_roles = roles if roles else ["anonymous"]
            ip_result = await self.ip_limiter.check_limit(ip, ip_roles, endpoint, requested)
            results["ip"] = ip_result
            if not ip_result.allowed:
                return False, ip_result, results
        
        # All passed - return most restrictive remaining
        most_restrictive = min(results.values(), key=lambda r: r.remaining)
        return True, most_restrictive, results
    
    async def get_headers(self, result: RateLimitResult) -> Dict[str, str]:
        """Generate rate limit headers."""
        headers = {
            "X-RateLimit-Limit": str(result.limit),
            "X-RateLimit-Remaining": str(result.remaining),
            "X-RateLimit-Scope": result.scope,
        }
        if result.reset_at:
            headers["X-RateLimit-Reset"] = str(result.reset_at // 1000)
        if result.retry_after:
            headers["Retry-After"] = str(result.retry_after // 1000 + 1)
        return headers


# Factory function
def create_rate_limiter(redis_client: Redis) -> MultiTierRateLimiter:
    """Create multi-tier rate limiter."""
    return MultiTierRateLimiter(redis_client)


# Dependency for FastAPI
async def get_rate_limiter(request: "Request") -> MultiTierRateLimiter:
    """Get rate limiter from app state."""
    return request.app.state.rate_limiter


async def check_rate_limit(
    request: "Request",
    user_id: Optional[str] = None,
    roles: list[str] = None
) -> RateLimitResult:
    """
    FastAPI dependency for rate limiting.
    
    Usage:
        @app.post("/endpoint")
        async def endpoint(
            request: Request,
            rate_limit: RateLimitResult = Depends(check_rate_limit)
        ):
            # rate_limit contains limit info
    """
    rate_limiter: MultiTierRateLimiter = request.app.state.rate_limiter
    
    # Extract identifier
    ip = request.client.host if request.client else "unknown"
    
    # Check all limits
    allowed, most_restrictive, all_results = await rate_limiter.check_all_limits(
        user_id=user_id,
        ip=ip,
        roles=roles,
        endpoint=request.url.path,
        requested=1
    )
    
    # Add rate limit headers to response
    request.state.rate_limit_headers = await rate_limiter.get_headers(most_restrictive)
    request.state.rate_limit_all = all_results
    
    if not allowed:
        from fastapi import HTTPException
        headers = await rate_limiter.get_headers(most_restrictive)
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers=headers
        )
    
    return most_restrictive