"""
Middleware for request size limits, timeouts, and circuit breaker.
"""

import time
import logging
from typing import Dict, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for an endpoint."""
    failures: int = 0
    successes: int = 0
    last_failure_time: float = 0
    last_success_time: float = 0
    state: str = "closed"  # closed, open, half-open
    next_attempt: float = 0


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 2          # Successes to close from half-open
    timeout_seconds: float = 60.0       # Time before half-open
    half_open_max_calls: int = 3        # Max calls in half-open state
    excluded_status_codes: set = field(default_factory=lambda: {400, 401, 403, 404, 422})


class CircuitBreaker:
    """Circuit breaker for endpoint protection."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self._states: Dict[str, CircuitBreakerState] = defaultdict(CircuitBreakerState)
        self._half_open_calls: Dict[str, int] = defaultdict(int)
    
    def _get_state(self, endpoint: str) -> CircuitBreakerState:
        return self._states[endpoint]
    
    def record_success(self, endpoint: str):
        """Record successful call."""
        state = self._get_state(endpoint)
        state.successes += 1
        state.last_success_time = time.time()
        
        if state.state == "half-open":
            if state.successes >= self.config.success_threshold:
                state.state = "closed"
                state.failures = 0
                state.successes = 0
                logger.info(f"Circuit breaker CLOSED for {endpoint}")
    
    def record_failure(self, endpoint: str, status_code: int = 500):
        """Record failed call."""
        # Don't count client errors as failures
        if status_code in self.config.excluded_status_codes:
            return
            
        state = self._get_state(endpoint)
        state.failures += 1
        state.last_failure_time = time.time()
        
        if state.state == "closed":
            if state.failures >= self.config.failure_threshold:
                state.state = "open"
                state.next_attempt = time.time() + self.config.timeout_seconds
                logger.warning(f"Circuit breaker OPENED for {endpoint} after {state.failures} failures")
        
        elif state.state == "half-open":
            # Any failure in half-open reopens the circuit
            state.state = "open"
            state.next_attempt = time.time() + self.config.timeout_seconds
            state.successes = 0
            logger.warning(f"Circuit breaker REOPENED for {endpoint}")
    
    def is_allowed(self, endpoint: str) -> Tuple[bool, Optional[int]]:
        """Check if call is allowed."""
        state = self._get_state(endpoint)
        
        if state.state == "closed":
            return True, None
        
        if state.state == "open":
            if time.time() >= state.next_attempt:
                state.state = "half-open"
                state.successes = 0
                state.failures = 0
                self._half_open_calls[endpoint] = 0
                logger.info(f"Circuit breaker HALF-OPEN for {endpoint}")
                return True, None
            else:
                retry_after = int(state.next_attempt - time.time()) + 1
                return False, retry_after
        
        if state.state == "half-open":
            calls = self._half_open_calls[endpoint]
            if calls >= self.config.half_open_max_calls:
                retry_after = int(state.next_attempt - time.time()) + 1
                return False, retry_after
            self._half_open_calls[endpoint] = calls + 1
            return True, None
        
        return True, None
    
    def get_status(self, endpoint: str) -> Dict[str, Any]:
        """Get circuit breaker status."""
        state = self._get_state(endpoint)
        return {
            "endpoint": endpoint,
            "state": state.state,
            "failures": state.failures,
            "successes": state.successes,
            "next_attempt": state.next_attempt if state.state == "open" else None
        }
    
    def reset(self, endpoint: str):
        """Manually reset circuit breaker."""
        if endpoint in self._states:
            del self._states[endpoint]
        if endpoint in self._half_open_calls:
            del self._half_open_calls[endpoint]
        logger.info(f"Circuit breaker RESET for {endpoint}")


class SizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce request body size limits."""
    
    def __init__(
        self,
        app: ASGIApp,
        max_size: int = 1_048_576,  # 1MB default
        max_size_per_endpoint: Optional[Dict[str, int]] = None
    ):
        super().__init__(app)
        self.max_size = max_size
        self.max_size_per_endpoint = max_size_per_endpoint or {
            "/recommend": 10_485,      # 10KB
            "/feedback": 5_242,        # 5KB
            "/recommend/batch": 104_857,  # 100KB
            "/feedback/batch": 1_048_576, # 1MB
        }
    
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        # Check Content-Length header
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                limit = self.max_size_per_endpoint.get(request.url.path, self.max_size)
                if size > limit:
                    logger.warning(
                        f"Request too large: {size} bytes for {request.url.path} "
                        f"(limit: {limit}) from {request.client.host}"
                    )
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": {
                                "code": 413,
                                "message": f"Request body too large. Maximum {limit} bytes allowed.",
                                "type": "PayloadTooLarge",
                                "limit": limit,
                                "received": size
                            }
                        }
                    )
            except ValueError:
                pass
        
        return await call_next(request)


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce request timeouts."""
    
    def __init__(
        self,
        app: ASGIApp,
        default_timeout: float = 30.0,
        timeout_per_endpoint: Optional[Dict[str, float]] = None
    ):
        super().__init__(app)
        self.default_timeout = default_timeout
        self.timeout_per_endpoint = timeout_per_endpoint or {
            "/recommend": 10.0,
            "/feedback": 5.0,
            "/user/{user_id}/features": 5.0,
            "/item/{item_id}/features": 5.0,
            "/metrics": 15.0,
            "/experiments": 10.0,
            "/health": 5.0,
        }
    
    def _match_endpoint(self, path: str) -> Optional[str]:
        """Match path to configured endpoint pattern."""
        if path in self.timeout_per_endpoint:
            return path
        # Try pattern matching for parameterized paths
        for pattern in self.timeout_per_endpoint:
            if "{" in pattern:
                import re
                regex = pattern.replace("{", "(?P<").replace("}", ">[^/]+)")
                if re.match(f"^{regex}$", path):
                    return pattern
        return None
    
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        matched = self._match_endpoint(request.url.path)
        timeout = self.timeout_per_endpoint.get(matched, self.default_timeout) if matched else self.default_timeout
        
        # Use asyncio.wait_for for timeout
        import asyncio
        try:
            return await asyncio.wait_for(call_next(request), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Request timeout: {request.url.path} after {timeout}s from {request.client.host}")
            return JSONResponse(
                status_code=504,
                content={
                    "error": {
                        "code": 504,
                        "message": f"Request timeout after {timeout} seconds",
                        "type": "TimeoutError",
                        "timeout": timeout
                    }
                }
            )


class CircuitBreakerMiddleware(BaseHTTPMiddleware):
    """Middleware for circuit breaker pattern."""
    
    def __init__(self, app: ASGIApp, config: Optional[CircuitBreakerConfig] = None):
        super().__init__(app)
        self.breaker = CircuitBreaker(config or CircuitBreakerConfig())
    
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        endpoint = request.url.path
        
        # Check if allowed
        allowed, retry_after = self.breaker.is_allowed(endpoint)
        if not allowed:
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "code": 503,
                        "message": "Service temporarily unavailable (circuit breaker open)",
                        "type": "CircuitBreakerOpen",
                        "retry_after": retry_after
                    }
                },
                headers={"Retry-After": str(retry_after)}
            )
        
        # Process request
        response = await call_next(request)
        
        # Record result
        if response.status_code >= 500:
            self.breaker.record_failure(endpoint, response.status_code)
        else:
            self.breaker.record_success(endpoint)
        
        # Add circuit breaker header
        status = self.breaker.get_status(endpoint)
        response.headers["X-Circuit-Breaker"] = status["state"]
        
        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add request ID to all requests/responses."""
    
    def __init__(self, app: ASGIApp, header_name: str = "X-Request-ID"):
        super().__init__(app)
        self.header_name = header_name
    
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        import uuid
        request_id = request.headers.get(self.header_name) or str(uuid.uuid4())
        request.state.request_id = request_id
        
        response = await call_next(request)
        response.headers[self.header_name] = request_id
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers."""
    
    def __init__(
        self,
        app: ASGIApp,
        hsts_max_age: int = 31536000,
        csp_policy: str = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'self'; form-action 'self';"
    ):
        super().__init__(app)
        self.hsts_max_age = hsts_max_age
        self.csp_policy = csp_policy
    
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        response = await call_next(request)
        
        # Security headers
        response.headers["Strict-Transport-Security"] = f"max-age={self.hsts_max_age}; includeSubDomains"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = self.csp_policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting request metrics."""
    
    def __init__(self, app: ASGIApp, metrics_collector=None):
        super().__init__(app)
        self.metrics_collector = metrics_collector
    
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        start_time = time.time()
        request_id = getattr(request.state, "request_id", "unknown")
        
        response = await call_next(request)
        
        duration = time.time() - start_time
        
        # Record metrics if collector available
        if self.metrics_collector:
            try:
                await self.metrics_collector.record_request(
                    method=request.method,
                    endpoint=request.url.path,
                    status_code=response.status_code,
                    duration=duration,
                    request_id=request_id
                )
            except Exception as e:
                logger.error(f"Failed to record metrics: {e}")
        
        # Add timing header
        response.headers["X-Response-Time"] = f"{duration*1000:.2f}ms"
        
        return response


# Combined middleware setup
def setup_middleware(app, config=None, metrics_collector=None):
    """Setup all middleware in correct order."""
    
    # Order matters - outermost first
    # 1. Security headers (outermost)
    security_config = config or {}
    app.add_middleware(
        SecurityHeadersMiddleware,
        hsts_max_age=security_config.get("hsts_max_age", 31536000),
        csp_policy=security_config.get("csp_policy", "default-src 'self';")
    )
    
    # 2. Request ID
    app.add_middleware(RequestIDMiddleware)
    
    # 3. Size limits
    app.add_middleware(
        SizeLimitMiddleware,
        max_size=security_config.get("max_request_size", 1_048_576),
        max_size_per_endpoint=security_config.get("max_size_per_endpoint")
    )
    
    # 4. Timeout
    app.add_middleware(
        TimeoutMiddleware,
        default_timeout=security_config.get("request_timeout", 30.0),
        timeout_per_endpoint=security_config.get("timeout_per_endpoint")
    )
    
    # 5. Circuit breaker
    cb_config = CircuitBreakerConfig(
        failure_threshold=security_config.get("circuit_breaker_threshold", 5),
        timeout_seconds=security_config.get("circuit_breaker_timeout", 60.0)
    )
    app.add_middleware(CircuitBreakerMiddleware, config=cb_config)
    
    # 6. Metrics (innermost, wraps actual handler)
    if metrics_collector:
        app.add_middleware(MetricsMiddleware, metrics_collector=metrics_collector)
    
    return app