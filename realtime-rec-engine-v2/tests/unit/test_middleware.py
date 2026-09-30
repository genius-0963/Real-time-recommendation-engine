"""
Unit tests for middleware module.
"""

import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from app.monitoring.middleware import (
    CircuitBreakerConfig,
    CircuitBreaker,
    SizeLimitMiddleware,
    TimeoutMiddleware,
    CircuitBreakerMiddleware,
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
    MetricsMiddleware,
    setup_middleware,
)


class TestCircuitBreaker:
    """Test circuit breaker (duplicate from rate_limiter test but for middleware)."""
    
    def test_circuit_breaker_basic(self):
        """Test basic circuit breaker functionality."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=1,
            timeout_seconds=0.1
        )
        breaker = CircuitBreaker(config)
        
        # Initially closed
        assert breaker.is_allowed("/test")[0] is True
        
        # Record failures
        breaker.record_failure("/test", 500)
        assert breaker.is_allowed("/test")[0] is True
        
        breaker.record_failure("/test", 500)
        assert breaker.is_allowed("/test")[0] is False
        
        # Wait for half-open
        time.sleep(0.15)
        assert breaker.is_allowed("/test")[0] is True
        
        # Success should close
        breaker.record_success("/test")
        state = breaker.get_status("/test")
        assert state["state"] == "closed"


class TestSizeLimitMiddleware:
    """Test size limit middleware."""
    
    def test_size_limit_allows_small_request(self):
        """Small requests should be allowed."""
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}
        
        app.add_middleware(SizeLimitMiddleware, max_size=1000)
        
        client = TestClient(app)
        response = client.get("/test")
        assert response.status_code == 200
    
    def test_size_limit_rejects_large_content_length(self):
        """Large Content-Length should be rejected."""
        app = FastAPI()
        
        @app.post("/test")
        async def test_endpoint():
            return {"status": "ok"}
        
        app.add_middleware(SizeLimitMiddleware, max_size=100)
        
        client = TestClient(app)
        # Send request with large Content-Length header
        response = client.post(
            "/test",
            json={"data": "x" * 200},
            headers={"Content-Length": "200"}
        )
        assert response.status_code == 413
        body = response.json()
        assert "too large" in body["error"]["message"].lower()
    
    def test_size_limit_per_endpoint(self):
        """Per-endpoint size limits should work."""
        app = FastAPI()
        
        @app.post("/small")
        async def small_endpoint():
            return {"status": "ok"}
        
        @app.post("/large")
        async def large_endpoint():
            return {"status": "ok"}
        
        app.add_middleware(
            SizeLimitMiddleware,
            max_size=1000,
            max_size_per_endpoint={"/small": 100}
        )
        
        client = TestClient(app)
        
        # Small endpoint should reject > 100 bytes
        response = client.post("/small", json={"data": "x" * 200}, headers={"Content-Length": "200"})
        assert response.status_code == 413
        
        # Large endpoint should allow (uses default 1000)
        response = client.post("/large", json={"data": "x" * 500}, headers={"Content-Length": "500"})
        assert response.status_code == 200


class TestTimeoutMiddleware:
    """Test timeout middleware."""
    
    def test_timeout_allows_fast_response(self):
        """Fast responses should complete."""
        app = FastAPI()
        
        @app.get("/fast")
        async def fast_endpoint():
            return {"status": "ok"}
        
        app.add_middleware(TimeoutMiddleware, default_timeout=1.0)
        
        client = TestClient(app)
        response = client.get("/fast")
        assert response.status_code == 200
    
    def test_timeout_rejects_slow_response(self):
        """Slow responses should timeout."""
        app = FastAPI()
        
        @app.get("/slow")
        async def slow_endpoint():
            import asyncio
            await asyncio.sleep(0.5)
            return {"status": "ok"}
        
        app.add_middleware(TimeoutMiddleware, default_timeout=0.1)
        
        client = TestClient(app)
        response = client.get("/slow")
        assert response.status_code == 504
        body = response.json()
        assert "timeout" in body["error"]["message"].lower()
    
    def test_timeout_per_endpoint(self):
        """Per-endpoint timeouts should work."""
        app = FastAPI()
        
        @app.get("/short")
        async def short_endpoint():
            import asyncio
            await asyncio.sleep(0.05)
            return {"status": "ok"}
        
        @app.get("/long")
        async def long_endpoint():
            import asyncio
            await asyncio.sleep(0.5)
            return {"status": "ok"}
        
        app.add_middleware(
            TimeoutMiddleware,
            default_timeout=0.1,
            timeout_per_endpoint={"/long": 1.0}
        )
        
        client = TestClient(app)
        
        # Short endpoint with 0.05s should pass with 0.1s default
        response = client.get("/short")
        assert response.status_code == 200
        
        # Long endpoint with 0.5s should pass with 1.0s override
        response = client.get("/long")
        assert response.status_code == 200


class TestCircuitBreakerMiddleware:
    """Test circuit breaker middleware."""
    
    def test_circuit_breaker_allows_initially(self):
        """Requests should be allowed initially."""
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}
        
        app.add_middleware(CircuitBreakerMiddleware)
        
        client = TestClient(app)
        response = client.get("/test")
        assert response.status_code == 200
        assert response.headers["X-Circuit-Breaker"] == "closed"
    
    def test_circuit_breaker_opens_on_failures(self):
        """Circuit should open after failures."""
        app = FastAPI()
        
        @app.get("/fail")
        async def fail_endpoint():
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail="Internal error")
        
        @app.get("/ok")
        async def ok_endpoint():
            return {"status": "ok"}
        
        config = CircuitBreakerConfig(failure_threshold=2, timeout_seconds=60)
        app.add_middleware(CircuitBreakerMiddleware, config=config)
        
        client = TestClient(app)
        
        # First failure
        response = client.get("/fail")
        assert response.status_code == 500
        
        # Second failure - should open circuit
        response = client.get("/fail")
        assert response.status_code == 500
        
        # Third request to same endpoint should be rejected by circuit breaker
        response = client.get("/fail")
        assert response.status_code == 503
        assert "circuit breaker" in response.json()["error"]["message"].lower()
        
        # Other endpoints should still work
        response = client.get("/ok")
        assert response.status_code == 200


class TestRequestIDMiddleware:
    """Test request ID middleware."""
    
    def test_request_id_generated(self):
        """Request ID should be generated if not provided."""
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint(request: Request):
            return {"request_id": request.state.request_id}
        
        app.add_middleware(RequestIDMiddleware)
        
        client = TestClient(app)
        response = client.get("/test")
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) > 0
        assert response.json()["request_id"] == response.headers["X-Request-ID"]
    
    def test_request_id_passthrough(self):
        """Custom request ID should be passed through."""
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint(request: Request):
            return {"request_id": request.state.request_id}
        
        app.add_middleware(RequestIDMiddleware)
        
        client = TestClient(app)
        custom_id = "custom-request-123"
        response = client.get("/test", headers={"X-Request-ID": custom_id})
        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == custom_id
        assert response.json()["request_id"] == custom_id


class TestSecurityHeadersMiddleware:
    """Test security headers middleware."""
    
    def test_security_headers_added(self):
        """Security headers should be added to responses."""
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}
        
        app.add_middleware(SecurityHeadersMiddleware)
        
        client = TestClient(app)
        response = client.get("/test")
        assert response.status_code == 200
        
        assert "Strict-Transport-Security" in response.headers
        assert "max-age=31536000" in response.headers["Strict-Transport-Security"]
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "Content-Security-Policy" in response.headers
        assert "Referrer-Policy" in response.headers
        assert "Permissions-Policy" in response.headers


class TestMetricsMiddleware:
    """Test metrics middleware."""
    
    def test_metrics_collector_called(self):
        """Metrics collector should be called."""
        mock_collector = AsyncMock()
        mock_collector.record_request = AsyncMock()
        
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}
        
        app.add_middleware(MetricsMiddleware, metrics_collector=mock_collector)
        
        client = TestClient(app)
        response = client.get("/test")
        assert response.status_code == 200
        
        mock_collector.record_request.assert_called_once()
        call_args = mock_collector.record_request.call_args
        assert call_args[1]["method"] == "GET"
        assert call_args[1]["endpoint"] == "/test"
        assert call_args[1]["status_code"] == 200
        assert "duration" in call_args[1]
    
    def test_response_time_header(self):
        """Response time header should be added."""
        mock_collector = AsyncMock()
        mock_collector.record_request = AsyncMock()
        
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}
        
        app.add_middleware(MetricsMiddleware, metrics_collector=mock_collector)
        
        client = TestClient(app)
        response = client.get("/test")
        assert response.status_code == 200
        assert "X-Response-Time" in response.headers
        assert "ms" in response.headers["X-Response-Time"]


class TestSetupMiddleware:
    """Test middleware setup function."""
    
    def test_setup_middleware_adds_all(self):
        """setup_middleware should add all middleware."""
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}
        
        config = {
            "hsts_max_age": 31536000,
            "csp_policy": "default-src 'self';",
            "max_request_size": 1000000,
            "request_timeout": 30,
            "circuit_breaker_threshold": 5,
            "circuit_breaker_timeout": 60,
        }
        
        mock_collector = AsyncMock()
        mock_collector.record_request = AsyncMock()
        
        setup_middleware(app, config, mock_collector)
        
        # Check middleware stack order (last added = first executed)
        middleware_types = [type(m) for m in app.user_middleware]
        
        # Should have all middleware types
        assert any(issubclass(m, SecurityHeadersMiddleware) for m in middleware_types)
        assert any(issubclass(m, RequestIDMiddleware) for m in middleware_types)
        assert any(issubclass(m, SizeLimitMiddleware) for m in middleware_types)
        assert any(issubclass(m, TimeoutMiddleware) for m in middleware_types)
        assert any(issubclass(m, CircuitBreakerMiddleware) for m in middleware_types)
        assert any(issubclass(m, MetricsMiddleware) for m in middleware_types)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])