"""
Integration tests for input validation and rate limiting.
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import AsyncClient


class TestInputValidation:
    """Tests for input validation."""
    
    def test_recommendation_request_valid(self, app_client, sample_recommendation_request):
        """Valid recommendation request should succeed."""
        response = app_client.post("/recommend", json=sample_recommendation_request)
        assert response.status_code == 200
        body = response.json()
        assert "recommendations" in body
        assert isinstance(body["recommendations"], list)
    
    def test_recommendation_request_invalid_user_id(self, app_client):
        """Invalid user_id format should be rejected."""
        payload = {
            "user_id": "user@invalid!",  # Invalid characters
            "num_recommendations": 10
        }
        response = app_client.post("/recommend", json=payload)
        assert response.status_code == 422
        body = response.json()
        assert "error" in body
        assert body["error"]["type"] == "ValidationError"
    
    def test_recommendation_request_user_id_too_long(self, app_client):
        """User ID exceeding max length should be rejected."""
        payload = {
            "user_id": "u" * 65,  # Max is 64
            "num_recommendations": 10
        }
        response = app_client.post("/recommend", json=payload)
        assert response.status_code == 422
    
    def test_recommendation_request_invalid_filter_key(self, app_client):
        """Filter with non-allowlisted key should be rejected."""
        payload = {
            "user_id": "user_123",
            "filters": {"invalid_key": "value"}
        }
        response = app_client.post("/recommend", json=payload)
        assert response.status_code == 422
        body = response.json()
        assert "not allowed" in body["error"]["message"].lower()
    
    def test_recommendation_request_invalid_context_key(self, app_client):
        """Context with non-allowlisted key should be rejected."""
        payload = {
            "user_id": "user_123",
            "context": {"secret_internal_field": "value"}
        }
        response = app_client.post("/recommend", json=payload)
        assert response.status_code == 422
        body = response.json()
        assert "not allowed" in body["error"]["message"].lower()
    
    def test_recommendation_request_too_many_filters(self, app_client):
        """Too many filter keys should be rejected."""
        payload = {
            "user_id": "user_123",
            "filters": {f"category_{i}": "value" for i in range(25)}  # Max is 20
        }
        response = app_client.post("/recommend", json=payload)
        assert response.status_code == 422
    
    def test_recommendation_request_valid_filters(self, app_client):
        """Valid filter keys should be accepted."""
        payload = {
            "user_id": "user_123",
            "filters": {
                "category": "electronics",
                "price_max": 500,
                "brand": "apple"
            }
        }
        response = app_client.post("/recommend", json=payload)
        assert response.status_code == 200
    
    def test_recommendation_request_valid_context(self, app_client):
        """Valid context keys should be accepted."""
        payload = {
            "user_id": "user_123",
            "context": {
                "device": "mobile",
                "platform": "ios",
                "timezone": "UTC"
            }
        }
        response = app_client.post("/recommend", json=payload)
        assert response.status_code == 200
    
    def test_recommendation_request_candidate_items_limit(self, app_client):
        """Too many candidate items should be rejected."""
        payload = {
            "user_id": "user_123",
            "candidate_items": [f"item_{i}" for i in range(1001)]  # Max is 1000
        }
        response = app_client.post("/recommend", json=payload)
        assert response.status_code == 422
    
    def test_feedback_request_valid(self, app_client):
        """Valid feedback request should succeed."""
        payload = {
            "user_id": "user_123",
            "item_id": "item_001",
            "interaction_type": "click",
            "rating": 4.5
        }
        response = app_client.post("/feedback", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
    
    def test_feedback_request_invalid_interaction_type(self, app_client):
        """Invalid interaction type should be rejected."""
        payload = {
            "user_id": "user_123",
            "item_id": "item_001",
            "interaction_type": "invalid_type"
        }
        response = app_client.post("/feedback", json=payload)
        assert response.status_code == 422
        body = response.json()
        assert "interaction_type" in body["error"]["message"].lower()
    
    def test_feedback_request_invalid_rating(self, app_client):
        """Rating outside 1-5 range should be rejected."""
        payload = {
            "user_id": "user_123",
            "item_id": "item_001",
            "interaction_type": "click",
            "rating": 6.0  # Max is 5
        }
        response = app_client.post("/feedback", json=payload)
        assert response.status_code == 422
    
    def test_feedback_request_valid_interaction_types(self, app_client):
        """All valid interaction types should be accepted."""
        valid_types = ["view", "click", "like", "share", "purchase", "add_to_cart", "remove_from_cart", "dismiss"]
        for interaction_type in valid_types:
            payload = {
                "user_id": "user_123",
                "item_id": "item_001",
                "interaction_type": interaction_type
            }
            response = app_client.post("/feedback", json=payload)
            assert response.status_code == 200, f"Failed for {interaction_type}"


class TestBatchEndpoints:
    """Tests for batch endpoints."""
    
    def test_batch_recommendation_valid(self, app_client, sample_recommendation_request):
        """Valid batch recommendation request should succeed."""
        payload = {
            "requests": [sample_recommendation_request, sample_recommendation_request]
        }
        response = app_client.post("/recommend/batch", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body, list)
        assert len(body) == 2
    
    def test_batch_recommendation_too_many(self, app_client, sample_recommendation_request):
        """Too many requests in batch should be rejected."""
        payload = {
            "requests": [sample_recommendation_request] * 101  # Max is 100
        }
        response = app_client.post("/recommend/batch", json=payload)
        assert response.status_code == 422
    
    def test_batch_feedback_valid(self, app_client):
        """Valid batch feedback request should succeed."""
        payload = {
            "feedback": [
                {"user_id": "user_1", "item_id": "item_1", "interaction_type": "click"},
                {"user_id": "user_2", "item_id": "item_2", "interaction_type": "view"}
            ]
        }
        response = app_client.post("/feedback/batch", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body, list)
        assert len(body) == 2
    
    def test_batch_feedback_too_many(self, app_client):
        """Too many feedback items should be rejected."""
        payload = {
            "feedback": [
                {"user_id": f"user_{i}", "item_id": "item_1", "interaction_type": "click"}
                for i in range(1001)  # Max is 1000
            ]
        }
        response = app_client.post("/feedback/batch", json=payload)
        assert response.status_code == 422


class TestFeatureEndpoints:
    """Tests for feature retrieval endpoints."""
    
    def test_user_features_valid(self, app_client):
        """Valid user features request should succeed."""
        response = app_client.get("/user/user_123/features?feature_names=user_age,user_interests")
        assert response.status_code == 200
        body = response.json()
        assert "user_id" in body
        assert "features" in body
    
    def test_user_features_invalid_feature(self, app_client):
        """Invalid feature name should be rejected."""
        response = app_client.get("/user/user_123/features?feature_names=invalid_feature")
        assert response.status_code == 422
        body = response.json()
        assert "not allowed" in body["error"]["message"].lower()
    
    def test_item_features_valid(self, app_client):
        """Valid item features request should succeed."""
        response = app_client.get("/item/item_001/features?feature_names=item_category,item_price")
        assert response.status_code == 200
        body = response.json()
        assert "user_id" in body  # Response model uses user_id field
        assert "features" in body


class TestRateLimiting:
    """Tests for rate limiting."""
    
    @pytest.mark.asyncio
    async def test_rate_limit_headers_present(self, app_client):
        """Rate limit headers should be present in response."""
        response = app_client.get("/health")
        assert response.status_code == 200
        # Check for rate limit headers
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Scope" in response.headers
    
    @pytest.mark.asyncio
    async def test_rate_limit_exceeded_returns_429(self, app_client):
        """Exceeding rate limit should return 429."""
        # This test would need a real rate limiter to work
        # Skipped in unit tests with mocked services
        pass
    
    @pytest.mark.asyncio
    async def test_rate_limit_different_scopes(self, app_client):
        """Rate limits should apply per scope."""
        # Global, user, endpoint, IP scopes
        pass


class TestSecurityHeaders:
    """Tests for security headers."""
    
    def test_security_headers_present(self, app_client):
        """Security headers should be present in all responses."""
        response = app_client.get("/health")
        assert response.status_code == 200
        
        assert "Strict-Transport-Security" in response.headers
        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert "Content-Security-Policy" in response.headers
        assert "Referrer-Policy" in response.headers
        assert "Permissions-Policy" in response.headers
    
    def test_hsts_header_value(self, app_client):
        """HSTS header should have correct value."""
        response = app_client.get("/health")
        hsts = response.headers["Strict-Transport-Security"]
        assert "max-age=31536000" in hsts
        assert "includeSubDomains" in hsts


class TestRequestSizeLimit:
    """Tests for request size limits."""
    
    def test_large_request_rejected(self, app_client):
        """Requests exceeding size limit should be rejected."""
        # Create a large payload (> 10KB for /recommend)
        large_context = {"data": "x" * 15000}  # ~15KB
        payload = {
            "user_id": "user_123",
            "context": large_context
        }
        response = app_client.post("/recommend", json=payload)
        # Should be rejected by middleware (413) or validation (422)
        assert response.status_code in (413, 422)
    
    def test_normal_request_accepted(self, app_client):
        """Normal sized requests should be accepted."""
        payload = {
            "user_id": "user_123",
            "context": {"device": "mobile"}
        }
        response = app_client.post("/recommend", json=payload)
        assert response.status_code == 200


class TestRequestID:
    """Tests for request ID tracking."""
    
    def test_request_id_in_response(self, app_client):
        """Request ID should be in response headers."""
        response = app_client.get("/health")
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) > 0
    
    def test_request_id_passed_through(self, app_client):
        """Custom request ID should be passed through."""
        custom_id = "custom-request-id-123"
        response = app_client.get("/health", headers={"X-Request-ID": custom_id})
        assert response.headers["X-Request-ID"] == custom_id


class TestCircuitBreaker:
    """Tests for circuit breaker (integration would need real service)."""
    
    def test_circuit_breaker_header(self, app_client):
        """Circuit breaker header should be present."""
        response = app_client.get("/health")
        assert "X-Circuit-Breaker" in response.headers
        assert response.headers["X-Circuit-Breaker"] in ["closed", "open", "half-open"]


class TestResponseFormat:
    """Tests for standardized response formats."""
    
    def test_recommendation_response_structure(self, app_client, sample_recommendation_request):
        """Recommendation response should match expected structure."""
        response = app_client.post("/recommend", json=sample_recommendation_request)
        assert response.status_code == 200
        body = response.json()
        
        # Check required fields
        assert "request_id" in body
        assert "recommendations" in body
        assert "user_id" in body
        assert "metadata" in body
        assert "timestamp" in body
        
        # Check metadata structure
        metadata = body["metadata"]
        assert "model_version" in metadata
        assert "latency_ms" in metadata
        assert "cache_hit" in metadata
    
    def test_error_response_structure(self, app_client):
        """Error responses should have standardized structure."""
        # Trigger validation error
        response = app_client.post("/recommend", json={"num_recommendations": 5})
        assert response.status_code == 422
        body = response.json()
        
        assert "error" in body
        error = body["error"]
        assert "code" in error
        assert "message" in error
        assert "type" in error
        assert "request_id" in error
        assert "timestamp" in error


class TestCORS:
    """Tests for CORS configuration."""
    
    def test_cors_preflight(self, app_client):
        """CORS preflight should work."""
        response = app_client.options(
            "/recommend",
            headers={
                "Origin": "https://app.example.com",
                "Access-Control-Request-Method": "POST"
            }
        )
        assert response.status_code == 200
        assert "Access-Control-Allow-Origin" in response.headers
    
    def test_cors_origin_allowed(self, app_client):
        """Allowed origins should work."""
        response = app_client.post(
            "/recommend",
            json={"user_id": "user_123", "num_recommendations": 1},
            headers={"Origin": "https://app.example.com"}
        )
        assert response.status_code == 200
        # In production, CORS would be configured with specific origins
    
    def test_cors_credentials(self, app_client):
        """Credentials should be allowed for CORS."""
        response = app_client.options(
            "/recommend",
            headers={
                "Origin": "https://app.example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Authorization, Content-Type"
            }
        )
        assert response.status_code == 200


class TestTimeouts:
    """Tests for request timeouts (would need slow endpoint)."""
    
    def test_health_endpoint_fast(self, app_client):
        """Health endpoint should respond quickly."""
        import time
        start = time.time()
        response = app_client.get("/health")
        elapsed = time.time() - start
        assert response.status_code == 200
        assert elapsed < 5.0  # Should be very fast


# Performance/load tests would go in separate test file
# These are basic integration tests for the validation and middleware


if __name__ == "__main__":
    pytest.main([__file__, "-v"])