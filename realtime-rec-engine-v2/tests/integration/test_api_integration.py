"""
Integration tests for the FastAPI recommendation API.

These tests require running backend services (Redis, PostgreSQL, Kafka, model).
They are marked with ``@pytest.mark.integration`` and skipped when services
are not reachable.
"""

import os
import pytest
from unittest.mock import patch, AsyncMock, MagicMock


def services_available() -> bool:
    """
    Quick check whether external services are reachable.
    Returns True only when INTEGRATION_TESTS=1 is set in the environment,
    meaning the operator has explicitly confirmed services are running.
    """
    return os.getenv("INTEGRATION_TESTS", "0") == "1"


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not services_available(), reason="Requires running services"),
]


class TestFullRecommendationFlow:
    """End-to-end recommendation request → response validation."""

    def test_full_recommendation_flow(self, app_client, sample_recommendation_request):
        """POST /recommend → verify response structure."""
        response = app_client.post("/recommend", json=sample_recommendation_request)
        assert response.status_code == 200

        body = response.json()
        assert "recommendations" in body
        assert "request_id" in body
        assert "user_id" in body
        assert body["user_id"] == sample_recommendation_request["user_id"]
        assert "metadata" in body
        assert "timestamp" in body

        # Each recommendation item must have the required fields
        for item in body["recommendations"]:
            assert "item_id" in item
            assert "score" in item
            assert "rank" in item


class TestEventIngestionFlow:
    """End-to-end event ingestion test."""

    def test_event_ingestion_flow(self, app_client):
        """POST /feedback → verify accepted."""
        payload = {
            "user_id": "integration_user",
            "item_id": "integration_item",
            "interaction_type": "click",
        }
        response = app_client.post("/feedback", json=payload)
        assert response.status_code == 200

        body = response.json()
        assert body.get("status") == "success"


class TestHealthWithServices:
    """Health check including downstream service status."""

    def test_health_with_services(self, app_client):
        """GET /health returns service status dict."""
        response = app_client.get("/health")
        assert response.status_code == 200

        body = response.json()
        assert "status" in body
        assert "services" in body
        assert isinstance(body["services"], dict)
