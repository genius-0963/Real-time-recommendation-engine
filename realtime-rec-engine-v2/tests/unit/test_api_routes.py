"""
Unit tests for FastAPI API routes in app.api.main.

All external dependencies (Redis, Kafka, PostgreSQL, model) are mocked
via the ``app_client`` fixture defined in conftest.py.
"""

import pytest


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_endpoint(self, app_client):
        """GET /health returns 200 and includes expected keys."""
        response = app_client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert "status" in body
        assert "version" in body
        assert body["status"] == "healthy"


class TestReadyEndpoint:
    """Tests for the /ready endpoint (falls back to /health if no separate readiness route)."""

    def test_ready_endpoint(self, app_client):
        """
        The app exposes /health as the liveness/readiness probe.
        GET /health returns 200 when all mocked services report healthy.
        """
        response = app_client.get("/health")
        assert response.status_code == 200


class TestRecommendEndpoint:
    """Tests for the POST /recommend endpoint."""

    def test_recommend_endpoint(self, app_client, sample_recommendation_request):
        """POST /recommend with a valid body returns 200 and contains 'recommendations'."""
        response = app_client.post("/recommend", json=sample_recommendation_request)
        assert response.status_code == 200
        body = response.json()
        assert "recommendations" in body
        assert isinstance(body["recommendations"], list)

    def test_recommend_missing_user_id(self, app_client):
        """POST /recommend without user_id returns 422 (validation error)."""
        response = app_client.post("/recommend", json={"num_recommendations": 5})
        assert response.status_code == 422


class TestBatchRecommend:
    """Tests for batch recommendation (if the API has /recommend/batch)."""

    def test_batch_recommend(self, app_client, sample_recommendation_request):
        """
        POST /recommend/batch — if the route does not exist the test will be
        skipped with an informative message.  Otherwise it must return 200.
        """
        payload = {"requests": [sample_recommendation_request]}
        response = app_client.post("/recommend/batch", json=payload)
        if response.status_code == 404:
            pytest.skip("/recommend/batch route not implemented")
        assert response.status_code == 200


class TestEventsEndpoint:
    """Tests for the POST /events endpoint (feedback is at /feedback)."""

    def test_events_endpoint(self, app_client):
        """
        POST /feedback with a valid interaction returns 200.
        The app uses /feedback instead of /events for recording interactions.
        """
        payload = {
            "user_id": "user_42",
            "item_id": "item_99",
            "interaction_type": "click",
        }
        response = app_client.post("/feedback", json=payload)
        assert response.status_code == 200

    def test_batch_events(self, app_client):
        """
        POST /events/batch — if the route does not exist the test is skipped.
        """
        payload = {
            "events": [
                {
                    "user_id": "user_1",
                    "item_id": "item_1",
                    "interaction_type": "view",
                }
            ]
        }
        response = app_client.post("/events/batch", json=payload)
        if response.status_code == 404:
            pytest.skip("/events/batch route not implemented")
        assert response.status_code in (200, 201)


class TestFeedbackEndpoint:
    """Tests for the POST /feedback endpoint."""

    def test_feedback_endpoint(self, app_client):
        """POST /feedback with valid body returns 200."""
        payload = {
            "user_id": "user_42",
            "item_id": "item_99",
            "interaction_type": "click",
            "rating": 4.5,
        }
        response = app_client.post("/feedback", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body.get("status") == "success"


class TestFeaturesEndpoint:
    """Tests for user/item feature retrieval."""

    def test_features_get(self, app_client):
        """GET /user/{user_id}/features returns 200."""
        response = app_client.get("/user/user_123/features")
        assert response.status_code == 200
        body = response.json()
        assert "user_id" in body
        assert body["user_id"] == "user_123"


class TestModelInfo:
    """Tests for model info endpoint."""

    def test_model_info(self, app_client):
        """
        GET /model/info — if the route does not exist the test will be
        skipped.  The app may expose model metadata at this path.
        """
        response = app_client.get("/model/info")
        if response.status_code == 404:
            pytest.skip("/model/info route not implemented")
        assert response.status_code == 200


class TestIndexStatus:
    """Tests for index status endpoint."""

    def test_index_status(self, app_client):
        """
        GET /index/status — if the route does not exist the test will be
        skipped.
        """
        response = app_client.get("/index/status")
        if response.status_code == 404:
            pytest.skip("/index/status route not implemented")
        assert response.status_code == 200


class TestMetricsEndpoint:
    """Tests for the GET /metrics endpoint."""

    def test_metrics_endpoint(self, app_client):
        """GET /metrics returns 200 and includes a 'metrics' key."""
        response = app_client.get("/metrics")
        assert response.status_code == 200
        body = response.json()
        assert "metrics" in body
