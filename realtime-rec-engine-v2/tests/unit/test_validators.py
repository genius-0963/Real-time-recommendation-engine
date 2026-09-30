"""
Unit tests for validators module.
"""

import pytest
from pydantic import ValidationError

from app.api.validators import (
    RecommendationRequest,
    FeedbackRequest,
    BatchRecommendationRequest,
    BatchFeedbackRequest,
    validate_filter_keys,
    validate_context_keys,
    validate_feature_names,
    validate_candidate_items,
    validate_interaction_type,
    validate_rating,
    validate_experiment_info,
    format_validation_error,
    UserID,
    ItemID,
    SessionID,
)


class TestValidators:
    """Test individual validator functions."""
    
    def test_validate_filter_keys_valid(self):
        """Valid filter keys should pass."""
        filters = {"category": "electronics", "price_max": 500}
        result = validate_filter_keys(filters)
        assert result == filters
    
    def test_validate_filter_keys_invalid(self):
        """Invalid filter key should raise ValueError."""
        filters = {"invalid_key": "value"}
        with pytest.raises(ValueError, match="not allowed"):
            validate_filter_keys(filters)
    
    def test_validate_filter_keys_too_many(self):
        """Too many filter keys should raise ValueError."""
        filters = {f"category_{i}": "value" for i in range(25)}
        with pytest.raises(ValueError, match="Too many filter keys"):
            validate_filter_keys(filters)
    
    def test_validate_filter_keys_none(self):
        """None should return None."""
        result = validate_filter_keys(None)
        assert result is None
    
    def test_validate_context_keys_valid(self):
        """Valid context keys should pass."""
        context = {"device": "mobile", "platform": "ios"}
        result = validate_context_keys(context)
        assert result == context
    
    def test_validate_context_keys_invalid(self):
        """Invalid context key should raise ValueError."""
        context = {"secret_field": "value"}
        with pytest.raises(ValueError, match="not allowed"):
            validate_context_keys(context)
    
    def test_validate_context_keys_invalid_value_type(self):
        """Non-simple context value should raise ValueError."""
        context = {"device": ["mobile", "desktop"]}  # List not allowed
        with pytest.raises(ValueError, match="must be string, number, or boolean"):
            validate_context_keys(context)
    
    def test_validate_feature_names_valid(self):
        """Valid feature names should pass."""
        features = ["user_age", "item_category"]
        result = validate_feature_names(features)
        assert result == features
    
    def test_validate_feature_names_invalid(self):
        """Invalid feature name should raise ValueError."""
        features = ["invalid_feature"]
        with pytest.raises(ValueError, match="not allowed"):
            validate_feature_names(features)
    
    def test_validate_candidate_items_valid(self):
        """Valid candidate items should pass."""
        items = ["item_001", "item_002", "item-003"]
        result = validate_candidate_items(items)
        assert result == items
    
    def test_validate_candidate_items_invalid_format(self):
        """Invalid item ID format should raise ValueError."""
        items = ["item@invalid"]
        with pytest.raises(ValueError, match="Invalid item_id format"):
            validate_candidate_items(items)
    
    def test_validate_candidate_items_too_many(self):
        """Too many candidate items should raise ValueError."""
        items = [f"item_{i}" for i in range(1001)]
        with pytest.raises(ValueError, match="Too many candidate items"):
            validate_candidate_items(items)
    
    def test_validate_interaction_type_valid(self):
        """Valid interaction types should pass."""
        for itype in ["view", "click", "like", "share", "purchase", "add_to_cart", "remove_from_cart", "dismiss"]:
            result = validate_interaction_type(itype)
            assert result == itype
    
    def test_validate_interaction_type_invalid(self):
        """Invalid interaction type should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid interaction_type"):
            validate_interaction_type("invalid")
    
    def test_validate_rating_valid(self):
        """Valid ratings should pass."""
        for rating in [1, 1.0, 3.5, 5, 5.0]:
            result = validate_rating(rating)
            assert result == float(rating)
    
    def test_validate_rating_invalid_low(self):
        """Rating below 1 should raise ValueError."""
        with pytest.raises(ValueError, match="between 1 and 5"):
            validate_rating(0.5)
    
    def test_validate_rating_invalid_high(self):
        """Rating above 5 should raise ValueError."""
        with pytest.raises(ValueError, match="between 1 and 5"):
            validate_rating(5.1)
    
    def test_validate_rating_none(self):
        """None rating should return None."""
        result = validate_rating(None)
        assert result is None
    
    def test_validate_experiment_info_valid(self):
        """Valid experiment info should pass."""
        info = {"experiment_id": "exp_001", "variant": "treatment"}
        result = validate_experiment_info(info)
        assert result == info
    
    def test_validate_experiment_info_invalid_id(self):
        """Invalid experiment ID should raise ValueError."""
        info = {"experiment_id": "exp@invalid"}
        with pytest.raises(ValueError, match="Invalid experiment_id format"):
            validate_experiment_info(info)
    
    def test_validate_experiment_info_invalid_variant(self):
        """Invalid variant should raise ValueError."""
        info = {"variant": "invalid_variant"}
        with pytest.raises(ValueError, match="Invalid variant"):
            validate_experiment_info(info)


class TestStringConstraints:
    """Test string constraint types."""
    
    def test_user_id_valid(self):
        """Valid user IDs should pass."""
        for uid in ["user_123", "user-123", "u1", "a" * 64]:
            assert UserID.validate(uid) == uid
    
    def test_user_id_invalid_chars(self):
        """Invalid characters should fail."""
        with pytest.raises(ValidationError):
            UserID.validate("user@123")
    
    def test_user_id_too_long(self):
        """Too long should fail."""
        with pytest.raises(ValidationError):
            UserID.validate("u" * 65)
    
    def test_item_id_valid(self):
        """Valid item IDs should pass."""
        for iid in ["item_001", "item-001", "i1", "a" * 64]:
            assert ItemID.validate(iid) == iid
    
    def test_session_id_valid(self):
        """Valid session IDs should pass."""
        for sid in ["sess_abc", "sess-123", "a" * 128]:
            assert SessionID.validate(sid) == sid


class TestRecommendationRequest:
    """Test RecommendationRequest model."""
    
    def test_minimal_valid_request(self):
        """Minimal valid request should pass."""
        req = RecommendationRequest(user_id="user_123")
        assert req.user_id == "user_123"
        assert req.num_recommendations == 10  # default
        assert req.candidate_items is None
        assert req.filters is None
        assert req.context is None
        assert req.ab_test_info is None
    
    def test_full_valid_request(self):
        """Full valid request should pass."""
        req = RecommendationRequest(
            user_id="user_123",
            session_id="sess_abc",
            num_recommendations=20,
            candidate_items=["item_1", "item_2"],
            filters={"category": "electronics", "price_max": 500},
            context={"device": "mobile", "timezone": "UTC"},
            ab_test_info={"experiment_id": "exp_001", "variant": "treatment"}
        )
        assert req.user_id == "user_123"
        assert req.num_recommendations == 20
        assert len(req.candidate_items) == 2
        assert req.filters["category"] == "electronics"
        assert req.context["device"] == "mobile"
        assert req.ab_test_info["variant"] == "treatment"
    
    def test_invalid_num_recommendations_low(self):
        """num_recommendations < 1 should fail."""
        with pytest.raises(ValidationError):
            RecommendationRequest(user_id="user_123", num_recommendations=0)
    
    def test_invalid_num_recommendations_high(self):
        """num_recommendations > 100 should fail."""
        with pytest.raises(ValidationError):
            RecommendationRequest(user_id="user_123", num_recommendations=101)
    
    def test_extra_fields_forbidden(self):
        """Extra fields should be forbidden."""
        with pytest.raises(ValidationError):
            RecommendationRequest(user_id="user_123", extra_field="value")


class TestFeedbackRequest:
    """Test FeedbackRequest model."""
    
    def test_minimal_valid_feedback(self):
        """Minimal valid feedback should pass."""
        fb = FeedbackRequest(
            user_id="user_123",
            item_id="item_001",
            interaction_type="click"
        )
        assert fb.user_id == "user_123"
        assert fb.rating is None
    
    def test_full_valid_feedback(self):
        """Full valid feedback should pass."""
        fb = FeedbackRequest(
            user_id="user_123",
            item_id="item_001",
            interaction_type="purchase",
            rating=4.5,
            timestamp="2024-01-15T10:30:00Z",
            metadata={"position": 1, "page": "home"}
        )
        assert fb.rating == 4.5
        assert fb.metadata["position"] == 1
    
    def test_all_valid_interaction_types(self):
        """All valid interaction types should pass."""
        valid_types = ["view", "click", "like", "share", "purchase", "add_to_cart", "remove_from_cart", "dismiss"]
        for itype in valid_types:
            fb = FeedbackRequest(user_id="u1", item_id="i1", interaction_type=itype)
            assert fb.interaction_type == itype
    
    def test_invalid_rating_high(self):
        """Rating > 5 should fail."""
        with pytest.raises(ValidationError):
            FeedbackRequest(user_id="u1", item_id="i1", interaction_type="click", rating=5.1)
    
    def test_invalid_rating_low(self):
        """Rating < 1 should fail."""
        with pytest.raises(ValidationError):
            FeedbackRequest(user_id="u1", item_id="i1", interaction_type="click", rating=0.5)


class TestBatchRequests:
    """Test batch request models."""
    
    def test_batch_recommendation_valid(self):
        """Valid batch recommendation should pass."""
        req = BatchRecommendationRequest(
            requests=[
                {"user_id": "user_1", "num_recommendations": 10},
                {"user_id": "user_2", "num_recommendations": 5}
            ]
        )
        assert len(req.requests) == 2
    
    def test_batch_recommendation_too_many(self):
        """Too many requests should fail."""
        requests = [{"user_id": f"user_{i}"} for i in range(101)]
        with pytest.raises(ValidationError):
            BatchRecommendationRequest(requests=requests)
    
    def test_batch_feedback_valid(self):
        """Valid batch feedback should pass."""
        fb = BatchFeedbackRequest(
            feedback=[
                {"user_id": "user_1", "item_id": "item_1", "interaction_type": "click"},
                {"user_id": "user_2", "item_id": "item_2", "interaction_type": "view"}
            ]
        )
        assert len(fb.feedback) == 2
    
    def test_batch_feedback_too_many(self):
        """Too many feedback items should fail."""
        feedback = [{"user_id": f"user_{i}", "item_id": "i1", "interaction_type": "click"} for i in range(1001)]
        with pytest.raises(ValidationError):
            BatchFeedbackRequest(feedback=feedback)


class TestFormatValidationError:
    """Test validation error formatting."""
    
    def test_format_validation_error(self):
        """Validation error should be formatted correctly."""
        try:
            RecommendationRequest(user_id="user@123")
        except ValidationError as e:
            formatted = format_validation_error(e, "req_123")
            
            assert formatted["error"]["code"] == 422
            assert formatted["error"]["type"] == "ValidationError"
            assert formatted["error"]["request_id"] == "req_123"
            assert "details" in formatted["error"]
            assert len(formatted["error"]["details"]) > 0
            detail = formatted["error"]["details"][0]
            assert "field" in detail
            assert "issue" in detail
            assert "type" in detail


if __name__ == "__main__":
    pytest.main([__file__, "-v"])