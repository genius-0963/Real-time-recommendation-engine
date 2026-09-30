"""
Input validation and sanitization for API endpoints.
Uses Pydantic v2 with strict validation and sanitization.
"""

import re
import json
from typing import Dict, List, Optional, Any, Set, Annotated
from datetime import datetime
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    StringConstraints,
    BeforeValidator,
    ValidationError,
)
from pydantic.types import PositiveInt, NonNegativeInt
from typing_extensions import Annotated as TypedAnnotated


# ─── String Constraints ──────────────────────────────────────────────
UserID = Annotated[str, StringConstraints(
    pattern=r'^[a-zA-Z0-9_\-]{1,64}$',
    min_length=1,
    max_length=64
)]
ItemID = Annotated[str, StringConstraints(
    pattern=r'^[a-zA-Z0-9_\-]{1,64}$',
    min_length=1,
    max_length=64
)]
SessionID = Annotated[str, StringConstraints(
    pattern=r'^[a-zA-Z0-9_\-]{1,128}$',
    max_length=128
)]
ExperimentID = Annotated[str, StringConstraints(
    pattern=r'^[a-zA-Z0-9_\-]{1,64}$',
    min_length=1,
    max_length=64
)]

# Sanitized string - no control chars, limited length
SanitizedString = Annotated[str, StringConstraints(
    pattern=r'^[\x20-\x7E]*$',  # printable ASCII only
    max_length=1024
)]

# JSON-serializable value with size limit
def validate_json_size(v: Any) -> Any:
    """Validate JSON-serializable and size."""
    if v is None:
        return v
    # Check serializable
    try:
        json.dumps(v)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Value must be JSON-serializable: {e}")
    # Check size (max 10KB)
    if len(json.dumps(v)) > 10240:
        raise ValueError("JSON value too large (max 10KB)")
    return v

JSONValue = Annotated[Any, BeforeValidator(validate_json_size)]


# ─── Allowed Filter Keys ─────────────────────────────────────────────
ALLOWED_FILTER_KEYS: Set[str] = {
    "category", "brand", "price_min", "price_max",
    "tags", "language", "region", "age_rating",
    "content_type", "freshness_days", "popularity_threshold"
}

ALLOWED_CONTEXT_KEYS: Set[str] = {
    "device", "platform", "os", "app_version",
    "timezone", "locale", "referrer", "utm_source",
    "utm_medium", "utm_campaign", "ab_test_bucket"
}

ALLOWED_FEATURE_NAMES: Set[str] = {
    "user_age", "user_gender", "user_location", "user_interests",
    "user_past_purchases", "user_session_count", "user_avg_session_duration",
    "item_category", "item_brand", "item_price", "item_popularity",
    "item_recency", "item_tags", "item_embedding",
    "context_device", "context_time_of_day", "context_day_of_week",
    "context_is_weekend", "context_is_holiday"
}


# ─── Validators ──────────────────────────────────────────────────────

def validate_filter_keys(filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Validate filter keys against allowlist."""
    if filters is None:
        return None
    if not isinstance(filters, dict):
        raise ValueError("filters must be a dictionary")
    if len(filters) > 20:
        raise ValueError("Too many filter keys (max 20)")
    for key in filters:
        if key not in ALLOWED_FILTER_KEYS:
            raise ValueError(f"Filter key not allowed: {key}. Allowed: {sorted(ALLOWED_FILTER_KEYS)}")
    return filters

def validate_context_keys(context: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Validate context keys against allowlist."""
    if context is None:
        return None
    if not isinstance(context, dict):
        raise ValueError("context must be a dictionary")
    if len(context) > 20:
        raise ValueError("Too many context keys (max 20)")
    for key in context:
        if key not in ALLOWED_CONTEXT_KEYS:
            raise ValueError(f"Context key not allowed: {key}. Allowed: {sorted(ALLOWED_CONTEXT_KEYS)}")
        # Validate context values are simple types
        val = context[key]
        if not isinstance(val, (str, int, float, bool)):
            raise ValueError(f"Context value for '{key}' must be string, number, or boolean")
    return context

def validate_feature_names(features: Optional[List[str]]) -> Optional[List[str]]:
    """Validate feature names against allowlist."""
    if features is None:
        return None
    if not isinstance(features, list):
        raise ValueError("feature_names must be a list")
    if len(features) > 50:
        raise ValueError("Too many features requested (max 50)")
    for f in features:
        if f not in ALLOWED_FEATURE_NAMES:
            raise ValueError(f"Feature not allowed: {f}")
    return features

def validate_candidate_items(items: Optional[List[str]]) -> Optional[List[str]]:
    """Validate candidate item IDs."""
    if items is None:
        return None
    if not isinstance(items, list):
        raise ValueError("candidate_items must be a list")
    if len(items) > 1000:
        raise ValueError("Too many candidate items (max 1000)")
    for item in items:
        if not re.match(r'^[a-zA-Z0-9_\-]{1,64}$', item):
            raise ValueError(f"Invalid item_id format: {item}")
    return items

def validate_interaction_type(interaction: str) -> str:
    """Validate interaction type."""
    allowed = {"view", "click", "like", "share", "purchase", "add_to_cart", "remove_from_cart", "dismiss"}
    if interaction not in allowed:
        raise ValueError(f"Invalid interaction_type: {interaction}. Allowed: {sorted(allowed)}")
    return interaction

def validate_rating(rating: Optional[float]) -> Optional[float]:
    """Validate rating is 1-5."""
    if rating is None:
        return None
    if not isinstance(rating, (int, float)):
        raise ValueError("Rating must be a number")
    if rating < 1 or rating > 5:
        raise ValueError("Rating must be between 1 and 5")
    return float(rating)

def validate_experiment_info(info: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """Validate A/B test info."""
    if info is None:
        return None
    if not isinstance(info, dict):
        raise ValueError("ab_test_info must be a dictionary")
    if "experiment_id" in info:
        if not re.match(r'^[a-zA-Z0-9_\-]{1,64}$', info["experiment_id"]):
            raise ValueError("Invalid experiment_id format")
    if "variant" in info:
        if info["variant"] not in {"control", "treatment", "variant_a", "variant_b"}:
            raise ValueError("Invalid variant")
    return info


# ─── Request Models ──────────────────────────────────────────────────

class RecommendationRequest(BaseModel):
    """Validated recommendation request."""
    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [{
                "user_id": "user_12345",
                "session_id": "sess_abcde",
                "num_recommendations": 10,
                "candidate_items": ["item_001", "item_002"],
                "filters": {"category": "electronics", "price_max": 500},
                "context": {"device": "mobile", "time_of_day": "evening"},
                "ab_test_info": {"experiment_id": "exp_001", "variant": "treatment"}
            }]
        }
    }
    
    user_id: UserID
    session_id: Optional[SessionID] = None
    num_recommendations: Annotated[int, Field(ge=1, le=100, default=10)]
    candidate_items: Annotated[Optional[List[ItemID]], BeforeValidator(validate_candidate_items)] = None
    filters: Annotated[Optional[Dict[str, JSONValue]], BeforeValidator(validate_filter_keys)] = None
    context: Annotated[Optional[Dict[str, JSONValue]], BeforeValidator(validate_context_keys)] = None
    ab_test_info: Annotated[Optional[Dict[str, str]], BeforeValidator(validate_experiment_info)] = None


class FeedbackRequest(BaseModel):
    """Validated feedback request."""
    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [{
                "user_id": "user_12345",
                "item_id": "item_001",
                "interaction_type": "click",
                "rating": 4.5,
                "timestamp": "2024-01-15T10:30:00Z",
                "metadata": {"position": 3, "page": "home"}
            }]
        }
    }
    
    user_id: UserID
    item_id: ItemID
    interaction_type: Annotated[str, BeforeValidator(validate_interaction_type)]
    rating: Annotated[Optional[float], BeforeValidator(validate_rating)] = None
    timestamp: Optional[datetime] = None
    metadata: Annotated[Optional[Dict[str, JSONValue]], BeforeValidator(validate_json_size)] = None


class BatchRecommendationRequest(BaseModel):
    """Batch recommendation request."""
    model_config = {"extra": "forbid"}
    
    requests: Annotated[List[RecommendationRequest], Field(min_length=1, max_length=100)]


class BatchFeedbackRequest(BaseModel):
    """Batch feedback request."""
    model_config = {"extra": "forbid"}
    
    feedback: Annotated[List[FeedbackRequest], Field(min_length=1, max_length=1000)]


class FeaturesRequest(BaseModel):
    """Feature retrieval request."""
    model_config = {"extra": "forbid"}
    
    feature_names: Annotated[Optional[List[str]], BeforeValidator(validate_feature_names)] = None


class ExperimentAssignRequest(BaseModel):
    """Experiment assignment request."""
    model_config = {"extra": "forbid"}
    
    user_id: UserID
    experiment_id: ExperimentID


# ─── Response Models ─────────────────────────────────────────────────

class RecommendationItem(BaseModel):
    """Single recommendation item."""
    item_id: ItemID
    score: Annotated[float, Field(ge=0.0, le=1.0)]
    rank: PositiveInt
    explanation: Optional[SanitizedString] = None
    metadata: Optional[Dict[str, JSONValue]] = None
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "item_id": "item_001",
                "score": 0.95,
                "rank": 1,
                "explanation": "Similar to your recent views",
                "metadata": {"category": "electronics", "price": 299}
            }
        }
    }


class RecommendationResponse(BaseModel):
    """Recommendation response."""
    request_id: str
    recommendations: List[RecommendationItem]
    user_id: UserID
    session_id: Optional[SessionID] = None
    metadata: Dict[str, JSONValue]
    timestamp: datetime
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "request_id": "req_abc123",
                "recommendations": [
                    {"item_id": "item_001", "score": 0.95, "rank": 1},
                    {"item_id": "item_002", "score": 0.87, "rank": 2}
                ],
                "user_id": "user_12345",
                "session_id": "sess_abcde",
                "metadata": {
                    "model_version": "2.1.0",
                    "latency_ms": 45.2,
                    "cache_hit": True,
                    "candidate_pool_size": 500,
                    "experiment_info": {"experiment_id": "exp_001", "variant": "treatment"}
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
    }


class FeedbackResponse(BaseModel):
    """Feedback response."""
    status: str = "success"
    message: str = "Feedback recorded successfully"
    request_id: str
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "success",
                "message": "Feedback recorded successfully",
                "request_id": "req_abc123"
            }
        }
    }


class FeaturesResponse(BaseModel):
    """Features response."""
    user_id: UserID
    features: Dict[str, JSONValue]
    timestamp: datetime
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "user_12345",
                "features": {"user_age": 30, "user_interests": ["tech", "gaming"]},
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
    }


class MetricsResponse(BaseModel):
    """Metrics response."""
    timestamp: datetime
    metrics: Dict[str, JSONValue]
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "timestamp": "2024-01-15T10:30:00Z",
                "metrics": {
                    "total_requests": 10000,
                    "avg_latency_ms": 25.0,
                    "p95_latency_ms": 85.0,
                    "cache_hit_rate": 0.72,
                    "error_rate": 0.001
                }
            }
        }
    }


class ExperimentResponse(BaseModel):
    """Experiment response."""
    experiment_id: ExperimentID
    user_id: UserID
    assignment: Dict[str, str]
    timestamp: datetime
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "experiment_id": "exp_001",
                "user_id": "user_12345",
                "assignment": {"variant": "treatment", "model_version": "2.1.0-exp"},
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
    }


class ErrorResponse(BaseModel):
    """Standardized error response."""
    error: Dict[str, Any]
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "error": {
                    "code": 400,
                    "message": "Invalid request: filter key 'invalid_key' not allowed",
                    "type": "ValidationError",
                    "details": [{"field": "filters", "issue": "invalid_key not in allowlist"}],
                    "request_id": "req_abc123",
                    "timestamp": "2024-01-15T10:30:00Z"
                }
            }
        }
    }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    services: Dict[str, str]
    version: str
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00Z",
                "services": {
                    "cache": "healthy",
                    "recommendation_service": "healthy",
                    "feature_service": "healthy"
                },
                "version": "2.0.0"
            }
        }
    }


# ─── Validation Error Formatting ────────────────────────────────────

def format_validation_error(exc: ValidationError, request_id: str) -> Dict[str, Any]:
    """Format Pydantic validation error for API response."""
    details = []
    for error in exc.errors():
        loc = " -> ".join(str(x) for x in error["loc"])
        details.append({
            "field": loc,
            "issue": error["msg"],
            "type": error["type"]
        })
    
    return {
        "error": {
            "code": 422,
            "message": "Request validation failed",
            "type": "ValidationError",
            "details": details,
            "request_id": request_id,
            "timestamp": datetime.now().isoformat()
        }
    }