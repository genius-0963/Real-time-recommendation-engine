"""
FastAPI application for real-time recommendation inference.
Provides REST endpoints for recommendations, feedback, and monitoring.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

from app.config import Config
from app.services.recommendation_service import RecommendationService
from app.services.feature_service import FeatureService
from app.cache.redis_cache import RedisCache
from app.monitoring.metrics_collector import MetricsCollector
from app.monitoring.rate_limiter import RateLimiter
from app.experiments.ab_testing import ABTestManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
config = Config()
recommendation_service = None
feature_service = None
cache = None
metrics = None
rate_limiter = None
ab_test_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting recommendation engine API")
    
    global recommendation_service, feature_service, cache, metrics, rate_limiter, ab_test_manager
    
    try:
        # Initialize services
        cache = RedisCache(config.redis)
        feature_service = FeatureService(config.feature_store, cache)
        recommendation_service = RecommendationService(
            config.model, feature_service, cache
        )
        metrics = MetricsCollector(config.monitoring)
        rate_limiter = RateLimiter(config.api)
        ab_test_manager = ABTestManager(config.experiments)
        
        # Load models and indexes
        await recommendation_service.initialize()
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down recommendation engine API")
    
    if recommendation_service:
        await recommendation_service.cleanup()
    
    if cache:
        await cache.close()


# Initialize FastAPI app
app = FastAPI(
    title="Real-time Recommendation Engine",
    description="Production-grade recommendation system with sub-100ms latency",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=config.api.cors_methods,
    allow_headers=config.api.cors_headers,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Pydantic models
class RecommendationRequest(BaseModel):
    """Request model for recommendations."""
    user_id: str = Field(..., description="Unique user identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    num_recommendations: int = Field(10, ge=1, le=100, description="Number of recommendations")
    candidate_items: Optional[List[str]] = Field(None, description="Specific candidate items to rank")
    filters: Optional[Dict[str, Any]] = Field(None, description="Request filters")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    ab_test_info: Optional[Dict[str, str]] = Field(None, description="A/B test information")
    
    @validator('num_recommendations')
    def validate_num_recommendations(cls, v):
        if v < 1 or v > 100:
            raise ValueError('num_recommendations must be between 1 and 100')
        return v


class RecommendationItem(BaseModel):
    """Single recommendation item."""
    item_id: str = Field(..., description="Item identifier")
    score: float = Field(..., description="Recommendation score")
    rank: int = Field(..., description="Recommendation rank")
    explanation: Optional[str] = Field(None, description="Explanation for recommendation")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    request_id: str = Field(..., description="Unique request identifier")
    recommendations: List[RecommendationItem] = Field(..., description="List of recommendations")
    user_id: str = Field(..., description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    metadata: Dict[str, Any] = Field(..., description="Response metadata")
    timestamp: datetime = Field(..., description="Response timestamp")


class FeedbackRequest(BaseModel):
    """Request model for user feedback."""
    user_id: str = Field(..., description="User identifier")
    item_id: str = Field(..., description="Item identifier")
    interaction_type: str = Field(..., description="Type of interaction")
    rating: Optional[float] = Field(None, ge=1, le=5, description="User rating")
    timestamp: Optional[datetime] = Field(None, description="Interaction timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class MetricsResponse(BaseModel):
    """Response model for metrics."""
    timestamp: datetime = Field(..., description="Metrics timestamp")
    metrics: Dict[str, Any] = Field(..., description="Metrics data")


# Dependency injection
async def get_rate_limit_key(request: Request) -> str:
    """Extract rate limit key from request."""
    # Use IP address or user ID for rate limiting
    user_id = request.headers.get("X-User-ID")
    if user_id:
        return f"user:{user_id}"
    return f"ip:{request.client.host}"


async def check_rate_limit(key: str = Depends(get_rate_limit_key)) -> None:
    """Check rate limits."""
    if not rate_limiter.is_allowed(key):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(rate_limiter.get_retry_after(key))}
        )


# Middleware for request tracking
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track requests and collect metrics."""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Add request ID to response headers
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    # Record metrics
    duration = time.time() - start_time
    await metrics.record_request(
        method=request.method,
        endpoint=str(request.url.path),
        status_code=response.status_code,
        duration=duration,
        request_id=request_id
    )
    
    return response


# API endpoints
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    try:
        # Check all services
        services_healthy = True
        service_status = {}
        
        # Check cache
        if cache:
            cache_healthy = await cache.health_check()
            service_status["cache"] = "healthy" if cache_healthy else "unhealthy"
            services_healthy = services_healthy and cache_healthy
        
        # Check recommendation service
        if recommendation_service:
            rec_healthy = await recommendation_service.health_check()
            service_status["recommendation_service"] = "healthy" if rec_healthy else "unhealthy"
            services_healthy = services_healthy and rec_healthy
        
        # Check feature service
        if feature_service:
            feature_healthy = await feature_service.health_check()
            service_status["feature_service"] = "healthy" if feature_healthy else "unhealthy"
            services_healthy = services_healthy and feature_healthy
        
        status_code = 200 if services_healthy else 503
        
        return JSONResponse(
            status_code=status_code,
            content={
                "status": "healthy" if services_healthy else "unhealthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "services": service_status,
                "version": "2.0.0"
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            }
        )


@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(
    request: RecommendationRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(check_rate_limit)
):
    """Get personalized recommendations for a user."""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        logger.info(f"Processing recommendation request {request_id} for user {request.user_id}")
        
        # A/B test routing
        experiment_config = None
        if request.ab_test_info:
            experiment_config = await ab_test_manager.get_experiment_config(
                request.user_id, 
                request.ab_test_info.get("experiment_id")
            )
        
        # Get recommendations
        recommendations = await recommendation_service.get_recommendations(
            user_id=request.user_id,
            session_id=request.session_id,
            num_recommendations=request.num_recommendations,
            candidate_items=request.candidate_items,
            filters=request.filters,
            context=request.context,
            experiment_config=experiment_config
        )
        
        # Prepare response
        response_items = []
        for i, rec in enumerate(recommendations):
            response_items.append(RecommendationItem(
                item_id=rec["item_id"],
                score=rec["score"],
                rank=i + 1,
                explanation=rec.get("explanation"),
                metadata=rec.get("metadata", {})
            ))
        
        # Prepare metadata
        metadata = {
            "model_version": recommendations[0].get("model_version", "unknown") if recommendations else "unknown",
            "latency_ms": (time.time() - start_time) * 1000,
            "cache_hit": recommendations[0].get("cache_hit", False) if recommendations else False,
            "candidate_pool_size": len(request.candidate_items) if request.candidate_items else "auto",
            "experiment_info": experiment_config
        }
        
        response = RecommendationResponse(
            request_id=request_id,
            recommendations=response_items,
            user_id=request.user_id,
            session_id=request.session_id,
            metadata=metadata,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Log recommendation for analytics (async)
        background_tasks.add_task(
            metrics.log_recommendation,
            request_id,
            request.user_id,
            response_items,
            metadata
        )
        
        # Record metrics
        await metrics.record_recommendation(
            user_id=request.user_id,
            num_recommendations=len(response_items),
            latency_ms=metadata["latency_ms"],
            cache_hit=metadata["cache_hit"],
            experiment_id=experiment_config.get("experiment_id") if experiment_config else None
        )
        
        logger.info(f"Generated {len(response_items)} recommendations for user {request.user_id} "
                   f"in {metadata['latency_ms']:.2f}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to generate recommendations for user {request.user_id}: {e}")
        
        # Record error metrics
        await metrics.record_error(
            endpoint="/recommend",
            error_type=str(type(e).__name__),
            user_id=request.user_id
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate recommendations: {str(e)}"
        )


@app.post("/feedback", tags=["Feedback"])
async def record_feedback(
    feedback: FeedbackRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(check_rate_limit)
):
    """Record user feedback for recommendations."""
    try:
        logger.info(f"Recording feedback for user {feedback.user_id}, item {feedback.item_id}")
        
        # Process feedback
        await recommendation_service.record_feedback(
            user_id=feedback.user_id,
            item_id=feedback.item_id,
            interaction_type=feedback.interaction_type,
            rating=feedback.rating,
            timestamp=feedback.timestamp or datetime.now(timezone.utc),
            metadata=feedback.metadata
        )
        
        # Record metrics
        await metrics.record_feedback(
            user_id=feedback.user_id,
            item_id=feedback.item_id,
            interaction_type=feedback.interaction_type,
            rating=feedback.rating
        )
        
        # Process feedback asynchronously (feature updates, model retraining, etc.)
        background_tasks.add_task(
            recommendation_service.process_feedback,
            feedback.user_id,
            feedback.item_id,
            feedback.interaction_type,
            feedback.rating,
            feedback.metadata
        )
        
        return {"status": "success", "message": "Feedback recorded successfully"}
        
    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")
        
        await metrics.record_error(
            endpoint="/feedback",
            error_type=str(type(e).__name__),
            user_id=feedback.user_id
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to record feedback: {str(e)}"
        )


@app.get("/user/{user_id}/features", tags=["Features"])
async def get_user_features(
    user_id: str,
    feature_names: Optional[str] = None,
    _: None = Depends(check_rate_limit)
):
    """Get features for a specific user."""
    try:
        features = []
        if feature_names:
            feature_list = feature_names.split(",")
            features = await feature_service.get_user_features(user_id, feature_list)
        else:
            features = await feature_service.get_all_user_features(user_id)
        
        return {
            "user_id": user_id,
            "features": features,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get user features: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get user features: {str(e)}"
        )


@app.get("/item/{item_id}/features", tags=["Features"])
async def get_item_features(
    item_id: str,
    feature_names: Optional[str] = None,
    _: None = Depends(check_rate_limit)
):
    """Get features for a specific item."""
    try:
        features = []
        if feature_names:
            feature_list = feature_names.split(",")
            features = await feature_service.get_item_features(item_id, feature_list)
        else:
            features = await feature_service.get_all_item_features(item_id)
        
        return {
            "item_id": item_id,
            "features": features,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get item features: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get item features: {str(e)}"
        )


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """Get system metrics."""
    try:
        metrics_data = await metrics.get_current_metrics()
        
        return MetricsResponse(
            timestamp=datetime.now(timezone.utc),
            metrics=metrics_data
        )
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get metrics: {str(e)}"
        )


@app.get("/experiments", tags=["Experiments"])
async def get_experiments():
    """Get active A/B experiments."""
    try:
        experiments = await ab_test_manager.get_active_experiments()
        
        return {
            "experiments": experiments,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get experiments: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get experiments: {str(e)}"
        )


@app.post("/experiments/{experiment_id}/assign", tags=["Experiments"])
async def assign_experiment(
    experiment_id: str,
    user_id: str,
    _: None = Depends(check_rate_limit)
):
    """Assign user to an A/B test experiment."""
    try:
        assignment = await ab_test_manager.assign_user(
            experiment_id=experiment_id,
            user_id=user_id
        )
        
        return {
            "experiment_id": experiment_id,
            "user_id": user_id,
            "assignment": assignment,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to assign experiment: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to assign experiment: {str(e)}"
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    await metrics.record_error(
        endpoint=str(request.url.path),
        error_type="HTTPException",
        status_code=exc.status_code
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "HTTPException",
                "message": exc.detail,
                "status_code": exc.status_code,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    await metrics.record_error(
        endpoint=str(request.url.path),
        error_type=type(exc).__name__
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": type(exc).__name__,
                "message": "Internal server error",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("Recommendation engine API started successfully")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("Recommendation engine API shutting down")


# Main execution
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers,
        reload=config.api.reload,
        log_level=config.api.log_level.lower(),
        access_log=True
    )
