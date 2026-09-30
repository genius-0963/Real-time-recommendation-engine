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
import uvicorn

from app.config import Config
from app.security import (
    AuthMiddleware,
    get_security_config,
    require_rec_read,
    require_rec_write,
    require_metrics_read,
    require_experiments_read,
    require_admin,
    TokenClaims,
)
from app.api.validators import (
    RecommendationRequest,
    FeedbackRequest,
    BatchRecommendationRequest,
    BatchFeedbackRequest,
    FeaturesRequest,
    ExperimentAssignRequest,
    RecommendationItem,
    RecommendationResponse,
    FeedbackResponse,
    FeaturesResponse,
    MetricsResponse,
    ExperimentResponse,
    ErrorResponse,
    HealthResponse,
    format_validation_error,
)
from app.services.recommendation_service import RecommendationService
from app.services.feature_service import FeatureService
from app.cache.redis_cache import RedisCache
from app.monitoring.metrics_collector import MetricsCollector
from app.monitoring.rate_limiter import create_rate_limiter, check_rate_limit, MultiTierRateLimiter
from app.monitoring.middleware import setup_middleware
from app.experiments.ab_testing import ABTestManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
config = Config()
recommendation_service = None
feature_service = None
cache = None
metrics = None
rate_limiter: MultiTierRateLimiter = None
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
        ab_test_manager = ABTestManager(config.experiments)
        
        # Initialize distributed rate limiter
        if cache and cache.redis:
            rate_limiter = create_rate_limiter(cache.redis)
            app.state.rate_limiter = rate_limiter
        
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
    lifespan=lifespan,
    # Response models for OpenAPI
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
        404: {"model": ErrorResponse, "description": "Not Found"},
        413: {"model": ErrorResponse, "description": "Payload Too Large"},
        422: {"model": ErrorResponse, "description": "Validation Error"},
        429: {"model": ErrorResponse, "description": "Rate Limit Exceeded"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
        503: {"model": ErrorResponse, "description": "Service Unavailable"},
        504: {"model": ErrorResponse, "description": "Gateway Timeout"},
    }
)

# Setup all middleware (security, size limits, timeout, circuit breaker, metrics)
security_config = get_security_config()
middleware_config = {
    "hsts_max_age": security_config.hsts_max_age,
    "csp_policy": security_config.csp_policy,
    "max_request_size": config.api.max_request_size if hasattr(config.api, 'max_request_size') else 1_048_576,
    "request_timeout": config.api.request_timeout,
    "circuit_breaker_threshold": config.api.circuit_breaker_threshold,
    "circuit_breaker_timeout": config.api.circuit_breaker_timeout,
}
setup_middleware(app, middleware_config, metrics)

# Add security middleware (must be added before other middleware)
AuthMiddleware(app, security_config)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=security_config.cors_allow_credentials,
    allow_methods=security_config.cors_methods,
    allow_headers=security_config.cors_headers,
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Rate limit dependency
async def rate_limit_dependency(request: Request, user: TokenClaims = Depends(require_rec_read)) -> None:
    """Rate limit dependency that uses user claims for role-based limits."""
    roles = user.roles if hasattr(user, 'roles') else []
    user_id = user.sub if hasattr(user, 'sub') else None
    await check_rate_limit(request, user_id=user_id, roles=roles)


# API endpoints
@app.get("/health", response_model=HealthResponse, tags=["Health"])
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
        
        return HealthResponse(
            status="healthy" if services_healthy else "unhealthy",
            timestamp=datetime.now(timezone.utc),
            services=service_status,
            version="2.0.0"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": 503,
                    "message": "Health check failed",
                    "type": "HealthCheckError",
                    "details": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
        )


@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(
    request: RecommendationRequest,
    background_tasks: BackgroundTasks,
    user: TokenClaims = Depends(require_rec_read),
    _: None = Depends(rate_limit_dependency),
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


@app.post("/recommend/batch", response_model=List[RecommendationResponse], tags=["Recommendations"])
async def get_batch_recommendations(
    request: BatchRecommendationRequest,
    background_tasks: BackgroundTasks,
    user: TokenClaims = Depends(require_rec_read),
    _: None = Depends(rate_limit_dependency),
):
    """Get personalized recommendations for multiple users in batch."""
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        logger.info(f"Processing batch recommendation request {request_id} with {len(request.requests)} requests")
        
        responses = []
        for req in request.requests:
            # A/B test routing per request
            experiment_config = None
            if req.ab_test_info:
                experiment_config = await ab_test_manager.get_experiment_config(
                    req.user_id, 
                    req.ab_test_info.get("experiment_id")
                )
            
            recommendations = await recommendation_service.get_recommendations(
                user_id=req.user_id,
                session_id=req.session_id,
                num_recommendations=req.num_recommendations,
                candidate_items=req.candidate_items,
                filters=req.filters,
                context=req.context,
                experiment_config=experiment_config
            )
            
            response_items = []
            for i, rec in enumerate(recommendations):
                response_items.append(RecommendationItem(
                    item_id=rec["item_id"],
                    score=rec["score"],
                    rank=i + 1,
                    explanation=rec.get("explanation"),
                    metadata=rec.get("metadata", {})
                ))
            
            metadata = {
                "model_version": recommendations[0].get("model_version", "unknown") if recommendations else "unknown",
                "latency_ms": (time.time() - start_time) * 1000,
                "cache_hit": recommendations[0].get("cache_hit", False) if recommendations else False,
                "candidate_pool_size": len(req.candidate_items) if req.candidate_items else "auto",
                "experiment_info": experiment_config
            }
            
            responses.append(RecommendationResponse(
                request_id=f"{request_id}_{req.user_id}",
                recommendations=response_items,
                user_id=req.user_id,
                session_id=req.session_id,
                metadata=metadata,
                timestamp=datetime.now(timezone.utc)
            ))
        
        logger.info(f"Generated batch recommendations for {len(responses)} users in {(time.time() - start_time)*1000:.2f}ms")
        return responses
        
    except Exception as e:
        logger.error(f"Failed to generate batch recommendations: {e}")
        await metrics.record_error(
            endpoint="/recommend/batch",
            error_type=str(type(e).__name__),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate batch recommendations: {str(e)}"
        )


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def record_feedback(
    feedback: FeedbackRequest,
    background_tasks: BackgroundTasks,
    user: TokenClaims = Depends(require_rec_write),
    _: None = Depends(rate_limit_dependency),
):
    """Record user feedback for recommendations."""
    request_id = str(uuid.uuid4())
    
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
        
        return FeedbackResponse(
            request_id=request_id
        )
        
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


@app.post("/feedback/batch", response_model=List[FeedbackResponse], tags=["Feedback"])
async def record_batch_feedback(
    request: BatchFeedbackRequest,
    background_tasks: BackgroundTasks,
    user: TokenClaims = Depends(require_rec_write),
    _: None = Depends(rate_limit_dependency),
):
    """Record feedback for multiple users in batch."""
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Recording batch feedback for {len(request.feedback)} events")
        
        responses = []
        for fb in request.feedback:
            await recommendation_service.record_feedback(
                user_id=fb.user_id,
                item_id=fb.item_id,
                interaction_type=fb.interaction_type,
                rating=fb.rating,
                timestamp=fb.timestamp or datetime.now(timezone.utc),
                metadata=fb.metadata
            )
            
            await metrics.record_feedback(
                user_id=fb.user_id,
                item_id=fb.item_id,
                interaction_type=fb.interaction_type,
                rating=fb.rating
            )
            
            background_tasks.add_task(
                recommendation_service.process_feedback,
                fb.user_id,
                fb.item_id,
                fb.interaction_type,
                fb.rating,
                fb.metadata
            )
            
            responses.append(FeedbackResponse(
                request_id=f"{request_id}_{fb.user_id}"
            ))
        
        return responses
        
    except Exception as e:
        logger.error(f"Failed to record batch feedback: {e}")
        await metrics.record_error(
            endpoint="/feedback/batch",
            error_type=str(type(e).__name__),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to record batch feedback: {str(e)}"
        )


@app.get("/user/{user_id}/features", response_model=FeaturesResponse, tags=["Features"])
async def get_user_features(
    user_id: str,
    feature_names: Optional[str] = None,
    user: TokenClaims = Depends(require_rec_read),
    _: None = Depends(rate_limit_dependency),
):
    """Get features for a specific user."""
    try:
        validated_features = None
        if feature_names:
            feature_list = feature_names.split(",")
            # Validate feature names
            from app.api.validators import validate_feature_names
            validated_features = validate_feature_names(feature_list)
            features = await feature_service.get_user_features(user_id, validated_features)
        else:
            features = await feature_service.get_all_user_features(user_id)
        
        return FeaturesResponse(
            user_id=user_id,
            features=features,
            timestamp=datetime.now(timezone.utc)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get user features: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get user features: {str(e)}"
        )


@app.get("/item/{item_id}/features", response_model=FeaturesResponse, tags=["Features"])
async def get_item_features(
    item_id: str,
    feature_names: Optional[str] = None,
    user: TokenClaims = Depends(require_rec_read),
    _: None = Depends(rate_limit_dependency),
):
    """Get features for a specific item."""
    try:
        validated_features = None
        if feature_names:
            feature_list = feature_names.split(",")
            from app.api.validators import validate_feature_names
            validated_features = validate_feature_names(feature_list)
            features = await feature_service.get_item_features(item_id, validated_features)
        else:
            features = await feature_service.get_all_item_features(item_id)
        
        return FeaturesResponse(
            user_id=item_id,  # Using user_id field for item_id in response model
            features=features,
            timestamp=datetime.now(timezone.utc)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get item features: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get item features: {str(e)}"
        )


@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics(
    user: TokenClaims = Depends(require_metrics_read),
    _: None = Depends(rate_limit_dependency),
):
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


@app.get("/experiments", response_model=List[ExperimentResponse], tags=["Experiments"])
async def get_experiments(
    user: TokenClaims = Depends(require_experiments_read),
    _: None = Depends(rate_limit_dependency),
):
    """Get active A/B experiments."""
    try:
        experiments = await ab_test_manager.get_active_experiments()
        
        return [
            ExperimentResponse(
                experiment_id=exp["experiment_id"],
                user_id="",  # Not user-specific
                assignment=exp,
                timestamp=datetime.now(timezone.utc)
            )
            for exp in experiments
        ]
        
    except Exception as e:
        logger.error(f"Failed to get experiments: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get experiments: {str(e)}"
        )


@app.post("/experiments/{experiment_id}/assign", response_model=ExperimentResponse, tags=["Experiments"])
async def assign_experiment(
    experiment_id: str,
    request: ExperimentAssignRequest,
    user: TokenClaims = Depends(require_experiments_read),
    _: None = Depends(rate_limit_dependency),
):
    """Assign user to an A/B test experiment."""
    try:
        assignment = await ab_test_manager.assign_user(
            experiment_id=experiment_id,
            user_id=request.user_id
        )
        
        return ExperimentResponse(
            experiment_id=experiment_id,
            user_id=request.user_id,
            assignment=assignment,
            timestamp=datetime.now(timezone.utc)
        )
        
    except Exception as e:
        logger.error(f"Failed to assign experiment: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to assign experiment: {str(e)}"
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with standardized error format."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    await metrics.record_error(
        endpoint=str(request.url.path),
        error_type="HTTPException",
        status_code=exc.status_code
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "type": "HTTPException",
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        },
        headers=getattr(request.state, "rate_limit_headers", {})
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with standardized error format."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    await metrics.record_error(
        endpoint=str(request.url.path),
        error_type=type(exc).__name__
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "type": type(exc).__name__,
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        },
        headers=getattr(request.state, "rate_limit_headers", {})
    )


# Validation error handler
@app.exception_handler(ValueError)
async def validation_exception_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": 422,
                "message": str(exc),
                "type": "ValidationError",
                "request_id": request_id,
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