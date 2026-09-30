"""
Authentication and Authorization middleware with JWT/OIDC validation and RBAC.
"""

import logging
import time
from typing import Dict, List, Optional, Set, Callable
from dataclasses import dataclass
from functools import lru_cache

import httpx
from jose import jwt, JWTError, jwk
from jose.utils import base64url_decode
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.security.config import get_security_config, SecurityConfig, SecretBackend

logger = logging.getLogger(__name__)

# Security scheme for OpenAPI
security_scheme = HTTPBearer(auto_error=False)


@dataclass
class TokenClaims:
    """Parsed and validated JWT claims."""
    sub: str  # user/subject identifier
    roles: List[str]
    scopes: List[str]
    iss: str  # issuer
    aud: str  # audience
    exp: int  # expiration timestamp
    iat: int  # issued at timestamp
    jti: Optional[str] = None  # JWT ID
    client_id: Optional[str] = None
    
    def has_role(self, role: str) -> bool:
        """Check if token has specific role."""
        return role in self.roles
    
    def has_any_role(self, roles: List[str]) -> bool:
        """Check if token has any of the specified roles."""
        return any(role in self.roles for role in roles)
    
    def has_scope(self, scope: str) -> bool:
        """Check if token has specific scope."""
        return scope in self.scopes
    
    def has_any_scope(self, scopes: List[str]) -> bool:
        """Check if token has any of the specified scopes."""
        return any(scope in self.scopes for scope in scopes)
    
    def is_admin(self, config: SecurityConfig) -> bool:
        """Check if token has admin role."""
        return self.has_any_role(config.admin_roles)
    
    def is_expired(self) -> bool:
        """Check if token is expired."""
        return time.time() > self.exp


class JWKSCache:
    """Cache for JWKS (JSON Web Key Set) with automatic refresh."""
    
    def __init__(self, jwks_url: str, cache_ttl: int = 300):
        self.jwks_url = jwks_url
        self.cache_ttl = cache_ttl
        self._keys: Dict[str, dict] = {}
        self._fetched_at: float = 0
    
    async def get_key(self, kid: str) -> Optional[dict]:
        """Get key by key ID, fetching JWKS if needed."""
        await self._ensure_fresh()
        return self._keys.get(kid)
    
    async def _ensure_fresh(self):
        """Fetch JWKS if cache is stale."""
        if time.time() - self._fetched_at < self.cache_ttl and self._keys:
            return
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(self.jwks_url)
                response.raise_for_status()
                jwks = response.json()
                
                self._keys = {key["kid"]: key for key in jwks.get("keys", []) if "kid" in key}
                self._fetched_at = time.time()
                logger.info(f"Refreshed JWKS cache: {len(self._keys)} keys")
        except Exception as e:
            logger.error(f"Failed to fetch JWKS from {self.jwks_url}: {e}")
            if not self._keys:
                raise HTTPException(status_code=503, detail="Unable to fetch signing keys")


class AuthMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for authentication and authorization."""
    
    # Paths that don't require authentication
    PUBLIC_PATHS = {
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/favicon.ico"
    }
    
    def __init__(self, app, config: Optional[SecurityConfig] = None):
        super().__init__(app)
        self.config = config or get_security_config()
        self.jwks_cache: Optional[JWKSCache] = None
        
        if self.config.jwks_url:
            self.jwks_cache = JWKSCache(
                self.config.jwks_url,
                self.config.token_cache_ttl_seconds
            )
    
    async def dispatch(self, request: Request, call_next):
        # Skip auth for public paths
        if request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)
        
        # Skip auth for OPTIONS (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)
        
        # Extract and validate token
        try:
            claims = await self._authenticate_request(request)
            request.state.auth = claims
        except HTTPException as e:
            return JSONResponse(
                status_code=e.status_code,
                content={"error": {"code": e.status_code, "message": e.detail}}
            )
        
        # Check authorization
        try:
            self._authorize_request(request, claims)
        except HTTPException as e:
            return JSONResponse(
                status_code=e.status_code,
                content={"error": {"code": e.status_code, "message": e.detail}}
            )
        
        return await call_next(request)
    
    async def _authenticate_request(self, request: Request) -> TokenClaims:
        """Extract and validate JWT from request."""
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=401,
                detail="Missing or invalid Authorization header",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        token = auth_header[7:]  # Remove "Bearer "
        
        if self.config.jwks_url:
            return await self._validate_jwt_rs256(token)
        else:
            return self._validate_jwt_hs256(token)
    
    async def _validate_jwt_rs256(self, token: str) -> TokenClaims:
        """Validate JWT using RS256 with JWKS."""
        if not self.jwks_cache:
            raise HTTPException(status_code=500, detail="JWKS not configured")
        
        try:
            # Get unverified header to find key ID
            unverified_header = jwt.get_unverified_header(token)
            kid = unverified_header.get("kid")
            if not kid:
                raise JWTError("Token missing 'kid' header")
            
            # Get signing key from JWKS
            jwk_data = await self.jwks_cache.get_key(kid)
            if not jwk_data:
                raise JWTError(f"Key not found in JWKS: {kid}")
            
            public_key = jwk.construct(jwk_data)
            
            # Decode and validate
            claims = jwt.decode(
                token,
                public_key.to_pem().decode(),
                algorithms=self.config.algorithms,
                audience=self.config.audience,
                issuer=self.config.issuer,
                options={"verify_exp": True, "verify_aud": True, "verify_iss": True}
            )
            
        except JWTError as e:
            logger.warning(f"JWT validation failed: {e}")
            raise HTTPException(
                status_code=401,
                detail=f"Invalid token: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        return self._parse_claims(claims)
    
    def _validate_jwt_hs256(self, token: str) -> TokenClaims:
        """Validate JWT using HS256 with secret from Vault/env."""
        from app.security.config import get_secret
        
        secret = get_secret("jwt", "secret_key")
        if not secret:
            raise HTTPException(status_code=500, detail="JWT secret not configured")
        
        try:
            claims = jwt.decode(
                token,
                secret,
                algorithms=["HS256"],
                audience=self.config.audience,
                issuer=self.config.issuer,
                options={"verify_exp": True}
            )
        except JWTError as e:
            logger.warning(f"JWT validation failed: {e}")
            raise HTTPException(
                status_code=401,
                detail=f"Invalid token: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        return self._parse_claims(claims)
    
    def _parse_claims(self, claims: dict) -> TokenClaims:
        """Parse raw claims into TokenClaims."""
        # Extract roles from various possible claim names
        roles = []
        for claim in ["roles", "realm_access.roles", "resource_access.client.roles", "groups"]:
            if "." in claim:
                parts = claim.split(".")
                value = claims
                for part in parts:
                    value = value.get(part, {})
                if isinstance(value, list):
                    roles.extend(value)
            else:
                value = claims.get(claim)
                if isinstance(value, list):
                    roles.extend(value)
        
        # Extract scopes
        scopes = claims.get("scope", "").split() if isinstance(claims.get("scope"), str) else []
        if "scopes" in claims and isinstance(claims["scopes"], list):
            scopes.extend(claims["scopes"])
        
        return TokenClaims(
            sub=claims.get("sub", ""),
            roles=list(set(roles)),  # deduplicate
            scopes=list(set(scopes)),
            iss=claims.get("iss", ""),
            aud=claims.get("aud", "") if isinstance(claims.get("aud"), str) else claims.get("aud", [""])[0],
            exp=claims.get("exp", 0),
            iat=claims.get("iat", 0),
            jti=claims.get("jti"),
            client_id=claims.get("client_id") or claims.get("azp")
        )
    
    def _authorize_request(self, request: Request, claims: TokenClaims):
        """Check if token has required scopes for the endpoint."""
        path = request.url.path
        
        # Find matching required scopes
        required_scopes = []
        for pattern, scopes in self.config.required_scopes.items():
            if self._path_matches(pattern, path):
                required_scopes = scopes
                break
        
        if not required_scopes:
            # No specific scopes required, but require authentication
            return
        
        if not claims.has_any_scope(required_scopes):
            # Check if admin role grants access
            if not claims.is_admin(self.config):
                logger.warning(
                    f"Authorization failed: user={claims.sub} "
                    f"path={path} required_scopes={required_scopes} "
                    f"user_scopes={claims.scopes} user_roles={claims.roles}"
                )
                raise HTTPException(
                    status_code=403,
                    detail=f"Insufficient permissions. Required scopes: {required_scopes}"
                )
    
    def _path_matches(self, pattern: str, path: str) -> bool:
        """Check if path matches pattern (supports wildcards)."""
        if pattern == path:
            return True
        if pattern.endswith("*"):
            return path.startswith(pattern[:-1])
        # Handle parameterized paths like /user/{user_id}/features
        import re
        regex_pattern = pattern.replace("{", "(?P<").replace("}", ">[^/]+)")
        return bool(re.match(f"^{regex_pattern}$", path))


# Dependency for extracting auth in route handlers
async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme)
) -> TokenClaims:
    """Dependency to get current authenticated user."""
    if not hasattr(request.state, "auth"):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return request.state.auth


async def require_scope(scope: str) -> Callable:
    """Dependency factory for requiring specific scope."""
    async def _check_scope(user: TokenClaims = Depends(get_current_user)) -> TokenClaims:
        config = get_security_config()
        if not user.has_scope(scope) and not user.is_admin(config):
            raise HTTPException(
                status_code=403,
                detail=f"Required scope: {scope}"
            )
        return user
    return _check_scope


async def require_role(role: str) -> Callable:
    """Dependency factory for requiring specific role."""
    async def _check_role(user: TokenClaims = Depends(get_current_user)) -> TokenClaims:
        config = get_security_config()
        if not user.has_role(role) and not user.is_admin(config):
            raise HTTPException(
                status_code=403,
                detail=f"Required role: {role}"
            )
        return user
    return _check_role


# Convenience dependencies
require_rec_read = require_scope("rec:read")
require_rec_write = require_scope("rec:write")
require_metrics_read = require_scope("metrics:read")
require_experiments_read = require_scope("experiments:read")
require_admin = require_role("rec-engine-admin")


def create_auth_middleware(app, config: Optional[SecurityConfig] = None) -> AuthMiddleware:
    """Factory to create and add auth middleware to app."""
    middleware = AuthMiddleware(app, config)
    app.add_middleware(BaseHTTPMiddleware, dispatch=middleware.dispatch)
    return middleware