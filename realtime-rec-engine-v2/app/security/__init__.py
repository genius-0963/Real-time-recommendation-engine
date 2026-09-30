"""
Security package for authentication, authorization, and secret management.
"""

from app.security.config import (
    SecurityConfig,
    SecretBackend,
    VaultConfig,
    AWSSecretsManagerConfig,
    TLSConfig,
    SecretClient,
    get_security_config,
    get_secret_client,
    get_secret,
)

from app.security.auth import (
    TokenClaims,
    AuthMiddleware,
    get_current_user,
    require_scope,
    require_role,
    require_rec_read,
    require_rec_write,
    require_metrics_read,
    require_experiments_read,
    require_admin,
    create_auth_middleware,
)

__all__ = [
    # Config
    "SecurityConfig",
    "SecretBackend",
    "VaultConfig",
    "AWSSecretsManagerConfig",
    "TLSConfig",
    "SecretClient",
    "get_security_config",
    "get_secret_client",
    "get_secret",
    # Auth
    "TokenClaims",
    "AuthMiddleware",
    "get_current_user",
    "require_scope",
    "require_role",
    "require_rec_read",
    "require_rec_write",
    "require_metrics_read",
    "require_experiments_read",
    "require_admin",
    "create_auth_middleware",
]