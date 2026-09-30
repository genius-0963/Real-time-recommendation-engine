"""
Security configuration management with Vault/AWS Secrets Manager integration.
Supports multiple secret backends with fallback and caching.
"""

import os
import logging
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import time
from functools import lru_cache

try:
    import hvac
    VAULT_AVAILABLE = True
except ImportError:
    VAULT_AVAILABLE = False

try:
    import boto3
    from botocore.exceptions import ClientError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

logger = logging.getLogger(__name__)


class SecretBackend(Enum):
    """Supported secret backends."""
    VAULT = "vault"
    AWS_SECRETS_MANAGER = "aws_secrets_manager"
    ENV = "env"


@dataclass
class VaultConfig:
    """HashiCorp Vault configuration."""
    url: str = "http://localhost:8200"
    token: Optional[str] = None
    role_id: Optional[str] = None
    secret_id: Optional[str] = None
    auth_method: str = "token"  # token, approle, kubernetes
    mount_point: str = "secret"
    kv_version: int = 2
    namespace: Optional[str] = None
    timeout: int = 10
    verify_ssl: bool = True


@dataclass
class AWSSecretsManagerConfig:
    """AWS Secrets Manager configuration."""
    region: str = "us-east-1"
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    endpoint_url: Optional[str] = None
    cache_ttl_seconds: int = 300


@dataclass
class TLSConfig:
    """TLS configuration for service connections."""
    enabled: bool = True
    cert_file: Optional[str] = None
    key_file: Optional[str] = None
    ca_file: Optional[str] = None
    verify_hostname: bool = True
    min_version: str = "TLSv1.3"


@dataclass
class SecurityConfig:
    """Main security configuration."""
    # Secret backend
    secret_backend: SecretBackend = SecretBackend.ENV
    vault: VaultConfig = field(default_factory=VaultConfig)
    aws_secrets_manager: AWSSecretsManagerConfig = field(default_factory=AWSSecretsManagerConfig)
    
    # JWT/OIDC
    jwks_url: Optional[str] = None
    issuer: Optional[str] = None
    audience: Optional[str] = None
    algorithms: List[str] = field(default_factory=lambda: ["RS256"])
    token_cache_ttl_seconds: int = 300
    
    # RBAC
    admin_roles: List[str] = field(default_factory=lambda: ["rec-engine-admin"])
    user_roles: List[str] = field(default_factory=lambda: ["rec-engine-user"])
    required_scopes: Dict[str, List[str]] = field(default_factory=lambda: {
        "/recommend": ["rec:read"],
        "/feedback": ["rec:write"],
        "/metrics": ["metrics:read"],
        "/experiments": ["experiments:read"],
        "/admin": ["rec:admin"]
    })
    
    # TLS per service
    kafka_tls: TLSConfig = field(default_factory=TLSConfig)
    redis_tls: TLSConfig = field(default_factory=TLSConfig)
    postgres_tls: TLSConfig = field(default_factory=TLSConfig)
    http_tls: TLSConfig = field(default_factory=TLSConfig)
    
    # CORS
    cors_origins: List[str] = field(default_factory=list)
    cors_methods: List[str] = field(default_factory=lambda: ["GET", "POST"])
    cors_headers: List[str] = field(default_factory=lambda: ["Authorization", "Content-Type"])
    cors_allow_credentials: bool = True
    
    # Security headers
    hsts_max_age: int = 31536000
    csp_policy: str = "default-src 'self'; script-src 'self'; object-src 'none'; frame-ancestors 'none';"
    
    @classmethod
    def from_env(cls) -> "SecurityConfig":
        """Load security configuration from environment variables."""
        config = cls()
        
        # Secret backend
        backend = os.getenv("SECURITY_SECRET_BACKEND", "env").lower()
        config.secret_backend = SecretBackend(backend)
        
        # Vault
        if VAULT_AVAILABLE:
            config.vault.url = os.getenv("VAULT_ADDR", config.vault.url)
            config.vault.token = os.getenv("VAULT_TOKEN", config.vault.token)
            config.vault.role_id = os.getenv("VAULT_ROLE_ID", config.vault.role_id)
            config.vault.secret_id = os.getenv("VAULT_SECRET_ID", config.vault.secret_id)
            config.vault.auth_method = os.getenv("VAULT_AUTH_METHOD", config.vault.auth_method)
            config.vault.mount_point = os.getenv("VAULT_MOUNT_POINT", config.vault.mount_point)
            config.vault.kv_version = int(os.getenv("VAULT_KV_VERSION", str(config.vault.kv_version)))
            config.vault.namespace = os.getenv("VAULT_NAMESPACE", config.vault.namespace)
        
        # AWS Secrets Manager
        if AWS_AVAILABLE:
            config.aws_secrets_manager.region = os.getenv("AWS_REGION", config.aws_secrets_manager.region)
            config.aws_secrets_manager.access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
            config.aws_secrets_manager.secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            config.aws_secrets_manager.endpoint_url = os.getenv("AWS_SECRETS_MANAGER_ENDPOINT")
        
        # JWT/OIDC
        config.jwks_url = os.getenv("OIDC_JWKS_URL")
        config.issuer = os.getenv("OIDC_ISSUER")
        config.audience = os.getenv("OIDC_AUDIENCE")
        
        # RBAC
        if admin_roles := os.getenv("SECURITY_ADMIN_ROLES"):
            config.admin_roles = admin_roles.split(",")
        if user_roles := os.getenv("SECURITY_USER_ROLES"):
            config.user_roles = user_roles.split(",")
        
        # CORS
        if cors_origins := os.getenv("SECURITY_CORS_ORIGINS"):
            config.cors_origins = cors_origins.split(",")
        
        # TLS
        config.kafka_tls.enabled = os.getenv("KAFKA_TLS_ENABLED", "true").lower() == "true"
        config.kafka_tls.cert_file = os.getenv("KAFKA_TLS_CERT_FILE")
        config.kafka_tls.key_file = os.getenv("KAFKA_TLS_KEY_FILE")
        config.kafka_tls.ca_file = os.getenv("KAFKA_TLS_CA_FILE")
        
        config.redis_tls.enabled = os.getenv("REDIS_TLS_ENABLED", "true").lower() == "true"
        config.redis_tls.cert_file = os.getenv("REDIS_TLS_CERT_FILE")
        config.redis_tls.key_file = os.getenv("REDIS_TLS_KEY_FILE")
        config.redis_tls.ca_file = os.getenv("REDIS_TLS_CA_FILE")
        
        config.postgres_tls.enabled = os.getenv("POSTGRES_TLS_ENABLED", "true").lower() == "true"
        config.postgres_tls.cert_file = os.getenv("POSTGRES_TLS_CERT_FILE")
        config.postgres_tls.key_file = os.getenv("POSTGRES_TLS_KEY_FILE")
        config.postgres_tls.ca_file = os.getenv("POSTGRES_TLS_CA_FILE")
        
        return config


class SecretClient:
    """Unified secret client supporting multiple backends."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self._vault_client: Optional[hvac.Client] = None
        self._aws_client: Optional[boto3.client] = None
        self._cache: Dict[str, tuple[Any, float]] = {}
        self._cache_ttl = 60  # seconds
    
    def _get_vault_client(self) -> hvac.Client:
        """Get or create Vault client."""
        if self._vault_client is None:
            if not VAULT_AVAILABLE:
                raise RuntimeError("hvac not installed. Install with: pip install hvac")
            
            client = hvac.Client(
                url=self.config.vault.url,
                token=self.config.vault.token,
                namespace=self.config.vault.namespace,
                verify=self.config.vault.verify_ssl,
                timeout=self.config.vault.timeout
            )
            
            if self.config.vault.auth_method == "approle":
                if not self.config.vault.role_id or not self.config.vault.secret_id:
                    raise ValueError("VAULT_ROLE_ID and VAULT_SECRET_ID required for approle auth")
                client.auth.approle.login(
                    role_id=self.config.vault.role_id,
                    secret_id=self.config.vault.secret_id
                )
            
            if not client.is_authenticated():
                raise RuntimeError("Failed to authenticate with Vault")
            
            self._vault_client = client
        
        return self._vault_client
    
    def _get_aws_client(self) -> boto3.client:
        """Get or create AWS Secrets Manager client."""
        if self._aws_client is None:
            if not AWS_AVAILABLE:
                raise RuntimeError("boto3 not installed. Install with: pip install boto3")
            
            session = boto3.Session(
                aws_access_key_id=self.config.aws_secrets_manager.access_key_id,
                aws_secret_access_key=self.config.aws_secrets_manager.secret_access_key,
                region_name=self.config.aws_secrets_manager.region
            )
            
            self._aws_client = session.client(
                "secretsmanager",
                endpoint_url=self.config.aws_secrets_manager.endpoint_url
            )
        
        return self._aws_client
    
    def get_secret(self, path: str, key: Optional[str] = None) -> Any:
        """
        Get secret from configured backend.
        
        Args:
            path: Secret path (Vault) or secret name (AWS)
            key: Specific key within secret (optional)
        
        Returns:
            Secret value or dict of all keys
        """
        cache_key = f"{path}:{key}"
        if cache_key in self._cache:
            value, expiry = self._cache[cache_key]
            if time.time() < expiry:
                return value
        
        try:
            if self.config.secret_backend == SecretBackend.VAULT:
                value = self._get_from_vault(path, key)
            elif self.config.secret_backend == SecretBackend.AWS_SECRETS_MANAGER:
                value = self._get_from_aws(path, key)
            else:
                value = self._get_from_env(path, key)
            
            self._cache[cache_key] = (value, time.time() + self._cache_ttl)
            return value
            
        except Exception as e:
            logger.error(f"Failed to get secret {path}: {e}")
            raise
    
    def _get_from_vault(self, path: str, key: Optional[str]) -> Any:
        """Get secret from Vault."""
        client = self._get_vault_client()
        
        if self.config.vault.kv_version == 2:
            secret = client.secrets.kv.v2.read_secret_version(
                path=path,
                mount_point=self.config.vault.mount_point
            )
            data = secret["data"]["data"]
        else:
            secret = client.secrets.kv.v1.read_secret(
                path=path,
                mount_point=self.config.vault.mount_point
            )
            data = secret["data"]
        
        if key:
            return data.get(key)
        return data
    
    def _get_from_aws(self, secret_name: str, key: Optional[str]) -> Any:
        """Get secret from AWS Secrets Manager."""
        client = self._get_aws_client()
        
        response = client.get_secret_value(SecretId=secret_name)
        secret_string = response.get("SecretString", "{}")
        data = json.loads(secret_string)
        
        if key:
            return data.get(key)
        return data
    
    def _get_from_env(self, prefix: str, key: Optional[str]) -> Any:
        """Get secret from environment variables."""
        if key:
            env_key = f"{prefix}_{key}".upper().replace("-", "_")
            return os.getenv(env_key)
        
        # Return all env vars with prefix
        result = {}
        for k, v in os.environ.items():
            if k.startswith(prefix.upper().replace("-", "_") + "_"):
                result[k[len(prefix) + 1:].lower()] = v
        return result
    
    def clear_cache(self):
        """Clear secret cache."""
        self._cache.clear()


@lru_cache(maxsize=1)
def get_security_config() -> SecurityConfig:
    """Get cached security configuration."""
    return SecurityConfig.from_env()


@lru_cache(maxsize=1)
def get_secret_client() -> SecretClient:
    """Get cached secret client."""
    return SecretClient(get_security_config())


def get_secret(path: str, key: Optional[str] = None) -> Any:
    """Convenience function to get a secret."""
    return get_secret_client().get_secret(path, key)