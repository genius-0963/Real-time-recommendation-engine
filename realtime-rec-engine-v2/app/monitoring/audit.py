"""
Structured audit logging for SOC 2 / GDPR compliance.
Logs all auth decisions, data access, config changes, and security events.
"""

import logging
import uuid
import json
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextvars import ContextVar
import structlog
from structlog.processors import JSONRenderer
import os


class AuditEventType(Enum):
    """Types of audit events."""
    # Authentication
    AUTH_LOGIN_SUCCESS = "auth.login.success"
    AUTH_LOGIN_FAILURE = "auth.login.failure"
    AUTH_LOGOUT = "auth.logout"
    AUTH_TOKEN_REFRESH = "auth.token.refresh"
    AUTH_TOKEN_REVOKED = "auth.token.revoked"
    
    # Authorization
    AUTHZ_ALLOW = "authz.allow"
    AUTHZ_DENY = "authz.deny"
    AUTHZ_ROLE_CHECK = "authz.role.check"
    
    # Data Access
    DATA_READ = "data.read"
    DATA_WRITE = "data.write"
    DATA_DELETE = "data.delete"
    DATA_EXPORT = "data.export"
    FEATURE_READ = "feature.read"
    FEATURE_WRITE = "feature.write"
    RECOMMENDATION_REQUEST = "recommendation.request"
    FEEDBACK_SUBMIT = "feedback.submit"
    
    # Configuration
    CONFIG_CHANGE = "config.change"
    INDEX_REBUILD = "index.rebuild"
    MODEL_DEPLOY = "model.deploy"
    EXPERIMENT_CREATE = "experiment.create"
    EXPERIMENT_ASSIGN = "experiment.assign"
    
    # Security
    RATE_LIMIT_EXCEEDED = "security.rate_limit_exceeded"
    CIRCUIT_BREAKER_OPEN = "security.circuit_breaker_open"
    INVALID_INPUT = "security.invalid_input"
    PATH_TRAVERSAL_ATTEMPT = "security.path_traversal"
    SCHEMA_VALIDATION_FAILURE = "security.schema_validation_failure"
    
    # System
    SERVICE_START = "system.service_start"
    SERVICE_STOP = "system.service_stop"
    HEALTH_CHECK = "system.health_check"
    BACKUP_START = "system.backup_start"
    BACKUP_COMPLETE = "system.backup_complete"


class AuditDecision(Enum):
    """Audit decision outcomes."""
    ALLOW = "allow"
    DENY = "deny"
    ERROR = "error"


@dataclass
class AuditLogEntry:
    """Structured audit log entry."""
    # Core fields
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_type: str = ""
    decision: str = ""
    
    # Actor (who)
    actor_id: str = ""           # user_id, service_account, or system
    actor_type: str = "user"     # user, service, system
    actor_roles: List[str] = field(default_factory=list)
    actor_ip: str = ""
    
    # Resource (what)
    resource_type: str = ""      # endpoint, feature, model, index, config
    resource_id: str = ""        # specific resource identifier
    resource_owner: str = ""     # owner of resource (for data access)
    
    # Action (what happened)
    action: str = ""             # HTTP method, operation name
    endpoint: str = ""           # API endpoint
    method: str = ""             # HTTP method
    
    # Context
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    request_id: str = ""
    trace_id: str = ""
    
    # Details (flexible)
    details: Dict[str, Any] = field(default_factory=dict)
    
    # Outcome
    status_code: int = 0
    error_message: str = ""
    latency_ms: float = 0.0
    
    # Compliance
    pii_accessed: bool = False
    data_classification: str = ""  # public, internal, confidential, restricted
    retention_days: int = 2555     # 7 years default
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


# Context variable for correlation ID propagation
correlation_id_var: ContextVar[str] = ContextVar('correlation_id', default='')
request_id_var: ContextVar[str] = ContextVar('request_id', default='')
trace_id_var: ContextVar[str] = ContextVar('trace_id', default='')


class AuditLogger:
    """
    Structured audit logger using structlog.
    Outputs JSON lines suitable for Loki, Elasticsearch, or S3 WORM storage.
    """
    
    def __init__(
        self,
        service_name: str = "rec-engine",
        environment: str = "production",
        log_level: str = "INFO",
        output_file: Optional[str] = None,
        enable_console: bool = True
    ):
        self.service_name = service_name
        self.environment = environment
        
        # Configure structlog
        self._configure_structlog(log_level, output_file, enable_console)
        self.logger = structlog.get_logger("audit")
        
        # Buffer for batch writes (optional)
        self._buffer: List[AuditLogEntry] = []
        self._buffer_lock = threading.Lock()
        self._buffer_size = 100
        
    def _configure_structlog(self, log_level: str, output_file: Optional[str], enable_console: bool):
        """Configure structlog processors."""
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            # Add service context
            self._add_service_context,
            # JSON output
            JSONRenderer()
        ]
        
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        # Configure stdlib logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(message)s",
            handlers=[]
        )
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter("%(message)s"))
            logging.getLogger().addHandler(console_handler)
        
        # File handler for immutable storage
        if output_file:
            file_handler = logging.FileHandler(output_file)
            file_handler.setFormatter(logging.Formatter("%(message)s"))
            logging.getLogger().addHandler(file_handler)
    
    def _add_service_context(self, logger, method_name, event_dict):
        """Add service context to all log entries."""
        event_dict["service"] = self.service_name
        event_dict["environment"] = self.environment
        
        # Add correlation IDs from context vars
        corr_id = correlation_id_var.get()
        if corr_id:
            event_dict["correlation_id"] = corr_id
        
        req_id = request_id_var.get()
        if req_id:
            event_dict["request_id"] = req_id
            
        trace_id = trace_id_var.get()
        if trace_id:
            event_dict["trace_id"] = trace_id
            
        return event_dict
    
    def log(
        self,
        event_type: AuditEventType,
        decision: AuditDecision,
        actor_id: str,
        actor_type: str = "user",
        actor_roles: List[str] = None,
        actor_ip: str = "",
        resource_type: str = "",
        resource_id: str = "",
        resource_owner: str = "",
        action: str = "",
        endpoint: str = "",
        method: str = "",
        session_id: str = "",
        details: Dict[str, Any] = None,
        status_code: int = 0,
        error_message: str = "",
        latency_ms: float = 0.0,
        pii_accessed: bool = False,
        data_classification: str = "internal",
        correlation_id: str = None,
        request_id: str = None,
        trace_id: str = None
    ) -> str:
        """
        Log an audit event.
        
        Returns:
            correlation_id for tracking
        """
        # Generate IDs if not provided
        if correlation_id is None:
            correlation_id = correlation_id_var.get() or str(uuid.uuid4())
        if request_id is None:
            request_id = request_id_var.get() or str(uuid.uuid4())
        if trace_id is None:
            trace_id = trace_id_var.get() or str(uuid.uuid4())
        
        entry = AuditLogEntry(
            event_type=event_type.value,
            decision=decision.value,
            actor_id=actor_id,
            actor_type=actor_type,
            actor_roles=actor_roles or [],
            actor_ip=actor_ip,
            resource_type=resource_type,
            resource_id=resource_id,
            resource_owner=resource_owner,
            action=action,
            endpoint=endpoint,
            method=method,
            session_id=session_id,
            correlation_id=correlation_id,
            request_id=request_id,
            trace_id=trace_id,
            details=details or {},
            status_code=status_code,
            error_message=error_message,
            latency_ms=latency_ms,
            pii_accessed=pii_accessed,
            data_classification=data_classification,
        )
        
        # Log via structlog
        self.logger.info(
            "audit_event",
            **entry.to_dict()
        )
        
        # Buffer for potential batch write to immutable storage
        with self._buffer_lock:
            self._buffer.append(entry)
            if len(self._buffer) >= self._buffer_size:
                self._flush_buffer()
        
        return correlation_id
    
    def _flush_buffer(self):
        """Flush buffer to immutable storage (S3 WORM, etc.)."""
        # In production, write to S3 with Object Lock, or append-only file
        # For now, just log buffer size
        if self._buffer:
            self.logger.debug("audit_buffer_flush", buffer_size=len(self._buffer))
            self._buffer.clear()
    
    # Convenience methods for common audit events
    
    def log_auth_success(
        self,
        user_id: str,
        roles: List[str],
        ip: str,
        method: str = "password",
        correlation_id: str = None
    ) -> str:
        """Log successful authentication."""
        return self.log(
            event_type=AuditEventType.AUTH_LOGIN_SUCCESS,
            decision=AuditDecision.ALLOW,
            actor_id=user_id,
            actor_type="user",
            actor_roles=roles,
            actor_ip=ip,
            resource_type="auth",
            resource_id="login",
            action=f"auth.{method}",
            endpoint="/auth/login",
            method="POST",
            status_code=200,
            correlation_id=correlation_id,
            data_classification="confidential"
        )
    
    def log_auth_failure(
        self,
        user_id: str,
        ip: str,
        reason: str,
        method: str = "password",
        correlation_id: str = None
    ) -> str:
        """Log failed authentication."""
        return self.log(
            event_type=AuditEventType.AUTH_LOGIN_FAILURE,
            decision=AuditDecision.DENY,
            actor_id=user_id,
            actor_type="user",
            actor_ip=ip,
            resource_type="auth",
            resource_id="login",
            action=f"auth.{method}",
            endpoint="/auth/login",
            method="POST",
            status_code=401,
            error_message=reason,
            correlation_id=correlation_id,
            data_classification="confidential"
        )
    
    def log_authz_allow(
        self,
        user_id: str,
        roles: List[str],
        resource_type: str,
        resource_id: str,
        action: str,
        endpoint: str,
        method: str,
        correlation_id: str = None
    ) -> str:
        """Log successful authorization."""
        return self.log(
            event_type=AuditEventType.AUTHZ_ALLOW,
            decision=AuditDecision.ALLOW,
            actor_id=user_id,
            actor_type="user",
            actor_roles=roles,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            endpoint=endpoint,
            method=method,
            status_code=200,
            correlation_id=correlation_id,
            data_classification="internal"
        )
    
    def log_authz_deny(
        self,
        user_id: str,
        roles: List[str],
        resource_type: str,
        resource_id: str,
        action: str,
        endpoint: str,
        method: str,
        reason: str,
        correlation_id: str = None
    ) -> str:
        """Log denied authorization."""
        return self.log(
            event_type=AuditEventType.AUTHZ_DENY,
            decision=AuditDecision.DENY,
            actor_id=user_id,
            actor_type="user",
            actor_roles=roles,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            endpoint=endpoint,
            method=method,
            status_code=403,
            error_message=reason,
            correlation_id=correlation_id,
            data_classification="internal"
        )
    
    def log_data_access(
        self,
        user_id: str,
        roles: List[str],
        resource_type: str,
        resource_id: str,
        resource_owner: str,
        action: str,
        endpoint: str,
        method: str,
        pii_accessed: bool = False,
        correlation_id: str = None
    ) -> str:
        """Log data access (read/write/delete)."""
        event_map = {
            "read": AuditEventType.DATA_READ,
            "write": AuditEventType.DATA_WRITE,
            "delete": AuditEventType.DATA_DELETE,
            "export": AuditEventType.DATA_EXPORT,
        }
        
        return self.log(
            event_type=event_map.get(action, AuditEventType.DATA_READ),
            decision=AuditDecision.ALLOW,
            actor_id=user_id,
            actor_type="user",
            actor_roles=roles,
            resource_type=resource_type,
            resource_id=resource_id,
            resource_owner=resource_owner,
            action=action,
            endpoint=endpoint,
            method="GET" if action == "read" else "POST" if action == "write" else "DELETE",
            status_code=200,
            pii_accessed=pii_accessed,
            data_classification="confidential" if pii_accessed else "internal",
            correlation_id=correlation_id
        )
    
    def log_recommendation_request(
        self,
        user_id: str,
        roles: List[str],
        num_recommendations: int,
        endpoint: str,
        method: str,
        latency_ms: float,
        correlation_id: str = None
    ) -> str:
        """Log recommendation request."""
        return self.log(
            event_type=AuditEventType.RECOMMENDATION_REQUEST,
            decision=AuditDecision.ALLOW,
            actor_id=user_id,
            actor_type="user",
            actor_roles=roles,
            resource_type="recommendation",
            resource_id=f"rec_{num_recommendations}",
            action="recommend",
            endpoint=endpoint,
            method=method,
            status_code=200,
            latency_ms=latency_ms,
            details={"num_recommendations": num_recommendations},
            pii_accessed=True,
            data_classification="confidential",
            correlation_id=correlation_id
        )
    
    def log_config_change(
        self,
        user_id: str,
        roles: List[str],
        config_key: str,
        old_value: str,
        new_value: str,
        correlation_id: str = None
    ) -> str:
        """Log configuration change."""
        return self.log(
            event_type=AuditEventType.CONFIG_CHANGE,
            decision=AuditDecision.ALLOW,
            actor_id=user_id,
            actor_type="user",
            actor_roles=roles,
            resource_type="config",
            resource_id=config_key,
            action="update",
            endpoint="/admin/config",
            method="PATCH",
            status_code=200,
            details={
                "config_key": config_key,
                "old_value": old_value,
                "new_value": new_value
            },
            data_classification="internal",
            correlation_id=correlation_id
        )
    
    def log_index_rebuild(
        self,
        user_id: str,
        roles: List[str],
        index_type: str,
        num_items: int,
        duration_ms: float,
        success: bool,
        correlation_id: str = None
    ) -> str:
        """Log ANN index rebuild."""
        return self.log(
            event_type=AuditEventType.INDEX_REBUILD,
            decision=AuditDecision.ALLOW if success else AuditDecision.ERROR,
            actor_id=user_id,
            actor_type="user",
            actor_roles=roles,
            resource_type="index",
            resource_id=index_type,
            action="rebuild",
            endpoint="/admin/index/rebuild",
            method="POST",
            status_code=200 if success else 500,
            latency_ms=duration_ms,
            details={
                "index_type": index_type,
                "num_items": num_items,
                "duration_ms": duration_ms
            },
            data_classification="internal",
            correlation_id=correlation_id
        )
    
    def log_rate_limit_exceeded(
        self,
        identifier: str,
        scope: str,
        limit: int,
        window_seconds: int,
        correlation_id: str = None
    ) -> str:
        """Log rate limit exceeded."""
        return self.log(
            event_type=AuditEventType.RATE_LIMIT_EXCEEDED,
            decision=AuditDecision.DENY,
            actor_id=identifier,
            actor_type="user" if scope == "user" else "ip",
            resource_type="rate_limit",
            resource_id=scope,
            action="rate_limit_check",
            endpoint="",
            method="",
            status_code=429,
            details={
                "limit": limit,
                "window_seconds": window_seconds,
                "scope": scope
            },
            correlation_id=correlation_id,
            data_classification="internal"
        )
    
    def log_circuit_breaker_open(
        self,
        endpoint: str,
        failure_count: int,
        correlation_id: str = None
    ) -> str:
        """Log circuit breaker opening."""
        return self.log(
            event_type=AuditEventType.CIRCUIT_BREAKER_OPEN,
            decision=AuditDecision.ERROR,
            actor_id="system",
            actor_type="system",
            resource_type="circuit_breaker",
            resource_id=endpoint,
            action="circuit_open",
            endpoint=endpoint,
            method="",
            status_code=503,
            details={
                "failure_count": failure_count,
                "endpoint": endpoint
            },
            correlation_id=correlation_id,
            data_classification="internal"
        )


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None
_audit_logger_lock = threading.Lock()


def get_audit_logger() -> AuditLogger:
    """Get or create global audit logger."""
    global _audit_logger
    if _audit_logger is None:
        with _audit_logger_lock:
            if _audit_logger is None:
                _audit_logger = AuditLogger(
                    service_name=os.getenv("SERVICE_NAME", "rec-engine"),
                    environment=os.getenv("ENVIRONMENT", "production"),
                    log_level=os.getenv("AUDIT_LOG_LEVEL", "INFO"),
                    output_file=os.getenv("AUDIT_LOG_FILE"),
                    enable_console=os.getenv("AUDIT_LOG_CONSOLE", "true").lower() == "true"
                )
    return _audit_logger


def set_correlation_id(correlation_id: str = None) -> str:
    """Set correlation ID for current context."""
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    correlation_id_var.set(correlation_id)
    return correlation_id


def set_request_id(request_id: str = None) -> str:
    """Set request ID for current context."""
    if request_id is None:
        request_id = str(uuid.uuid4())
    request_id_var.set(request_id)
    return request_id


def set_trace_id(trace_id: str = None) -> str:
    """Set trace ID for current context."""
    if trace_id is None:
        trace_id = str(uuid.uuid4())
    trace_id_var.set(trace_id)
    return trace_id


def clear_context():
    """Clear all context variables."""
    correlation_id_var.set('')
    request_id_var.set('')
    trace_id_var.set('')