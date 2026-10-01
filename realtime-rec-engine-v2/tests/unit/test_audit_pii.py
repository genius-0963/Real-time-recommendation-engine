"""
Unit tests for AuditLogger and PII Scrubber.
"""

import pytest
import json
import logging
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timezone

from app.monitoring.audit import (
    AuditLogger,
    AuditEventType,
    AuditDecision,
    AuditLogEntry,
    get_audit_logger,
    set_correlation_id,
    set_request_id,
    clear_context,
)
from app.monitoring.pii_scrubber import (
    PIIScrubber,
    PIIScrubbingFilter,
    get_pii_scrubber,
    setup_pii_scrubbing,
    PIIScrubbingMiddleware,
)


class TestAuditLogEntry:
    """Test AuditLogEntry dataclass."""
    
    def test_audit_log_entry_creation(self):
        """Test creating an audit log entry."""
        entry = AuditLogEntry(
            event_type="test.event",
            decision="allow",
            actor_id="user_123",
            actor_type="user",
            resource_type="api",
            resource_id="test",
            action="read",
            endpoint="/test",
            method="GET",
            status_code=200,
        )
        
        assert entry.event_type == "test.event"
        assert entry.decision == "allow"
        assert entry.actor_id == "user_123"
        assert entry.timestamp is not None
    
    def test_audit_log_entry_to_dict(self):
        """Test converting to dictionary."""
        entry = AuditLogEntry(
            event_type="test.event",
            decision="allow",
            actor_id="user_123",
        )
        
        d = entry.to_dict()
        assert isinstance(d, dict)
        assert d["event_type"] == "test.event"
        assert d["decision"] == "allow"
        assert d["actor_id"] == "user_123"
        assert "timestamp" in d
        assert "correlation_id" in d
    
    def test_audit_log_entry_to_json(self):
        """Test converting to JSON."""
        entry = AuditLogEntry(
            event_type="test.event",
            decision="allow",
            actor_id="user_123",
        )
        
        json_str = entry.to_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["event_type"] == "test.event"


class TestAuditLogger:
    """Test AuditLogger functionality."""
    
    def setup_method(self):
        """Reset singleton for each test."""
        import app.monitoring.audit as audit_module
        audit_module._audit_logger = None
    
    def test_audit_logger_creation(self):
        """Test creating audit logger."""
        logger = AuditLogger(
            service_name="test-service",
            environment="test",
            log_level="DEBUG",
            enable_console=False
        )
        
        assert logger.service_name == "test-service"
        assert logger.environment == "test"
        assert logger.logger is not None
    
    def test_log_auth_success(self):
        """Test logging auth success."""
        logger = AuditLogger(
            service_name="test",
            environment="test",
            log_level="DEBUG",
            enable_console=False
        )
        
        corr_id = logger.log_auth_success(
            user_id="user_123",
            roles=["rec-engine-user"],
            ip="192.168.1.1",
            method="password"
        )
        
        assert corr_id is not None
        assert len(corr_id) > 0
    
    def test_log_auth_failure(self):
        """Test logging auth failure."""
        logger = AuditLogger(
            service_name="test",
            environment="test",
            log_level="DEBUG",
            enable_console=False
        )
        
        corr_id = logger.log_auth_failure(
            user_id="user_123",
            ip="192.168.1.1",
            reason="Invalid password",
            method="password"
        )
        
        assert corr_id is not None
    
    def test_log_authz_allow(self):
        """Test logging authorization allow."""
        logger = AuditLogger(
            service_name="test",
            environment="test",
            log_level="DEBUG",
            enable_console=False
        )
        
        corr_id = logger.log_authz_allow(
            user_id="user_123",
            roles=["rec-engine-user"],
            resource_type="recommendation",
            resource_id="rec_10",
            action="recommend",
            endpoint="/recommend",
            method="POST"
        )
        
        assert corr_id is not None
    
    def test_log_authz_deny(self):
        """Test logging authorization deny."""
        logger = AuditLogger(
            service_name="test",
            environment="test",
            log_level="DEBUG",
            enable_console=False
        )
        
        corr_id = logger.log_authz_deny(
            user_id="user_123",
            roles=["rec-engine-user"],
            resource_type="admin",
            resource_id="admin_panel",
            action="admin_access",
            endpoint="/admin",
            method="GET",
            reason="Insufficient permissions"
        )
        
        assert corr_id is not None
    
    def test_log_data_access(self):
        """Test logging data access."""
        logger = AuditLogger(
            service_name="test",
            environment="test",
            log_level="DEBUG",
            enable_console=False
        )
        
        corr_id = logger.log_data_access(
            user_id="user_123",
            roles=["rec-engine-user"],
            resource_type="feature",
            resource_id="user_age",
            resource_owner="user_123",
            action="read",
            endpoint="/user/user_123/features",
            method="GET",
            pii_accessed=True
        )
        
        assert corr_id is not None
    
    def test_log_recommendation_request(self):
        """Test logging recommendation request."""
        logger = AuditLogger(
            service_name="test",
            environment="test",
            log_level="DEBUG",
            enable_console=False
        )
        
        corr_id = logger.log_recommendation_request(
            user_id="user_123",
            roles=["rec-engine-user"],
            num_recommendations=10,
            endpoint="/recommend",
            method="POST",
            latency_ms=45.2
        )
        
        assert corr_id is not None
    
    def test_log_config_change(self):
        """Test logging config change."""
        logger = AuditLogger(
            service_name="test",
            environment="test",
            log_level="DEBUG",
            enable_console=False
        )
        
        corr_id = logger.log_config_change(
            user_id="admin_123",
            roles=["rec-engine-admin"],
            config_key="rate_limit",
            old_value="1000",
            new_value="2000"
        )
        
        assert corr_id is not None
    
    def test_log_index_rebuild(self):
        """Test logging index rebuild."""
        logger = AuditLogger(
            service_name="test",
            environment="test",
            log_level="DEBUG",
            enable_console=False
        )
        
        corr_id = logger.log_index_rebuild(
            user_id="admin_123",
            roles=["rec-engine-admin"],
            index_type="scann",
            num_items=1000000,
            duration_ms=45000.0,
            success=True
        )
        
        assert corr_id is not None
    
    def test_log_rate_limit_exceeded(self):
        """Test logging rate limit exceeded."""
        logger = AuditLogger(
            service_name="test",
            environment="test",
            log_level="DEBUG",
            enable_console=False
        )
        
        corr_id = logger.log_rate_limit_exceeded(
            identifier="user_123",
            scope="user",
            limit=1000,
            window_seconds=60
        )
        
        assert corr_id is not None
    
    def test_log_circuit_breaker_open(self):
        """Test logging circuit breaker open."""
        logger = AuditLogger(
            service_name="test",
            environment="test",
            log_level="DEBUG",
            enable_console=False
        )
        
        corr_id = logger.log_circuit_breaker_open(
            endpoint="/recommend",
            failure_count=10
        )
        
        assert corr_id is not None
    
    def test_general_log(self):
        """Test general log method."""
        logger = AuditLogger(
            service_name="test",
            environment="test",
            log_level="DEBUG",
            enable_console=False
        )
        
        corr_id = logger.log(
            event_type=AuditEventType.DATA_READ,
            decision=AuditDecision.ALLOW,
            actor_id="user_123",
            actor_type="user",
            actor_roles=["rec-engine-user"],
            resource_type="recommendation",
            resource_id="rec_10",
            action="recommend",
            endpoint="/recommend",
            method="POST",
            status_code=200,
            correlation_id="test_corr_id"
        )
        
        assert corr_id == "test_corr_id"
    
    def test_global_functions(self):
        """Test global getter functions."""
        # Reset singleton
        import app.monitoring.audit as audit_module
        audit_module._audit_logger = None
        
        logger = get_audit_logger()
        assert logger is not None
        assert isinstance(logger, AuditLogger)
        
        # Second call should return same instance
        logger2 = get_audit_logger()
        assert logger is logger2
    
    def test_correlation_id_context(self):
        """Test correlation ID context management."""
        corr_id = set_correlation_id("test_corr_123")
        assert corr_id == "test_corr_123"
        
        req_id = set_request_id("test_req_456")
        assert req_id == "test_req_456"
        
        clear_context()
        # After clear, should generate new IDs
        new_corr = set_correlation_id()
        assert new_corr != "test_corr_123"


class TestPIIScrubber:
    """Test PIIScrubber functionality."""
    
    def test_scrub_email(self):
        """Test scrubbing email addresses."""
        scrubber = PIIScrubber()
        
        text = "Contact user@example.com for support"
        result = scrubber.scrub_string(text)
        
        assert "user@example.com" not in result
        assert "[EMAIL_REDACTED]" in result
    
    def test_scrub_phone(self):
        """Test scrubbing phone numbers."""
        scrubber = PIIScrubber()
        
        text = "Call 555-123-4567 or (555) 123-4567"
        result = scrubber.scrub_string(text)
        
        assert "555-123-4567" not in result
        assert "(555) 123-4567" not in result
        assert "[PHONE_REDACTED]" in result
    
    def test_scrub_ipv4(self):
        """Test scrubbing IPv4 addresses."""
        scrubber = PIIScrubber()
        
        text = "Client IP: 192.168.1.100"
        result = scrubber.scrub_string(text)
        
        assert "192.168.1.100" not in result
        assert "[IP_REDACTED]" in result
    
    def test_scrub_uuid(self):
        """Test scrubbing UUIDs."""
        scrubber = PIIScrubber()
        
        text = "User ID: 123e4567-e89b-12d3-a456-426614174000"
        result = scrubber.scrub_string(text)
        
        assert "123e4567-e89b-12d3-a456-426614174000" not in result
        assert "[UUID_REDACTED]" in result
    
    def test_scrub_jwt(self):
        """Test scrubbing JWT tokens."""
        scrubber = PIIScrubber()
        
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        text = f"Token: {jwt}"
        result = scrubber.scrub_string(text)
        
        assert jwt not in result
        assert "[JWT_REDACTED]" in result
    
    def test_scrub_api_key(self):
        """Test scrubbing API keys."""
        scrubber = PIIScrubber()
        
        text = "API Key: api_key=sk-1234567890abcdef1234"
        result = scrubber.scrub_string(text)
        
        assert "sk-1234567890abcdef1234" not in result
        assert "[API_KEY_REDACTED]" in result
    
    def test_scrub_aws_key(self):
        """Test scrubbing AWS access keys."""
        scrubber = PIIScrubber()
        
        text = "AKIAIOSFODNN7EXAMPLE"
        result = scrubber.scrub_string(text)
        
        assert "AKIAIOSFODNN7EXAMPLE" not in result
        assert "[AWS_KEY_REDACTED]" in result
    
    def test_scrub_auth_header(self):
        """Test scrubbing authorization headers."""
        scrubber = PIIScrubber()
        
        text = 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9'
        result = scrubber.scrub_string(text)
        
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in result
        assert "Bearer [REDACTED]" in result or "[REDACTED]" in result
    
    def test_scrub_dict(self):
        """Test scrubbing dictionary values."""
        scrubber = PIIScrubber()
        
        data = {
            "user_id": "user_123",
            "email": "user@example.com",
            "name": "John Doe",
            "api_key": "sk-1234567890",
            "preferences": {"theme": "dark"}
        }
        
        result = scrubber.scrub_dict(data)
        
        assert result["user_id"] == "[REDACTED]"
        assert result["email"] == "[REDACTED]"
        assert result["api_key"] == "[REDACTED]"
        assert result["name"] == "John Doe"  # Not in default PII fields
        assert result["preferences"]["theme"] == "dark"
    
    def test_scrub_nested_dict(self):
        """Test scrubbing nested dictionaries."""
        scrubber = PIIScrubber()
        
        data = {
            "user": {
                "email": "nested@example.com",
                "profile": {
                    "phone": "555-123-4567"
                }
            }
        }
        
        result = scrubber.scrub_dict(data)
        
        assert result["user"]["email"] == "[REDACTED]"
        assert "555-123-4567" not in str(result)
    
    def test_scrub_list(self):
        """Test scrubbing lists."""
        scrubber = PIIScrubber()
        
        data = {
            "emails": ["a@b.com", "c@d.com"],
            "names": ["Alice", "Bob"]
        }
        
        result = scrubber.scrub_dict(data)
        
        assert "[EMAIL_REDACTED]" in str(result["emails"])
        assert result["names"] == ["Alice", "Bob"]
    
    def test_allowlist(self):
        """Test allowlist functionality."""
        scrubber = PIIScrubber(allowlist={"test@example.com"})
        
        text = "Email: test@example.com and other@example.com"
        result = scrubber.scrub_string(text)
        
        # Allowlisted email should not be redacted
        assert "test@example.com" in result
        # Other email should be redacted
        assert "other@example.com" not in result
        assert "[EMAIL_REDACTED]" in result


class TestPIIScrubbingFilter:
    """Test PIIScrubbingFilter logging filter."""
    
    def test_filter_scrubs_log_record(self):
        """Test that filter scrubs log record."""
        scrubber = PIIScrubber()
        filter_obj = PIIScrubbingFilter(scrubber, enabled=True)
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="User email: user@example.com",
            args=(),
            exc_info=None
        )
        
        result = filter_obj.filter(record)
        
        assert result is True
        assert "user@example.com" not in record.msg
        assert "[EMAIL_REDACTED]" in record.msg
    
    def test_filter_disabled(self):
        """Test filter when disabled."""
        scrubber = PIIScrubber()
        filter_obj = PIIScrubbingFilter(scrubber, enabled=False)
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="User email: user@example.com",
            args=(),
            exc_info=None
        )
        
        result = filter_obj.filter(record)
        
        assert result is True
        assert "user@example.com" in record.msg


class TestSetupPIIScrubbing:
    """Test setup_pii_scrubbing function."""
    
    def test_setup_adds_filter(self):
        """Test that setup adds filter to logger."""
        logger = logging.getLogger("test_pii_setup")
        logger.handlers = []
        logger.setLevel(logging.DEBUG)
        
        scrubber = setup_pii_scrubbing(logger_name="test_pii_setup", enabled=True)
        
        assert scrubber is not None
        assert isinstance(scrubber, PIIScrubber)
        
        # Check filter was added
        filters = logger.filters
        assert len(filters) > 0
        assert any(isinstance(f, PIIScrubbingFilter) for f in filters)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])