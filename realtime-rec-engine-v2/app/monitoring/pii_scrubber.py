"""
PII Scrubbing Middleware - Removes PII from log output.
Uses regex patterns and optional Presidio for detection.
"""

import re
import logging
import os
from typing import Dict, List, Optional, Set, Callable
from dataclasses import dataclass
from functools import wraps

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False


@dataclass
class PIIPattern:
    """PII detection pattern."""
    name: str
    pattern: re.Pattern
    replacement: str
    flags: int = 0


class PIIScrubber:
    """
    Scrubs PII from log messages and structured data.
    Supports both regex-based and Presidio-based detection.
    """
    
    # Default PII patterns
    DEFAULT_PATTERNS = [
        # Email addresses
        PIIPattern(
            name="email",
            pattern=re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            replacement="[EMAIL_REDACTED]"
        ),
        # Phone numbers (US format)
        PIIPattern(
            name="phone_us",
            pattern=re.compile(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'),
            replacement="[PHONE_REDACTED]"
        ),
        # IP addresses (IPv4)
        PIIPattern(
            name="ipv4",
            pattern=re.compile(r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'),
            replacement="[IP_REDACTED]"
        ),
        # IPv6 addresses (simplified)
        PIIPattern(
            name="ipv6",
            pattern=re.compile(r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'),
            replacement="[IP_REDACTED]"
        ),
        # Credit card numbers (basic Luhn-agnostic pattern)
        PIIPattern(
            name="credit_card",
            pattern=re.compile(r'\b(?:\d[ -]*?){13,16}\b'),
            replacement="[CARD_REDACTED]"
        ),
        # Social Security Numbers (US)
        PIIPattern(
            name="ssn",
            pattern=re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            replacement="[SSN_REDACTED]"
        ),
        # UUIDs (potential user identifiers)
        PIIPattern(
            name="uuid",
            pattern=re.compile(r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b'),
            replacement="[UUID_REDACTED]"
        ),
        # JWT tokens
        PIIPattern(
            name="jwt",
            pattern=re.compile(r'eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+'),
            replacement="[JWT_REDACTED]"
        ),
        # API Keys (common patterns)
        PIIPattern(
            name="api_key",
            pattern=re.compile(r'\b(?:api[_-]?key|apikey|access[_-]?token|secret[_-]?key)[\s:=]+[A-Za-z0-9_\-]{20,}\b', re.IGNORECASE),
            replacement="[API_KEY_REDACTED]"
        ),
        # AWS Access Keys
        PIIPattern(
            name="aws_access_key",
            pattern=re.compile(r'\bAKIA[0-9A-Z]{16}\b'),
            replacement="[AWS_KEY_REDACTED]"
        ),
        # Generic authorization headers
        PIIPattern(
            name="auth_header",
            pattern=re.compile(r'(?i)(authorization|x-api-key|x-auth-token)\s*[:=]\s*[^\s,}]+', re.IGNORECASE),
            replacement=r'\1=[REDACTED]'
        ),
        # User IDs in common formats
        PIIPattern(
            name="user_id",
            pattern=re.compile(r'\b(?:user[_-]?id|userid|uid)[\s:=]+[A-Za-z0-9_\-]{1,64}\b', re.IGNORECASE),
            replacement=r'\1=[USER_ID_REDACTED]'
        ),
    ]
    
    def __init__(
        self,
        patterns: List[PIIPattern] = None,
        custom_patterns: List[PIIPattern] = None,
        use_presidio: bool = False,
        presidio_languages: List[str] = None,
        allowlist: Set[str] = None
    ):
        self.patterns = patterns or self.DEFAULT_PATTERNS
        if custom_patterns:
            self.patterns.extend(custom_patterns)
        
        self.allowlist = allowlist or set()
        self.use_presidio = use_presidio and PRESIDIO_AVAILABLE
        self.presidio_analyzer = None
        
        if self.use_presidio:
            self._init_presidio(presidio_languages or ["en"])
    
    def _init_presidio(self, languages: List[str]):
        """Initialize Presidio analyzer."""
        try:
            nlp_engine = NlpEngineProvider(nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": lang, "model_name": f"{lang}_core_web_sm"} for lang in languages]
            }).create_engine()
            
            self.presidio_analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
        except Exception as e:
            logging.warning(f"Failed to initialize Presidio: {e}. Falling back to regex only.")
            self.use_presidio = False
    
    def scrub_string(self, text: str) -> str:
        """Scrub PII from a string using regex patterns."""
        if not text:
            return text
        
        # Check allowlist first
        for allowed in self.allowlist:
            if allowed in text:
                return text
        
        result = text
        for pattern in self.patterns:
            result = pattern.pattern.sub(pattern.replacement, result)
        
        # Use Presidio for additional detection
        if self.use_presidio and self.presidio_analyzer:
            result = self._scrub_with_presidio(result)
        
        return result
    
    def _scrub_with_presidio(self, text: str) -> str:
        """Scrub using Presidio analyzer."""
        try:
            results = self.presidio_analyzer.analyze(text=text, language="en")
            # Sort by start position descending to avoid index shifting
            results.sort(key=lambda x: x.start, reverse=True)
            
            chars = list(text)
            for result in results:
                if result.score >= 0.7:  # High confidence threshold
                    entity_text = text[result.start:result.end]
                    if entity_text not in self.allowlist:
                        replacement = f"[{result.entity_type}_REDACTED]"
                        for i in range(result.start, result.end):
                            if i < len(chars):
                                chars[i] = ''
                        chars[result.start] = replacement
            
            return ''.join(chars)
        except Exception:
            return text
    
    def scrub_dict(self, data: Dict[str, Any], pii_fields: Set[str] = None) -> Dict[str, Any]:
        """Scrub PII from dictionary values."""
        if pii_fields is None:
            pii_fields = {
                "email", "phone", "ip", "ip_address", "ipv4", "ipv6",
                "user_id", "user_id", "username", "full_name", "name",
                "address", "street", "city", "zip", "postal_code",
                "ssn", "credit_card", "card_number", "cvv",
                "api_key", "secret", "token", "password", "authorization",
                "device_id", "device_id", "imei", "mac_address",
                "location", "latitude", "longitude", "geo"
            }
        
        result = {}
        for key, value in data.items():
            if key.lower() in pii_fields:
                result[key] = "[REDACTED]"
            elif isinstance(value, str):
                result[key] = self.scrub_string(value)
            elif isinstance(value, dict):
                result[key] = self.scrub_dict(value, pii_fields)
            elif isinstance(value, list):
                result[key] = [
                    self.scrub_dict(item, pii_fields) if isinstance(item, dict)
                    else self.scrub_string(item) if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                result[key] = value
        return result
    
    def scrub_log_record(self, record: logging.LogRecord) -> logging.LogRecord:
        """Scrub PII from a log record."""
        # Scrub message
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            record.msg = self.scrub_string(record.msg)
        
        # Scrub args
        if record.args:
            record.args = tuple(
                self.scrub_string(arg) if isinstance(arg, str) else arg
                for arg in record.args
            )
        
        # Scrub extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'exc_info', 'exc_text', 'stack_info']:
                if isinstance(value, str):
                    record.__dict__[key] = self.scrub_string(value)
                elif isinstance(value, dict):
                    record.__dict__[key] = self.scrub_dict(value)
        
        return record


class PIIScrubbingFilter(logging.Filter):
    """Logging filter that scrubs PII from log records."""
    
    def __init__(self, scrubber: PIIScrubber = None, enabled: bool = True):
        super().__init__()
        self.scrubber = scrubber or PIIScrubber()
        self.enabled = enabled
    
    def filter(self, record: logging.LogRecord) -> bool:
        if self.enabled:
            self.scrubber.scrub_log_record(record)
        return True


class PIIScrubbingMiddleware:
    """ASGI middleware that scrubs PII from request/response logging."""
    
    def __init__(self, app, scrubber: PIIScrubber = None, exclude_paths: List[str] = None):
        self.app = app
        self.scrubber = scrubber or PIIScrubber()
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/favicon.ico"]
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        path = scope.get("path", "")
        if path in self.exclude_paths:
            await self.app(scope, receive, send)
            return
        
        # Capture request/response for logging
        request_body = b""
        response_body = b""
        status_code = 200
        
        async def receive_wrapper():
            nonlocal request_body
            message = await receive()
            if message["type"] == "http.request":
                request_body += message.get("body", b"")
            return message
        
        async def send_wrapper(message):
            nonlocal response_body, status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            elif message["type"] == "http.response.body":
                response_body += message.get("body", b"")
            await send(message)
        
        try:
            await self.app(scope, receive_wrapper, send_wrapper)
        finally:
            # Log scrubbed request/response
            self._log_request_response(scope, request_body, response_body, status_code)
    
    def _log_request_response(self, scope, request_body, response_body, status_code):
        """Log scrubbed request/response."""
        method = scope.get("method", "")
        path = scope.get("path", "")
        query_string = scope.get("query_string", b"").decode("utf-8", errors="ignore")
        client = scope.get("client", ("unknown", 0))
        
        # Scrub request body
        scrubbed_request = self._scrub_body(request_body, path)
        scrubbed_response = self._scrub_body(response_body, path)
        
        logger = logging.getLogger("http.access")
        logger.info(
            "http_request",
            method=method,
            path=path,
            query=query_string,
            client_ip=client[0],
            status_code=status_code,
            request_body=scrubbed_request,
            response_body=scrubbed_response
        )
    
    def _scrub_body(self, body: bytes, path: str) -> str:
        """Scrub PII from request/response body."""
        if not body:
            return ""
        
        try:
            text = body.decode("utf-8", errors="ignore")
            return self.scrubber.scrub_string(text)
        except Exception:
            return "[BODY_SCRUB_ERROR]"


# Decorator for function-level PII scrubbing
def scrub_pii(pii_fields: Set[str] = None):
    """Decorator to scrub PII from function return value."""
    def decorator(func: Callable) -> Callable:
        scrubber = PIIScrubber()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, dict):
                return scrubber.scrub_dict(result, pii_fields)
            elif isinstance(result, list):
                return [scrubber.scrub_dict(item, pii_fields) if isinstance(item, dict) else item for item in result]
            elif isinstance(result, str):
                return scrubber.scrub_string(result)
            return result
        return wrapper
    return decorator


# Global scrubber instance
_pii_scrubber: Optional[PIIScrubber] = None


def get_pii_scrubber() -> PIIScrubber:
    """Get global PII scrubber instance."""
    global _pii_scrubber
    if _pii_scrubber is None:
        use_presidio = os.getenv("PII_USE_PRESIDIO", "false").lower() == "true"
        _pii_scrubber = PIIScrubber(use_presidio=use_presidio)
    return _pii_scrubber


def setup_pii_scrubbing(logger_name: str = None, enabled: bool = True):
    """Setup PII scrubbing for a logger."""
    scrubber = get_pii_scrubber()
    target_logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
    target_logger.addFilter(PIIScrubbingFilter(scrubber, enabled))
    return scrubber