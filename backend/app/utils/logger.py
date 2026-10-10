"""
Structured logging configuration for FaceMatric API.
"""

import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict
from app.config import settings


class JSONFormatter(logging.Formatter):
    """Custom formatter to output logs as JSON."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        
        if hasattr(record, "component"):
            log_data["component"] = record.component
        
        if hasattr(record, "processing_time_ms"):
            log_data["processing_time_ms"] = record.processing_time_ms
        
        return json.dumps(log_data)


class TextFormatter(logging.Formatter):
    """Standard text formatter for human-readable logs."""
    
    def __init__(self):
        super().__init__(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )


def setup_logging() -> None:
    """Configure application logging based on settings."""
    
    # Get log level from settings
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    
    # Set formatter based on configuration
    if settings.log_format.lower() == "json":
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter()
    
    handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(handler)
    
    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    logging.getLogger("mediapipe").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a component.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


# Privacy-safe logging helpers
def log_processing_start(logger: logging.Logger, request_id: str) -> None:
    """Log the start of image processing (no sensitive data)."""
    logger.info(
        "Processing started",
        extra={"request_id": request_id}
    )


def log_processing_complete(
    logger: logging.Logger,
    request_id: str,
    processing_time_ms: int,
    component: str
) -> None:
    """Log successful processing completion."""
    logger.info(
        f"{component} completed",
        extra={
            "request_id": request_id,
            "processing_time_ms": processing_time_ms,
            "component": component
        }
    )


def log_processing_error(
    logger: logging.Logger,
    request_id: str,
    error_code: str,
    component: str,
    error_message: str
) -> None:
    """Log processing error without sensitive data."""
    logger.error(
        f"{component} failed: {error_code}",
        extra={
            "request_id": request_id,
            "component": component,
            "error_code": error_code,
            "error_message": error_message
        }
    )
