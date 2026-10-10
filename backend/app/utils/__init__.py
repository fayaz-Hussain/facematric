"""
Utility modules for FaceMatric backend.
"""

from .errors import (
    ErrorCode,
    ValidationError,
    ProcessingError,
    get_error_message,
    ERROR_MESSAGES
)
from .logger import (
    setup_logging,
    get_logger,
    log_processing_start,
    log_processing_complete,
    log_processing_error
)

__all__ = [
    "ErrorCode",
    "ValidationError",
    "ProcessingError",
    "get_error_message",
    "ERROR_MESSAGES",
    "setup_logging",
    "get_logger",
    "log_processing_start",
    "log_processing_complete",
    "log_processing_error",
]
