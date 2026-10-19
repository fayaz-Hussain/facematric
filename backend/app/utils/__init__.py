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
from .face_crop import extract_face_crop

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
    "extract_face_crop",
]
