"""
Custom error classes and error codes for FaceMatric API.
"""

from enum import Enum
from typing import Optional, Dict, Any


class ErrorCode(str, Enum):
    """Standardized error codes for API responses."""
    
    # Request/Input validation errors (0xx)
    INVALID_REQUEST = "INVALID_REQUEST"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    
    # Image-related errors (1xx)
    INVALID_IMAGE = "INVALID_IMAGE"
    POOR_QUALITY = "POOR_QUALITY"
    NO_IMAGE = "NO_IMAGE"
    
    # Face detection errors (2xx)
    NO_FACE_DETECTED = "NO_FACE_DETECTED"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    OCCLUSION_DETECTED = "OCCLUSION_DETECTED"
    
    # Processing errors (3xx)
    PROCESSING_TIMEOUT = "PROCESSING_TIMEOUT"
    PROCESSING_FAILED = "PROCESSING_FAILED"
    
    # Celebrity matching errors (4xx)
    CELEBRITY_MATCHING_FAILED = "CELEBRITY_MATCHING_FAILED"
    CELEBRITY_DB_UNAVAILABLE = "CELEBRITY_DB_UNAVAILABLE"
    
    # System errors (5xx)
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"


class ValidationError(Exception):
    """
    Custom exception for validation errors.
    
    Includes error code and user-friendly message.
    """
    
    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        retry: bool = True,
        details: Optional[Dict[str, Any]] = None
    ):
        self.error_code = error_code
        self.message = message
        self.retry = retry
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API response."""
        response = {
            "status": "error",
            "error_code": self.error_code.value,
            "message": self.message,
            "retry": self.retry
        }
        
        if self.details:
            response["details"] = self.details
        
        return response


class ProcessingError(Exception):
    """Exception for processing-related failures."""
    
    def __init__(self, message: str, component: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.component = component
        self.details = details or {}
        super().__init__(self.message)


# User-friendly error messages
ERROR_MESSAGES = {
    ErrorCode.INVALID_REQUEST: "Invalid request format. Please provide either a file upload or base64 image string, not both.",
    ErrorCode.FILE_TOO_LARGE: "File size exceeds 10MB limit. Please upload a smaller image.",
    ErrorCode.INVALID_IMAGE: "Corrupt or unsupported file format. Please upload a valid image (JPEG, PNG, HEIC, WebP, BMP, or GIF).",
    ErrorCode.POOR_QUALITY: "Image quality is too low for accurate analysis. Please ensure your photo is at least 300x300 pixels, well-lit, and facing the camera directly.",
    ErrorCode.NO_IMAGE: "No image provided. Please upload or capture a photo.",
    ErrorCode.NO_FACE_DETECTED: "We couldn't detect a face in your photo. Please take a clear, front-facing photo with good lighting.",
    ErrorCode.MULTIPLE_FACES: "Multiple faces detected. Please upload a photo with only one person.",
    ErrorCode.OCCLUSION_DETECTED: "We detected accessories covering your face. Please upload a photo without glasses, masks, or other obstructions for accurate analysis.",
    ErrorCode.PROCESSING_TIMEOUT: "Processing took too long. Please try again with a clearer photo.",
    ErrorCode.PROCESSING_FAILED: "Failed to process your image. Please try again.",
    ErrorCode.CELEBRITY_MATCHING_FAILED: "Celebrity matching is temporarily unavailable. Your golden ratio score is still available.",
    ErrorCode.CELEBRITY_DB_UNAVAILABLE: "Celebrity database is not available.",
    ErrorCode.INTERNAL_ERROR: "An unexpected error occurred. Please try again.",
    ErrorCode.SERVICE_UNAVAILABLE: "Service is temporarily unavailable. Please try again later."
}


def get_error_message(error_code: ErrorCode, custom_message: Optional[str] = None) -> str:
    """
    Get user-friendly error message for error code.
    
    Args:
        error_code: The error code
        custom_message: Optional custom message to override default
    
    Returns:
        User-friendly error message
    """
    return custom_message or ERROR_MESSAGES.get(error_code, "An error occurred.")
