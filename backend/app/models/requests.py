"""
Pydantic request models for FaceMatric API.

This module defines request schemas for the FaceMatric facial analysis API,
including validation rules and documentation examples.

Requirements Validated:
- 14.1: API endpoint accepts structured requests
- 14.2: API response includes status field
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
import base64
import binascii


class AnalyzeRequest(BaseModel):
    """
    Request model for facial analysis endpoint.
    
    Supports two input methods:
    1. Base64-encoded image string (image_base64 field)
    2. Multipart file upload (handled separately by FastAPI UploadFile)
    
    The endpoint can accept either method. When using JSON body, provide image_base64.
    When using multipart/form-data, the file is handled directly by FastAPI.
    
    Attributes:
        image_base64: Optional base64-encoded image string. Accepts both raw base64
                     and data URI format (e.g., "data:image/jpeg;base64,...")
        include_celebrity_match: Whether to perform celebrity matching. Default is True.
                                Set to False to skip celebrity matching and reduce processing time.
    
    Validation Rules:
        - image_base64 must be valid base64 encoding if provided
        - Data URI prefixes are automatically stripped
        - At least one input method (image_base64 or file upload) must be provided at endpoint level
    
    Example JSON body:
        {
            "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAA...",
            "include_celebrity_match": true
        }
    
    Example minimal request (skipping celebrity matching):
        {
            "image_base64": "/9j/4AAQSkZJRgABAQAA...",
            "include_celebrity_match": false
        }
    """
    
    image_base64: Optional[str] = Field(
        None,
        description=(
            "Base64-encoded image string. Supports JPEG, PNG, HEIC, WebP, BMP, and GIF formats. "
            "Can include data URI prefix (e.g., 'data:image/jpeg;base64,...') or raw base64 string. "
            "Maximum recommended size: 10MB (approximately 13.7MB base64)."
        ),
        examples=[
            "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAA...",
            "/9j/4AAQSkZJRgABAQAA..."
        ]
    )
    
    include_celebrity_match: bool = Field(
        True,
        description=(
            "Whether to perform celebrity face matching analysis. "
            "Set to False to skip celebrity matching and reduce processing time by ~2 seconds. "
            "Golden ratio scoring will still be performed regardless of this setting."
        ),
        examples=[True, False]
    )
    
    @field_validator('image_base64')
    @classmethod
    def validate_base64(cls, v: Optional[str]) -> Optional[str]:
        """
        Validate and normalize base64 image string.
        
        This validator:
        1. Returns None if no value provided (file upload will be used instead)
        2. Strips data URI prefix if present (e.g., "data:image/jpeg;base64,")
        3. Validates that remaining string is valid base64 encoding
        
        Args:
            v: Raw base64 string or data URI
        
        Returns:
            Cleaned base64 string without data URI prefix
        
        Raises:
            ValueError: If base64 encoding is invalid
        """
        if v is None:
            return v
        
        # Strip whitespace
        v = v.strip()
        
        if not v:
            return None
        
        # Strip data URI prefix if present
        # Format: data:[<mediatype>][;base64],<data>
        if v.startswith('data:'):
            comma_index = v.find(',')
            if comma_index != -1:
                # Extract the data portion after the comma
                v = v[comma_index + 1:]
            else:
                # Malformed data URI without comma
                raise ValueError(
                    "Invalid data URI format. Expected format: 'data:image/jpeg;base64,<data>'"
                )
        
        # Remove any whitespace that might have been inserted during transmission
        v = v.replace(' ', '').replace('\n', '').replace('\r', '').replace('\t', '')
        
        # Validate it's proper base64 encoding
        try:
            # Attempt to decode to verify validity
            # validate=True ensures strict base64 alphabet checking
            base64.b64decode(v, validate=True)
        except (binascii.Error, ValueError) as e:
            raise ValueError(
                f"Invalid base64 encoding: {str(e)}. "
                "Ensure the string contains only valid base64 characters (A-Z, a-z, 0-9, +, /, =)"
            )
        
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAAyADIBAREA/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/AP38oooooooooooooooooooooooor//Z",
                    "include_celebrity_match": True
                },
                {
                    "image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8U...",
                    "include_celebrity_match": False
                }
            ]
        }
    }


class HealthCheckRequest(BaseModel):
    """Request model for health check endpoint (typically no body needed)."""
    pass
