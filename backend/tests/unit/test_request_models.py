"""
Unit tests for Pydantic request models.

Tests validation logic, field defaults, and error handling for API request schemas.
"""

import pytest
from pydantic import ValidationError
from app.models.requests import AnalyzeRequest
import base64


class TestAnalyzeRequest:
    """Test suite for AnalyzeRequest model."""
    
    def test_valid_raw_base64(self):
        """Test that valid raw base64 string is accepted."""
        # Create a simple 1x1 red pixel JPEG
        valid_base64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCwAA8A/9k="
        
        request = AnalyzeRequest(
            image_base64=valid_base64,
            include_celebrity_match=True
        )
        
        assert request.image_base64 == valid_base64
        assert request.include_celebrity_match is True
    
    def test_valid_data_uri_base64(self):
        """Test that base64 with data URI prefix is accepted and cleaned."""
        valid_base64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCwAA8A/9k="
        data_uri = f"data:image/jpeg;base64,{valid_base64}"
        
        request = AnalyzeRequest(
            image_base64=data_uri,
            include_celebrity_match=True
        )
        
        # Validator should strip the data URI prefix
        assert request.image_base64 == valid_base64
        assert not request.image_base64.startswith("data:")
    
    def test_default_include_celebrity_match(self):
        """Test that include_celebrity_match defaults to True."""
        valid_base64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCwAA8A/9k="
        
        request = AnalyzeRequest(image_base64=valid_base64)
        
        assert request.include_celebrity_match is True
    
    def test_celebrity_match_can_be_false(self):
        """Test that include_celebrity_match can be explicitly set to False."""
        valid_base64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCwAA8A/9k="
        
        request = AnalyzeRequest(
            image_base64=valid_base64,
            include_celebrity_match=False
        )
        
        assert request.include_celebrity_match is False
    
    def test_none_image_base64_allowed(self):
        """Test that image_base64 can be None (for file uploads)."""
        request = AnalyzeRequest(
            image_base64=None,
            include_celebrity_match=True
        )
        
        assert request.image_base64 is None
        assert request.include_celebrity_match is True
    
    def test_empty_string_returns_none(self):
        """Test that empty string for image_base64 is converted to None."""
        request = AnalyzeRequest(
            image_base64="",
            include_celebrity_match=True
        )
        
        assert request.image_base64 is None
    
    def test_whitespace_only_returns_none(self):
        """Test that whitespace-only string is converted to None."""
        request = AnalyzeRequest(
            image_base64="   \n\t  ",
            include_celebrity_match=True
        )
        
        assert request.image_base64 is None
    
    def test_invalid_base64_raises_error(self):
        """Test that invalid base64 encoding raises ValidationError."""
        invalid_base64 = "this is not valid base64!!@#$%"
        
        with pytest.raises(ValidationError) as exc_info:
            AnalyzeRequest(
                image_base64=invalid_base64,
                include_celebrity_match=True
            )
        
        errors = exc_info.value.errors()
        assert len(errors) > 0
        assert "image_base64" in str(errors[0])
        assert "base64" in str(errors[0]).lower()
    
    def test_malformed_data_uri_raises_error(self):
        """Test that data URI without comma separator raises error."""
        malformed_uri = "data:image/jpeg;base64/9j/4AAQSkZJRg"  # Missing comma
        
        with pytest.raises(ValidationError) as exc_info:
            AnalyzeRequest(
                image_base64=malformed_uri,
                include_celebrity_match=True
            )
        
        errors = exc_info.value.errors()
        assert len(errors) > 0
        assert "data URI" in str(errors[0])
    
    def test_base64_with_whitespace_is_cleaned(self):
        """Test that base64 with embedded whitespace is properly cleaned."""
        # Base64 with spaces and newlines (common in some encoders)
        valid_base64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCwAA8A/9k="
        base64_with_whitespace = valid_base64[:20] + " \n " + valid_base64[20:]
        
        request = AnalyzeRequest(
            image_base64=base64_with_whitespace,
            include_celebrity_match=True
        )
        
        # Should remove whitespace and validate successfully
        assert " " not in request.image_base64
        assert "\n" not in request.image_base64
    
    def test_model_json_schema_has_examples(self):
        """Test that the model's JSON schema includes example values."""
        schema = AnalyzeRequest.model_json_schema()
        
        assert "examples" in schema
        assert len(schema["examples"]) > 0
        
        # Check first example has required structure
        example = schema["examples"][0]
        assert "image_base64" in example
        assert "include_celebrity_match" in example
    
    def test_model_fields_have_descriptions(self):
        """Test that model fields have proper descriptions."""
        schema = AnalyzeRequest.model_json_schema()
        
        assert "properties" in schema
        assert "image_base64" in schema["properties"]
        assert "include_celebrity_match" in schema["properties"]
        
        # Check descriptions exist
        assert "description" in schema["properties"]["image_base64"]
        assert "description" in schema["properties"]["include_celebrity_match"]
        
        # Verify descriptions contain key information
        image_desc = schema["properties"]["image_base64"]["description"]
        assert "base64" in image_desc.lower()
        
        celeb_desc = schema["properties"]["include_celebrity_match"]["description"]
        assert "celebrity" in celeb_desc.lower()
    
    def test_both_fields_can_be_provided(self):
        """Test that both fields can be provided together."""
        valid_base64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCwAA8A/9k="
        
        request = AnalyzeRequest(
            image_base64=valid_base64,
            include_celebrity_match=False
        )
        
        assert request.image_base64 == valid_base64
        assert request.include_celebrity_match is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
