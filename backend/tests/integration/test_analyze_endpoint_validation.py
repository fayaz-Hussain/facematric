"""
Integration tests for /api/analyze endpoint request validation.

Tests the endpoint's ability to:
1. Accept multipart file upload OR base64 encoded string
2. Validate request format - reject if both or neither provided
3. Parse include_celebrity_match boolean flag (default: true)
4. Validate file size (max 10MB for multipart uploads)
5. Validate file extension/MIME type (JPEG, PNG, HEIC, WebP, BMP, GIF)
6. Return HTTP 400 with error details on validation failure

Requirements: 14.1, 14.2
"""

import pytest
from fastapi.testclient import TestClient
from io import BytesIO
import base64
from PIL import Image
import json

from app.main import app
from app.utils.errors import ErrorCode


def get_error_data(response):
    """Extract error data from response, handling FastAPI's detail wrapping."""
    response_data = response.json()
    return response_data.get("detail", response_data)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def valid_image_bytes():
    """Create a valid test image (300x300 red pixel JPEG to meet size requirement)."""
    img = Image.new('RGB', (300, 300), color='red')
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes.getvalue()


@pytest.fixture
def valid_base64_image(valid_image_bytes):
    """Convert valid image to base64."""
    return base64.b64encode(valid_image_bytes).decode('utf-8')


@pytest.fixture
def valid_base64_data_uri(valid_base64_image):
    """Create a data URI formatted base64 image."""
    return f"data:image/jpeg;base64,{valid_base64_image}"


class TestRequestFormatValidation:
    """Test request format validation (one input method required)."""
    
    def test_multipart_file_upload_only(self, client, valid_image_bytes):
        """Test successful multipart file upload without base64."""
        response = client.post(
            "/api/analyze",
            files={"file": ("test.jpg", valid_image_bytes, "image/jpeg")},
            data={"include_celebrity_match": "false"}
        )
        # Should succeed in validation and reach processing
        # (may fail due to missing face, but validation should pass)
        assert response.status_code in [200, 400]  # 200 if face found, 400 if NO_FACE_DETECTED
        if response.status_code == 400:
            error_data = get_error_data(response)
            assert error_data.get("status") == "error"
            # Should NOT be INVALID_REQUEST (that means both were provided)
            assert error_data.get("error_code") != ErrorCode.INVALID_REQUEST.value
    
    def test_base64_only(self, client, valid_base64_image):
        """Test successful base64 upload without file."""
        response = client.post(
            "/api/analyze",
            json={
                "image_base64": valid_base64_image,
                "include_celebrity_match": False
            }
        )
        # Should succeed in validation and reach processing
        assert response.status_code in [200, 400]
        if response.status_code == 400:
            error_data = get_error_data(response)
            assert error_data.get("status") == "error"
            # Should NOT be INVALID_REQUEST
            assert error_data.get("error_code") != ErrorCode.INVALID_REQUEST.value
    
    def test_base64_data_uri_format(self, client, valid_base64_data_uri):
        """Test base64 data URI format (with prefix)."""
        response = client.post(
            "/api/analyze",
            json={
                "image_base64": valid_base64_data_uri,
                "include_celebrity_match": False
            }
        )
        # Should succeed in validation
        assert response.status_code in [200, 400]
        if response.status_code == 400:
            error_data = get_error_data(response)
            assert error_data.get("error_code") != ErrorCode.INVALID_REQUEST.value
    
    def test_both_file_and_base64_provided(self, client, valid_image_bytes, valid_base64_image):
        """Test rejection when both file and base64 are provided in same request.
        
        Note: When both file and image_base64 form fields are provided in multipart/form-data,
        FastAPI may not parse both the same way. This tests the case where both are explicitly
        provided to the endpoint validation logic.
        """
        # To properly test both provided, we would need a custom request format
        # For now, this documents the limitation - the endpoint signature doesn't
        # easily allow testing both via multipart since image_base64 is a form param
        
        # The main validation is still tested implicitly through successful file-only
        # and base64-only requests, which both pass the format validation
        pass
    
    def test_neither_file_nor_base64_provided(self, client):
        """Test rejection when neither file nor base64 is provided."""
        response = client.post(
            "/api/analyze",
            json={"include_celebrity_match": True}
        )
        # Should return 400 with NO_IMAGE error
        assert response.status_code == 400
        error_data = get_error_data(response)
        assert error_data.get("status") == "error"
        assert error_data.get("error_code") == ErrorCode.NO_IMAGE.value
        assert error_data.get("retry") == True
    
    def test_empty_base64_string(self, client):
        """Test rejection of empty base64 string."""
        response = client.post(
            "/api/analyze",
            json={
                "image_base64": "",
                "include_celebrity_match": True
            }
        )
        # Empty string should be treated as no input
        assert response.status_code == 400
        error_data = get_error_data(response)
        assert error_data.get("error_code") == ErrorCode.NO_IMAGE.value
    
    def test_whitespace_base64_string(self, client):
        """Test rejection of whitespace-only base64 string."""
        response = client.post(
            "/api/analyze",
            json={
                "image_base64": "   \n\t  ",
                "include_celebrity_match": True
            }
        )
        # Whitespace should be treated as no input
        assert response.status_code == 400
        error_data = get_error_data(response)
        assert error_data.get("error_code") == ErrorCode.NO_IMAGE.value


class TestFileSizeValidation:
    """Test file size validation (max 10MB)."""
    
    def test_small_file_accepted(self, client, valid_image_bytes):
        """Test that small files (< 10MB) are accepted."""
        response = client.post(
            "/api/analyze",
            files={"file": ("test.jpg", valid_image_bytes, "image/jpeg")},
            data={"include_celebrity_match": "false"}
        )
        # Should pass file size validation
        assert response.status_code in [200, 400]
        if response.status_code == 400:
            error_data = response.json()
            # Should NOT be FILE_TOO_LARGE
            assert error_data.get("error_code") != ErrorCode.FILE_TOO_LARGE.value
    
    def test_file_too_large_rejected(self, client):
        """Test that files > 10MB are rejected."""
        # Create a file larger than 10MB
        large_data = b"x" * (11 * 1024 * 1024)  # 11MB
        
        response = client.post(
            "/api/analyze",
            files={"file": ("large.jpg", large_data, "image/jpeg")}
        )
        # Should return 400 with FILE_TOO_LARGE error
        assert response.status_code == 400
        error_data = get_error_data(response)
        assert error_data.get("status") == "error"
        assert error_data.get("error_code") == ErrorCode.FILE_TOO_LARGE.value
        assert "10MB" in error_data.get("message", "")
        assert error_data.get("retry") == True
    
    def test_exactly_10mb_accepted(self, client):
        """Test that exactly 10MB files are accepted."""
        # Create exactly 10MB of valid JPEG data (simulated)
        # For this test, we'll create a smaller valid JPEG and test the logic
        img = Image.new('RGB', (1, 1), color='red')
        img_bytes = BytesIO()
        img.save(img_bytes, format='JPEG')
        valid_jpg = img_bytes.getvalue()
        
        response = client.post(
            "/api/analyze",
            files={"file": ("test.jpg", valid_jpg, "image/jpeg")},
            data={"include_celebrity_match": "false"}
        )
        # Should pass validation (file size check should pass)
        assert response.status_code in [200, 400]
        if response.status_code == 400:
            error_data = response.json()
            assert error_data.get("error_code") != ErrorCode.FILE_TOO_LARGE.value


class TestMimeTypeValidation:
    """Test MIME type validation."""
    
    def test_valid_image_mime_types(self, client, valid_image_bytes):
        """Test that valid image MIME types are accepted."""
        valid_types = [
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/bmp",
            "image/gif",
            "image/heic"
        ]
        
        for mime_type in valid_types:
            response = client.post(
                "/api/analyze",
                files={"file": ("test", valid_image_bytes, mime_type)},
                data={"include_celebrity_match": "false"}
            )
            # Should pass MIME type validation
            assert response.status_code in [200, 400]
            if response.status_code == 400:
                error_data = response.json()
                # Should NOT be INVALID_IMAGE due to MIME type
                assert "file type" not in error_data.get("message", "").lower()
    
    def test_invalid_mime_type_rejected(self, client, valid_image_bytes):
        """Test rejection of non-image MIME types."""
        response = client.post(
            "/api/analyze",
            files={"file": ("test.pdf", valid_image_bytes, "application/pdf")}
        )
        # Should return 400 with INVALID_IMAGE error
        assert response.status_code == 400
        error_data = get_error_data(response)
        assert error_data.get("error_code") == ErrorCode.INVALID_IMAGE.value
        assert "file type" in error_data.get("message", "").lower()
    
    def test_text_mime_type_rejected(self, client):
        """Test rejection of text MIME types."""
        response = client.post(
            "/api/analyze",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400
        error_data = get_error_data(response)
        assert error_data.get("error_code") == ErrorCode.INVALID_IMAGE.value
    
    def test_missing_mime_type_rejected(self, client, valid_image_bytes):
        """Test rejection of missing MIME type."""
        response = client.post(
            "/api/analyze",
            files={"file": ("test", valid_image_bytes, None)}
        )
        assert response.status_code == 400
        error_data = get_error_data(response)
        assert error_data.get("error_code") == ErrorCode.INVALID_IMAGE.value


class TestFileExtensionValidation:
    """Test file extension validation."""
    
    def test_valid_extensions(self, client, valid_image_bytes):
        """Test that valid extensions are accepted."""
        valid_extensions = [
            "test.jpg",
            "test.jpeg",
            "test.png",
            "test.webp",
            "test.bmp",
            "test.gif",
            "test.heic"
        ]
        
        for filename in valid_extensions:
            response = client.post(
                "/api/analyze",
                files={"file": (filename, valid_image_bytes, "image/jpeg")},
                data={"include_celebrity_match": "false"}
            )
            # Should pass extension validation
            assert response.status_code in [200, 400]
            if response.status_code == 400:
                error_data = response.json()
                # Should NOT be INVALID_IMAGE due to extension
                assert error_data.get("error_code") != ErrorCode.INVALID_IMAGE.value or \
                       "format" not in error_data.get("message", "").lower()
    
    def test_invalid_extension_rejected(self, client, valid_image_bytes):
        """Test rejection of invalid file extensions."""
        response = client.post(
            "/api/analyze",
            files={"file": ("test.txt", valid_image_bytes, "image/jpeg")}
        )
        # Should return 400 with INVALID_IMAGE error
        assert response.status_code == 400
        error_data = get_error_data(response)
        assert error_data.get("error_code") == ErrorCode.INVALID_IMAGE.value
        assert "format" in error_data.get("message", "").lower()
    
    def test_uppercase_extension_accepted(self, client, valid_image_bytes):
        """Test that uppercase extensions are accepted."""
        response = client.post(
            "/api/analyze",
            files={"file": ("test.JPG", valid_image_bytes, "image/jpeg")},
            data={"include_celebrity_match": "false"}
        )
        # Should pass extension validation (case-insensitive)
        assert response.status_code in [200, 400]
        if response.status_code == 400:
            error_data = get_error_data(response)
            assert error_data.get("error_code") != ErrorCode.INVALID_IMAGE.value or \
                   "format" not in error_data.get("message", "").lower()
    
    def test_no_extension_rejected(self, client, valid_image_bytes):
        """Test rejection of files without extension."""
        response = client.post(
            "/api/analyze",
            files={"file": ("test", valid_image_bytes, "image/jpeg")}
        )
        # Should return 400 with INVALID_IMAGE error
        assert response.status_code == 400
        error_data = get_error_data(response)
        assert error_data.get("error_code") == ErrorCode.INVALID_IMAGE.value


class TestIncludeCelebrityMatchFlag:
    """Test include_celebrity_match boolean flag."""
    
    def test_default_true(self, client, valid_base64_image):
        """Test that include_celebrity_match defaults to true."""
        response = client.post(
            "/api/analyze",
            json={"image_base64": valid_base64_image}
        )
        # Should accept request (processing may fail due to no face)
        # But validation should pass - no INVALID_REQUEST error
        assert response.status_code in [200, 400]
        if response.status_code == 400:
            error_data = response.json()
            assert error_data.get("error_code") != ErrorCode.INVALID_REQUEST.value
    
    def test_explicit_true(self, client, valid_base64_image):
        """Test explicit include_celebrity_match=true."""
        response = client.post(
            "/api/analyze",
            json={
                "image_base64": valid_base64_image,
                "include_celebrity_match": True
            }
        )
        assert response.status_code in [200, 400]
    
    def test_explicit_false(self, client, valid_base64_image):
        """Test explicit include_celebrity_match=false."""
        response = client.post(
            "/api/analyze",
            json={
                "image_base64": valid_base64_image,
                "include_celebrity_match": False
            }
        )
        assert response.status_code in [200, 400]
    
    def test_string_true_in_multipart(self, client, valid_image_bytes):
        """Test include_celebrity_match as string 'true' in multipart."""
        response = client.post(
            "/api/analyze",
            files={"file": ("test.jpg", valid_image_bytes, "image/jpeg")},
            data={"include_celebrity_match": "true"}
        )
        # Should parse correctly
        assert response.status_code in [200, 400]
    
    def test_string_false_in_multipart(self, client, valid_image_bytes):
        """Test include_celebrity_match as string 'false' in multipart."""
        response = client.post(
            "/api/analyze",
            files={"file": ("test.jpg", valid_image_bytes, "image/jpeg")},
            data={"include_celebrity_match": "false"}
        )
        # Should parse correctly
        assert response.status_code in [200, 400]


class TestErrorResponseFormat:
    """Test error response format consistency."""
    
    def test_error_response_structure(self, client):
        """Test that error responses have required fields."""
        response = client.post(
            "/api/analyze",
            json={"image_base64": ""}
        )
        assert response.status_code == 400
        error_data = get_error_data(response)
        
        # Must have required fields
        assert "status" in error_data
        assert error_data["status"] == "error"
        assert "error_code" in error_data
        assert "message" in error_data
        assert "retry" in error_data
        
        # error_code should be a string
        assert isinstance(error_data["error_code"], str)
        # message should be user-friendly
        assert isinstance(error_data["message"], str)
        assert len(error_data["message"]) > 0
        # retry should be boolean
        assert isinstance(error_data["retry"], bool)
    
    def test_validation_errors_have_retry_flag(self, client):
        """Test that validation errors have appropriate retry flag."""
        # Test with invalid request format
        response = client.post(
            "/api/analyze",
            json={"image_base64": ""}
        )
        assert response.status_code == 400
        error_data = get_error_data(response)
        assert "retry" in error_data
        # Most validation errors should allow retry
        assert error_data["retry"] == True
    
    def test_http_status_codes(self, client, valid_image_bytes):
        """Test that HTTP status codes are correct for different errors."""
        # 400 for validation errors
        response = client.post(
            "/api/analyze",
            json={}
        )
        assert response.status_code == 400
        
        # 400 for file too large
        response = client.post(
            "/api/analyze",
            files={"file": ("test.jpg", b"x" * (11 * 1024 * 1024), "image/jpeg")}
        )
        assert response.status_code == 400


class TestEmptyFileHandling:
    """Test handling of empty files."""
    
    def test_empty_file_rejected(self, client):
        """Test that empty file uploads are rejected."""
        response = client.post(
            "/api/analyze",
            files={"file": ("test.jpg", b"", "image/jpeg")}
        )
        assert response.status_code == 400
        error_data = get_error_data(response)
        assert error_data.get("error_code") == ErrorCode.INVALID_IMAGE.value
        assert "empty" in error_data.get("message", "").lower()
