"""
Unit tests for backend infrastructure setup.

Tests configuration, logging, and error handling infrastructure.
"""

import pytest
import logging
import json
from app.config import settings
from app.utils.errors import ValidationError, ErrorCode, get_error_message
from app.utils.logger import setup_logging, get_logger, JSONFormatter


class TestConfiguration:
    """Test application configuration."""
    
    def test_settings_loaded(self):
        """Test that settings are properly loaded."""
        assert settings.app_name == "FaceMatric API"
        assert settings.app_version == "1.0.0"
        assert settings.processing_timeout == 30
        assert settings.phi == 1.618
    
    def test_allowed_origins_parsing(self):
        """Test that CORS origins are parsed correctly."""
        origins = settings.get_allowed_origins_list()
        assert isinstance(origins, list)
        assert len(origins) > 0
        assert all(isinstance(origin, str) for origin in origins)
    
    def test_paths_configured(self):
        """Test that paths are configured."""
        assert settings.celebrity_db_path is not None
        assert settings.mediapipe_model_path is not None
    
    def test_image_constraints(self):
        """Test image processing constraints."""
        assert settings.max_image_size_mb == 10
        assert settings.min_image_dimension == 300
        assert settings.max_image_dimension == 640
        assert settings.jpeg_quality == 85
    
    def test_mediapipe_configuration(self):
        """Test MediaPipe configuration."""
        assert settings.mediapipe_max_faces == 1
        assert settings.mediapipe_min_detection_confidence == 0.5
        assert settings.landmark_confidence_threshold == 0.6


class TestErrorHandling:
    """Test error handling infrastructure."""
    
    def test_error_codes_defined(self):
        """Test that all error codes are defined."""
        assert ErrorCode.INVALID_IMAGE is not None
        assert ErrorCode.NO_FACE_DETECTED is not None
        assert ErrorCode.MULTIPLE_FACES is not None
        assert ErrorCode.POOR_QUALITY is not None
        assert ErrorCode.OCCLUSION_DETECTED is not None
        assert ErrorCode.PROCESSING_TIMEOUT is not None
        assert ErrorCode.INTERNAL_ERROR is not None
    
    def test_validation_error_creation(self):
        """Test ValidationError can be created."""
        error = ValidationError(
            ErrorCode.NO_FACE_DETECTED,
            "Test message",
            retry=True
        )
        assert error.error_code == ErrorCode.NO_FACE_DETECTED
        assert error.message == "Test message"
        assert error.retry is True
    
    def test_validation_error_to_dict(self):
        """Test ValidationError conversion to dictionary."""
        error = ValidationError(
            ErrorCode.INVALID_IMAGE,
            "Invalid image format",
            retry=True
        )
        error_dict = error.to_dict()
        
        assert error_dict["status"] == "error"
        assert error_dict["error_code"] == "INVALID_IMAGE"
        assert error_dict["message"] == "Invalid image format"
        assert error_dict["retry"] is True
    
    def test_error_messages_exist(self):
        """Test that all error codes have messages."""
        for error_code in ErrorCode:
            message = get_error_message(error_code)
            assert message is not None
            assert len(message) > 0
    
    def test_custom_error_message(self):
        """Test custom error message override."""
        custom = "Custom error message"
        message = get_error_message(ErrorCode.INTERNAL_ERROR, custom)
        assert message == custom


class TestLogging:
    """Test logging infrastructure."""
    
    def test_setup_logging(self):
        """Test that logging can be initialized."""
        setup_logging()
        logger = get_logger(__name__)
        assert logger is not None
        assert isinstance(logger, logging.Logger)
    
    def test_get_logger(self):
        """Test logger creation."""
        logger = get_logger("test_module")
        assert logger is not None
        assert logger.name == "test_module"
    
    def test_json_formatter(self):
        """Test JSON log formatter."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        
        # Should be valid JSON
        log_data = json.loads(formatted)
        assert log_data["message"] == "Test message"
        assert log_data["level"] == "INFO"
        assert "timestamp" in log_data
    
    def test_json_formatter_with_extra_fields(self):
        """Test JSON formatter with extra fields."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Add extra fields
        record.request_id = "test-123"
        record.component = "test_component"
        record.processing_time_ms = 100
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["request_id"] == "test-123"
        assert log_data["component"] == "test_component"
        assert log_data["processing_time_ms"] == 100


class TestProjectStructure:
    """Test that project structure is correct."""
    
    def test_app_directory_exists(self):
        """Test that app directory exists."""
        import os
        assert os.path.exists("app")
        assert os.path.isdir("app")
    
    def test_components_directory_exists(self):
        """Test that components directory exists."""
        import os
        assert os.path.exists("app/components")
        assert os.path.isdir("app/components")
    
    def test_models_directory_exists(self):
        """Test that models directory exists."""
        import os
        assert os.path.exists("app/models")
        assert os.path.isdir("app/models")
    
    def test_utils_directory_exists(self):
        """Test that utils directory exists."""
        import os
        assert os.path.exists("app/utils")
        assert os.path.isdir("app/utils")
    
    def test_data_directory_exists(self):
        """Test that data directory exists."""
        import os
        assert os.path.exists("data")
        assert os.path.isdir("data")
    
    def test_tests_directory_exists(self):
        """Test that tests directory exists."""
        import os
        assert os.path.exists("tests")
        assert os.path.isdir("tests")
        assert os.path.exists("tests/unit")
        assert os.path.exists("tests/integration")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
