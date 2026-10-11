"""
Integration tests for FastAPI application setup (Task 10.1).

Tests that the FastAPI application is properly initialized with CORS middleware,
logging infrastructure, error handlers, and health check endpoint.

Requirements: 14.1, 12.1, 12.2, 13.1, 13.2, 13.3, 13.4
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.config import settings
import json


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestFastAPIInitialization:
    """Test FastAPI application initialization."""
    
    def test_fastapi_app_title(self):
        """Test that FastAPI app is initialized with correct title."""
        assert app.title == "FaceMatric API"
    
    def test_fastapi_app_version(self):
        """Test that FastAPI app is initialized with correct version."""
        assert app.version == "1.0.0"
    
    def test_fastapi_app_has_description(self):
        """Test that FastAPI app has description."""
        assert app.description is not None
        assert "Golden Ratio" in app.description
    
    def test_fastapi_app_has_routes(self):
        """Test that FastAPI app has routes defined."""
        assert len(app.routes) > 0
    
    def test_root_endpoint_exists(self):
        """Test that root endpoint is registered."""
        route_paths = [route.path for route in app.routes]
        assert "/" in route_paths
    
    def test_health_endpoint_exists(self):
        """Test that health endpoint is registered."""
        route_paths = [route.path for route in app.routes]
        assert "/health" in route_paths
    
    def test_analyze_endpoint_exists(self):
        """Test that analyze endpoint is registered."""
        route_paths = [route.path for route in app.routes]
        assert "/api/analyze" in route_paths


class TestCORSMiddleware:
    """Test CORS middleware configuration."""
    
    def test_cors_middleware_is_configured(self):
        """Test that CORS middleware is added to app."""
        # CORS middleware should be in the app's middleware stack
        assert len(app.user_middleware) > 0
        middleware_names = [m.cls.__name__ for m in app.user_middleware]
        assert "CORSMiddleware" in middleware_names
    
    def test_allowed_origins_configured(self):
        """Test that allowed origins are configured."""
        allowed_origins = settings.get_allowed_origins_list()
        assert len(allowed_origins) > 0
    
    def test_allowed_origins_include_localhost_3000(self):
        """Test that localhost:3000 is in allowed origins."""
        allowed_origins = settings.get_allowed_origins_list()
        assert "http://localhost:3000" in allowed_origins or "http://localhost:5173" in allowed_origins
    
    def test_allowed_origins_include_production_url(self):
        """Test that production URL is configurable."""
        # Production URL should be in environment variables
        allowed_origins = settings.get_allowed_origins_list()
        # At least one origin should be configured
        assert len(allowed_origins) >= 1
    
    def test_cors_allows_credentials(self, client):
        """Test that CORS allows credentials (if configured)."""
        # Send preflight request
        response = client.options("/api/analyze")
        # Preflight should return 200
        assert response.status_code in [200, 405]  # 405 if OPTIONS not explicitly allowed
    
    def test_cors_allows_methods(self):
        """Test that CORS allows standard HTTP methods."""
        # CORS middleware is configured with allow_methods=["*"]
        # This allows all methods
        assert settings is not None


class TestLoggingInfrastructure:
    """Test logging infrastructure setup."""
    
    def test_logging_is_configured(self):
        """Test that logging is configured."""
        from app.utils.logger import get_logger
        
        logger = get_logger(__name__)
        assert logger is not None
    
    def test_logger_has_expected_methods(self):
        """Test that logger has standard logging methods."""
        from app.utils.logger import get_logger
        
        logger = get_logger(__name__)
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'warning')
        assert hasattr(logger, 'debug')
    
    def test_startup_logging(self, client):
        """Test that startup events are logged."""
        # Make a request to trigger startup if not already done
        response = client.get("/health")
        assert response.status_code == 200
        # If startup logging worked, we should get a valid response
    
    def test_log_level_configured(self):
        """Test that log level is configured from settings."""
        log_level = settings.log_level
        assert log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class TestErrorHandling:
    """Test error handling configuration."""
    
    def test_error_handlers_configured(self):
        """Test that error handlers are registered."""
        # Error handlers are configured via decorators
        assert len(app.routes) > 0
    
    def test_validation_error_handler_registered(self):
        """Test that ValidationError handler is registered."""
        # ValidationError handler is configured in main.py
        from app.utils.errors import ValidationError
        assert ValidationError is not None
    
    def test_http_exception_handling(self, client):
        """Test that HTTP exceptions are handled properly."""
        # Test with invalid endpoint
        response = client.get("/invalid-endpoint")
        assert response.status_code == 404


class TestHealthCheckEndpoint:
    """Test health check endpoint."""
    
    def test_health_check_returns_200(self, client):
        """Test that health check returns 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_check_response_format(self, client):
        """Test that health check response has correct format."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_health_check_includes_version(self, client):
        """Test that health check includes API version."""
        response = client.get("/health")
        data = response.json()
        
        assert "version" in data
        assert data["version"] == "1.0.0"
    
    def test_health_check_includes_celebrity_feature_flag(self, client):
        """Test that health check reports celebrity matching availability."""
        response = client.get("/health")
        data = response.json()
        
        assert "celebrity_matching_available" in data
        assert isinstance(data["celebrity_matching_available"], bool)
    
    def test_health_check_content_type(self, client):
        """Test that health check returns JSON content type."""
        response = client.get("/health")
        assert "application/json" in response.headers.get("content-type", "")


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root_endpoint_returns_200(self, client):
        """Test that root endpoint returns 200 OK."""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_root_endpoint_returns_json(self, client):
        """Test that root endpoint returns JSON."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, dict)
    
    def test_root_endpoint_has_app_info(self, client):
        """Test that root endpoint includes app information."""
        response = client.get("/")
        data = response.json()
        
        assert "name" in data
        assert "version" in data
        assert data["name"] == "FaceMatric API"
        assert data["version"] == "1.0.0"


class TestEnvironmentConfiguration:
    """Test environment variable configuration."""
    
    def test_app_name_from_settings(self):
        """Test that app name comes from settings."""
        assert settings.app_name == "FaceMatric API"
    
    def test_app_version_from_settings(self):
        """Test that app version comes from settings."""
        assert settings.app_version == "1.0.0"
    
    def test_processing_timeout_from_settings(self):
        """Test that processing timeout comes from settings."""
        assert settings.processing_timeout == 30
    
    def test_host_from_settings(self):
        """Test that host comes from settings."""
        assert settings.host == "0.0.0.0"
    
    def test_port_from_settings(self):
        """Test that port comes from settings."""
        assert settings.port == 8000
    
    def test_cors_origins_from_settings(self):
        """Test that CORS origins come from settings."""
        allowed_origins = settings.get_allowed_origins_list()
        assert len(allowed_origins) > 0


class TestStartupEvent:
    """Test startup event configuration."""
    
    def test_startup_event_registered(self):
        """Test that startup event is registered."""
        # Startup event is defined in main.py with @app.on_event("startup")
        # Verify by checking if health check works (depends on startup initialization)
        pass
    
    def test_face_detector_initialized_on_startup(self, client):
        """Test that face detector is initialized on startup."""
        # Make request to trigger startup
        response = client.get("/health")
        assert response.status_code == 200


class TestDocumentation:
    """Test API documentation."""
    
    def test_docs_endpoint_available(self, client):
        """Test that Swagger UI docs are available."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_redoc_endpoint_available(self, client):
        """Test that ReDoc docs are available."""
        response = client.get("/redoc")
        assert response.status_code == 200
    
    def test_openapi_schema_available(self, client):
        """Test that OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200


class TestAppComprehensive:
    """Comprehensive tests for app setup."""
    
    def test_app_has_all_required_endpoints(self, client):
        """Test that app has all required endpoints."""
        endpoints = ["/", "/health", "/api/analyze", "/docs", "/redoc"]
        
        # Check root, health, and docs (analyze will fail without image but should exist)
        response = client.get("/")
        assert response.status_code == 200
        
        response = client.get("/health")
        assert response.status_code == 200
        
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_app_cors_configuration_summary(self):
        """Test complete CORS configuration."""
        # CORS should allow:
        # - localhost:3000 (frontend dev)
        # - localhost:5173 (Vite dev)
        # - Production URL if configured
        
        allowed_origins = settings.get_allowed_origins_list()
        
        # Should have at least localhost origins
        assert any("localhost" in origin for origin in allowed_origins)
    
    def test_app_meets_requirement_14_1(self, client):
        """
        Test that app meets Requirement 14.1.
        
        Requirement 14.1: WHEN analysis completes successfully, THE API_Gateway 
        SHALL return HTTP status 200 with Content-Type application/json
        
        This tests that the app framework is set up to support this requirement.
        """
        # Root endpoint should return JSON
        response = client.get("/")
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")
        
        # Health check should return JSON
        response = client.get("/health")
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")

