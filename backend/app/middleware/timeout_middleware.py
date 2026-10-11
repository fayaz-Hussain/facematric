"""
Timeout middleware for processing request timeout handling.

Implements 30-second timeout for processing endpoints with proper
resource cleanup, monitoring, and error handling.

Features:
- Enforces configurable timeout (default: 30 seconds) via environment variable
- Applies timeout to /api/analyze endpoint only
- Skips health checks and quick endpoints
- Returns HTTP 408 Request Timeout on timeout
- Logs timeout events for monitoring
- Configurable via PROCESSING_TIMEOUT environment variable
"""

import asyncio
import time
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from app.utils.logger import get_logger
from app.config import settings
from app.utils.errors import ErrorCode, get_error_message

logger = get_logger(__name__)


class TimeoutMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce timeout on processing endpoints.
    
    Configuration:
    - PROCESSING_TIMEOUT env variable (default: 30 seconds)
    - Applies to /api/analyze endpoint
    - Skips health checks and quick endpoints
    
    Response format on timeout:
    {
        "status": "error",
        "error_code": "PROCESSING_TIMEOUT",
        "message": "Processing took too long. Please try again with a clearer photo.",
        "retry": true
    }
    """
    
    # Endpoints that should have timeout enforcement
    TIMEOUT_ENDPOINTS = ["/api/analyze"]
    
    # Endpoints that should be excluded from timeout (quick endpoints)
    EXCLUDED_ENDPOINTS = ["/health", "/", "/docs", "/redoc", "/openapi.json"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> JSONResponse:
        """
        Process request with timeout enforcement.
        
        - Skips excluded endpoints (health checks, docs)
        - Enforces timeout on /api/analyze endpoint
        - Logs timeout events for monitoring
        - Cleans up resources on timeout
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain
            
        Returns:
            Response from handler or HTTP 408 on timeout
        """
        path = request.url.path
        
        # Skip excluded endpoints (quick endpoints, health checks)
        if any(path.startswith(ep) for ep in self.EXCLUDED_ENDPOINTS):
            return await call_next(request)
        
        # Check if this is a timeout endpoint
        should_timeout = any(
            path.startswith(ep) for ep in self.TIMEOUT_ENDPOINTS
        )
        
        if not should_timeout:
            return await call_next(request)
        
        # Generate request ID if not already present
        request_id = getattr(request.state, "request_id", None)
        if not request_id:
            import uuid
            request_id = str(uuid.uuid4())
            request.state.request_id = request_id
        
        timeout_seconds = settings.processing_timeout
        logger.debug(
            f"[{request_id}] Timeout middleware: enforcing {timeout_seconds}s timeout for {path}"
        )
        
        start_time = time.time()
        
        try:
            # Wrap handler with timeout
            response = await asyncio.wait_for(
                call_next(request),
                timeout=float(timeout_seconds)
            )
            
            return response
            
        except asyncio.TimeoutError:
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            logger.error(
                f"[{request_id}] Request timeout exceeded: "
                f"exceeded {timeout_seconds}s limit after {elapsed_ms}ms. "
                f"Path: {path}"
            )
            
            # Return HTTP 408 with timeout error response
            # Proper resource cleanup happens via context manager unwinding
            return JSONResponse(
                status_code=408,
                content={
                    "status": "error",
                    "error_code": ErrorCode.PROCESSING_TIMEOUT.value,
                    "message": get_error_message(ErrorCode.PROCESSING_TIMEOUT),
                    "retry": True
                }
            )
        
        except Exception as e:
            # Let other exceptions propagate
            logger.error(f"[{request_id}] Timeout middleware error: {str(e)}", exc_info=True)
            raise
