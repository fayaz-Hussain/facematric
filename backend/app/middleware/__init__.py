"""
Middleware components for FaceMatric API.
"""

from app.middleware.timeout_middleware import TimeoutMiddleware

__all__ = ["TimeoutMiddleware"]
