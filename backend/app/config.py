"""
Application configuration using environment variables.
"""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Configuration
    app_name: str = "FaceMatric API"
    app_version: str = "1.0.0"
    allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000"
    processing_timeout: int = 30
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Paths
    celebrity_db_path: str = "./data/celebrity_embeddings.pkl"
    mediapipe_model_path: str = "./data/face_landmarker.task"
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Feature Flags
    enable_celebrity_matching: bool = True
    
    # Image Processing Constraints
    max_image_size_mb: int = 10
    min_image_dimension: int = 300
    max_image_dimension: int = 640
    jpeg_quality: int = 85
    
    # Golden Ratio
    phi: float = 1.618
    
    # Celebrity Matching Thresholds
    celebrity_threshold_strong: float = 0.70
    celebrity_threshold_moderate: float = 0.55
    celebrity_confidence_threshold: float = 0.70  # Deprecated alias pointing to strong threshold
    
    # MediaPipe Configuration
    mediapipe_max_faces: int = 1
    mediapipe_min_detection_confidence: float = 0.5
    landmark_confidence_threshold: float = 0.6
    
    # Pose Quality Gates
    max_pose_yaw_deg: float = 25.0
    max_pose_roll_deg: float = 20.0
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def get_allowed_origins_list(self) -> List[str]:
        """Parse allowed origins from comma-separated string."""
        return [origin.strip() for origin in self.allowed_origins.split(",")]


# Global settings instance
settings = Settings()
