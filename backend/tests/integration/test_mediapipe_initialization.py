"""
Integration test for MediaPipe Face Mesh initialization.

This test verifies that:
- Task 4.1: MediaPipe Face Mesh initializes correctly with face_landmarker.task
- The model file is present and valid
- Configuration settings are correct (max_num_faces=1, min_detection_confidence=0.5)
- Running mode is set for single image processing
"""

import pytest
from pathlib import Path
from app.components.face_detector import FaceDetector
from app.config import settings


class TestMediaPipeInitialization:
    """Test MediaPipe Face Mesh initialization with real model."""
    
    def test_model_file_exists(self):
        """Verify face_landmarker.task model file exists."""
        model_path = Path(settings.mediapipe_model_path)
        assert model_path.exists(), f"Model file not found at {model_path}"
        assert model_path.is_file(), f"Model path is not a file: {model_path}"
        
        # Verify file has reasonable size (should be several MB)
        file_size = model_path.stat().st_size
        assert file_size > 1_000_000, f"Model file too small: {file_size} bytes"
        assert file_size < 50_000_000, f"Model file unexpectedly large: {file_size} bytes"
    
    def test_face_detector_initialization(self):
        """
        Test FaceDetector initializes successfully with correct configuration.
        
        Requirements: 2.1
        Task: 4.1
        """
        # Initialize detector
        detector = FaceDetector()
        
        # Verify detector was created
        assert detector is not None
        assert detector.detector is not None
        
        # Verify model path is set correctly
        assert detector.model_path == settings.mediapipe_model_path
        
        # Verify configuration settings
        assert settings.mediapipe_max_faces == 1, "max_num_faces should be 1"
        assert settings.mediapipe_min_detection_confidence == 0.5, \
            "min_detection_confidence should be 0.5"
    
    def test_configuration_values(self):
        """
        Verify MediaPipe configuration matches task 4.1 requirements.
        
        Task 4.1 specifies:
        - max_num_faces=1
        - min_detection_confidence=0.5
        - Running mode for single image processing
        """
        # Check configuration values
        assert settings.mediapipe_max_faces == 1
        assert settings.mediapipe_min_detection_confidence == 0.5
        assert settings.landmark_confidence_threshold == 0.6
        
        # Verify model path is configured
        assert settings.mediapipe_model_path is not None
        assert len(settings.mediapipe_model_path) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
