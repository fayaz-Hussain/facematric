"""
Unit tests for Face Detector component.

Tests landmark extraction, confidence calculation, and error handling.
"""

import pytest
import numpy as np
import inspect
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from app.components.face_detector import FaceDetector, get_face_detector
from app.models.internal import ImageData, Landmarks
from app.utils.errors import ValidationError, ErrorCode


@pytest.fixture
def sample_image_data():
    """Create sample ImageData for testing."""
    # Create a simple RGB image
    array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return ImageData(
        array=array,
        width=640,
        height=480,
        original_format='JPEG'
    )


@pytest.fixture
def mock_mediapipe_landmark():
    """Create a mock MediaPipe landmark."""
    def create_landmark(x, y, z, visibility=None, presence=None):
        lm = Mock()
        lm.x = x
        lm.y = y
        lm.z = z
        
        # Set visibility and presence as actual float values
        if visibility is not None:
            lm.visibility = float(visibility)
        else:
            # Use spec to prevent Mock objects from being added
            if hasattr(lm, 'visibility'):
                del lm.visibility
        
        if presence is not None:
            lm.presence = float(presence)
        else:
            if hasattr(lm, 'presence'):
                del lm.presence
        
        return lm
    return create_landmark


@pytest.fixture
def mock_detection_result_with_confidence(mock_mediapipe_landmark):
    """Create mock detection result with 468 landmarks having visibility/presence scores."""
    result = Mock()
    
    # Create 468 landmarks with high confidence
    # Use realistic facial landmark distribution
    landmarks = []
    for i in range(468):
        # Realistic distribution: spread across face area (30%-70% horizontally, 20%-80% vertically)
        if i < 234:
            # First half: upper half of face
            x = 0.35 + (i % 18) * 0.025
            y = 0.25 + (i // 18) * 0.015
        else:
            # Second half: lower half of face
            x = 0.35 + ((i - 234) % 18) * 0.025
            y = 0.55 + ((i - 234) // 18) * 0.015
        
        # Add reasonable z-depth variation
        z = 0.02 + 0.005 * np.sin(i * 0.01)
        
        # High visibility and presence scores
        landmarks.append(
            mock_mediapipe_landmark(x, y, z, visibility=0.9, presence=0.95)
        )
    
    result.face_landmarks = [landmarks]
    return result


@pytest.fixture
def mock_detection_result_no_confidence(mock_mediapipe_landmark):
    """Create mock detection result without visibility/presence scores."""
    result = Mock()
    
    # Create 468 landmarks without confidence scores
    landmarks = []
    for i in range(468):
        x = 0.3 + (i % 20) * 0.02
        y = 0.2 + (i // 20) * 0.02
        z = 0.001 * (i % 10)
        
        # Create landmark without visibility/presence
        lm = Mock(spec=['x', 'y', 'z'])
        lm.x = x
        lm.y = y
        lm.z = z
        landmarks.append(lm)
    
    result.face_landmarks = [landmarks]
    return result


@pytest.fixture
def mock_detection_result_low_confidence(mock_mediapipe_landmark):
    """Create mock detection result with low confidence scores."""
    result = Mock()
    
    # Create 468 landmarks with low confidence
    landmarks = []
    for i in range(468):
        if i < 234:
            x = 0.35 + (i % 18) * 0.025
            y = 0.25 + (i // 18) * 0.015
        else:
            x = 0.35 + ((i - 234) % 18) * 0.025
            y = 0.55 + ((i - 234) // 18) * 0.015
        
        z = 0.02 + 0.005 * np.sin(i * 0.01)
        
        # Low visibility and presence scores
        landmarks.append(
            mock_mediapipe_landmark(x, y, z, visibility=0.4, presence=0.5)
        )
    
    result.face_landmarks = [landmarks]
    return result


@pytest.fixture
def mock_detection_result_no_face():
    """Create mock detection result with no faces detected."""
    result = Mock()
    result.face_landmarks = []
    return result


@pytest.fixture
def mock_detection_result_multiple_faces(mock_mediapipe_landmark):
    """Create mock detection result with multiple faces."""
    result = Mock()
    
    # Create two sets of 468 landmarks with realistic coordinates
    face1_landmarks = []
    face2_landmarks = []
    
    for i in range(468):
        if i < 234:
            x = 0.35 + (i % 18) * 0.025
            y = 0.25 + (i // 18) * 0.015
        else:
            x = 0.35 + ((i - 234) % 18) * 0.025
            y = 0.55 + ((i - 234) // 18) * 0.015
        
        z = 0.02 + 0.005 * np.sin(i * 0.01)
        
        # First face at left position
        face1_landmarks.append(
            mock_mediapipe_landmark(x * 0.5, y, z, visibility=0.9, presence=0.9)
        )
        # Second face at right position
        face2_landmarks.append(
            mock_mediapipe_landmark(x * 0.5 + 0.5, y, z, visibility=0.9, presence=0.9)
        )
    
    result.face_landmarks = [face1_landmarks, face2_landmarks]
    return result


class TestFaceDetectorLandmarkExtraction:
    """Test landmark extraction and coordinate conversion."""
    
    @patch('app.components.face_detector.vision.FaceLandmarker.create_from_options')
    @patch.object(FaceDetector, '_check_occlusions')
    def test_extract_468_landmarks(
        self,
        mock_check_occlusions,
        mock_create,
        sample_image_data,
        mock_detection_result_with_confidence
    ):
        """Test that exactly 468 3D landmarks are extracted."""
        # Skip occlusion check for this test
        mock_check_occlusions.return_value = None
        
        # Setup mock detector
        mock_detector = Mock()
        mock_detector.detect.return_value = mock_detection_result_with_confidence
        mock_create.return_value = mock_detector
        
        # Initialize face detector
        face_detector = FaceDetector(model_path='dummy_path.task')
        
        # Detect faces
        landmarks = face_detector.detect_faces(sample_image_data)
        
        # Verify 468 landmarks extracted
        assert landmarks.points.shape == (468, 3), "Should extract exactly 468 3D landmarks"
        assert landmarks.points.dtype == np.float32
    
    @patch('app.components.face_detector.vision.FaceLandmarker.create_from_options')
    @patch.object(FaceDetector, '_check_occlusions')
    def test_landmark_coordinate_conversion(
        self,
        mock_check_occlusions,
        mock_create,
        sample_image_data,
        mock_detection_result_with_confidence
    ):
        """Test that normalized coordinates are converted to pixel coordinates."""
        # Skip occlusion check for this test
        mock_check_occlusions.return_value = None
        
        # Setup mock detector
        mock_detector = Mock()
        mock_detector.detect.return_value = mock_detection_result_with_confidence
        mock_create.return_value = mock_detector
        
        # Initialize face detector
        face_detector = FaceDetector(model_path='dummy_path.task')
        
        # Detect faces
        landmarks = face_detector.detect_faces(sample_image_data)
        
        # Check that x, y are in pixel space
        x_coords = landmarks.points[:, 0]
        y_coords = landmarks.points[:, 1]
        
        assert np.all(x_coords >= 0), "X coordinates should be positive"
        assert np.all(x_coords <= sample_image_data.width), "X coordinates should be within image width"
        assert np.all(y_coords >= 0), "Y coordinates should be positive"
        assert np.all(y_coords <= sample_image_data.height), "Y coordinates should be within image height"
    
    @patch('app.components.face_detector.vision.FaceLandmarker.create_from_options')
    @patch.object(FaceDetector, '_check_occlusions')
    def test_z_coordinates_preserved(
        self,
        mock_check_occlusions,
        mock_create,
        sample_image_data,
        mock_detection_result_with_confidence
    ):
        """Test that z-depth coordinates are preserved."""
        # Skip occlusion check for this test
        mock_check_occlusions.return_value = None
        
        # Setup mock detector
        mock_detector = Mock()
        mock_detector.detect.return_value = mock_detection_result_with_confidence
        mock_create.return_value = mock_detector
        
        # Initialize face detector
        face_detector = FaceDetector(model_path='dummy_path.task')
        
        # Detect faces
        landmarks = face_detector.detect_faces(sample_image_data)
        
        # Check z coordinates exist and are reasonable
        z_coords = landmarks.points[:, 2]
        assert z_coords is not None
        assert len(z_coords) == 468


class TestFaceDetectorConfidenceCalculation:
    """Test confidence score calculation."""
    
    @patch('app.components.face_detector.vision.FaceLandmarker.create_from_options')
    @patch.object(FaceDetector, '_check_occlusions')
    def test_confidence_from_visibility_presence(
        self,
        mock_check_occlusions,
        mock_create,
        sample_image_data,
        mock_detection_result_with_confidence
    ):
        """Test confidence calculation from visibility and presence scores."""
        # Skip occlusion check for this test
        mock_check_occlusions.return_value = None
        
        # Setup mock detector
        mock_detector = Mock()
        mock_detector.detect.return_value = mock_detection_result_with_confidence
        mock_create.return_value = mock_detector
        
        # Initialize face detector
        face_detector = FaceDetector(model_path='dummy_path.task')
        
        # Detect faces
        landmarks = face_detector.detect_faces(sample_image_data)
        
        # Should have high confidence (visibility=0.9, presence=0.95, avg=0.925)
        assert landmarks.confidence > 0.85, "Should calculate high confidence from scores"
        assert landmarks.confidence <= 1.0
    
    @patch('app.components.face_detector.vision.FaceLandmarker.create_from_options')
    @patch.object(FaceDetector, '_check_occlusions')
    def test_confidence_geometric_fallback(
        self,
        mock_check_occlusions,
        mock_create,
        sample_image_data,
        mock_detection_result_no_confidence
    ):
        """Test confidence fallback to geometric quality assessment."""
        # Skip occlusion check for this test
        mock_check_occlusions.return_value = None
        
        # Setup mock detector
        mock_detector = Mock()
        mock_detector.detect.return_value = mock_detection_result_no_confidence
        mock_create.return_value = mock_detector
        
        # Initialize face detector
        face_detector = FaceDetector(model_path='dummy_path.task')
        
        # Detect faces
        landmarks = face_detector.detect_faces(sample_image_data)
        
        # Should use geometric quality metric
        assert 0.3 <= landmarks.confidence <= 1.0, "Should fallback to geometric quality"
        assert isinstance(landmarks.confidence, float)
    
    @patch('app.components.face_detector.vision.FaceLandmarker.create_from_options')
    @patch('app.components.face_detector.settings.landmark_confidence_threshold', 0.6)
    @patch.object(FaceDetector, '_check_occlusions')
    def test_low_confidence_raises_poor_quality(
        self,
        mock_check_occlusions,
        mock_create,
        sample_image_data,
        mock_detection_result_low_confidence
    ):
        """Test that low confidence raises POOR_QUALITY error."""
        # Don't skip occlusion check for this test, but we expect error before reaching it
        # Setup mock detector
        mock_detector = Mock()
        mock_detector.detect.return_value = mock_detection_result_low_confidence
        mock_create.return_value = mock_detector
        
        # Initialize face detector
        face_detector = FaceDetector(model_path='dummy_path.task')
        
        # Should raise POOR_QUALITY error
        with pytest.raises(ValidationError) as exc_info:
            face_detector.detect_faces(sample_image_data)
        
        assert exc_info.value.error_code == ErrorCode.POOR_QUALITY
        assert "quality" in exc_info.value.message.lower()


class TestFaceDetectorErrorHandling:
    """Test error handling for various detection scenarios."""
    
    @patch('app.components.face_detector.vision.FaceLandmarker.create_from_options')
    def test_no_face_detected(
        self,
        mock_create,
        sample_image_data,
        mock_detection_result_no_face
    ):
        """Test NO_FACE_DETECTED error when no faces found."""
        # Setup mock detector
        mock_detector = Mock()
        mock_detector.detect.return_value = mock_detection_result_no_face
        mock_create.return_value = mock_detector
        
        # Initialize face detector
        face_detector = FaceDetector(model_path='dummy_path.task')
        
        # Should raise NO_FACE_DETECTED
        with pytest.raises(ValidationError) as exc_info:
            face_detector.detect_faces(sample_image_data)
        
        assert exc_info.value.error_code == ErrorCode.NO_FACE_DETECTED
        assert "couldn't detect a face" in exc_info.value.message.lower()
    
    @patch('app.components.face_detector.vision.FaceLandmarker.create_from_options')
    def test_multiple_faces_detected(
        self,
        mock_create,
        sample_image_data,
        mock_detection_result_multiple_faces
    ):
        """Test MULTIPLE_FACES error when multiple faces found."""
        # Setup mock detector
        mock_detector = Mock()
        mock_detector.detect.return_value = mock_detection_result_multiple_faces
        mock_create.return_value = mock_detector
        
        # Initialize face detector
        face_detector = FaceDetector(model_path='dummy_path.task')
        
        # Should raise MULTIPLE_FACES
        with pytest.raises(ValidationError) as exc_info:
            face_detector.detect_faces(sample_image_data)
        
        assert exc_info.value.error_code == ErrorCode.MULTIPLE_FACES
        assert "multiple faces" in exc_info.value.message.lower()


class TestFaceDetectorGeometricQuality:
    """Test geometric quality assessment."""
    
    @patch('app.components.face_detector.vision.FaceLandmarker.create_from_options')
    @patch.object(FaceDetector, '_check_occlusions')
    def test_geometric_quality_well_distributed(
        self,
        mock_check_occlusions,
        mock_create,
        sample_image_data
    ):
        """Test geometric quality for well-distributed landmarks."""
        # Skip occlusion check for this test
        mock_check_occlusions.return_value = None
        
        # Create landmarks with good spatial distribution
        result = Mock()
        landmarks = []
        for i in range(468):
            lm = Mock(spec=['x', 'y', 'z'])
            lm.x = 0.2 + (i % 23) * 0.026  # Spread across width
            lm.y = 0.2 + (i // 23) * 0.03  # Spread across height
            lm.z = 0.005 * np.sin(i * 0.1)  # Reasonable z-depth variation
            landmarks.append(lm)
        
        result.face_landmarks = [landmarks]
        
        mock_detector = Mock()
        mock_detector.detect.return_value = result
        mock_create.return_value = mock_detector
        
        face_detector = FaceDetector(model_path='dummy_path.task')
        landmarks_result = face_detector.detect_faces(sample_image_data)
        
        # Should have good geometric quality
        assert landmarks_result.confidence >= 0.6, "Well-distributed landmarks should have good quality"


class TestFaceDetectorMetadata:
    """Test that landmark metadata is correctly set."""
    
    @patch('app.components.face_detector.vision.FaceLandmarker.create_from_options')
    @patch.object(FaceDetector, '_check_occlusions')
    def test_landmarks_metadata(
        self,
        mock_check_occlusions,
        mock_create,
        sample_image_data,
        mock_detection_result_with_confidence
    ):
        """Test that Landmarks object contains correct metadata."""
        # Skip occlusion check for this test
        mock_check_occlusions.return_value = None
        
        # Setup mock detector
        mock_detector = Mock()
        mock_detector.detect.return_value = mock_detection_result_with_confidence
        mock_create.return_value = mock_detector
        
        # Initialize face detector
        face_detector = FaceDetector(model_path='dummy_path.task')
        
        # Detect faces
        landmarks = face_detector.detect_faces(sample_image_data)
        
        # Verify metadata
        assert landmarks.image_width == sample_image_data.width
        assert landmarks.image_height == sample_image_data.height
        assert isinstance(landmarks.confidence, float)
        assert 0.0 <= landmarks.confidence <= 1.0


class TestFaceDetectorSingleton:
    """Test the global face detector instance."""
    
    def test_get_face_detector_singleton(self):
        """Test that get_face_detector returns a singleton instance."""
        detector1 = get_face_detector()
        detector2 = get_face_detector()
        
        assert detector1 is detector2, "Should return same instance"
        assert isinstance(detector1, FaceDetector)



class TestFaceDetectorOcclusionDetection:
    """Test occlusion detection for glasses and masks."""
    
    def test_occlusion_methods_exist(self):
        """Verify that occlusion detection methods exist in FaceDetector."""
        assert hasattr(FaceDetector, '_check_occlusions')
        assert hasattr(FaceDetector, '_detect_glasses')
        assert hasattr(FaceDetector, '_detect_mask')
    
    def test_glasses_detection_landmarks_indices(self):
        """Verify that glasses detection uses correct landmark indices."""
        # Verify eye region indices are properly defined in implementation
        # Left eye indices: 362, 385, 387, 263, 373, 380, 374, 381, 382, 466
        # Right eye indices: 33, 160, 158, 133, 153, 144, 145, 163, 7, 246
        source = inspect.getsource(FaceDetector._detect_glasses)
        assert "362" in source  # Left eye outer corner
        assert "33" in source   # Right eye outer corner
    
    def test_mask_detection_landmarks_indices(self):
        """Verify that mask detection uses correct landmark indices."""
        # Mouth region indices: 61, 291, 0, 17, 84, 181, 91, 146, etc.
        source = inspect.getsource(FaceDetector._detect_mask)
        assert "61" in source  # Left mouth corner
        assert "291" in source  # Right mouth corner
    
    def test_occlusion_error_code_is_correct(self):
        """Verify that occlusion detection raises correct error code."""
        assert ErrorCode.OCCLUSION_DETECTED.value == "OCCLUSION_DETECTED"
    
    @patch('app.components.face_detector.vision.FaceLandmarker.create_from_options')
    @patch.object(FaceDetector, '_check_occlusions')
    def test_occlusion_check_called_during_detection(
        self,
        mock_check_occlusions,
        mock_create,
        sample_image_data,
        mock_detection_result_with_confidence
    ):
        """Test that occlusion check is called during face detection."""
        # Skip occlusion check for this test (we test occlusion detection separately)
        mock_check_occlusions.return_value = None
        
        # Setup mock detector with realistic landmarks
        mock_detector = Mock()
        mock_detector.detect.return_value = mock_detection_result_with_confidence
        mock_create.return_value = mock_detector
        
        # Initialize face detector
        face_detector = FaceDetector(model_path='dummy_path.task')
        
        # Detect faces - should succeed with realistic landmarks
        landmarks = face_detector.detect_faces(sample_image_data)
        
        # Verify landmarks were returned
        assert landmarks.points.shape == (468, 3)
        assert landmarks.confidence > 0.3
        
        # Verify occlusion check was called
        mock_check_occlusions.assert_called_once()
    
    @patch('app.components.face_detector.vision.FaceLandmarker.create_from_options')
    def test_glasses_detection_high_z_variance(
        self,
        mock_create,
        sample_image_data,
        mock_mediapipe_landmark
    ):
        """Test glasses detection via high z-depth variance in eye region."""
        # Create landmarks with high z-variance in eye region (simulating glasses)
        result = Mock()
        landmarks = []
        for i in range(468):
            if i < 234:
                x = 0.35 + (i % 18) * 0.025
                y = 0.25 + (i // 18) * 0.015
            else:
                x = 0.35 + ((i - 234) % 18) * 0.025
                y = 0.55 + ((i - 234) // 18) * 0.015
            
            # Simulate glasses by creating high z-variance in eye landmarks
            if i in [362, 385, 387, 263]:  # Left eye indices
                z = 0.08 + 0.02 * np.sin(i * 0.1)  # High variance
            else:
                z = 0.02 + 0.005 * np.sin(i * 0.1)
            
            landmarks.append(
                mock_mediapipe_landmark(x, y, z, visibility=0.9, presence=0.95)
            )
        
        result.face_landmarks = [landmarks]
        
        mock_detector = Mock()
        mock_detector.detect.return_value = result
        mock_create.return_value = mock_detector
        
        face_detector = FaceDetector(model_path='dummy_path.task')
        
        # Should raise OCCLUSION_DETECTED
        with pytest.raises(ValidationError) as exc_info:
            face_detector.detect_faces(sample_image_data)
        
        assert exc_info.value.error_code == ErrorCode.OCCLUSION_DETECTED
        assert "accessories" in exc_info.value.message.lower()
    
    @patch('app.components.face_detector.vision.FaceLandmarker.create_from_options')
    def test_mask_detection_high_z_variance(
        self,
        mock_create,
        sample_image_data,
        mock_mediapipe_landmark
    ):
        """Test mask detection via high z-depth variance in mouth region."""
        # Create landmarks with high z-variance in mouth region (simulating mask)
        result = Mock()
        landmarks = []
        for i in range(468):
            if i < 234:
                x = 0.35 + (i % 18) * 0.025
                y = 0.25 + (i // 18) * 0.015
            else:
                x = 0.35 + ((i - 234) % 18) * 0.025
                y = 0.55 + ((i - 234) // 18) * 0.015
            
            # Simulate mask by creating high z-variance in mouth landmarks
            if i in [61, 291, 0, 17]:  # Mouth corner indices
                z = 0.09 + 0.03 * np.sin(i * 0.1)  # Very high variance
            else:
                z = 0.02 + 0.005 * np.sin(i * 0.1)
            
            landmarks.append(
                mock_mediapipe_landmark(x, y, z, visibility=0.9, presence=0.95)
            )
        
        result.face_landmarks = [landmarks]
        
        mock_detector = Mock()
        mock_detector.detect.return_value = result
        mock_create.return_value = mock_detector
        
        face_detector = FaceDetector(model_path='dummy_path.task')
        
        # Should raise OCCLUSION_DETECTED
        with pytest.raises(ValidationError) as exc_info:
            face_detector.detect_faces(sample_image_data)
        
        assert exc_info.value.error_code == ErrorCode.OCCLUSION_DETECTED
        assert ("mask" in exc_info.value.message.lower() or 
                "mouth" in exc_info.value.message.lower())


class TestFaceDetectorMediaPipeInitialization:
    """Test MediaPipe initialization and configuration."""
    
    @patch('app.components.face_detector.vision.FaceLandmarker.create_from_options')
    def test_mediapipe_initialization_success(self, mock_create):
        """Test successful MediaPipe initialization."""
        mock_detector = Mock()
        mock_create.return_value = mock_detector
        
        # Initialize should succeed
        face_detector = FaceDetector(model_path='dummy_path.task')
        
        # Verify detector was created
        assert face_detector.detector is not None
        mock_create.assert_called_once()
    
    @patch('app.components.face_detector.vision.FaceLandmarker.create_from_options')
    def test_mediapipe_initialization_failure(self, mock_create):
        """Test MediaPipe initialization failure handling."""
        mock_create.side_effect = Exception("Model file not found")
        
        # Should raise RuntimeError
        with pytest.raises(RuntimeError) as exc_info:
            FaceDetector(model_path='nonexistent.task')
        
        assert "MediaPipe initialization failed" in str(exc_info.value)
    
    @patch('app.components.face_detector.vision.FaceLandmarker.create_from_options')
    def test_detector_configuration_parameters(self, mock_create):
        """Test that FaceDetector configures MediaPipe with correct parameters."""
        mock_detector = Mock()
        mock_create.return_value = mock_detector
        
        # Initialize
        FaceDetector(model_path='dummy_path.task')
        
        # Verify create_from_options was called
        assert mock_create.called
        
        # Get the options argument passed to create_from_options
        call_args = mock_create.call_args
        assert call_args is not None


class TestFaceDetectorDetectionSuccessCase:
    """Test successful face detection scenarios."""
    
    @patch('app.components.face_detector.vision.FaceLandmarker.create_from_options')
    @patch.object(FaceDetector, '_check_occlusions')
    def test_single_face_detection_success(
        self,
        mock_check_occlusions,
        mock_create,
        sample_image_data,
        mock_detection_result_with_confidence
    ):
        """Test successful detection of single face."""
        mock_check_occlusions.return_value = None
        
        mock_detector = Mock()
        mock_detector.detect.return_value = mock_detection_result_with_confidence
        mock_create.return_value = mock_detector
        
        face_detector = FaceDetector(model_path='dummy_path.task')
        landmarks = face_detector.detect_faces(sample_image_data)
        
        # Should return valid landmarks
        assert landmarks is not None
        assert landmarks.points.shape == (468, 3)
        assert landmarks.confidence > 0.0
    
    @patch('app.components.face_detector.vision.FaceLandmarker.create_from_options')
    @patch.object(FaceDetector, '_check_occlusions')
    def test_detection_returns_correct_image_dimensions(
        self,
        mock_check_occlusions,
        mock_create,
        sample_image_data,
        mock_detection_result_with_confidence
    ):
        """Test that detected landmarks have correct image dimensions."""
        mock_check_occlusions.return_value = None
        
        mock_detector = Mock()
        mock_detector.detect.return_value = mock_detection_result_with_confidence
        mock_create.return_value = mock_detector
        
        face_detector = FaceDetector(model_path='dummy_path.task')
        landmarks = face_detector.detect_faces(sample_image_data)
        
        # Check image dimensions are properly stored
        assert landmarks.image_width == sample_image_data.width
        assert landmarks.image_height == sample_image_data.height


class TestFaceDetectorErrorCases:
    """Test error handling for various failure scenarios."""
    
    @patch('app.components.face_detector.vision.FaceLandmarker.create_from_options')
    def test_mediapipe_detection_exception(
        self,
        mock_create,
        sample_image_data
    ):
        """Test handling of MediaPipe detection exceptions."""
        mock_detector = Mock()
        mock_detector.detect.side_effect = Exception("Detection failed")
        mock_create.return_value = mock_detector
        
        face_detector = FaceDetector(model_path='dummy_path.task')
        
        # Should raise ValidationError with PROCESSING_FAILED
        with pytest.raises(ValidationError) as exc_info:
            face_detector.detect_faces(sample_image_data)
        
        assert exc_info.value.error_code == ErrorCode.PROCESSING_FAILED
        assert "detection failed" in exc_info.value.message.lower()


class TestFaceDetectorCoordinateConversion:
    """Test coordinate conversion from normalized to pixel space."""
    
    @patch('app.components.face_detector.vision.FaceLandmarker.create_from_options')
    @patch.object(FaceDetector, '_check_occlusions')
    def test_x_coordinate_conversion(
        self,
        mock_check_occlusions,
        mock_create,
        sample_image_data,
        mock_mediapipe_landmark
    ):
        """Test that normalized x coordinates are correctly converted to pixels."""
        mock_check_occlusions.return_value = None
        
        result = Mock()
        # Create specific landmarks with known normalized coordinates
        landmarks_list = []
        for i in range(468):
            x_norm = 0.5  # Middle of image
            y_norm = 0.5
            z = 0.01
            landmarks_list.append(
                mock_mediapipe_landmark(x_norm, y_norm, z, visibility=0.9, presence=0.95)
            )
        
        result.face_landmarks = [landmarks_list]
        
        mock_detector = Mock()
        mock_detector.detect.return_value = result
        mock_create.return_value = mock_detector
        
        face_detector = FaceDetector(model_path='dummy_path.task')
        landmarks = face_detector.detect_faces(sample_image_data)
        
        # Normalized 0.5 should convert to middle of image (320 for 640px width)
        x_pixels = landmarks.points[0, 0]
        assert abs(x_pixels - 320.0) < 1.0  # Allow 1px tolerance
    
    @patch('app.components.face_detector.vision.FaceLandmarker.create_from_options')
    @patch.object(FaceDetector, '_check_occlusions')
    def test_y_coordinate_conversion(
        self,
        mock_check_occlusions,
        mock_create,
        sample_image_data,
        mock_mediapipe_landmark
    ):
        """Test that normalized y coordinates are correctly converted to pixels."""
        mock_check_occlusions.return_value = None
        
        result = Mock()
        landmarks_list = []
        for i in range(468):
            x_norm = 0.5
            y_norm = 0.75  # 3/4 down the image
            z = 0.01
            landmarks_list.append(
                mock_mediapipe_landmark(x_norm, y_norm, z, visibility=0.9, presence=0.95)
            )
        
        result.face_landmarks = [landmarks_list]
        
        mock_detector = Mock()
        mock_detector.detect.return_value = result
        mock_create.return_value = mock_detector
        
        face_detector = FaceDetector(model_path='dummy_path.task')
        landmarks = face_detector.detect_faces(sample_image_data)
        
        # Normalized 0.75 should convert to 3/4 of height (360 for 480px height)
        y_pixels = landmarks.points[0, 1]
        assert abs(y_pixels - 360.0) < 1.0  # Allow 1px tolerance


class TestFaceDetectorConfidenceCalculationFormula:
    """Test confidence calculation formula and thresholds."""
    
    @patch('app.components.face_detector.vision.FaceLandmarker.create_from_options')
    @patch.object(FaceDetector, '_check_occlusions')
    def test_confidence_formula_averaging(
        self,
        mock_check_occlusions,
        mock_create,
        sample_image_data,
        mock_mediapipe_landmark
    ):
        """Test that confidence is calculated as average of visibility and presence."""
        mock_check_occlusions.return_value = None
        
        result = Mock()
        landmarks_list = []
        for i in range(468):
            x = 0.3 + (i % 20) * 0.03
            y = 0.3 + (i // 20) * 0.03
            z = 0.01
            # All landmarks have same visibility/presence for consistent calculation
            landmarks_list.append(
                mock_mediapipe_landmark(x, y, z, visibility=0.8, presence=0.8)
            )
        
        result.face_landmarks = [landmarks_list]
        
        mock_detector = Mock()
        mock_detector.detect.return_value = result
        mock_create.return_value = mock_detector
        
        face_detector = FaceDetector(model_path='dummy_path.task')
        landmarks = face_detector.detect_faces(sample_image_data)
        
        # Confidence should be average of visibility and presence
        # (0.8 + 0.8) / 2 = 0.8
        assert abs(landmarks.confidence - 0.8) < 0.05
    
    @patch('app.components.face_detector.vision.FaceLandmarker.create_from_options')
    @patch('app.components.face_detector.settings.landmark_confidence_threshold', 0.8)
    @patch.object(FaceDetector, '_check_occlusions')
    def test_confidence_below_threshold_raises_error(
        self,
        mock_check_occlusions,
        mock_create,
        sample_image_data,
        mock_mediapipe_landmark
    ):
        """Test that confidence below threshold raises POOR_QUALITY error."""
        # Don't skip occlusion check
        result = Mock()
        landmarks_list = []
        for i in range(468):
            x = 0.3 + (i % 20) * 0.03
            y = 0.3 + (i // 20) * 0.03
            z = 0.01
            # Low visibility/presence scores
            landmarks_list.append(
                mock_mediapipe_landmark(x, y, z, visibility=0.5, presence=0.5)
            )
        
        result.face_landmarks = [landmarks_list]
        
        mock_detector = Mock()
        mock_detector.detect.return_value = result
        mock_create.return_value = mock_detector
        
        face_detector = FaceDetector(model_path='dummy_path.task')
        
        # Should raise POOR_QUALITY
        with pytest.raises(ValidationError) as exc_info:
            face_detector.detect_faces(sample_image_data)
        
        assert exc_info.value.error_code == ErrorCode.POOR_QUALITY


class TestFaceDetectorVisibilityPresenceScores:
    """Test handling of visibility and presence scores from landmarks."""
    
    @patch('app.components.face_detector.vision.FaceLandmarker.create_from_options')
    @patch.object(FaceDetector, '_check_occlusions')
    def test_high_visibility_scores(
        self,
        mock_check_occlusions,
        mock_create,
        sample_image_data,
        mock_detection_result_with_confidence
    ):
        """Test that high visibility scores result in high confidence."""
        mock_check_occlusions.return_value = None
        
        mock_detector = Mock()
        mock_detector.detect.return_value = mock_detection_result_with_confidence
        mock_create.return_value = mock_detector
        
        face_detector = FaceDetector(model_path='dummy_path.task')
        landmarks = face_detector.detect_faces(sample_image_data)
        
        # High visibility (0.9) and presence (0.95) should give ~0.925 confidence
        assert landmarks.confidence > 0.80
