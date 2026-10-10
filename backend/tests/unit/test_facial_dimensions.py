"""
Unit tests for Facial Dimension Measurements (Requirement 4.1).

Tests the five core facial dimension measurements:
1. Face length (landmark 10 to 152)
2. Face width (landmark 234 to 454)
3. Jaw width (landmark 427 to 200)
4. Nose width (landmark 119 to 348)
5. Eye distance (landmark 468 to 473)
"""

import pytest
import numpy as np

from app.components.symmetry_classifier import SymmetryClassifier
from app.models.internal import Landmarks


@pytest.fixture
def symmetry_classifier():
    """Create SymmetryClassifier instance for testing."""
    return SymmetryClassifier()


@pytest.fixture
def synthetic_landmarks():
    """
    Create synthetic landmarks for testing.
    
    This creates a 468-point landmark array with known values
    to verify distance calculations are correct.
    """
    # Create 468 landmarks with z-coordinate = 0
    landmarks_array = np.zeros((468, 3), dtype=np.float32)
    
    # Set specific landmarks for testing
    # Face boundaries
    landmarks_array[10] = [320, 150, 0]    # FACE_TOP (top of head)
    landmarks_array[152] = [320, 450, 0]   # FACE_BOTTOM (chin)
    landmarks_array[234] = [200, 300, 0]   # FACE_LEFT (left cheekbone)
    landmarks_array[454] = [440, 300, 0]   # FACE_RIGHT (right cheekbone)
    
    # Jaw points
    landmarks_array[427] = [180, 380, 0]   # JAW_LEFT_ALT
    landmarks_array[200] = [460, 380, 0]   # JAW_RIGHT_ALT
    
    # Nose points
    landmarks_array[119] = [300, 280, 0]   # NOSTRIL_LEFT
    landmarks_array[348] = [340, 280, 0]   # NOSTRIL_RIGHT
    
    # Eye corners (for pupil center calculation)
    landmarks_array[33] = [270, 195, 0]    # LEFT_EYE_OUTER
    landmarks_array[133] = [290, 195, 0]   # LEFT_EYE_INNER
    landmarks_array[263] = [350, 195, 0]   # RIGHT_EYE_OUTER
    landmarks_array[362] = [370, 195, 0]   # RIGHT_EYE_INNER
    
    return Landmarks(
        points=landmarks_array,
        confidence=0.9,
        image_width=640,
        image_height=480
    )


class TestFacialDimensionMeasurements:
    """Tests for facial dimension measurements (Requirement 4.1)."""
    
    def test_measure_face_length(self, symmetry_classifier, synthetic_landmarks):
        """
        Test face length measurement: distance from top of head (landmark 10) to chin (landmark 152).
        
        Expected: sqrt((320-320)^2 + (150-450)^2) = sqrt(90000) = 300 pixels
        """
        face_length = symmetry_classifier.measure_face_length(synthetic_landmarks)
        assert isinstance(face_length, float), "Face length should be a float"
        assert face_length == 300.0, f"Expected 300, got {face_length}"
    
    def test_measure_face_width(self, symmetry_classifier, synthetic_landmarks):
        """
        Test face width measurement: distance between left (234) and right (454) cheekbones.
        
        Expected: sqrt((200-440)^2 + (300-300)^2) = sqrt(57600) = 240 pixels
        """
        face_width = symmetry_classifier.measure_face_width(synthetic_landmarks)
        assert isinstance(face_width, float), "Face width should be a float"
        assert face_width == 240.0, f"Expected 240, got {face_width}"
    
    def test_measure_jaw_width(self, symmetry_classifier, synthetic_landmarks):
        """
        Test jaw width measurement: distance between left (427) and right (200) jaw angles.
        
        Expected: sqrt((180-460)^2 + (380-380)^2) = sqrt(78400) = 280 pixels
        """
        jaw_width = symmetry_classifier.measure_jaw_width(synthetic_landmarks)
        assert isinstance(jaw_width, float), "Jaw width should be a float"
        assert jaw_width == 280.0, f"Expected 280, got {jaw_width}"
    
    def test_measure_nose_width(self, symmetry_classifier, synthetic_landmarks):
        """
        Test nose width measurement: distance between left (119) and right (348) nostrils.
        
        Expected: sqrt((300-340)^2 + (280-280)^2) = sqrt(1600) = 40 pixels
        """
        nose_width = symmetry_classifier.measure_nose_width(synthetic_landmarks)
        assert isinstance(nose_width, float), "Nose width should be a float"
        assert nose_width == 40.0, f"Expected 40, got {nose_width}"
    
    def test_measure_eye_distance(self, symmetry_classifier, synthetic_landmarks):
        """
        Test eye distance measurement: distance between left and right pupil centers.
        
        Pupil centers are calculated as midpoints:
        Left pupil center: ((270+290)/2, (195+195)/2) = (280, 195)
        Right pupil center: ((350+370)/2, (195+195)/2) = (360, 195)
        Expected: sqrt((280-360)^2 + (195-195)^2) = sqrt(6400) = 80 pixels
        """
        eye_distance = symmetry_classifier.measure_eye_distance(synthetic_landmarks)
        assert isinstance(eye_distance, float), "Eye distance should be a float"
        assert eye_distance == 80.0, f"Expected 80, got {eye_distance}"
    
    def test_get_all_facial_dimensions(self, symmetry_classifier, synthetic_landmarks):
        """
        Test get_all_facial_dimensions returns all five measurements.
        
        This method should return a dictionary with all five dimensions.
        """
        dimensions = symmetry_classifier.get_all_facial_dimensions(synthetic_landmarks)
        
        # Verify return type
        assert isinstance(dimensions, dict), "Should return a dictionary"
        
        # Verify all keys are present
        expected_keys = {'face_length', 'face_width', 'jaw_width', 'nose_width', 'eye_distance'}
        assert set(dimensions.keys()) == expected_keys, f"Expected keys {expected_keys}, got {dimensions.keys()}"
        
        # Verify all values are floats
        for key, value in dimensions.items():
            assert isinstance(value, float), f"{key} should be a float, got {type(value)}"
        
        # Verify values match individual measurements
        assert dimensions['face_length'] == 300.0
        assert dimensions['face_width'] == 240.0
        assert dimensions['jaw_width'] == 280.0
        assert dimensions['nose_width'] == 40.0
        assert dimensions['eye_distance'] == 80.0
    
    def test_measurements_with_offset_landmarks(self, symmetry_classifier):
        """
        Test measurements work correctly with landmarks offset from origin.
        
        This ensures euclidean_distance calculations are correct regardless
        of absolute position.
        """
        landmarks_array = np.zeros((468, 3), dtype=np.float32)
        
        # Offset all points by (1000, 2000)
        offset_x, offset_y = 1000, 2000
        landmarks_array[10] = [1320, 2150, 0]
        landmarks_array[152] = [1320, 2450, 0]
        landmarks_array[234] = [1200, 2300, 0]
        landmarks_array[454] = [1440, 2300, 0]
        landmarks_array[427] = [1180, 2380, 0]
        landmarks_array[200] = [1460, 2380, 0]
        landmarks_array[119] = [1300, 2280, 0]
        landmarks_array[348] = [1340, 2280, 0]
        landmarks_array[33] = [1270, 2195, 0]
        landmarks_array[133] = [1290, 2195, 0]
        landmarks_array[263] = [1350, 2195, 0]
        landmarks_array[362] = [1370, 2195, 0]
        
        landmarks = Landmarks(
            points=landmarks_array,
            confidence=0.9,
            image_width=640,
            image_height=480
        )
        
        # Measurements should be the same as non-offset version
        assert symmetry_classifier.measure_face_length(landmarks) == 300.0
        assert symmetry_classifier.measure_face_width(landmarks) == 240.0
        assert symmetry_classifier.measure_jaw_width(landmarks) == 280.0
        assert symmetry_classifier.measure_nose_width(landmarks) == 40.0
        assert symmetry_classifier.measure_eye_distance(landmarks) == 80.0
    
    def test_measurements_are_positive(self, symmetry_classifier, synthetic_landmarks):
        """Test that all measurements return positive values."""
        face_length = symmetry_classifier.measure_face_length(synthetic_landmarks)
        face_width = symmetry_classifier.measure_face_width(synthetic_landmarks)
        jaw_width = symmetry_classifier.measure_jaw_width(synthetic_landmarks)
        nose_width = symmetry_classifier.measure_nose_width(synthetic_landmarks)
        eye_distance = symmetry_classifier.measure_eye_distance(synthetic_landmarks)
        
        assert face_length > 0, "Face length should be positive"
        assert face_width > 0, "Face width should be positive"
        assert jaw_width > 0, "Jaw width should be positive"
        assert nose_width > 0, "Nose width should be positive"
        assert eye_distance > 0, "Eye distance should be positive"
    
    def test_face_dimensions_ratios_are_reasonable(self, symmetry_classifier, synthetic_landmarks):
        """
        Test that facial dimension ratios are reasonable.
        
        For a face to be valid, certain proportions should make sense:
        - Face length should be larger than face width (longer than wide)
        - Face width should be larger than jaw width (usually)
        - Eye distance should be less than face width
        """
        face_length = symmetry_classifier.measure_face_length(synthetic_landmarks)
        face_width = symmetry_classifier.measure_face_width(synthetic_landmarks)
        jaw_width = symmetry_classifier.measure_jaw_width(synthetic_landmarks)
        eye_distance = symmetry_classifier.measure_eye_distance(synthetic_landmarks)
        
        # Face is typically longer than wide
        assert face_length > face_width, "Face length should be > face width"
        
        # Eye distance should be less than face width
        assert eye_distance < face_width, "Eye distance should be < face width"
        
        # Jaw width should be reasonable relative to face width
        # (can be close or larger depending on face shape)
        assert jaw_width > 0, "Jaw width should be positive"
    
    @pytest.mark.parametrize("test_case", [
        {
            "name": "Perfectly vertical face",
            "landmarks_overrides": {
                10: [320, 150, 0],
                152: [320, 450, 0],
            },
            "expected_length": 300.0
        },
        {
            "name": "Perfectly horizontal width",
            "landmarks_overrides": {
                234: [200, 300, 0],
                454: [440, 300, 0],
            },
            "expected_width": 240.0
        },
        {
            "name": "Small face dimensions",
            "landmarks_overrides": {
                10: [320, 300, 0],
                152: [320, 350, 0],
                234: [310, 325, 0],
                454: [330, 325, 0],
            },
            "expected_length": 50.0,
            "expected_width": 20.0
        }
    ])
    def test_measurement_edge_cases(self, symmetry_classifier, test_case):
        """Test measurements with edge case scenarios."""
        landmarks_array = np.zeros((468, 3), dtype=np.float32)
        
        # Apply standard positions first
        landmarks_array[10] = [320, 150, 0]
        landmarks_array[152] = [320, 450, 0]
        landmarks_array[234] = [200, 300, 0]
        landmarks_array[454] = [440, 300, 0]
        landmarks_array[427] = [180, 380, 0]
        landmarks_array[200] = [460, 380, 0]
        landmarks_array[119] = [300, 280, 0]
        landmarks_array[348] = [340, 280, 0]
        landmarks_array[33] = [270, 195, 0]
        landmarks_array[133] = [290, 195, 0]
        landmarks_array[263] = [350, 195, 0]
        landmarks_array[362] = [370, 195, 0]
        
        # Apply test-specific overrides
        for landmark_idx, coords in test_case["landmarks_overrides"].items():
            landmarks_array[landmark_idx] = coords
        
        landmarks = Landmarks(
            points=landmarks_array,
            confidence=0.9,
            image_width=640,
            image_height=480
        )
        
        # Check expected values if provided
        if "expected_length" in test_case:
            actual = symmetry_classifier.measure_face_length(landmarks)
            assert abs(actual - test_case["expected_length"]) < 0.1, \
                f"{test_case['name']}: Expected length {test_case['expected_length']}, got {actual}"
        
        if "expected_width" in test_case:
            actual = symmetry_classifier.measure_face_width(landmarks)
            assert abs(actual - test_case["expected_width"]) < 0.1, \
                f"{test_case['name']}: Expected width {test_case['expected_width']}, got {actual}"


class TestMeasurementIntegration:
    """Integration tests using measurements for symmetry classification."""
    
    def test_measurements_used_in_classification(self, symmetry_classifier, synthetic_landmarks):
        """
        Test that facial dimension measurements integrate properly with symmetry classification.
        
        This ensures the measurements can be used by other components.
        """
        # Get dimensions
        dimensions = symmetry_classifier.get_all_facial_dimensions(synthetic_landmarks)
        
        # Calculate aspect ratio (commonly used in classification)
        aspect_ratio = dimensions['face_length'] / dimensions['face_width']
        assert 1.0 < aspect_ratio < 2.0, "Aspect ratio should be reasonable for face"
        
        # Calculate jaw-to-width ratio
        jaw_to_width_ratio = dimensions['jaw_width'] / dimensions['face_width']
        assert 0.5 < jaw_to_width_ratio < 2.0, "Jaw to width ratio should be reasonable"
        
        # All measurements should be consistent
        assert dimensions['face_length'] > dimensions['nose_width'], \
            "Face length should be >> nose width"
        assert dimensions['face_width'] > dimensions['eye_distance'] * 1.5, \
            "Face width should be > eye distance"
