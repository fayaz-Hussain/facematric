"""
Integration tests for OverlayGenerator component.

Tests the complete overlay generation pipeline including:
- 2D landmark extraction from 3D landmarks
- Golden ratio guide calculations
- OverlayData structure generation

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

import pytest
import numpy as np
from app.components.overlay_generator import OverlayGenerator
from app.models.internal import Landmarks, OverlayDataInternal


class TestOverlayGeneratorIntegration:
    """Integration tests for complete overlay generation."""
    
    @pytest.fixture
    def overlay_generator(self):
        """Create OverlayGenerator instance."""
        return OverlayGenerator()
    
    @pytest.fixture
    def realistic_landmarks(self):
        """
        Create realistic 3D landmarks in pixel coordinates.
        
        Landmarks are converted to pixel coordinates by the FaceDetector
        before being stored in the Landmarks object.
        Width: 640px, Height: 480px
        """
        # 3D landmarks (468, 3) with (x, y, z) in pixel coordinates
        # Random landmarks in valid image range
        landmarks_3d = np.zeros((468, 3), dtype=np.float32)
        landmarks_3d[:, 0] = np.random.uniform(0, 640, 468)  # x in pixels
        landmarks_3d[:, 1] = np.random.uniform(0, 480, 468)  # y in pixels
        landmarks_3d[:, 2] = np.random.uniform(-0.5, 0.5, 468)  # z depth
        
        # Set key facial landmarks for proper overlay generation
        # Face boundaries (in pixel coordinates)
        landmarks_3d[10] = [320.0, 72.0, 0.0]    # Forehead top (x_center, 15% of height)
        landmarks_3d[152] = [320.0, 408.0, 0.0]  # Chin bottom (x_center, 85% of height)
        landmarks_3d[234] = [128.0, 240.0, 0.0]  # Left face contour (20% of width, 50% of height)
        landmarks_3d[454] = [512.0, 240.0, 0.0]  # Right face contour (80% of width, 50% of height)
        
        # Nose bridge (important for spiral center)
        landmarks_3d[6] = [320.0, 192.0, 0.2]  # (x_center, 40% of height)
        
        return landmarks_3d
    
    @pytest.fixture
    def image_dimensions(self):
        """Standard image dimensions."""
        return (640, 480)
    
    def test_generate_produces_valid_overlay_data(
        self, overlay_generator, realistic_landmarks, image_dimensions
    ):
        """
        Test that generate() produces valid OverlayDataInternal.
        
        **Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5**
        """
        width, height = image_dimensions
        
        # Create Landmarks object
        landmarks = Landmarks(
            points=realistic_landmarks,
            confidence=0.85,
            image_width=width,
            image_height=height
        )
        
        # Generate overlay data
        overlay_data = overlay_generator.generate(landmarks)
        
        # Verify type
        assert isinstance(overlay_data, OverlayDataInternal)
        
        # Verify landmarks_2d shape (Requirement 9.1)
        assert overlay_data.landmarks_2d.shape == (468, 2)
        
        # Verify all coordinates are within bounds (Requirement 9.5)
        assert np.all(overlay_data.landmarks_2d[:, 0] >= 0)
        assert np.all(overlay_data.landmarks_2d[:, 0] <= width)
        assert np.all(overlay_data.landmarks_2d[:, 1] >= 0)
        assert np.all(overlay_data.landmarks_2d[:, 1] <= height)
        
        # Verify midline_x is within bounds (Requirement 9.3)
        assert 0 <= overlay_data.midline_x <= width
        
        # Verify horizontal thirds count and bounds (Requirement 9.2)
        assert len(overlay_data.horizontal_thirds_y) == 4
        for y in overlay_data.horizontal_thirds_y:
            assert 0 <= y <= height
        
        # Verify thirds are in ascending order
        for i in range(len(overlay_data.horizontal_thirds_y) - 1):
            assert overlay_data.horizontal_thirds_y[i] < overlay_data.horizontal_thirds_y[i+1]
        
        # Verify phi spiral center is within bounds (Requirement 9.4)
        assert isinstance(overlay_data.phi_spiral_center, tuple)
        assert len(overlay_data.phi_spiral_center) == 2
        cx, cy = overlay_data.phi_spiral_center
        assert 0 <= cx <= width
        assert 0 <= cy <= height
    
    def test_2d_landmark_extraction(
        self, overlay_generator, realistic_landmarks, image_dimensions
    ):
        """
        Test that 3D landmarks are properly converted to 2D.
        
        Requirement 9.1: 2D landmark extraction from 3D landmarks
        """
        width, height = image_dimensions
        
        landmarks = Landmarks(
            points=realistic_landmarks,
            confidence=0.85,
            image_width=width,
            image_height=height
        )
        
        overlay_data = overlay_generator.generate(landmarks)
        
        # Verify z-coordinate was dropped
        assert overlay_data.landmarks_2d.ndim == 2
        assert overlay_data.landmarks_2d.shape[1] == 2
        
        # Verify all 468 landmarks are present
        assert overlay_data.landmarks_2d.shape[0] == 468
        
        # Verify landmarks are in pixel space (not normalized)
        # After conversion, values should be scaled to image dimensions
        assert np.all(overlay_data.landmarks_2d[:, 0] >= 0)
        assert np.all(overlay_data.landmarks_2d[:, 0] <= width)
        assert np.all(overlay_data.landmarks_2d[:, 1] >= 0)
        assert np.all(overlay_data.landmarks_2d[:, 1] <= height)
    
    def test_vertical_midline_calculation(
        self, overlay_generator, realistic_landmarks, image_dimensions
    ):
        """
        Test vertical midline calculation.
        
        Requirement 9.3: Calculate vertical midline coordinate
        """
        width, height = image_dimensions
        
        landmarks = Landmarks(
            points=realistic_landmarks,
            confidence=0.85,
            image_width=width,
            image_height=height
        )
        
        overlay_data = overlay_generator.generate(landmarks)
        
        # Midline should be average of left and right face x-coordinates
        # Left face (234) is at 0.2 * 640 = 128
        # Right face (454) is at 0.8 * 640 = 512
        # Expected midline: (128 + 512) / 2 = 320
        expected_midline = (128.0 + 512.0) / 2.0
        assert abs(overlay_data.midline_x - expected_midline) < 1.0
    
    def test_horizontal_thirds_calculation(
        self, overlay_generator, realistic_landmarks, image_dimensions
    ):
        """
        Test horizontal thirds calculation.
        
        Requirement 9.2: Calculate horizontal third divisions
        """
        width, height = image_dimensions
        
        landmarks = Landmarks(
            points=realistic_landmarks,
            confidence=0.85,
            image_width=width,
            image_height=height
        )
        
        overlay_data = overlay_generator.generate(landmarks)
        
        # Thirds should divide face height using golden ratio proportions
        # Face top (10) is at 72.0
        # Face bottom (152) is at 408.0
        # Face height: 408 - 72 = 336
        # Third 0 (top): 72.0
        # Third 1 at 0.382 ratio: 72 + 336*0.382 ≈ 200.352
        # Third 2 at 0.618 ratio: 72 + 336*0.618 ≈ 279.648
        # Third 3 (bottom): 408.0
        
        expected_top = 72.0
        expected_bottom = 408.0
        expected_face_height = expected_bottom - expected_top
        expected_third_1 = expected_top + expected_face_height * 0.382
        expected_third_2 = expected_top + expected_face_height * 0.618
        expected_third_3 = expected_bottom
        
        assert abs(overlay_data.horizontal_thirds_y[0] - expected_top) < 1.0
        assert abs(overlay_data.horizontal_thirds_y[1] - expected_third_1) < 2.0
        assert abs(overlay_data.horizontal_thirds_y[2] - expected_third_2) < 2.0
        assert abs(overlay_data.horizontal_thirds_y[3] - expected_third_3) < 1.0
    
    def test_phi_spiral_center_calculation(
        self, overlay_generator, realistic_landmarks, image_dimensions
    ):
        """
        Test phi spiral center calculation.
        
        Requirement 9.4: Calculate phi spiral center (near nose bridge)
        """
        width, height = image_dimensions
        
        landmarks = Landmarks(
            points=realistic_landmarks,
            confidence=0.85,
            image_width=width,
            image_height=height
        )
        
        overlay_data = overlay_generator.generate(landmarks)
        
        # Spiral center should be at nose bridge (landmark 6)
        # Nose bridge is at (0.5, 0.4) normalized = (320, 192) in pixels
        expected_cx = 0.5 * width
        expected_cy = 0.4 * height
        
        cx, cy = overlay_data.phi_spiral_center
        assert abs(cx - expected_cx) < 1.0
        assert abs(cy - expected_cy) < 1.0
    
    def test_coordinates_within_image_bounds(
        self, overlay_generator, realistic_landmarks, image_dimensions
    ):
        """
        Test that all guide coordinates fall within image bounds.
        
        Requirement 9.5: Ensure all coordinates within image bounds
        """
        width, height = image_dimensions
        
        landmarks = Landmarks(
            points=realistic_landmarks,
            confidence=0.85,
            image_width=width,
            image_height=height
        )
        
        overlay_data = overlay_generator.generate(landmarks)
        
        # Check landmarks
        assert np.all(overlay_data.landmarks_2d >= 0)
        assert np.all(overlay_data.landmarks_2d[:, 0] <= width)
        assert np.all(overlay_data.landmarks_2d[:, 1] <= height)
        
        # Check midline
        assert 0 <= overlay_data.midline_x <= width
        
        # Check horizontal thirds
        for y in overlay_data.horizontal_thirds_y:
            assert 0 <= y <= height
        
        # Check phi spiral
        cx, cy = overlay_data.phi_spiral_center
        assert 0 <= cx <= width
        assert 0 <= cy <= height
    
    def test_different_image_sizes(self, overlay_generator):
        """
        Test overlay generation with different image dimensions.
        """
        test_cases = [
            (320, 240),
            (640, 480),
            (1280, 720),
            (1920, 1440),
            (800, 800),
        ]
        
        for width, height in test_cases:
            # Create landmarks appropriate for this image size
            landmarks_3d = np.zeros((468, 3), dtype=np.float32)
            landmarks_3d[:, 0] = np.random.uniform(0, width, 468)
            landmarks_3d[:, 1] = np.random.uniform(0, height, 468)
            landmarks_3d[:, 2] = np.random.uniform(-0.5, 0.5, 468)
            
            # Set key landmarks
            landmarks_3d[10] = [width/2, height*0.15, 0.0]    # Forehead
            landmarks_3d[152] = [width/2, height*0.85, 0.0]   # Chin
            landmarks_3d[234] = [width*0.2, height*0.5, 0.0]  # Left face
            landmarks_3d[454] = [width*0.8, height*0.5, 0.0]  # Right face
            landmarks_3d[6] = [width/2, height*0.4, 0.2]      # Nose bridge
            
            landmarks = Landmarks(
                points=landmarks_3d,
                confidence=0.85,
                image_width=width,
                image_height=height
            )
            
            overlay_data = overlay_generator.generate(landmarks)
            
            # Verify all coordinates are within bounds
            assert np.all(overlay_data.landmarks_2d[:, 0] >= 0)
            assert np.all(overlay_data.landmarks_2d[:, 0] <= width)
            assert np.all(overlay_data.landmarks_2d[:, 1] >= 0)
            assert np.all(overlay_data.landmarks_2d[:, 1] <= height)
            
            assert 0 <= overlay_data.midline_x <= width
            
            for y in overlay_data.horizontal_thirds_y:
                assert 0 <= y <= height
            
            cx, cy = overlay_data.phi_spiral_center
            assert 0 <= cx <= width
            assert 0 <= cy <= height
    
    def test_overlay_data_structure_consistency(
        self, overlay_generator, realistic_landmarks, image_dimensions
    ):
        """
        Test that OverlayDataInternal has consistent structure.
        """
        width, height = image_dimensions
        
        landmarks = Landmarks(
            points=realistic_landmarks,
            confidence=0.85,
            image_width=width,
            image_height=height
        )
        
        overlay_data = overlay_generator.generate(landmarks)
        
        # Verify all attributes exist
        assert hasattr(overlay_data, 'landmarks_2d')
        assert hasattr(overlay_data, 'midline_x')
        assert hasattr(overlay_data, 'horizontal_thirds_y')
        assert hasattr(overlay_data, 'phi_spiral_center')
        
        # Verify types
        assert isinstance(overlay_data.landmarks_2d, np.ndarray)
        assert isinstance(overlay_data.midline_x, float)
        assert isinstance(overlay_data.horizontal_thirds_y, list)
        assert isinstance(overlay_data.phi_spiral_center, tuple)
    
    def test_no_nan_or_inf_values(
        self, overlay_generator, realistic_landmarks, image_dimensions
    ):
        """
        Test that overlay data contains no NaN or Inf values.
        """
        width, height = image_dimensions
        
        landmarks = Landmarks(
            points=realistic_landmarks,
            confidence=0.85,
            image_width=width,
            image_height=height
        )
        
        overlay_data = overlay_generator.generate(landmarks)
        
        # Check landmarks
        assert not np.any(np.isnan(overlay_data.landmarks_2d))
        assert not np.any(np.isinf(overlay_data.landmarks_2d))
        
        # Check midline
        assert not np.isnan(overlay_data.midline_x)
        assert not np.isinf(overlay_data.midline_x)
        
        # Check thirds
        for y in overlay_data.horizontal_thirds_y:
            assert not np.isnan(y)
            assert not np.isinf(y)
        
        # Check spiral center
        cx, cy = overlay_data.phi_spiral_center
        assert not np.isnan(cx) and not np.isnan(cy)
        assert not np.isinf(cx) and not np.isinf(cy)
    
    def test_multiple_calls_produce_consistent_results(
        self, overlay_generator, realistic_landmarks, image_dimensions
    ):
        """
        Test that calling generate multiple times with same data produces consistent results.
        """
        width, height = image_dimensions
        
        landmarks = Landmarks(
            points=realistic_landmarks.copy(),
            confidence=0.85,
            image_width=width,
            image_height=height
        )
        
        # Generate twice
        result1 = overlay_generator.generate(landmarks)
        result2 = overlay_generator.generate(landmarks)
        
        # Compare results
        np.testing.assert_array_almost_equal(result1.landmarks_2d, result2.landmarks_2d)
        assert abs(result1.midline_x - result2.midline_x) < 0.001
        assert len(result1.horizontal_thirds_y) == len(result2.horizontal_thirds_y)
        for y1, y2 in zip(result1.horizontal_thirds_y, result2.horizontal_thirds_y):
            assert abs(y1 - y2) < 0.001
        assert abs(result1.phi_spiral_center[0] - result2.phi_spiral_center[0]) < 0.001
        assert abs(result1.phi_spiral_center[1] - result2.phi_spiral_center[1]) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
