"""
Unit tests for golden ratio guide calculations in OverlayGenerator.

Requirements: 9.2, 9.3, 9.4, 9.5
"""

import pytest
import numpy as np
from app.components.overlay_generator import OverlayGenerator


class TestGoldenRatioGuideCalculations:
    """Tests for calculate_golden_ratio_guides method."""
    
    @pytest.fixture
    def overlay_generator(self):
        """Create OverlayGenerator instance."""
        return OverlayGenerator()
    
    @pytest.fixture
    def mock_landmarks_2d(self):
        """
        Create mock 468 2D landmarks for testing.
        
        Key landmark indices used:
        - 10: Forehead top (face_top_y)
        - 152: Chin bottom (face_bottom_y)
        - 234: Left face contour (for midline and phi spiral)
        - 454: Right face contour (for midline and phi spiral)
        - 6: Nose bridge (for phi spiral center)
        """
        landmarks = np.zeros((468, 2), dtype=np.float32)
        
        # Face vertical boundaries
        landmarks[10] = [320.0, 100.0]   # Forehead top
        landmarks[152] = [320.0, 400.0]  # Chin bottom
        
        # Face horizontal boundaries
        landmarks[234] = [200.0, 250.0]  # Left face contour
        landmarks[454] = [440.0, 250.0]  # Right face contour
        
        # Nose bridge (for phi spiral center)
        landmarks[6] = [320.0, 220.0]
        
        return landmarks
    
    def test_calculate_golden_ratio_guides_basic(
        self, overlay_generator, mock_landmarks_2d
    ):
        """
        Test basic calculation of golden ratio guides.
        
        **Validates: Requirements 9.2, 9.3, 9.4, 9.5**
        """
        image_width = 640
        image_height = 480
        
        guides = overlay_generator.calculate_golden_ratio_guides(
            mock_landmarks_2d,
            image_width,
            image_height
        )
        
        # Verify structure
        assert isinstance(guides, dict)
        assert 'horizontal_thirds' in guides
        assert 'vertical_midline' in guides
        assert 'phi_spiral' in guides
        
        # Verify horizontal_thirds structure
        assert isinstance(guides['horizontal_thirds'], list)
        assert len(guides['horizontal_thirds']) == 2
        assert all(isinstance(x, (int, float)) for x in guides['horizontal_thirds'])
        
        # Verify vertical_midline
        assert isinstance(guides['vertical_midline'], (int, float))
        
        # Verify phi_spiral structure
        assert isinstance(guides['phi_spiral'], dict)
        assert 'cx' in guides['phi_spiral']
        assert 'cy' in guides['phi_spiral']
    
    def test_horizontal_thirds_divisions(
        self, overlay_generator, mock_landmarks_2d
    ):
        """
        Test that horizontal thirds divide image width into thirds.
        
        Requirement 9.2: Calculate horizontal third divisions
        """
        image_width = 600
        image_height = 480
        
        guides = overlay_generator.calculate_golden_ratio_guides(
            mock_landmarks_2d,
            image_width,
            image_height
        )
        
        x1, x2 = guides['horizontal_thirds']
        
        # Should divide into thirds: x1 at 1/3, x2 at 2/3
        assert abs(x1 - 200.0) < 0.1  # 600 / 3 = 200
        assert abs(x2 - 400.0) < 0.1  # 2 * 600 / 3 = 400
        
        # x1 should be before x2
        assert x1 < x2
        
        # Both should be within image bounds
        assert x1 > 0
        assert x2 < image_width
    
    def test_vertical_midline_calculation(
        self, overlay_generator, mock_landmarks_2d
    ):
        """
        Test vertical midline is at image center.
        
        Requirement 9.3: Calculate vertical midline coordinate
        """
        image_width = 640
        image_height = 480
        
        guides = overlay_generator.calculate_golden_ratio_guides(
            mock_landmarks_2d,
            image_width,
            image_height
        )
        
        # Vertical midline should be at image height / 2
        assert abs(guides['vertical_midline'] - 240.0) < 0.1
    
    def test_phi_spiral_center_coordinates(
        self, overlay_generator, mock_landmarks_2d
    ):
        """
        Test phi spiral center is based on face geometry.
        
        Requirement 9.4: Calculate phi spiral center coordinates
        """
        image_width = 640
        image_height = 480
        
        guides = overlay_generator.calculate_golden_ratio_guides(
            mock_landmarks_2d,
            image_width,
            image_height
        )
        
        cx = guides['phi_spiral']['cx']
        cy = guides['phi_spiral']['cy']
        
        # Center x should be average of left (200) and right (440) face
        assert abs(cx - 320.0) < 0.1
        
        # Center y should be average of top (100) and bottom (400) face
        # (100 + 400) / 2 = 250
        assert abs(cy - 250.0) < 0.1
    
    def test_coordinates_within_image_bounds(
        self, overlay_generator, mock_landmarks_2d
    ):
        """
        Test all guide coordinates fall within image dimensions.
        
        Requirement 9.5: Ensure all coordinates within image bounds
        """
        image_width = 640
        image_height = 480
        
        guides = overlay_generator.calculate_golden_ratio_guides(
            mock_landmarks_2d,
            image_width,
            image_height
        )
        
        # Check horizontal_thirds
        for x in guides['horizontal_thirds']:
            assert 0 <= x <= image_width
        
        # Check vertical_midline
        assert 0 <= guides['vertical_midline'] <= image_height
        
        # Check phi_spiral
        assert 0 <= guides['phi_spiral']['cx'] <= image_width
        assert 0 <= guides['phi_spiral']['cy'] <= image_height
    
    def test_different_image_dimensions(self, overlay_generator, mock_landmarks_2d):
        """
        Test with various image dimensions.
        
        Ensures calculations work correctly for different aspect ratios.
        """
        test_cases = [
            (640, 480),   # 4:3 aspect ratio
            (800, 600),   # 4:3 aspect ratio
            (1920, 1440), # 4:3 aspect ratio
            (640, 640),   # 1:1 aspect ratio (square)
            (720, 480),   # Different proportions
        ]
        
        for width, height in test_cases:
            guides = overlay_generator.calculate_golden_ratio_guides(
                mock_landmarks_2d,
                width,
                height
            )
            
            # Verify all values are within bounds
            assert all(0 <= x <= width for x in guides['horizontal_thirds'])
            assert 0 <= guides['vertical_midline'] <= height
            assert 0 <= guides['phi_spiral']['cx'] <= width
            assert 0 <= guides['phi_spiral']['cy'] <= height
    
    def test_invalid_landmarks_shape(self, overlay_generator):
        """
        Test error handling for invalid landmark shape.
        
        Should raise ValueError for incorrect shape.
        """
        invalid_landmarks = np.zeros((467, 2))  # Wrong number of landmarks
        
        with pytest.raises(ValueError, match="landmarks_2d must have shape"):
            overlay_generator.calculate_golden_ratio_guides(
                invalid_landmarks,
                640,
                480
            )
    
    def test_invalid_image_dimensions(self, overlay_generator, mock_landmarks_2d):
        """
        Test error handling for invalid image dimensions.
        
        Should raise ValueError for zero or negative dimensions.
        """
        with pytest.raises(ValueError, match="image_width and image_height must be positive"):
            overlay_generator.calculate_golden_ratio_guides(
                mock_landmarks_2d,
                0,
                480
            )
        
        with pytest.raises(ValueError, match="image_width and image_height must be positive"):
            overlay_generator.calculate_golden_ratio_guides(
                mock_landmarks_2d,
                640,
                0
            )
        
        with pytest.raises(ValueError, match="image_width and image_height must be positive"):
            overlay_generator.calculate_golden_ratio_guides(
                mock_landmarks_2d,
                -640,
                480
            )
    
    def test_return_types(self, overlay_generator, mock_landmarks_2d):
        """
        Test that all return values have correct types.
        
        All numeric values should be float type for consistency.
        """
        guides = overlay_generator.calculate_golden_ratio_guides(
            mock_landmarks_2d,
            640,
            480
        )
        
        # Check types
        assert isinstance(guides['horizontal_thirds'], list)
        assert isinstance(guides['vertical_midline'], float)
        assert isinstance(guides['phi_spiral'], dict)
        assert isinstance(guides['phi_spiral']['cx'], float)
        assert isinstance(guides['phi_spiral']['cy'], float)
        
        # All numeric values should be float
        assert all(isinstance(x, float) for x in guides['horizontal_thirds'])
    
    def test_thirds_ordering(self, overlay_generator, mock_landmarks_2d):
        """
        Test that horizontal thirds are in correct order (x1 < x2).
        """
        guides = overlay_generator.calculate_golden_ratio_guides(
            mock_landmarks_2d,
            640,
            480
        )
        
        x1, x2 = guides['horizontal_thirds']
        assert x1 < x2, "First third should be less than second third"
    
    def test_with_realistic_face_landmarks(self, overlay_generator):
        """
        Test with more realistic facial landmark data.
        
        Uses actual proportions that might occur in real face detection.
        """
        landmarks = np.zeros((468, 2), dtype=np.float32)
        
        # Realistic face dimensions (portrait orientation)
        landmarks[10] = [320.0, 80.0]    # Forehead top
        landmarks[152] = [320.0, 420.0]  # Chin bottom (340px height)
        landmarks[234] = [180.0, 250.0]  # Left face
        landmarks[454] = [460.0, 250.0]  # Right face (280px width)
        landmarks[6] = [320.0, 200.0]    # Nose bridge
        
        guides = overlay_generator.calculate_golden_ratio_guides(
            landmarks,
            640,
            500
        )
        
        # Verify all coordinates are reasonable
        assert 0 < guides['horizontal_thirds'][0] < guides['horizontal_thirds'][1] < 640
        assert 0 < guides['vertical_midline'] < 500
        assert 0 < guides['phi_spiral']['cx'] < 640
        assert 0 < guides['phi_spiral']['cy'] < 500
    
    def test_golden_ratio_grid_proportions(self, overlay_generator, mock_landmarks_2d):
        """
        Test that horizontal thirds approximate golden ratio divisions.
        
        The thirds should divide the image into sections that approximate
        golden ratio proportions (0.382 and 0.618 of the height).
        """
        image_width = 600
        image_height = 480
        
        landmarks = np.zeros((468, 2), dtype=np.float32)
        landmarks[10] = [300.0, 100.0]    # Face top
        landmarks[152] = [300.0, 400.0]   # Face bottom (300px height)
        landmarks[234] = [150.0, 250.0]   # Left face
        landmarks[454] = [450.0, 250.0]   # Right face
        landmarks[6] = [300.0, 220.0]     # Nose bridge
        
        guides = overlay_generator.calculate_golden_ratio_guides(
            landmarks,
            image_width,
            image_height
        )
        
        # Verify the thirds divide width correctly (1/3 and 2/3)
        x1, x2 = guides['horizontal_thirds']
        expected_x1 = image_width / 3.0
        expected_x2 = 2 * image_width / 3.0
        
        assert abs(x1 - expected_x1) < 0.1
        assert abs(x2 - expected_x2) < 0.1
