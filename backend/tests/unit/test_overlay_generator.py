"""
Comprehensive unit tests for OverlayGenerator component.

Tests validate:
- Landmark count (exactly 468 2D points)
- Guide coordinate bounds (within image dimensions)
- Golden ratio divisions correctness
- 2D landmark extraction from 3D landmarks
- Vertical midline calculation
- Horizontal thirds using golden ratio proportions
- Phi spiral center positioning at nose bridge
- Bounds validation
- Edge cases (different image sizes, boundary conditions)
- Data consistency (multiple calls produce same results)

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

import pytest
import numpy as np
from app.components.overlay_generator import OverlayGenerator
from app.models.internal import Landmarks, OverlayDataInternal


class TestOverlayGeneratorLandmarkExtraction:
    """Tests for 2D landmark extraction from 3D landmarks (Requirement 9.1)."""
    
    @pytest.fixture
    def overlay_generator(self):
        """Create OverlayGenerator instance."""
        return OverlayGenerator()
    
    @pytest.fixture
    def landmarks_3d(self):
        """Create 468 3D landmarks with proper face geometry."""
        landmarks = np.zeros((468, 3), dtype=np.float32)
        # Random values for most landmarks
        landmarks[:, 0] = np.random.uniform(0, 640, 468)
        landmarks[:, 1] = np.random.uniform(0, 480, 468)
        landmarks[:, 2] = np.random.uniform(-0.5, 0.5, 468)
        
        # Set key facial landmarks for proper geometry
        landmarks[10] = [320.0, 100.0, 0.0]    # Forehead top
        landmarks[152] = [320.0, 400.0, 0.0]   # Chin bottom
        landmarks[234] = [200.0, 250.0, 0.0]   # Left face
        landmarks[454] = [440.0, 250.0, 0.0]   # Right face
        landmarks[6] = [320.0, 200.0, 0.0]     # Nose bridge
        
        return landmarks
    
    def test_landmark_count_exactly_468(self, overlay_generator, landmarks_3d):
        """
        Test that exactly 468 2D landmarks are extracted.
        
        **Validates: Requirement 9.1**
        
        The OverlayGenerator must convert all 468 3D landmarks to 2D points
        by dropping the z-coordinate.
        """
        landmarks = Landmarks(
            points=landmarks_3d,
            confidence=0.85,
            image_width=640,
            image_height=480
        )
        
        overlay_data = overlay_generator.generate(landmarks)
        
        # Verify exactly 468 landmarks
        assert overlay_data.landmarks_2d.shape[0] == 468, \
            f"Expected 468 landmarks, got {overlay_data.landmarks_2d.shape[0]}"
    
    def test_2d_landmarks_have_two_coordinates(self, overlay_generator, landmarks_3d):
        """
        Test that 2D landmarks have exactly 2 coordinates (x, y).
        
        **Validates: Requirement 9.1**
        """
        landmarks = Landmarks(
            points=landmarks_3d,
            confidence=0.85,
            image_width=640,
            image_height=480
        )
        
        overlay_data = overlay_generator.generate(landmarks)
        
        assert overlay_data.landmarks_2d.shape[1] == 2, \
            f"Expected 2 coordinates per landmark, got {overlay_data.landmarks_2d.shape[1]}"
    
    def test_2d_landmarks_are_from_xy_coordinates(self, overlay_generator):
        """
        Test that 2D landmarks correctly extract x and y from 3D landmarks.
        
        The z-coordinate should be dropped, preserving only x and y.
        """
        # Create landmarks with specific values
        landmarks_3d = np.zeros((468, 3), dtype=np.float32)
        
        # Set some landmarks with known values
        for i in range(468):
            landmarks_3d[i, 0] = float(i)  # x = index
            landmarks_3d[i, 1] = float(i * 2)  # y = index * 2
            landmarks_3d[i, 2] = float(i * 3)  # z = index * 3 (should be dropped)
        
        landmarks = Landmarks(
            points=landmarks_3d,
            confidence=0.85,
            image_width=1000,
            image_height=1000
        )
        
        overlay_data = overlay_generator.generate(landmarks)
        
        # Verify x and y are preserved correctly
        for i in range(468):
            assert overlay_data.landmarks_2d[i, 0] == float(i), \
                f"X coordinate mismatch at landmark {i}"
            assert overlay_data.landmarks_2d[i, 1] == float(i * 2), \
                f"Y coordinate mismatch at landmark {i}"
    
    def test_2d_landmarks_in_valid_pixel_range(self, overlay_generator):
        """
        Test that 2D landmarks are within valid image pixel ranges.
        
        All landmark coordinates should be within [0, image_dimension].
        """
        width, height = 640, 480
        
        # Create landmarks that should already be in pixel coordinates
        landmarks_3d = np.zeros((468, 3), dtype=np.float32)
        landmarks_3d[:, 0] = np.random.uniform(0, width, 468)
        landmarks_3d[:, 1] = np.random.uniform(0, height, 468)
        landmarks_3d[:, 2] = np.random.uniform(-0.5, 0.5, 468)
        
        landmarks = Landmarks(
            points=landmarks_3d,
            confidence=0.85,
            image_width=width,
            image_height=height
        )
        
        overlay_data = overlay_generator.generate(landmarks)
        
        # All x coordinates should be in [0, width]
        assert np.all(overlay_data.landmarks_2d[:, 0] >= 0), \
            "Some x coordinates are negative"
        assert np.all(overlay_data.landmarks_2d[:, 0] <= width), \
            "Some x coordinates exceed image width"
        
        # All y coordinates should be in [0, height]
        assert np.all(overlay_data.landmarks_2d[:, 1] >= 0), \
            "Some y coordinates are negative"
        assert np.all(overlay_data.landmarks_2d[:, 1] <= height), \
            "Some y coordinates exceed image height"


class TestOverlayGeneratorVerticalMidline:
    """Tests for vertical midline calculation (Requirement 9.3)."""
    
    @pytest.fixture
    def overlay_generator(self):
        return OverlayGenerator()
    
    def test_midline_at_center_x_coordinate(self, overlay_generator):
        """
        Test that vertical midline is at the center x-coordinate of the face.
        
        **Validates: Requirement 9.3**
        
        The midline should be the average of left and right face contours.
        """
        landmarks_3d = np.zeros((468, 3), dtype=np.float32)
        
        # Set left and right face landmarks
        left_face_x = 150.0
        right_face_x = 450.0
        expected_midline = (left_face_x + right_face_x) / 2.0
        
        landmarks_3d[234] = [left_face_x, 250.0, 0.0]  # Left face
        landmarks_3d[454] = [right_face_x, 250.0, 0.0]  # Right face
        landmarks_3d[10] = [300.0, 100.0, 0.0]   # Face top
        landmarks_3d[152] = [300.0, 400.0, 0.0]  # Face bottom
        
        landmarks = Landmarks(
            points=landmarks_3d,
            confidence=0.85,
            image_width=640,
            image_height=480
        )
        
        overlay_data = overlay_generator.generate(landmarks)
        
        assert abs(overlay_data.midline_x - expected_midline) < 0.1, \
            f"Expected midline at {expected_midline}, got {overlay_data.midline_x}"
    
    def test_midline_within_image_bounds(self, overlay_generator):
        """
        Test that midline x-coordinate is within image width bounds.
        
        **Validates: Requirement 9.5**
        """
        width = 640
        
        landmarks_3d = np.zeros((468, 3), dtype=np.float32)
        landmarks_3d[234] = [50.0, 250.0, 0.0]   # Left face (5% width)
        landmarks_3d[454] = [590.0, 250.0, 0.0]  # Right face (92% width)
        landmarks_3d[10] = [320.0, 100.0, 0.0]   # Face top
        landmarks_3d[152] = [320.0, 400.0, 0.0]  # Face bottom
        
        landmarks = Landmarks(
            points=landmarks_3d,
            confidence=0.85,
            image_width=width,
            image_height=480
        )
        
        overlay_data = overlay_generator.generate(landmarks)
        
        assert 0 <= overlay_data.midline_x <= width, \
            f"Midline {overlay_data.midline_x} outside bounds [0, {width}]"
    
    def test_midline_with_symmetric_face(self, overlay_generator):
        """
        Test midline calculation with perfectly symmetric face.
        """
        landmarks_3d = np.zeros((468, 3), dtype=np.float32)
        
        center_x = 320.0
        left_offset = 100.0
        
        landmarks_3d[234] = [center_x - left_offset, 250.0, 0.0]  # Left
        landmarks_3d[454] = [center_x + left_offset, 250.0, 0.0]  # Right
        landmarks_3d[10] = [320.0, 100.0, 0.0]  # Face top
        landmarks_3d[152] = [320.0, 400.0, 0.0]  # Face bottom
        
        landmarks = Landmarks(
            points=landmarks_3d,
            confidence=0.85,
            image_width=640,
            image_height=480
        )
        
        overlay_data = overlay_generator.generate(landmarks)
        
        assert abs(overlay_data.midline_x - center_x) < 0.1, \
            f"Midline should be at center {center_x}, got {overlay_data.midline_x}"


class TestOverlayGeneratorHorizontalThirds:
    """Tests for horizontal thirds using golden ratio divisions (Requirement 9.2)."""
    
    @pytest.fixture
    def overlay_generator(self):
        return OverlayGenerator()
    
    def test_horizontal_thirds_count_is_four(self, overlay_generator):
        """
        Test that exactly 4 y-values are returned for horizontal thirds.
        
        **Validates: Requirement 9.2**
        
        The four values represent:
        - 0: Face top (forehead)
        - 1: Upper third (at ~38.2% of face height using φ)
        - 2: Lower third (at ~61.8% of face height using φ)
        - 3: Face bottom (chin)
        """
        landmarks_3d = np.zeros((468, 3), dtype=np.float32)
        landmarks_3d[10] = [320.0, 100.0, 0.0]   # Face top
        landmarks_3d[152] = [320.0, 400.0, 0.0]  # Face bottom
        
        landmarks = Landmarks(
            points=landmarks_3d,
            confidence=0.85,
            image_width=640,
            image_height=480
        )
        
        overlay_data = overlay_generator.generate(landmarks)
        
        assert len(overlay_data.horizontal_thirds_y) == 4, \
            f"Expected 4 thirds, got {len(overlay_data.horizontal_thirds_y)}"
    
    def test_horizontal_thirds_use_golden_ratio_proportions(self, overlay_generator):
        """
        Test that horizontal thirds divide face using golden ratio.
        
        **Validates: Requirement 9.2**
        
        Using golden ratio (φ ≈ 1.618):
        - First third at 1/φ ≈ 0.382 of face height
        - Second third at φ/(φ+1) ≈ 0.618 of face height
        """
        landmarks_3d = np.zeros((468, 3), dtype=np.float32)
        
        # Face from y=100 to y=400 (height = 300)
        face_top_y = 100.0
        face_bottom_y = 400.0
        face_height = face_bottom_y - face_top_y
        
        landmarks_3d[10] = [320.0, face_top_y, 0.0]   # Face top
        landmarks_3d[152] = [320.0, face_bottom_y, 0.0]  # Face bottom
        
        landmarks = Landmarks(
            points=landmarks_3d,
            confidence=0.85,
            image_width=640,
            image_height=480
        )
        
        overlay_data = overlay_generator.generate(landmarks)
        
        thirds = overlay_data.horizontal_thirds_y
        
        # Third 0 should be at face_top_y
        assert abs(thirds[0] - face_top_y) < 0.1
        
        # Third 1 should be at face_top + 0.382 * height
        expected_third_1 = face_top_y + face_height * 0.382
        assert abs(thirds[1] - expected_third_1) < 1.0, \
            f"Expected third 1 at {expected_third_1}, got {thirds[1]}"
        
        # Third 2 should be at face_top + 0.618 * height
        expected_third_2 = face_top_y + face_height * 0.618
        assert abs(thirds[2] - expected_third_2) < 1.0, \
            f"Expected third 2 at {expected_third_2}, got {thirds[2]}"
        
        # Third 3 should be at face_bottom
        assert abs(thirds[3] - face_bottom_y) < 0.1
    
    def test_horizontal_thirds_in_ascending_order(self, overlay_generator):
        """
        Test that horizontal thirds are in strictly ascending order.
        
        This ensures the divisions properly partition the face vertically.
        """
        landmarks_3d = np.zeros((468, 3), dtype=np.float32)
        landmarks_3d[10] = [320.0, 100.0, 0.0]
        landmarks_3d[152] = [320.0, 400.0, 0.0]
        
        landmarks = Landmarks(
            points=landmarks_3d,
            confidence=0.85,
            image_width=640,
            image_height=480
        )
        
        overlay_data = overlay_generator.generate(landmarks)
        
        thirds = overlay_data.horizontal_thirds_y
        
        # Verify each value is less than the next
        for i in range(len(thirds) - 1):
            assert thirds[i] < thirds[i + 1], \
                f"Third {i} ({thirds[i]}) should be less than third {i+1} ({thirds[i+1]})"
    
    def test_horizontal_thirds_within_image_bounds(self, overlay_generator):
        """
        Test that all third y-coordinates fall within image height.
        
        **Validates: Requirement 9.5**
        """
        height = 480
        
        landmarks_3d = np.zeros((468, 3), dtype=np.float32)
        landmarks_3d[10] = [320.0, 50.0, 0.0]      # Face top (near top)
        landmarks_3d[152] = [320.0, 430.0, 0.0]    # Face bottom (near bottom)
        
        landmarks = Landmarks(
            points=landmarks_3d,
            confidence=0.85,
            image_width=640,
            image_height=height
        )
        
        overlay_data = overlay_generator.generate(landmarks)
        
        for i, y in enumerate(overlay_data.horizontal_thirds_y):
            assert 0 <= y <= height, \
                f"Third {i} y-coordinate {y} outside bounds [0, {height}]"


class TestOverlayGeneratorPhiSpiralCenter:
    """Tests for phi spiral center positioning (Requirement 9.4)."""
    
    @pytest.fixture
    def overlay_generator(self):
        return OverlayGenerator()
    
    def test_phi_spiral_center_is_tuple(self, overlay_generator):
        """
        Test that phi spiral center is returned as (x, y) tuple.
        
        **Validates: Requirement 9.4**
        """
        landmarks_3d = np.zeros((468, 3), dtype=np.float32)
        landmarks_3d[6] = [320.0, 200.0, 0.0]     # Nose bridge
        landmarks_3d[234] = [200.0, 250.0, 0.0]   # Left face
        landmarks_3d[454] = [440.0, 250.0, 0.0]   # Right face
        landmarks_3d[10] = [320.0, 100.0, 0.0]    # Face top
        landmarks_3d[152] = [320.0, 400.0, 0.0]   # Face bottom
        
        landmarks = Landmarks(
            points=landmarks_3d,
            confidence=0.85,
            image_width=640,
            image_height=480
        )
        
        overlay_data = overlay_generator.generate(landmarks)
        
        assert isinstance(overlay_data.phi_spiral_center, tuple), \
            f"Expected tuple, got {type(overlay_data.phi_spiral_center)}"
        assert len(overlay_data.phi_spiral_center) == 2, \
            f"Expected 2-element tuple, got {len(overlay_data.phi_spiral_center)}"
    
    def test_phi_spiral_center_at_nose_bridge(self, overlay_generator):
        """
        Test that phi spiral center is positioned at nose bridge.
        
        **Validates: Requirement 9.4**
        
        The nose bridge (landmark 6) is the focal point for golden ratio spirals.
        """
        landmarks_3d = np.zeros((468, 3), dtype=np.float32)
        
        nose_bridge_x = 320.0
        nose_bridge_y = 200.0
        
        landmarks_3d[6] = [nose_bridge_x, nose_bridge_y, 0.2]  # Nose bridge
        landmarks_3d[234] = [200.0, 250.0, 0.0]   # Left face
        landmarks_3d[454] = [440.0, 250.0, 0.0]   # Right face
        landmarks_3d[10] = [320.0, 100.0, 0.0]    # Face top
        landmarks_3d[152] = [320.0, 400.0, 0.0]   # Face bottom
        
        landmarks = Landmarks(
            points=landmarks_3d,
            confidence=0.85,
            image_width=640,
            image_height=480
        )
        
        overlay_data = overlay_generator.generate(landmarks)
        
        cx, cy = overlay_data.phi_spiral_center
        
        assert abs(cx - nose_bridge_x) < 0.1, \
            f"Expected spiral center x at {nose_bridge_x}, got {cx}"
        assert abs(cy - nose_bridge_y) < 0.1, \
            f"Expected spiral center y at {nose_bridge_y}, got {cy}"
    
    def test_phi_spiral_center_within_image_bounds(self, overlay_generator):
        """
        Test that phi spiral center coordinates are within image bounds.
        
        **Validates: Requirement 9.5**
        """
        width, height = 640, 480
        
        landmarks_3d = np.zeros((468, 3), dtype=np.float32)
        landmarks_3d[6] = [320.0, 200.0, 0.0]     # Nose bridge (center)
        landmarks_3d[234] = [50.0, 250.0, 0.0]    # Left face
        landmarks_3d[454] = [590.0, 250.0, 0.0]   # Right face
        landmarks_3d[10] = [320.0, 100.0, 0.0]    # Face top
        landmarks_3d[152] = [320.0, 400.0, 0.0]   # Face bottom
        
        landmarks = Landmarks(
            points=landmarks_3d,
            confidence=0.85,
            image_width=width,
            image_height=height
        )
        
        overlay_data = overlay_generator.generate(landmarks)
        
        cx, cy = overlay_data.phi_spiral_center
        
        assert 0 <= cx <= width, \
            f"Spiral center x {cx} outside bounds [0, {width}]"
        assert 0 <= cy <= height, \
            f"Spiral center y {cy} outside bounds [0, {height}]"
    
    def test_phi_spiral_center_at_different_positions(self, overlay_generator):
        """
        Test phi spiral center with nose bridge at various positions.
        """
        test_positions = [
            (100.0, 50.0),
            (320.0, 200.0),
            (500.0, 350.0),
            (50.0, 100.0),
        ]
        
        for nose_x, nose_y in test_positions:
            landmarks_3d = np.zeros((468, 3), dtype=np.float32)
            landmarks_3d[6] = [nose_x, nose_y, 0.0]
            landmarks_3d[234] = [50.0, 250.0, 0.0]
            landmarks_3d[454] = [590.0, 250.0, 0.0]
            landmarks_3d[10] = [320.0, 100.0, 0.0]
            landmarks_3d[152] = [320.0, 400.0, 0.0]
            
            landmarks = Landmarks(
                points=landmarks_3d,
                confidence=0.85,
                image_width=640,
                image_height=480
            )
            
            overlay_data = overlay_generator.generate(landmarks)
            
            cx, cy = overlay_data.phi_spiral_center
            
            assert abs(cx - nose_x) < 0.1
            assert abs(cy - nose_y) < 0.1


class TestOverlayGeneratorBoundsValidation:
    """Tests for coordinate bounds validation (Requirement 9.5)."""
    
    @pytest.fixture
    def overlay_generator(self):
        return OverlayGenerator()
    
    def test_all_landmark_coordinates_within_bounds(self, overlay_generator):
        """
        Test that all 468 landmark coordinates are within image dimensions.
        
        **Validates: Requirement 9.5**
        """
        width, height = 640, 480
        
        landmarks_3d = np.random.rand(468, 3) * 100
        landmarks_3d[:, 0] = np.random.uniform(10, width - 10, 468)
        landmarks_3d[:, 1] = np.random.uniform(10, height - 10, 468)
        
        # Ensure key landmarks are set properly
        landmarks_3d[10] = [320.0, 100.0, 0.0]   # Face top
        landmarks_3d[152] = [320.0, 400.0, 0.0]  # Face bottom
        landmarks_3d[234] = [200.0, 250.0, 0.0]  # Left face
        landmarks_3d[454] = [440.0, 250.0, 0.0]  # Right face
        landmarks_3d[6] = [320.0, 200.0, 0.0]    # Nose bridge
        
        landmarks = Landmarks(
            points=landmarks_3d.astype(np.float32),
            confidence=0.85,
            image_width=width,
            image_height=height
        )
        
        overlay_data = overlay_generator.generate(landmarks)
        
        # Check all landmarks
        assert np.all(overlay_data.landmarks_2d[:, 0] >= 0), \
            "Some x coordinates are negative"
        assert np.all(overlay_data.landmarks_2d[:, 0] <= width), \
            "Some x coordinates exceed image width"
        assert np.all(overlay_data.landmarks_2d[:, 1] >= 0), \
            "Some y coordinates are negative"
        assert np.all(overlay_data.landmarks_2d[:, 1] <= height), \
            "Some y coordinates exceed image height"
    
    def test_all_guide_coordinates_within_bounds(self, overlay_generator):
        """
        Test that all golden ratio guide coordinates are within bounds.
        
        Includes midline, thirds, and spiral center.
        """
        width, height = 640, 480
        
        landmarks_3d = np.zeros((468, 3), dtype=np.float32)
        landmarks_3d[10] = [320.0, 100.0, 0.0]
        landmarks_3d[152] = [320.0, 400.0, 0.0]
        landmarks_3d[234] = [150.0, 250.0, 0.0]
        landmarks_3d[454] = [490.0, 250.0, 0.0]
        landmarks_3d[6] = [320.0, 200.0, 0.0]
        
        landmarks = Landmarks(
            points=landmarks_3d,
            confidence=0.85,
            image_width=width,
            image_height=height
        )
        
        overlay_data = overlay_generator.generate(landmarks)
        
        # Check midline
        assert 0 <= overlay_data.midline_x <= width
        
        # Check thirds
        for y in overlay_data.horizontal_thirds_y:
            assert 0 <= y <= height
        
        # Check spiral center
        cx, cy = overlay_data.phi_spiral_center
        assert 0 <= cx <= width
        assert 0 <= cy <= height
    
    def test_bounds_with_extreme_image_sizes(self, overlay_generator):
        """
        Test coordinate bounds with extreme image dimensions.
        """
        test_cases = [
            (100, 100),      # Very small
            (4000, 3000),    # Very large
            (1920, 1080),    # Common large size
            (320, 240),      # Small size
        ]
        
        for width, height in test_cases:
            landmarks_3d = np.zeros((468, 3), dtype=np.float32)
            landmarks_3d[:, 0] = np.random.uniform(0, width, 468)
            landmarks_3d[:, 1] = np.random.uniform(0, height, 468)
            
            landmarks_3d[10] = [width/2, height*0.15, 0.0]
            landmarks_3d[152] = [width/2, height*0.85, 0.0]
            landmarks_3d[234] = [width*0.2, height*0.5, 0.0]
            landmarks_3d[454] = [width*0.8, height*0.5, 0.0]
            landmarks_3d[6] = [width/2, height*0.4, 0.0]
            
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


class TestOverlayGeneratorEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    @pytest.fixture
    def overlay_generator(self):
        return OverlayGenerator()
    
    def test_face_at_image_edges(self, overlay_generator):
        """
        Test with face positioned at image edges.
        """
        width, height = 640, 480
        
        landmarks_3d = np.zeros((468, 3), dtype=np.float32)
        
        # Face at top-left
        landmarks_3d[10] = [50.0, 20.0, 0.0]      # Top
        landmarks_3d[152] = [50.0, 200.0, 0.0]    # Bottom
        landmarks_3d[234] = [20.0, 110.0, 0.0]    # Left
        landmarks_3d[454] = [80.0, 110.0, 0.0]    # Right
        landmarks_3d[6] = [50.0, 80.0, 0.0]       # Nose
        
        landmarks = Landmarks(
            points=landmarks_3d,
            confidence=0.85,
            image_width=width,
            image_height=height
        )
        
        overlay_data = overlay_generator.generate(landmarks)
        
        # Verify all coordinates are still within bounds
        assert np.all(overlay_data.landmarks_2d[:, 0] >= 0)
        assert np.all(overlay_data.landmarks_2d[:, 0] <= width)
        assert np.all(overlay_data.landmarks_2d[:, 1] >= 0)
        assert np.all(overlay_data.landmarks_2d[:, 1] <= height)
    
    def test_small_face(self, overlay_generator):
        """
        Test with very small face relative to image.
        """
        width, height = 1920, 1440
        
        landmarks_3d = np.zeros((468, 3), dtype=np.float32)
        
        # Small face centered
        landmarks_3d[10] = [960.0, 500.0, 0.0]
        landmarks_3d[152] = [960.0, 700.0, 0.0]
        landmarks_3d[234] = [900.0, 600.0, 0.0]
        landmarks_3d[454] = [1020.0, 600.0, 0.0]
        landmarks_3d[6] = [960.0, 600.0, 0.0]
        
        landmarks = Landmarks(
            points=landmarks_3d,
            confidence=0.85,
            image_width=width,
            image_height=height
        )
        
        overlay_data = overlay_generator.generate(landmarks)
        
        # Should still have valid output
        assert overlay_data.landmarks_2d.shape == (468, 2)
        assert len(overlay_data.horizontal_thirds_y) == 4
        assert isinstance(overlay_data.phi_spiral_center, tuple)
    
    def test_large_face_relative_to_image(self, overlay_generator):
        """
        Test with large face relative to image size.
        """
        width, height = 320, 240
        
        landmarks_3d = np.zeros((468, 3), dtype=np.float32)
        
        # Large face that fills most of image
        landmarks_3d[10] = [160.0, 30.0, 0.0]
        landmarks_3d[152] = [160.0, 210.0, 0.0]
        landmarks_3d[234] = [30.0, 120.0, 0.0]
        landmarks_3d[454] = [290.0, 120.0, 0.0]
        landmarks_3d[6] = [160.0, 100.0, 0.0]
        
        landmarks = Landmarks(
            points=landmarks_3d,
            confidence=0.85,
            image_width=width,
            image_height=height
        )
        
        overlay_data = overlay_generator.generate(landmarks)
        
        # Verify bounds are still respected
        assert np.all(overlay_data.landmarks_2d[:, 0] >= 0)
        assert np.all(overlay_data.landmarks_2d[:, 0] <= width)
        assert np.all(overlay_data.landmarks_2d[:, 1] >= 0)
        assert np.all(overlay_data.landmarks_2d[:, 1] <= height)
    
    def test_perfectly_square_image(self, overlay_generator):
        """
        Test with square image (width == height).
        """
        size = 512
        
        landmarks_3d = np.zeros((468, 3), dtype=np.float32)
        landmarks_3d[10] = [256.0, 100.0, 0.0]
        landmarks_3d[152] = [256.0, 400.0, 0.0]
        landmarks_3d[234] = [156.0, 250.0, 0.0]
        landmarks_3d[454] = [356.0, 250.0, 0.0]
        landmarks_3d[6] = [256.0, 200.0, 0.0]
        
        landmarks = Landmarks(
            points=landmarks_3d,
            confidence=0.85,
            image_width=size,
            image_height=size
        )
        
        overlay_data = overlay_generator.generate(landmarks)
        
        # Should work correctly with square image
        assert overlay_data.landmarks_2d.shape == (468, 2)
        assert 0 <= overlay_data.midline_x <= size


class TestOverlayGeneratorDataConsistency:
    """Tests for data consistency (Requirement 9.5)."""
    
    @pytest.fixture
    def overlay_generator(self):
        return OverlayGenerator()
    
    def test_multiple_calls_with_same_data_produce_same_results(
        self, overlay_generator
    ):
        """
        Test that calling generate multiple times with the same data
        produces identical results.
        
        **Validates: Data consistency requirement**
        """
        landmarks_3d = np.zeros((468, 3), dtype=np.float32)
        landmarks_3d[:, 0] = np.arange(468) * 0.1
        landmarks_3d[:, 1] = np.arange(468) * 0.1
        landmarks_3d[10] = [320.0, 100.0, 0.0]
        landmarks_3d[152] = [320.0, 400.0, 0.0]
        landmarks_3d[234] = [200.0, 250.0, 0.0]
        landmarks_3d[454] = [440.0, 250.0, 0.0]
        landmarks_3d[6] = [320.0, 200.0, 0.0]
        
        landmarks = Landmarks(
            points=landmarks_3d,
            confidence=0.85,
            image_width=640,
            image_height=480
        )
        
        # Generate multiple times
        result1 = overlay_generator.generate(landmarks)
        result2 = overlay_generator.generate(landmarks)
        result3 = overlay_generator.generate(landmarks)
        
        # Compare landmarks
        np.testing.assert_array_almost_equal(
            result1.landmarks_2d, result2.landmarks_2d
        )
        np.testing.assert_array_almost_equal(
            result2.landmarks_2d, result3.landmarks_2d
        )
        
        # Compare midline
        assert abs(result1.midline_x - result2.midline_x) < 0.001
        assert abs(result2.midline_x - result3.midline_x) < 0.001
        
        # Compare thirds
        for i in range(4):
            assert abs(result1.horizontal_thirds_y[i] - result2.horizontal_thirds_y[i]) < 0.001
            assert abs(result2.horizontal_thirds_y[i] - result3.horizontal_thirds_y[i]) < 0.001
        
        # Compare spiral center
        assert abs(result1.phi_spiral_center[0] - result2.phi_spiral_center[0]) < 0.001
        assert abs(result1.phi_spiral_center[1] - result2.phi_spiral_center[1]) < 0.001
        assert abs(result2.phi_spiral_center[0] - result3.phi_spiral_center[0]) < 0.001
        assert abs(result2.phi_spiral_center[1] - result3.phi_spiral_center[1]) < 0.001
    
    def test_no_nan_or_inf_values_in_output(self, overlay_generator):
        """
        Test that output never contains NaN or infinity values.
        """
        landmarks_3d = np.random.rand(468, 3).astype(np.float32) * 100
        landmarks_3d[10] = [320.0, 100.0, 0.0]
        landmarks_3d[152] = [320.0, 400.0, 0.0]
        landmarks_3d[234] = [200.0, 250.0, 0.0]
        landmarks_3d[454] = [440.0, 250.0, 0.0]
        landmarks_3d[6] = [320.0, 200.0, 0.0]
        
        landmarks = Landmarks(
            points=landmarks_3d,
            confidence=0.85,
            image_width=640,
            image_height=480
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
    
    def test_output_types_are_correct(self, overlay_generator):
        """
        Test that output has correct data types for JSON serialization.
        """
        landmarks_3d = np.zeros((468, 3), dtype=np.float32)
        landmarks_3d[10] = [320.0, 100.0, 0.0]
        landmarks_3d[152] = [320.0, 400.0, 0.0]
        landmarks_3d[234] = [200.0, 250.0, 0.0]
        landmarks_3d[454] = [440.0, 250.0, 0.0]
        landmarks_3d[6] = [320.0, 200.0, 0.0]
        
        landmarks = Landmarks(
            points=landmarks_3d,
            confidence=0.85,
            image_width=640,
            image_height=480
        )
        
        overlay_data = overlay_generator.generate(landmarks)
        
        # Verify types
        assert isinstance(overlay_data.landmarks_2d, np.ndarray)
        assert isinstance(overlay_data.midline_x, float)
        assert isinstance(overlay_data.horizontal_thirds_y, list)
        assert all(isinstance(y, (int, float)) for y in overlay_data.horizontal_thirds_y)
        assert isinstance(overlay_data.phi_spiral_center, tuple)
        assert all(isinstance(v, (int, float)) for v in overlay_data.phi_spiral_center)


class TestOverlayGeneratorIntegration:
    """Integration tests combining multiple requirements."""
    
    @pytest.fixture
    def overlay_generator(self):
        return OverlayGenerator()
    
    def test_complete_overlay_generation_workflow(self, overlay_generator):
        """
        Test complete workflow: landmarks → overlay data.
        
        **Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5**
        """
        # Create realistic landmarks
        landmarks_3d = np.zeros((468, 3), dtype=np.float32)
        landmarks_3d[:, 0] = np.random.uniform(0, 640, 468)
        landmarks_3d[:, 1] = np.random.uniform(0, 480, 468)
        landmarks_3d[:, 2] = np.random.uniform(-0.5, 0.5, 468)
        
        # Set key landmarks
        landmarks_3d[10] = [320.0, 80.0, 0.0]      # Top
        landmarks_3d[152] = [320.0, 420.0, 0.0]    # Bottom
        landmarks_3d[234] = [180.0, 250.0, 0.0]    # Left
        landmarks_3d[454] = [460.0, 250.0, 0.0]    # Right
        landmarks_3d[6] = [320.0, 200.0, 0.2]      # Nose
        
        landmarks = Landmarks(
            points=landmarks_3d,
            confidence=0.85,
            image_width=640,
            image_height=480
        )
        
        overlay_data = overlay_generator.generate(landmarks)
        
        # Validate all requirements in one test
        # Requirement 9.1: 468 2D landmarks
        assert overlay_data.landmarks_2d.shape == (468, 2)
        
        # Requirement 9.2: 4 horizontal thirds
        assert len(overlay_data.horizontal_thirds_y) == 4
        assert all(overlay_data.horizontal_thirds_y[i] < overlay_data.horizontal_thirds_y[i+1]
                   for i in range(3))
        
        # Requirement 9.3: Vertical midline present
        assert isinstance(overlay_data.midline_x, float)
        
        # Requirement 9.4: Phi spiral center
        assert isinstance(overlay_data.phi_spiral_center, tuple)
        assert len(overlay_data.phi_spiral_center) == 2
        
        # Requirement 9.5: All coordinates within bounds
        assert np.all(overlay_data.landmarks_2d >= 0)
        assert np.all(overlay_data.landmarks_2d[:, 0] <= 640)
        assert np.all(overlay_data.landmarks_2d[:, 1] <= 480)
        assert 0 <= overlay_data.midline_x <= 640
        assert all(0 <= y <= 480 for y in overlay_data.horizontal_thirds_y)
        cx, cy = overlay_data.phi_spiral_center
        assert 0 <= cx <= 640 and 0 <= cy <= 480


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
