"""
Unit Tests for 2D Landmark Extraction

Tests the extract_2d_landmarks function which converts 3D landmarks to 2D
by dropping the z-coordinate and handling scaling for normalized vs pixel coordinates.

Requirement: 9.1
"""

import pytest
import numpy as np
from app.utils.landmark_extraction import extract_2d_landmarks


class TestExtract2DLandmarks:
    """Test suite for extract_2d_landmarks function."""
    
    def test_normalized_landmarks_scaling(self):
        """Test extraction with normalized landmarks (0-1 range)."""
        # Create mock normalized landmarks (0-1 range)
        landmarks_3d = np.random.uniform(0, 1, (468, 3)).astype(np.float32)
        
        image_width = 640
        image_height = 480
        
        landmarks_2d = extract_2d_landmarks(landmarks_3d, image_width, image_height)
        
        # Verify shape
        assert landmarks_2d.shape == (468, 2)
        
        # Verify all coordinates are within image bounds
        assert np.all(landmarks_2d[:, 0] >= 0)
        assert np.all(landmarks_2d[:, 0] <= image_width)
        assert np.all(landmarks_2d[:, 1] >= 0)
        assert np.all(landmarks_2d[:, 1] <= image_height)
        
        # Verify scaling occurred (max values should be close to image dimensions)
        assert np.max(landmarks_2d[:, 0]) <= image_width
        assert np.max(landmarks_2d[:, 1]) <= image_height
    
    def test_pixel_coordinates_no_scaling(self):
        """Test extraction with pixel coordinates (no scaling needed)."""
        # Create mock pixel coordinate landmarks
        image_width = 640
        image_height = 480
        
        landmarks_3d = np.random.uniform(
            [0, 0, -0.5],
            [image_width, image_height, 0.5],
            (468, 3)
        ).astype(np.float32)
        
        landmarks_2d = extract_2d_landmarks(landmarks_3d, image_width, image_height)
        
        # Verify shape
        assert landmarks_2d.shape == (468, 2)
        
        # Verify all coordinates are within image bounds
        assert np.all(landmarks_2d[:, 0] >= 0)
        assert np.all(landmarks_2d[:, 0] <= image_width)
        assert np.all(landmarks_2d[:, 1] >= 0)
        assert np.all(landmarks_2d[:, 1] <= image_height)
    
    def test_exact_normalized_coordinates(self):
        """Test with known normalized coordinates."""
        # Create known normalized landmarks
        landmarks_3d = np.zeros((468, 3), dtype=np.float32)
        
        # Set some specific test points
        landmarks_3d[0] = [0.0, 0.0, 0.0]      # Top-left corner
        landmarks_3d[1] = [1.0, 1.0, 0.0]      # Bottom-right corner
        landmarks_3d[2] = [0.5, 0.5, 0.0]      # Center
        
        image_width = 640
        image_height = 480
        
        landmarks_2d = extract_2d_landmarks(landmarks_3d, image_width, image_height)
        
        # Verify exact scaling
        assert landmarks_2d[0, 0] == 0.0       # x: 0.0 * 640 = 0.0
        assert landmarks_2d[0, 1] == 0.0       # y: 0.0 * 480 = 0.0
        
        assert landmarks_2d[1, 0] == 640.0     # x: 1.0 * 640 = 640.0
        assert landmarks_2d[1, 1] == 480.0     # y: 1.0 * 480 = 480.0
        
        assert landmarks_2d[2, 0] == 320.0     # x: 0.5 * 640 = 320.0
        assert landmarks_2d[2, 1] == 240.0     # y: 0.5 * 480 = 240.0
    
    def test_clipping_over_bounds_normalized(self):
        """Test that landmarks exceeding bounds are clipped (normalized)."""
        # Create landmarks with some values slightly over 1.0
        landmarks_3d = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 1.5, 0.0],  # Will scale to beyond image bounds
            [0.5, 0.5, 0.0],
        ], dtype=np.float32)
        
        # Pad to 468 landmarks
        landmarks_3d = np.vstack([
            landmarks_3d,
            np.random.uniform(0, 1, (465, 3)).astype(np.float32)
        ])
        
        image_width = 640
        image_height = 480
        
        landmarks_2d = extract_2d_landmarks(landmarks_3d, image_width, image_height)
        
        # Verify clipping occurred for out-of-bounds coordinates
        # 1.5 * 640 = 960, but should be clipped to 640
        assert landmarks_2d[1, 0] <= image_width
        assert landmarks_2d[1, 1] <= image_height
        
        # All values should be within bounds
        assert np.all(landmarks_2d[:, 0] >= 0)
        assert np.all(landmarks_2d[:, 0] <= image_width)
        assert np.all(landmarks_2d[:, 1] >= 0)
        assert np.all(landmarks_2d[:, 1] <= image_height)
    
    def test_clipping_over_bounds_pixel(self):
        """Test that pixel landmarks exceeding bounds are clipped."""
        landmarks_3d = np.array([
            [0.0, 0.0, 0.0],
            [700.0, 500.0, 0.0],  # Beyond image bounds
            [320.0, 240.0, 0.0],
        ], dtype=np.float32)
        
        # Pad to 468 landmarks
        landmarks_3d = np.vstack([
            landmarks_3d,
            np.random.uniform([0, 0, -0.5], [640, 480, 0.5], (465, 3)).astype(np.float32)
        ])
        
        image_width = 640
        image_height = 480
        
        landmarks_2d = extract_2d_landmarks(landmarks_3d, image_width, image_height)
        
        # Verify clipping occurred
        assert landmarks_2d[1, 0] == image_width  # Clipped to 640
        assert landmarks_2d[1, 1] == image_height  # Clipped to 480
        
        # All values should be within bounds
        assert np.all(landmarks_2d[:, 0] >= 0)
        assert np.all(landmarks_2d[:, 0] <= image_width)
        assert np.all(landmarks_2d[:, 1] >= 0)
        assert np.all(landmarks_2d[:, 1] <= image_height)
    
    def test_z_coordinate_dropped(self):
        """Test that z-coordinate is properly dropped."""
        # Create landmarks with distinct z values
        landmarks_3d = np.random.uniform(0, 1, (468, 3)).astype(np.float32)
        landmarks_3d[:, 2] = np.linspace(-1, 1, 468)  # Varying z values
        
        image_width = 640
        image_height = 480
        
        landmarks_2d = extract_2d_landmarks(landmarks_3d, image_width, image_height)
        
        # Verify shape is (468, 2) - no z coordinate
        assert landmarks_2d.shape == (468, 2)
        assert landmarks_2d.ndim == 2
        assert landmarks_2d.shape[1] == 2
    
    def test_different_image_dimensions(self):
        """Test with various image dimensions."""
        test_cases = [
            (320, 240),
            (1280, 720),
            (640, 480),
            (1024, 768),
            (300, 300),  # Square
            (1920, 1080),  # 16:9
        ]
        
        for width, height in test_cases:
            landmarks_3d = np.random.uniform(0, 1, (468, 3)).astype(np.float32)
            landmarks_2d = extract_2d_landmarks(landmarks_3d, width, height)
            
            # Verify bounds for all test cases
            assert np.all(landmarks_2d[:, 0] >= 0)
            assert np.all(landmarks_2d[:, 0] <= width)
            assert np.all(landmarks_2d[:, 1] >= 0)
            assert np.all(landmarks_2d[:, 1] <= height)
    
    def test_negative_coordinates_clamped(self):
        """Test that negative coordinates are clamped to 0."""
        landmarks_3d = np.array([
            [-0.1, -0.1, 0.0],  # Negative normalized coords
            [0.5, 0.5, 0.0],
        ], dtype=np.float32)
        
        # Pad to 468
        landmarks_3d = np.vstack([
            landmarks_3d,
            np.random.uniform(0, 1, (466, 3)).astype(np.float32)
        ])
        
        image_width = 640
        image_height = 480
        
        landmarks_2d = extract_2d_landmarks(landmarks_3d, image_width, image_height)
        
        # Negative values should be clamped to 0
        assert landmarks_2d[0, 0] == 0.0  # -0.1 * 640 = -64, clipped to 0
        assert landmarks_2d[0, 1] == 0.0  # -0.1 * 480 = -48, clipped to 0
        
        # All values should be >= 0
        assert np.all(landmarks_2d >= 0)
    
    def test_invalid_shape_raises_error(self):
        """Test that invalid input shape raises ValueError."""
        # Wrong number of landmarks
        landmarks_3d = np.random.rand(100, 3)
        with pytest.raises(ValueError, match="must have shape \\(468, 3\\)"):
            extract_2d_landmarks(landmarks_3d, 640, 480)
        
        # Wrong number of dimensions
        landmarks_3d = np.random.rand(468, 2)
        with pytest.raises(ValueError, match="must have shape \\(468, 3\\)"):
            extract_2d_landmarks(landmarks_3d, 640, 480)
        
        # 1D array
        landmarks_3d = np.random.rand(1404)  # 468 * 3
        with pytest.raises(ValueError, match="must have shape \\(468, 3\\)"):
            extract_2d_landmarks(landmarks_3d, 640, 480)
    
    def test_invalid_image_dimensions_raises_error(self):
        """Test that invalid image dimensions raise ValueError."""
        landmarks_3d = np.random.rand(468, 3)
        
        # Zero width
        with pytest.raises(ValueError, match="must be positive"):
            extract_2d_landmarks(landmarks_3d, 0, 480)
        
        # Zero height
        with pytest.raises(ValueError, match="must be positive"):
            extract_2d_landmarks(landmarks_3d, 640, 0)
        
        # Negative width
        with pytest.raises(ValueError, match="must be positive"):
            extract_2d_landmarks(landmarks_3d, -640, 480)
        
        # Negative height
        with pytest.raises(ValueError, match="must be positive"):
            extract_2d_landmarks(landmarks_3d, 640, -480)
    
    def test_output_dtype_is_float32(self):
        """Test that output dtype is float32."""
        landmarks_3d = np.random.rand(468, 3).astype(np.float32)
        landmarks_2d = extract_2d_landmarks(landmarks_3d, 640, 480)
        
        assert landmarks_2d.dtype == np.float32
    
    def test_no_mutation_of_input(self):
        """Test that input array is not mutated."""
        landmarks_3d = np.random.rand(468, 3).astype(np.float32)
        landmarks_3d_copy = landmarks_3d.copy()
        
        extract_2d_landmarks(landmarks_3d, 640, 480)
        
        # Verify input was not modified
        np.testing.assert_array_equal(landmarks_3d, landmarks_3d_copy)
    
    def test_edge_case_all_zeros(self):
        """Test with all zero landmarks."""
        landmarks_3d = np.zeros((468, 3), dtype=np.float32)
        landmarks_2d = extract_2d_landmarks(landmarks_3d, 640, 480)
        
        # All should be zero
        np.testing.assert_array_equal(landmarks_2d, np.zeros((468, 2)))
    
    def test_edge_case_all_ones(self):
        """Test with all ones (normalized max)."""
        landmarks_3d = np.ones((468, 3), dtype=np.float32)
        landmarks_2d = extract_2d_landmarks(landmarks_3d, 640, 480)
        
        # Should be scaled to image dimensions
        expected = np.array([[640.0, 480.0]] * 468, dtype=np.float32)
        np.testing.assert_array_almost_equal(landmarks_2d, expected)


class TestExtract2DLandmarksIntegration:
    """Integration tests for landmark extraction with overlay generation."""
    
    def test_extracted_landmarks_valid_for_overlay(self):
        """Test that extracted landmarks are suitable for overlay generation."""
        # Simulate MediaPipe output with normalized coordinates
        landmarks_3d = np.random.uniform(0, 1, (468, 3)).astype(np.float32)
        
        image_width = 640
        image_height = 480
        
        landmarks_2d = extract_2d_landmarks(landmarks_3d, image_width, image_height)
        
        # Key requirements for overlay generation:
        # 1. Exactly 468 landmarks
        assert landmarks_2d.shape[0] == 468
        
        # 2. Each landmark has x, y coordinates
        assert landmarks_2d.shape[1] == 2
        
        # 3. All coordinates within image bounds
        assert np.all(landmarks_2d >= 0)
        assert np.all(landmarks_2d[:, 0] <= image_width)
        assert np.all(landmarks_2d[:, 1] <= image_height)
        
        # 4. No NaN or Inf values
        assert not np.any(np.isnan(landmarks_2d))
        assert not np.any(np.isinf(landmarks_2d))
    
    def test_specific_landmark_indices_preserved(self):
        """Test that specific landmark indices maintain their positions."""
        # Create landmarks with known positions
        landmarks_3d = np.zeros((468, 3), dtype=np.float32)
        
        # Key landmarks for overlay
        landmarks_3d[10] = [0.1, 0.1, 0.0]    # Face top (forehead)
        landmarks_3d[152] = [0.1, 0.9, 0.0]   # Face bottom (chin)
        landmarks_3d[234] = [0.0, 0.5, 0.0]   # Left face contour
        landmarks_3d[454] = [1.0, 0.5, 0.0]   # Right face contour
        landmarks_3d[6] = [0.5, 0.5, 0.0]     # Nose bridge (spiral center)
        
        image_width = 640
        image_height = 480
        
        landmarks_2d = extract_2d_landmarks(landmarks_3d, image_width, image_height)
        
        # Verify key landmarks are preserved
        assert landmarks_2d[10, 0] == pytest.approx(0.1 * 640)
        assert landmarks_2d[10, 1] == pytest.approx(0.1 * 480)
        
        assert landmarks_2d[152, 0] == pytest.approx(0.1 * 640)
        assert landmarks_2d[152, 1] == pytest.approx(0.9 * 480)
        
        assert landmarks_2d[234, 0] == pytest.approx(0.0)
        assert landmarks_2d[234, 1] == pytest.approx(0.5 * 480)
        
        assert landmarks_2d[454, 0] == pytest.approx(640.0)
        assert landmarks_2d[454, 1] == pytest.approx(0.5 * 480)
        
        assert landmarks_2d[6, 0] == pytest.approx(0.5 * 640)
        assert landmarks_2d[6, 1] == pytest.approx(0.5 * 480)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
