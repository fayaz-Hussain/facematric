"""
Standalone test for resize functionality without importing MediaPipe.
Tests PIL resize implementation directly.
"""

import pytest
from PIL import Image
from io import BytesIO
import numpy as np


def create_test_image(width: int, height: int, color: tuple = (128, 128, 128)) -> Image.Image:
    """Create test image with specified dimensions."""
    return Image.new('RGB', (width, height), color)


def resize_image(image: Image.Image, max_dimension: int = 640) -> Image.Image:
    """
    Resize image maintaining aspect ratio (same logic as ImageProcessor).
    
    Args:
        image: PIL Image object
        max_dimension: Maximum dimension (default 640)
    
    Returns:
        Resized PIL Image object
    """
    width, height = image.size
    
    # Calculate scale factor
    scale_factor = min(
        max_dimension / width,
        max_dimension / height
    )
    
    # Only resize if image is too large
    if scale_factor < 1.0:
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Use LANCZOS for high-quality downsampling
        image = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )
    
    return image


class TestResizeLogic:
    """Test resize logic in isolation."""
    
    def test_resize_width_exceeds_640(self):
        """Test resizing when width exceeds 640px."""
        image = create_test_image(800, 400)
        resized = resize_image(image)
        
        # Scale factor should be 640/800 = 0.8
        # New dimensions: 640x320
        assert resized.width == 640
        assert resized.height == 320
        
        # Aspect ratio preserved
        original_aspect = 800 / 400
        new_aspect = resized.width / resized.height
        assert abs(original_aspect - new_aspect) < 0.01
    
    def test_resize_height_exceeds_640(self):
        """Test resizing when height exceeds 640px."""
        image = create_test_image(400, 1000)
        resized = resize_image(image)
        
        # Scale factor should be 640/1000 = 0.64
        # New dimensions: 256x640
        assert resized.width == 256
        assert resized.height == 640
        
        # Aspect ratio preserved
        original_aspect = 400 / 1000
        new_aspect = resized.width / resized.height
        assert abs(original_aspect - new_aspect) < 0.01
    
    def test_resize_square_image(self):
        """Test resizing square image."""
        image = create_test_image(1000, 1000)
        resized = resize_image(image)
        
        # Should be 640x640
        assert resized.width == 640
        assert resized.height == 640
    
    def test_no_resize_when_within_640(self):
        """Test that images within 640px are NOT resized."""
        image = create_test_image(500, 400)
        resized = resize_image(image)
        
        # Should NOT be resized
        assert resized.width == 500
        assert resized.height == 400
    
    def test_resize_various_aspect_ratios(self):
        """Test aspect ratio preservation across various sizes."""
        test_cases = [
            (1920, 1080),  # 16:9 aspect ratio
            (1200, 800),   # 3:2 aspect ratio
            (1000, 1500),  # 2:3 aspect ratio (portrait)
            (2000, 1000),  # 2:1 aspect ratio (panoramic)
        ]
        
        for width, height in test_cases:
            image = create_test_image(width, height)
            resized = resize_image(image)
            
            # Calculate aspect ratios
            original_aspect = width / height
            new_aspect = resized.width / resized.height
            
            # Aspect ratio should be preserved
            assert abs(original_aspect - new_aspect) < 0.01, (
                f"Aspect ratio not preserved for {width}x{height}: "
                f"original={original_aspect:.3f}, new={new_aspect:.3f}"
            )
            
            # Dimensions should be <= 640
            assert resized.width <= 640 and resized.height <= 640
            
            # At least one dimension should be 640 (the constraining dimension)
            if width > 640 or height > 640:
                assert resized.width == 640 or resized.height == 640
    
    def test_scale_factor_calculation(self):
        """Test that scale factor is correctly calculated."""
        image = create_test_image(1280, 720)
        resized = resize_image(image)
        
        # Scale factor should be min(640/1280, 640/720) = min(0.5, 0.889) = 0.5
        # New dimensions: 640x360
        assert resized.width == 640
        assert resized.height == 360
    
    def test_lanczos_resampling_preserves_detail(self):
        """Test that Lanczos resampling produces high-quality results."""
        # Create checkerboard pattern
        width, height = 1000, 1000
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        square_size = 50
        for i in range(0, height, square_size):
            for j in range(0, width, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    img_array[i:i+square_size, j:j+square_size] = [255, 255, 255]
                else:
                    img_array[i:i+square_size, j:j+square_size] = [0, 0, 0]
        
        image = Image.fromarray(img_array)
        resized = resize_image(image)
        
        # Verify resize happened
        assert resized.width == 640
        assert resized.height == 640
        
        # Convert back to array and verify contrast preserved
        resized_array = np.array(resized)
        std_value = np.std(resized_array)
        
        # Standard deviation should be significant (pattern preserved)
        assert std_value > 50, f"Image lost detail after resize (std={std_value})"
    
    def test_color_preservation(self):
        """Test that resize preserves color information."""
        red_color = (255, 0, 0)
        image = create_test_image(800, 600, red_color)
        resized = resize_image(image)
        
        # Convert to array
        resized_array = np.array(resized)
        
        # Check that the resized image is still predominantly red
        mean_red = np.mean(resized_array[:, :, 0])
        mean_green = np.mean(resized_array[:, :, 1])
        mean_blue = np.mean(resized_array[:, :, 2])
        
        assert mean_red > 200, f"Red channel too low: {mean_red}"
        assert mean_green < 50, f"Green channel too high: {mean_green}"
        assert mean_blue < 50, f"Blue channel too high: {mean_blue}"
    
    def test_extreme_panoramic(self):
        """Test resizing with extreme panoramic aspect ratio."""
        # 2400x800 image (3:1 ratio)
        image = create_test_image(2400, 800)
        resized = resize_image(image)
        
        # Scale factor: min(640/2400, 640/800) = min(0.267, 0.8) = 0.267
        # New dimensions: 640x213 (approximately)
        assert resized.width == 640
        assert 213 <= resized.height <= 214  # Allow for rounding
    
    def test_extreme_portrait(self):
        """Test resizing with extreme portrait aspect ratio."""
        # 600x1800 image (1:3 ratio)
        image = create_test_image(600, 1800)
        resized = resize_image(image)
        
        # Scale factor: min(640/600, 640/1800) = min(1.067, 0.356) = 0.356
        # New dimensions: 213x640 (approximately)
        assert 213 <= resized.width <= 214
        assert resized.height == 640


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
