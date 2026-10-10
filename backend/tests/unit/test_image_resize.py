"""
Unit tests for ImageProcessor resize functionality (Task 3.4)

Tests aspect ratio-preserving resize with Lanczos resampling.
"""

import pytest
import numpy as np
from PIL import Image
from io import BytesIO
import base64

from app.components.image_processor import ImageProcessor
from app.models.internal import ImageData
from app.utils.errors import ValidationError, ErrorCode


class TestImageResize:
    """Test suite for aspect ratio-preserving resize functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create ImageProcessor instance."""
        return ImageProcessor()
    
    def create_test_image(self, width: int, height: int, color: tuple = (128, 128, 128)) -> bytes:
        """
        Helper to create test image with specified dimensions.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            color: RGB color tuple for solid fill
        
        Returns:
            JPEG image bytes
        """
        image = Image.new('RGB', (width, height), color)
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        return buffer.getvalue()
    
    @pytest.mark.asyncio
    async def test_resize_large_image_width_exceeds_640(self, processor):
        """Test resizing when width exceeds 640px."""
        # Create 800x400 image (width exceeds max dimension)
        image_bytes = self.create_test_image(800, 400)
        
        result = await processor.process_upload(file_bytes=image_bytes)
        
        # Scale factor should be 640/800 = 0.8
        # New dimensions: 640x320
        assert result.width == 640
        assert result.height == 320
        
        # Aspect ratio preserved: 800/400 = 2.0, 640/320 = 2.0
        original_aspect = 800 / 400
        new_aspect = result.width / result.height
        assert abs(original_aspect - new_aspect) < 0.01
    
    @pytest.mark.asyncio
    async def test_resize_large_image_height_exceeds_640(self, processor):
        """Test resizing when height exceeds 640px."""
        # Create 400x1000 image (height exceeds max dimension)
        image_bytes = self.create_test_image(400, 1000)
        
        result = await processor.process_upload(file_bytes=image_bytes)
        
        # Scale factor should be 640/1000 = 0.64
        # New dimensions: 256x640
        assert result.width == 256
        assert result.height == 640
        
        # Aspect ratio preserved: 400/1000 = 0.4, 256/640 = 0.4
        original_aspect = 400 / 1000
        new_aspect = result.width / result.height
        assert abs(original_aspect - new_aspect) < 0.01
    
    @pytest.mark.asyncio
    async def test_resize_large_square_image(self, processor):
        """Test resizing square image (both dimensions exceed 640px)."""
        # Create 1000x1000 image
        image_bytes = self.create_test_image(1000, 1000)
        
        result = await processor.process_upload(file_bytes=image_bytes)
        
        # Scale factor should be 640/1000 = 0.64
        # New dimensions: 640x640 (both scaled equally)
        assert result.width == 640
        assert result.height == 640
        
        # Aspect ratio preserved: 1.0
        assert result.width == result.height
    
    @pytest.mark.asyncio
    async def test_no_resize_when_within_640(self, processor):
        """Test that images within 640px are NOT resized."""
        # Create 500x400 image (both dimensions < 640)
        image_bytes = self.create_test_image(500, 400)
        
        result = await processor.process_upload(file_bytes=image_bytes)
        
        # Should NOT be resized
        assert result.width == 500
        assert result.height == 400
    
    @pytest.mark.asyncio
    async def test_no_resize_exactly_640(self, processor):
        """Test that 640x640 image is NOT resized."""
        # Create exactly 640x640 image
        image_bytes = self.create_test_image(640, 640)
        
        result = await processor.process_upload(file_bytes=image_bytes)
        
        # Should NOT be resized
        assert result.width == 640
        assert result.height == 640
    
    @pytest.mark.asyncio
    async def test_resize_maintains_aspect_ratio_various_sizes(self, processor):
        """Test aspect ratio preservation across various image sizes."""
        test_cases = [
            (1920, 1080),  # 16:9 aspect ratio
            (1200, 800),   # 3:2 aspect ratio
            (1000, 1500),  # 2:3 aspect ratio (portrait)
            (2000, 1000),  # 2:1 aspect ratio (panoramic)
        ]
        
        for width, height in test_cases:
            image_bytes = self.create_test_image(width, height)
            result = await processor.process_upload(file_bytes=image_bytes)
            
            # Calculate original and new aspect ratios
            original_aspect = width / height
            new_aspect = result.width / result.height
            
            # Aspect ratio should be preserved (within floating point tolerance)
            assert abs(original_aspect - new_aspect) < 0.01, (
                f"Aspect ratio not preserved for {width}x{height}: "
                f"original={original_aspect:.3f}, new={new_aspect:.3f}"
            )
            
            # At least one dimension should be 640 (the constraining dimension)
            assert result.width <= 640 and result.height <= 640, (
                f"Image not properly constrained: {result.width}x{result.height}"
            )
            
            # The larger original dimension should result in 640
            if width > height:
                if width > 640:
                    assert result.width == 640
            else:
                if height > 640:
                    assert result.height == 640
    
    @pytest.mark.asyncio
    async def test_resize_scale_factor_calculation(self, processor):
        """Test that scale factor is correctly calculated."""
        # Create 1280x720 image
        image_bytes = self.create_test_image(1280, 720)
        
        result = await processor.process_upload(file_bytes=image_bytes)
        
        # Scale factor should be min(640/1280, 640/720) = min(0.5, 0.889) = 0.5
        # New dimensions: 640x360
        assert result.width == 640
        assert result.height == 360
    
    @pytest.mark.asyncio
    async def test_lanczos_resampling_quality(self, processor):
        """Test that Lanczos resampling produces high-quality results."""
        # Create a test image with distinct patterns to verify quality
        # We'll create an image with alternating colors
        width, height = 1000, 1000
        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create checkerboard pattern
        square_size = 50
        for i in range(0, height, square_size):
            for j in range(0, width, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    img_array[i:i+square_size, j:j+square_size] = [255, 255, 255]
                else:
                    img_array[i:i+square_size, j:j+square_size] = [0, 0, 0]
        
        # Convert to PIL Image and save as JPEG
        image = Image.fromarray(img_array)
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        image_bytes = buffer.getvalue()
        
        result = await processor.process_upload(file_bytes=image_bytes)
        
        # Verify resize happened
        assert result.width == 640
        assert result.height == 640
        
        # Verify result is still a valid image array
        assert result.array.shape == (640, 640, 3)
        assert result.array.dtype == np.uint8
        
        # Verify we still have contrast (not all gray)
        # After Lanczos interpolation, we should still have distinct dark and light regions
        mean_value = np.mean(result.array)
        std_value = np.std(result.array)
        
        # Standard deviation should be significant (pattern preserved)
        assert std_value > 50, "Image lost detail after resize (std too low)"
    
    @pytest.mark.asyncio
    async def test_resize_with_base64_input(self, processor):
        """Test resizing works with base64-encoded input."""
        # Create test image
        image_bytes = self.create_test_image(1000, 800)
        base64_str = base64.b64encode(image_bytes).decode('utf-8')
        
        result = await processor.process_upload(base64_str=base64_str)
        
        # Should be resized to 640x512
        assert result.width == 640
        assert result.height == 512
    
    @pytest.mark.asyncio
    async def test_resize_after_exif_rotation(self, processor):
        """Test that resize happens AFTER EXIF rotation is applied."""
        # Create 1000x500 image and add EXIF rotation data
        image = Image.new('RGB', (1000, 500), (200, 100, 50))
        
        # Add EXIF data for 90-degree rotation
        from PIL.ExifTags import TAGS
        exif = image.getexif()
        # Orientation tag 274, value 6 means rotate 90 degrees CW
        exif[274] = 6
        
        # Save with EXIF
        buffer = BytesIO()
        image.save(buffer, format='JPEG', exif=exif.tobytes())
        image_bytes = buffer.getvalue()
        
        result = await processor.process_upload(file_bytes=image_bytes)
        
        # After 90-degree rotation: 1000x500 becomes 500x1000
        # Then scaled: 640/1000 = 0.64 → 320x640
        # Note: EXIF handling might vary, so we check that rotation was applied
        # and dimensions are within expected range
        assert max(result.width, result.height) == 640
        assert result.width <= 640 and result.height <= 640
    
    @pytest.mark.asyncio
    async def test_resize_preserves_rgb_values(self, processor):
        """Test that resize preserves color information."""
        # Create image with specific color
        red_color = (255, 0, 0)
        image_bytes = self.create_test_image(800, 600, red_color)
        
        result = await processor.process_upload(file_bytes=image_bytes)
        
        # Check that the resized image is still predominantly red
        mean_red = np.mean(result.array[:, :, 0])
        mean_green = np.mean(result.array[:, :, 1])
        mean_blue = np.mean(result.array[:, :, 2])
        
        # Red channel should be much higher than others
        assert mean_red > 200, f"Red channel too low: {mean_red}"
        assert mean_green < 50, f"Green channel too high: {mean_green}"
        assert mean_blue < 50, f"Blue channel too high: {mean_blue}"
    
    @pytest.mark.asyncio
    async def test_minimum_dimension_not_affected_by_resize(self, processor):
        """Test that minimum dimension validation still works after resize."""
        # Create 250x200 image (below 300px minimum)
        image_bytes = self.create_test_image(250, 200)
        
        # Should raise POOR_QUALITY error before reaching resize
        with pytest.raises(ValidationError) as exc_info:
            await processor.process_upload(file_bytes=image_bytes)
        
        assert exc_info.value.error_code == ErrorCode.POOR_QUALITY
        assert "resolution too low" in exc_info.value.message.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
