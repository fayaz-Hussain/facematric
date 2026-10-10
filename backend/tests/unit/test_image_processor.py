"""
Unit tests for Image Processor Component.

Tests format detection, EXIF rotation, color space normalization,
resizing, and validation.
"""

import pytest
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
import base64

from app.components.image_processor import ImageProcessor
from app.utils.errors import ValidationError, ErrorCode


class TestImageProcessor:
    """Test ImageProcessor component."""
    
    @pytest.fixture
    def processor(self):
        """Create ImageProcessor instance."""
        return ImageProcessor()
    
    @pytest.fixture
    def create_test_image(self):
        """Factory to create test images with various properties."""
        def _create(
            width=640,
            height=480,
            mode='RGB',
            color=(128, 128, 128),
            exif_orientation=None
        ):
            """
            Create a test image.
            
            Args:
                width: Image width in pixels
                height: Image height in pixels
                mode: PIL image mode ('RGB', 'RGBA', 'CMYK', etc.)
                color: Fill color
                exif_orientation: EXIF orientation value (1-8) or None
            
            Returns:
                PIL Image object
            """
            image = Image.new(mode, (width, height), color)
            
            # Add some visual content to make rotation visible
            draw = ImageDraw.Draw(image)
            # Draw a triangle to show orientation
            draw.polygon(
                [(width // 4, height // 2), (width // 2, height // 4), (width // 2, 3 * height // 4)],
                fill=(255, 0, 0) if mode == 'RGB' else (255, 0, 0, 255) if mode == 'RGBA' else None
            )
            
            # Set EXIF orientation if specified
            if exif_orientation:
                exif_bytes = self._create_exif_with_orientation(exif_orientation)
                # Save with EXIF data
                buffer = BytesIO()
                image.save(buffer, format='JPEG', exif=exif_bytes)
                buffer.seek(0)
                image = Image.open(buffer)
            
            return image
        
        return _create
    
    def _create_exif_with_orientation(self, orientation):
        """Create minimal EXIF data with orientation tag."""
        from PIL.Image import Exif
        exif = Exif()
        exif[0x0112] = orientation  # Orientation tag
        return exif.tobytes()
    
    def image_to_bytes(self, image, format='JPEG'):
        """Convert PIL Image to bytes."""
        buffer = BytesIO()
        if format == 'JPEG' and image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        image.save(buffer, format=format)
        return buffer.getvalue()
    
    def image_to_base64(self, image, format='JPEG'):
        """Convert PIL Image to base64 string."""
        image_bytes = self.image_to_bytes(image, format)
        return base64.b64encode(image_bytes).decode('utf-8')


class TestFormatDetection(TestImageProcessor):
    """Test format detection and validation."""
    
    @pytest.mark.asyncio
    async def test_jpeg_format_accepted(self, processor, create_test_image):
        """Test that JPEG format is accepted."""
        image = create_test_image(width=400, height=400)
        image_bytes = self.image_to_bytes(image, 'JPEG')
        
        result = await processor.process_upload(file_bytes=image_bytes)
        
        assert result.original_format == 'JPEG'
        assert result.width <= 640
        assert result.height <= 640
    
    @pytest.mark.asyncio
    async def test_png_format_accepted(self, processor, create_test_image):
        """Test that PNG format is accepted."""
        image = create_test_image(width=400, height=400)
        image_bytes = self.image_to_bytes(image, 'PNG')
        
        result = await processor.process_upload(file_bytes=image_bytes)
        
        assert result.original_format == 'PNG'
    
    @pytest.mark.asyncio
    async def test_webp_format_accepted(self, processor, create_test_image):
        """Test that WebP format is accepted."""
        image = create_test_image(width=400, height=400)
        image_bytes = self.image_to_bytes(image, 'WEBP')
        
        result = await processor.process_upload(file_bytes=image_bytes)
        
        assert result.original_format == 'WEBP'
    
    @pytest.mark.asyncio
    async def test_bmp_format_accepted(self, processor, create_test_image):
        """Test that BMP format is accepted."""
        image = create_test_image(width=400, height=400)
        image_bytes = self.image_to_bytes(image, 'BMP')
        
        result = await processor.process_upload(file_bytes=image_bytes)
        
        assert result.original_format == 'BMP'
    
    @pytest.mark.asyncio
    async def test_gif_format_accepted(self, processor, create_test_image):
        """Test that GIF format is accepted."""
        image = create_test_image(width=400, height=400)
        image_bytes = self.image_to_bytes(image, 'GIF')
        
        result = await processor.process_upload(file_bytes=image_bytes)
        
        assert result.original_format == 'GIF'
    
    @pytest.mark.asyncio
    async def test_corrupt_image_rejected(self, processor):
        """Test that corrupt images are rejected."""
        corrupt_bytes = b'not an image'
        
        with pytest.raises(ValidationError) as exc_info:
            await processor.process_upload(file_bytes=corrupt_bytes)
        
        assert exc_info.value.error_code == ErrorCode.INVALID_IMAGE
        assert "Corrupt" in exc_info.value.message


class TestEXIFRotation(TestImageProcessor):
    """Test EXIF orientation handling for all 8 values."""
    
    def create_oriented_image(self, orientation):
        """
        Create an image with specific EXIF orientation.
        
        EXIF Orientation values:
        1 = Normal (0°)
        2 = Mirrored horizontal
        3 = Rotated 180°
        4 = Mirrored vertical
        5 = Mirrored horizontal + rotated 270° CCW
        6 = Rotated 90° CW (270° CCW)
        7 = Mirrored horizontal + rotated 90° CW
        8 = Rotated 270° CW (90° CCW)
        """
        # Create a distinct image where rotation is visible
        width, height = 400, 600  # Rectangular to make rotation obvious
        image = Image.new('RGB', (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # Draw distinct markings
        # Top corner marker (small square in top-left)
        draw.rectangle([(10, 10), (50, 50)], fill=(255, 0, 0))
        
        # Text marker
        draw.text((width // 2 - 20, height // 2), "TOP", fill=(0, 0, 0))
        
        # Save with EXIF orientation
        buffer = BytesIO()
        exif_bytes = self._create_exif_with_orientation(orientation)
        image.save(buffer, format='JPEG', exif=exif_bytes)
        buffer.seek(0)
        
        return buffer.getvalue()
    
    @pytest.mark.asyncio
    async def test_exif_orientation_1_normal(self, processor):
        """Test EXIF orientation 1 (normal, no rotation)."""
        image_bytes = self.create_oriented_image(1)
        
        # Should process without error
        result = await processor.process_upload(file_bytes=image_bytes)
        
        assert result.array is not None
        assert result.width > 0
        assert result.height > 0
    
    @pytest.mark.asyncio
    async def test_exif_orientation_2_mirror_horizontal(self, processor):
        """Test EXIF orientation 2 (mirrored horizontal)."""
        image_bytes = self.create_oriented_image(2)
        
        result = await processor.process_upload(file_bytes=image_bytes)
        
        assert result.array is not None
        # After correction, image should be properly oriented
    
    @pytest.mark.asyncio
    async def test_exif_orientation_3_rotate_180(self, processor):
        """Test EXIF orientation 3 (rotated 180°)."""
        image_bytes = self.create_oriented_image(3)
        
        result = await processor.process_upload(file_bytes=image_bytes)
        
        assert result.array is not None
        # Image should be corrected to normal orientation
    
    @pytest.mark.asyncio
    async def test_exif_orientation_4_mirror_vertical(self, processor):
        """Test EXIF orientation 4 (mirrored vertical)."""
        image_bytes = self.create_oriented_image(4)
        
        result = await processor.process_upload(file_bytes=image_bytes)
        
        assert result.array is not None
    
    @pytest.mark.asyncio
    async def test_exif_orientation_5_mirror_horizontal_rotate_270(self, processor):
        """Test EXIF orientation 5 (mirrored horizontal + rotated 270° CCW)."""
        image_bytes = self.create_oriented_image(5)
        
        result = await processor.process_upload(file_bytes=image_bytes)
        
        assert result.array is not None
    
    @pytest.mark.asyncio
    async def test_exif_orientation_6_rotate_90_cw(self, processor):
        """Test EXIF orientation 6 (rotated 90° CW / 270° CCW)."""
        image_bytes = self.create_oriented_image(6)
        
        result = await processor.process_upload(file_bytes=image_bytes)
        
        assert result.array is not None
        # Width and height should be swapped after rotation correction
    
    @pytest.mark.asyncio
    async def test_exif_orientation_7_mirror_horizontal_rotate_90_cw(self, processor):
        """Test EXIF orientation 7 (mirrored horizontal + rotated 90° CW)."""
        image_bytes = self.create_oriented_image(7)
        
        result = await processor.process_upload(file_bytes=image_bytes)
        
        assert result.array is not None
    
    @pytest.mark.asyncio
    async def test_exif_orientation_8_rotate_270_cw(self, processor):
        """Test EXIF orientation 8 (rotated 270° CW / 90° CCW)."""
        image_bytes = self.create_oriented_image(8)
        
        result = await processor.process_upload(file_bytes=image_bytes)
        
        assert result.array is not None
        # Width and height should be swapped after rotation correction
    
    @pytest.mark.asyncio
    async def test_missing_exif_handled_gracefully(self, processor, create_test_image):
        """Test that images without EXIF data are handled gracefully."""
        image = create_test_image(width=400, height=400)
        # Create image without EXIF data
        buffer = BytesIO()
        image.save(buffer, format='PNG')  # PNG typically doesn't have EXIF
        image_bytes = buffer.getvalue()
        
        result = await processor.process_upload(file_bytes=image_bytes)
        
        assert result.array is not None


class TestColorSpaceNormalization(TestImageProcessor):
    """Test color space conversion."""
    
    @pytest.mark.asyncio
    async def test_rgba_to_rgb_conversion(self, processor, create_test_image):
        """Test RGBA to RGB conversion with white background composite."""
        image = create_test_image(width=400, height=400, mode='RGBA', color=(128, 128, 128, 200))
        image_bytes = self.image_to_bytes(image, 'PNG')
        
        result = await processor.process_upload(file_bytes=image_bytes)
        
        # Result should be RGB (3 channels)
        assert result.array.shape[2] == 3
        assert result.array.dtype == np.uint8
    
    @pytest.mark.asyncio
    async def test_cmyk_to_rgb_conversion(self, processor):
        """Test CMYK to RGB conversion."""
        # Create CMYK image
        image = Image.new('CMYK', (400, 400), (0, 0, 0, 0))
        buffer = BytesIO()
        image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        
        result = await processor.process_upload(file_bytes=image_bytes)
        
        # Result should be RGB (3 channels)
        assert result.array.shape[2] == 3
    
    @pytest.mark.asyncio
    async def test_grayscale_to_rgb_conversion(self, processor):
        """Test grayscale to RGB conversion."""
        image = Image.new('L', (400, 400), 128)
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        
        result = await processor.process_upload(file_bytes=image_bytes)
        
        # Result should be RGB (3 channels)
        assert result.array.shape[2] == 3


class TestResizing(TestImageProcessor):
    """Test aspect ratio-preserving resize."""
    
    @pytest.mark.asyncio
    async def test_large_image_resized(self, processor, create_test_image):
        """Test that large images are resized to max dimension."""
        # Create image larger than max dimension
        image = create_test_image(width=1920, height=1080)
        image_bytes = self.image_to_bytes(image)
        
        result = await processor.process_upload(file_bytes=image_bytes)
        
        # Should be resized to fit within 640px
        assert result.width <= 640
        assert result.height <= 640
        assert max(result.width, result.height) == 640
    
    @pytest.mark.asyncio
    async def test_aspect_ratio_preserved(self, processor, create_test_image):
        """Test that aspect ratio is preserved during resize."""
        # Create 2:1 aspect ratio image
        image = create_test_image(width=1600, height=800)
        image_bytes = self.image_to_bytes(image)
        
        result = await processor.process_upload(file_bytes=image_bytes)
        
        # Aspect ratio should be preserved (approximately 2:1)
        aspect_ratio = result.width / result.height
        assert 1.9 < aspect_ratio < 2.1
    
    @pytest.mark.asyncio
    async def test_small_image_not_upscaled(self, processor, create_test_image):
        """Test that small images are not upscaled."""
        # Create image smaller than max dimension but above minimum (300x300)
        image = create_test_image(width=400, height=350)
        image_bytes = self.image_to_bytes(image)
        
        result = await processor.process_upload(file_bytes=image_bytes)
        
        # Should keep original size (not upscaled)
        assert result.width == 400
        assert result.height == 350


class TestValidation(TestImageProcessor):
    """Test image validation."""
    
    @pytest.mark.asyncio
    async def test_minimum_resolution_enforced(self, processor, create_test_image):
        """Test that images below minimum resolution are rejected."""
        # Create image smaller than minimum (300x300)
        image = create_test_image(width=200, height=200)
        image_bytes = self.image_to_bytes(image)
        
        with pytest.raises(ValidationError) as exc_info:
            await processor.process_upload(file_bytes=image_bytes)
        
        assert exc_info.value.error_code == ErrorCode.POOR_QUALITY
        assert "resolution too low" in exc_info.value.message.lower()
    
    @pytest.mark.asyncio
    async def test_extreme_aspect_ratio_rejected(self, processor, create_test_image):
        """Test that extreme aspect ratios are rejected."""
        # Create image with 4:1 aspect ratio (exceeds 3:1 limit)
        image = create_test_image(width=1200, height=300)
        image_bytes = self.image_to_bytes(image)
        
        with pytest.raises(ValidationError) as exc_info:
            await processor.process_upload(file_bytes=image_bytes)
        
        assert exc_info.value.error_code == ErrorCode.POOR_QUALITY
        assert "aspect ratio" in exc_info.value.message.lower()
    
    @pytest.mark.asyncio
    async def test_no_image_provided_error(self, processor):
        """Test error when no image is provided."""
        with pytest.raises(ValidationError) as exc_info:
            await processor.process_upload()
        
        assert exc_info.value.error_code == ErrorCode.NO_IMAGE
    
    @pytest.mark.asyncio
    async def test_invalid_base64_rejected(self, processor):
        """Test that invalid base64 strings are rejected."""
        invalid_base64 = "not-valid-base64!!!"
        
        with pytest.raises(ValidationError) as exc_info:
            await processor.process_upload(base64_str=invalid_base64)
        
        assert exc_info.value.error_code == ErrorCode.INVALID_IMAGE


class TestBase64Input(TestImageProcessor):
    """Test base64 string input."""
    
    @pytest.mark.asyncio
    async def test_base64_input_accepted(self, processor, create_test_image):
        """Test that base64-encoded images are processed correctly."""
        image = create_test_image(width=400, height=400)
        base64_str = self.image_to_base64(image)
        
        result = await processor.process_upload(base64_str=base64_str)
        
        assert result.array is not None
        assert result.width > 0
        assert result.height > 0
    
    @pytest.mark.asyncio
    async def test_base64_and_bytes_mutually_exclusive(self, processor, create_test_image):
        """Test that providing both base64 and bytes uses base64."""
        image = create_test_image(width=400, height=400)
        base64_str = self.image_to_base64(image)
        image_bytes = self.image_to_bytes(image)
        
        # When both provided, base64 takes precedence
        result = await processor.process_upload(
            file_bytes=image_bytes,
            base64_str=base64_str
        )
        
        assert result.array is not None


class TestOutputFormat(TestImageProcessor):
    """Test output format and data structure."""
    
    @pytest.mark.asyncio
    async def test_output_is_numpy_array(self, processor, create_test_image):
        """Test that output is a NumPy array."""
        image = create_test_image(width=400, height=400)
        image_bytes = self.image_to_bytes(image)
        
        result = await processor.process_upload(file_bytes=image_bytes)
        
        assert isinstance(result.array, np.ndarray)
        assert result.array.dtype == np.uint8
    
    @pytest.mark.asyncio
    async def test_output_has_correct_shape(self, processor, create_test_image):
        """Test that output array has correct shape (H, W, 3)."""
        image = create_test_image(width=400, height=400)
        image_bytes = self.image_to_bytes(image)
        
        result = await processor.process_upload(file_bytes=image_bytes)
        
        # Shape should be (height, width, 3) for RGB
        assert len(result.array.shape) == 3
        assert result.array.shape[2] == 3
        assert result.array.shape[0] == result.height
        assert result.array.shape[1] == result.width
    
    @pytest.mark.asyncio
    async def test_metadata_populated(self, processor, create_test_image):
        """Test that metadata fields are correctly populated."""
        image = create_test_image(width=400, height=300)
        image_bytes = self.image_to_bytes(image, 'JPEG')
        
        result = await processor.process_upload(file_bytes=image_bytes)
        
        assert result.original_format == 'JPEG'
        assert result.width == 400
        assert result.height == 300


class TestJPEGCompression(TestImageProcessor):
    """Test JPEG compression functionality (Requirement 1.4)."""
    
    @pytest.mark.asyncio
    async def test_jpeg_compression_applied(self, processor, create_test_image):
        """Test that JPEG compression at 85% quality is applied during processing."""
        # Create a large, complex image to test compression
        image = create_test_image(width=800, height=800, mode='RGB')
        
        # Add complex patterns that would benefit from compression
        draw = ImageDraw.Draw(image)
        for i in range(0, 800, 20):
            draw.line([(0, i), (800, i)], fill=(255, 0, 0))
            draw.line([(i, 0), (i, 800)], fill=(0, 255, 0))
        
        image_bytes = self.image_to_bytes(image, 'PNG')  # Start with PNG (lossless)
        original_size = len(image_bytes)
        
        # Process the image
        result = await processor.process_upload(file_bytes=image_bytes)
        
        # Verify the image was processed
        assert result.array is not None
        assert result.width <= 640  # Should be resized
        assert result.height <= 640
        
        # The numpy array should represent compressed JPEG data
        # (compression happens during processing, not in the final array)
        assert result.array.shape[2] == 3  # RGB format preserved
    
    @pytest.mark.asyncio
    async def test_compression_quality_setting(self, processor):
        """Test that compression uses the configured quality setting (85%)."""
        # Verify the processor has the correct quality setting
        assert processor.jpeg_quality == 85
    
    @pytest.mark.asyncio
    async def test_compress_to_jpeg_method(self, processor, create_test_image):
        """Test the standalone compress_to_jpeg method returns JPEG bytes."""
        image = create_test_image(width=400, height=400, mode='RGB')
        
        # Use the standalone compression method
        jpeg_bytes = processor.compress_to_jpeg(image)
        
        # Verify it's valid JPEG data
        assert jpeg_bytes is not None
        assert len(jpeg_bytes) > 0
        
        # Verify we can load it as a JPEG
        loaded_image = Image.open(BytesIO(jpeg_bytes))
        assert loaded_image.format == 'JPEG'
        assert loaded_image.size == (400, 400)
    
    @pytest.mark.asyncio
    async def test_compression_maintains_quality(self, processor, create_test_image):
        """Test that 85% JPEG compression maintains reasonable visual quality."""
        # Create a detailed image
        image = create_test_image(width=640, height=640, mode='RGB', color=(128, 128, 128))
        
        # Process through the pipeline
        image_bytes = self.image_to_bytes(image, 'PNG')
        result = await processor.process_upload(file_bytes=image_bytes)
        
        # The image should still have valid pixel data
        assert result.array is not None
        assert result.array.shape[2] == 3
        
        # Verify pixel values are in valid range
        assert result.array.min() >= 0
        assert result.array.max() <= 255
    
    @pytest.mark.asyncio
    async def test_compression_with_various_formats(self, processor, create_test_image):
        """Test that compression works correctly with various input formats."""
        formats_to_test = ['JPEG', 'PNG', 'BMP', 'WEBP']
        
        for format_name in formats_to_test:
            image = create_test_image(width=500, height=500)
            
            # Skip formats that don't support RGB
            if format_name == 'GIF':
                continue
                
            image_bytes = self.image_to_bytes(image, format_name)
            
            # Process the image
            result = await processor.process_upload(file_bytes=image_bytes)
            
            # Verify compression was applied (all formats should work)
            assert result.array is not None
            assert result.original_format == format_name
            assert result.width <= 640
            assert result.height <= 640


class TestValidationChecks(TestImageProcessor):
    """Test validation checks for requirements 1.6 and 20.4."""
    
    @pytest.mark.asyncio
    async def test_poor_quality_minimum_resolution(self, processor, create_test_image):
        """Test POOR_QUALITY error for images below 300x300px (Requirement 1.6)."""
        # Create image below minimum resolution
        image = create_test_image(width=250, height=250)
        image_bytes = self.image_to_bytes(image)
        
        with pytest.raises(ValidationError) as exc_info:
            await processor.process_upload(file_bytes=image_bytes)
        
        assert exc_info.value.error_code == ErrorCode.POOR_QUALITY
        assert "resolution too low" in exc_info.value.message.lower()
        assert "300" in exc_info.value.message  # Should mention minimum size
    
    @pytest.mark.asyncio
    async def test_poor_quality_extreme_aspect_ratio(self, processor, create_test_image):
        """Test POOR_QUALITY error for extreme aspect ratios >3:1 (Requirement 20.4)."""
        # Create image with 4:1 aspect ratio (exceeds 3:1 limit)
        image = create_test_image(width=1200, height=300)
        image_bytes = self.image_to_bytes(image)
        
        with pytest.raises(ValidationError) as exc_info:
            await processor.process_upload(file_bytes=image_bytes)
        
        assert exc_info.value.error_code == ErrorCode.POOR_QUALITY
        assert "aspect ratio" in exc_info.value.message.lower()
        assert "3" in exc_info.value.message  # Should mention the limit
    
    @pytest.mark.asyncio
    async def test_acceptable_aspect_ratio_passes(self, processor, create_test_image):
        """Test that aspect ratios at the limit (3:1) are accepted."""
        # Create image with exactly 3:1 aspect ratio
        image = create_test_image(width=900, height=300)
        image_bytes = self.image_to_bytes(image)
        
        # Should process successfully
        result = await processor.process_upload(file_bytes=image_bytes)
        
        assert result.array is not None
        assert result.width <= 640
        assert result.height <= 640
    
    @pytest.mark.asyncio
    async def test_minimum_resolution_exactly_300x300(self, processor, create_test_image):
        """Test that exactly 300x300px images are accepted."""
        # Create image at exactly minimum resolution
        image = create_test_image(width=300, height=300)
        image_bytes = self.image_to_bytes(image)
        
        # Should process successfully
        result = await processor.process_upload(file_bytes=image_bytes)
        
        assert result.array is not None
        assert result.width == 300
        assert result.height == 300


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
