"""
Image Processor Component

Handles image upload, normalization, validation, and preprocessing.
Supports JPEG, PNG, HEIC, WebP, BMP, and GIF formats.
"""

from PIL import Image, ImageOps
from io import BytesIO
import numpy as np
import base64
from typing import Optional

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

from app.models.internal import ImageData
from app.utils.errors import ValidationError, ErrorCode
from app.utils.logger import get_logger
from app.config import settings

logger = get_logger(__name__)


class ImageProcessor:
    """
    Processes and validates uploaded images for facial analysis.
    
    Features:
    - Multi-format support (JPEG, PNG, HEIC, WebP, BMP, GIF)
    - EXIF rotation handling
    - Color space normalization (CMYK→RGB, RGBA→RGB)
    - Aspect ratio-preserving resize
    - JPEG compression
    - Validation (resolution, aspect ratio, format)
    """
    
    SUPPORTED_FORMATS = {'JPEG', 'PNG', 'HEIC', 'HEIF', 'WEBP', 'BMP', 'GIF'}
    
    def __init__(self):
        """Initialize ImageProcessor with configuration from settings."""
        self.min_dimension = settings.min_image_dimension
        self.max_dimension = settings.max_image_dimension
        self.jpeg_quality = settings.jpeg_quality
        self.max_aspect_ratio = 3.0  # Maximum ratio of longest to shortest side
    
    async def process_upload(
        self,
        file_bytes: Optional[bytes] = None,
        base64_str: Optional[str] = None
    ) -> ImageData:
        """
        Main entry point for image processing.
        
        Args:
            file_bytes: Raw image bytes from file upload
            base64_str: Base64-encoded image string (already validated)
        
        Returns:
            ImageData: Normalized image data ready for analysis
        
        Raises:
            ValidationError: If image is invalid, corrupt, or doesn't meet requirements
        """
        # Decode input
        if base64_str:
            clean_str = base64_str
            if "," in clean_str and clean_str.startswith("data:"):
                clean_str = clean_str.split(",", 1)[1]
            clean_str = clean_str.strip()
            
            # Enforce 10MB limit on decoded size early without full decode (Requirement 3c)
            max_size = settings.max_image_size_mb * 1024 * 1024
            estimated_size = (len(clean_str) * 3) // 4
            if estimated_size > max_size:
                raise ValidationError(
                    ErrorCode.FILE_TOO_LARGE,
                    f"File size exceeds {settings.max_image_size_mb}MB limit"
                )
                
            try:
                image_bytes = base64.b64decode(clean_str, validate=True)
            except Exception as e:
                logger.error(f"Base64 decode failed: {str(e)}")
                raise ValidationError(
                    ErrorCode.INVALID_IMAGE,
                    "Malformed base64 image data"
                )
                
            if len(image_bytes) > max_size:
                raise ValidationError(
                    ErrorCode.FILE_TOO_LARGE,
                    f"File size exceeds {settings.max_image_size_mb}MB limit"
                )
        elif file_bytes:
            image_bytes = file_bytes
        else:
            raise ValidationError(
                ErrorCode.NO_IMAGE,
                "No image provided"
            )
        
        # Open and validate image
        try:
            image = Image.open(BytesIO(image_bytes))
            original_format = image.format or "UNKNOWN"
        except Exception as e:
            logger.error(f"PIL image open failed: {str(e)}")
            raise ValidationError(
                ErrorCode.INVALID_IMAGE,
                "Corrupt or unsupported file format. Please upload a valid image."
            )
        
        # Validate format
        if image.format not in self.SUPPORTED_FORMATS:
            raise ValidationError(
                ErrorCode.INVALID_IMAGE,
                f"Unsupported format: {image.format}. Supported formats: JPEG, PNG, HEIC, WebP, BMP, GIF"
            )
        
        # Apply EXIF rotation
        image = self._apply_exif_rotation(image)
        
        # Validate dimensions
        width, height = image.size
        self._validate_dimensions(width, height)
        
        # Normalize color space
        image = self._normalize_color_space(image)
        
        # Resize maintaining aspect ratio
        image = self._resize_image(image)
        
        # Compress to JPEG at 85% quality (requirement 1.4)
        # This reduces memory usage and standardizes format for processing
        image = self._compress_image(image)
        
        # Convert to numpy array (RGB format)
        image_array = np.array(image, dtype=np.uint8)
        
        logger.info(
            f"Image processed successfully: {original_format}, "
            f"original size: {width}x{height}, "
            f"final size: {image.width}x{image.height}"
        )
        
        return ImageData(
            array=image_array,
            width=image.width,
            height=image.height,
            original_format=original_format
        )
    
    def _apply_exif_rotation(self, image: Image.Image) -> Image.Image:
        """
        Apply EXIF orientation to rotate image correctly.
        
        Handles all 8 EXIF orientation values.
        
        Args:
            image: PIL Image object
        
        Returns:
            Rotated PIL Image object
        """
        try:
            # ImageOps.exif_transpose handles all EXIF orientation cases
            image = ImageOps.exif_transpose(image)
        except Exception as e:
            # If EXIF processing fails, just continue with original orientation
            logger.warning(f"EXIF rotation failed: {str(e)}")
        
        return image
    
    def _validate_dimensions(self, width: int, height: int) -> None:
        """
        Validate image dimensions meet requirements.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
        
        Raises:
            ValidationError: If dimensions don't meet requirements
        """
        # Check minimum resolution
        if width < self.min_dimension or height < self.min_dimension:
            raise ValidationError(
                ErrorCode.POOR_QUALITY,
                f"Image resolution too low for accurate analysis. "
                f"Minimum size: {self.min_dimension}x{self.min_dimension} pixels. "
                f"Your image: {width}x{height} pixels."
            )
        
        # Check aspect ratio
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > self.max_aspect_ratio:
            raise ValidationError(
                ErrorCode.POOR_QUALITY,
                f"Image aspect ratio too extreme for accurate face detection. "
                f"Maximum ratio: {self.max_aspect_ratio}:1. "
                f"Your image ratio: {aspect_ratio:.2f}:1."
            )
    
    def _normalize_color_space(self, image: Image.Image) -> Image.Image:
        """
        Convert image to RGB color space.
        
        Handles:
        - RGBA → RGB (composite on white background)
        - CMYK → RGB
        - Grayscale → RGB
        - Other modes → RGB
        
        Args:
            image: PIL Image object
        
        Returns:
            RGB PIL Image object
        """
        if image.mode == 'RGBA':
            # Composite onto white background
            background = Image.new('RGB', image.size, (255, 255, 255))
            # Use alpha channel for compositing
            background.paste(image, mask=image.split()[3])
            logger.debug("Converted RGBA to RGB with white background")
            return background
        
        elif image.mode == 'CMYK':
            # Convert CMYK to RGB
            image = image.convert('RGB')
            logger.debug("Converted CMYK to RGB")
            return image
        
        elif image.mode != 'RGB':
            # Convert any other mode to RGB
            logger.debug(f"Converting {image.mode} to RGB")
            return image.convert('RGB')
        
        return image
    
    def _resize_image(self, image: Image.Image) -> Image.Image:
        """
        Resize image maintaining aspect ratio.
        
        If either dimension exceeds max_dimension, scales down proportionally.
        Uses high-quality Lanczos resampling.
        
        Args:
            image: PIL Image object
        
        Returns:
            Resized PIL Image object
        """
        width, height = image.size
        
        # Calculate scale factor
        scale_factor = min(
            self.max_dimension / width,
            self.max_dimension / height
        )
        
        # Only resize if image is too large
        if scale_factor < 1.0:
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            logger.debug(f"Resizing from {width}x{height} to {new_width}x{new_height}")
            
            # Use LANCZOS for high-quality downsampling
            image = image.resize(
                (new_width, new_height),
                Image.Resampling.LANCZOS
            )
        
        return image
    
    def _compress_image(self, image: Image.Image) -> Image.Image:
        """
        Compress image to JPEG format at configured quality (85%).
        
        This reduces memory usage and standardizes the image format for processing.
        Per requirement 1.4, images are compressed at 85% JPEG quality.
        
        Args:
            image: PIL Image object (must be RGB)
        
        Returns:
            Compressed PIL Image object
        """
        # Compress to JPEG at 85% quality
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=self.jpeg_quality, optimize=True)
        buffer.seek(0)
        
        # Reload the compressed image
        compressed_image = Image.open(buffer)
        
        logger.debug(f"Compressed image to JPEG at {self.jpeg_quality}% quality")
        return compressed_image
    
    def compress_to_jpeg(self, image: Image.Image) -> bytes:
        """
        Compress image to JPEG format at configured quality and return bytes.
        
        This method can be used when JPEG bytes are needed (e.g., for saving thumbnails).
        
        Args:
            image: PIL Image object (must be RGB)
        
        Returns:
            JPEG image bytes
        """
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=self.jpeg_quality, optimize=True)
        return buffer.getvalue()


# Global instance
image_processor = ImageProcessor()
