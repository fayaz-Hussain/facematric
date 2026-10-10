"""
Landmark Extraction Utilities

Provides functions for extracting and transforming facial landmarks.
"""

import numpy as np
from typing import Tuple


def extract_2d_landmarks(
    landmarks_3d: np.ndarray,
    image_width: int,
    image_height: int
) -> np.ndarray:
    """
    Extract 2D landmarks from 3D landmarks by dropping z-coordinate and scaling.
    
    Handles conversion of MediaPipe Face Mesh landmarks to 2D coordinates.
    
    Requirements:
    - Input: landmarks_3d shape (468, 3) with (x, y, z) coordinates
    - Output: shape (468, 2) with (x, y) coordinates
    - If landmarks are in [0, 1] range (normalized): scale to image dimensions
    - If landmarks are in pixel range: just extract (x, y) and clip to bounds
    - Clip output to valid ranges: 0 <= x < image_width, 0 <= y < image_height
    
    Args:
        landmarks_3d: 3D landmarks array with shape (468, 3)
                      Can contain either:
                      - Normalized coordinates in range [0, 1]
                      - Pixel coordinates in range [0, image_width] x [0, image_height]
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
    
    Returns:
        np.ndarray: 2D landmarks with shape (468, 2) containing (x, y) coordinates
                    All coordinates clipped to valid ranges:
                    - 0 <= x <= image_width
                    - 0 <= y <= image_height
    
    Raises:
        ValueError: If landmarks_3d is not shape (468, 3)
        ValueError: If image dimensions are not positive
    
    Examples:
        >>> import numpy as np
        >>> from app.utils.landmark_extraction import extract_2d_landmarks
        
        # Normalized landmarks (0-1 range)
        >>> landmarks_normalized = np.random.rand(468, 3)
        >>> landmarks_2d = extract_2d_landmarks(landmarks_normalized, 640, 480)
        >>> landmarks_2d.shape
        (468, 2)
        >>> (landmarks_2d >= 0).all() and (landmarks_2d[:, 0] <= 640).all()
        True
        
        # Pixel coordinates
        >>> landmarks_pixels = np.random.uniform(0, 640, (468, 3))
        >>> landmarks_pixels[:, 1] = np.random.uniform(0, 480, 468)
        >>> landmarks_2d = extract_2d_landmarks(landmarks_pixels, 640, 480)
        >>> landmarks_2d.shape
        (468, 2)
    """
    # Validate input shape
    if landmarks_3d.shape != (468, 3):
        raise ValueError(
            f"landmarks_3d must have shape (468, 3), got {landmarks_3d.shape}"
        )
    
    # Validate image dimensions
    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            f"image_width and image_height must be positive, "
            f"got {image_width}x{image_height}"
        )
    
    # Extract x and y coordinates (drop z)
    landmarks_2d = landmarks_3d[:, :2].copy()
    
    # Detect if landmarks are normalized (0-1 range) or pixel coordinates
    # Check if all x coordinates are in [0, 1] range
    max_x = np.max(landmarks_2d[:, 0])
    max_y = np.max(landmarks_2d[:, 1])
    
    # If max values are close to 1 or less, assume normalized coordinates
    # Use a threshold of 1.5 to handle potential floating point variations
    if max_x <= 1.5 and max_y <= 1.5:
        # Assume normalized coordinates - scale to image dimensions
        landmarks_2d[:, 0] = landmarks_2d[:, 0] * image_width
        landmarks_2d[:, 1] = landmarks_2d[:, 1] * image_height
    # Otherwise assume pixel coordinates, no scaling needed
    
    # Clip to image bounds
    landmarks_2d[:, 0] = np.clip(landmarks_2d[:, 0], 0, image_width)
    landmarks_2d[:, 1] = np.clip(landmarks_2d[:, 1], 0, image_height)
    
    return landmarks_2d.astype(np.float32)
