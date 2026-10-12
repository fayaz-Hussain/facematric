"""
Internal data structures for component communication.

These dataclasses are used for passing data between backend components.
They are NOT exposed in the API (use requests.py and responses.py for that).

Requirements validated:
- 1.2: ImageData normalized to RGB color space
- 1.3: ImageData with aspect ratio preserved dimensions
- 2.2: Landmarks with exactly 468 3D coordinates
- 4.8: SymmetryType enum with 6 classification types
- 5.1-5.8: GoldenRatioResult with 8 feature measurements
- 6.4: FeatureScoreInternal with measured_ratio, ideal_ratio, score, deviation
- 7.1-7.3: GoldenRatioResult overall_score bounded [0, 100] with validation
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum
import numpy as np


class SymmetryType(Enum):
    """
    Face symmetry type classifications.
    
    Requirements: 4.8
    
    Six face shape classifications based on geometric proportions:
    - OVAL: Balanced proportions (length/width ≈ 1.5, rounded jawline)
    - RECTANGLE: Elongated face (length > width, angular jawline)
    - ROUND: Equal dimensions (length ≈ width, soft angles)
    - SQUARE: Equal dimensions (length ≈ width, angular jawline)
    - HEART: Forehead wider than chin
    - DIAMOND: Cheekbones wider than forehead and chin
    """
    OVAL = "oval"
    RECTANGLE = "rectangle"
    ROUND = "round"
    SQUARE = "square"
    HEART = "heart"
    DIAMOND = "diamond"


@dataclass
class ImageData:
    """
    Normalized image data after preprocessing.
    
    Requirements: 1.2 (RGB normalization), 1.3 (aspect ratio preservation)
    
    Represents a processed image ready for face detection. All images are:
    - Converted to RGB color space (no CMYK, alpha channels handled)
    - Resized to max 640px dimension maintaining aspect ratio
    - Validated for minimum resolution (300x300px)
    
    Attributes:
        array: RGB image array with shape (height, width, 3), dtype uint8
        width: Image width in pixels (after resize)
        height: Image height in pixels (after resize)
        original_format: Original image format (JPEG, PNG, HEIC, WebP, BMP, GIF)
    """
    array: np.ndarray  # Shape: (H, W, 3), dtype: uint8, RGB format
    width: int
    height: int
    original_format: str
    
    def __post_init__(self):
        """Validate image data after initialization."""
        if self.array.ndim != 3:
            raise ValueError(f"Image array must be 3D, got {self.array.ndim}D")
        if self.array.shape[2] != 3:
            raise ValueError(f"Image must have 3 channels (RGB), got {self.array.shape[2]}")
        if self.array.shape[0] != self.height or self.array.shape[1] != self.width:
            raise ValueError(
                f"Array shape {self.array.shape[:2]} doesn't match dimensions "
                f"(height={self.height}, width={self.width})"
            )
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Dimensions must be positive, got {self.width}x{self.height}")


@dataclass
class Landmarks:
    """
    468 3D facial landmarks from MediaPipe Face Mesh.
    
    Requirements: 2.2 (468 landmark extraction)
    
    MediaPipe Face Mesh produces 468 facial landmarks covering:
    - Face oval contours
    - Left and right eyes (16 points each)
    - Eyebrows
    - Nose bridge and nostrils
    - Lips (inner and outer contours)
    
    Attributes:
        points: Array of 468 landmark coordinates with shape (468, 3)
                Each row is [x, y, z] where:
                - x, y are pixel coordinates in the image
                - z is depth relative to face plane (normalized, approximate)
        confidence: Average landmark detection confidence (0.0-1.0)
                   Thresholded at 0.6 for quality validation
        image_width: Width of source image (for coordinate validation)
        image_height: Height of source image (for coordinate validation)
    """
    points: np.ndarray  # Shape: (468, 3), dtype: float32
    confidence: float
    image_width: int
    image_height: int
    
    def __post_init__(self):
        """Validate landmarks after initialization."""
        if self.points.shape != (468, 3):
            raise ValueError(f"Landmarks must have shape (468, 3), got {self.points.shape}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {self.confidence}")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError(
                f"Image dimensions must be positive, got {self.image_width}x{self.image_height}"
            )
    
    def get_2d_landmarks(self) -> np.ndarray:
        """
        Extract 2D landmarks by dropping z-coordinate.
        
        Used for overlay generation and visualization.
        
        Returns:
            Array of shape (468, 2) with [x, y] coordinates
        """
        return self.points[:, :2].copy()
    
    def get_landmark(self, index: int) -> np.ndarray:
        """
        Get a specific landmark by index.
        
        Args:
            index: Landmark index (0-467)
        
        Returns:
            Array of shape (3,) with [x, y, z] coordinates
        
        Raises:
            IndexError: If index is out of range
        """
        if not 0 <= index < 468:
            raise IndexError(f"Landmark index must be in [0, 467], got {index}")
        return self.points[index].copy()


@dataclass
class FeatureScoreInternal:
    """
    Internal representation of a single facial ratio score.
    
    Requirements: 6.4 (feature breakdown with measured_ratio, ideal_ratio, score, deviation)
    
    Each of the 8 golden ratio measurements is scored individually using:
        score = 100 × (1 - |measured_ratio - 1.618| / 1.618)
    
    Used during golden ratio scoring computation. Converted to FeatureScore
    Pydantic model for API response.
    
    Attributes:
        feature: Human-readable feature name (e.g., "Face Length ÷ Face Width")
        measured_ratio: Actual ratio measured from facial landmarks
        ideal_ratio: Target golden ratio (φ = 1.618)
        score: Computed score in range [0, 100]
        deviation: Difference from ideal (measured_ratio - ideal_ratio)
    """
    feature: str
    measured_ratio: float
    ideal_ratio: float  # Always 1.618 for golden ratio
    score: float
    deviation: float
    
    def __post_init__(self):
        """Validate feature score data."""
        if not 0.0 <= self.score <= 100.0:
            raise ValueError(f"Score must be in [0, 100], got {self.score}")
        if self.measured_ratio < 0.0:
            raise ValueError(f"Measured ratio must be non-negative, got {self.measured_ratio}")
        # Verify deviation consistency
        expected_deviation = self.measured_ratio - self.ideal_ratio
        if abs(self.deviation - expected_deviation) > 0.001:
            raise ValueError(
                f"Deviation {self.deviation} inconsistent with "
                f"measured_ratio {self.measured_ratio} - ideal_ratio {self.ideal_ratio}"
            )


@dataclass
class GoldenRatioResult:
    """
    Complete golden ratio analysis result.
    
    Requirements:
    - 5.1-5.8: Computation of 8 facial ratios
    - 7.1: Overall score as arithmetic mean of feature scores
    - 7.2: Overall score rounded to one decimal place
    - 7.3: Overall score bounded [0, 100]
    
    Contains the overall golden ratio score and breakdown of 8 individual
    facial ratio measurements. The overall score is the arithmetic mean of
    the 8 feature scores.
    
    Attributes:
        overall_score: Mean of all feature scores, range [0, 100], rounded to 1 decimal
        features: List of feature individual feature scores. Typically 8 for symmetric features,
                  but may be >8 if bilateral features are detected as asymmetric (>10% difference).
                  See Requirement 19.4 for asymmetric feature handling.
    """
    overall_score: float
    features: List[FeatureScoreInternal]
    
    def __post_init__(self):
        """Validate golden ratio result."""
        if not 0.0 <= self.overall_score <= 100.0:
            raise ValueError(f"Overall score must be in [0, 100], got {self.overall_score}")
        
        # Requirement 5.1-5.8: Must have at least 8 features (may be >8 with asymmetric features)
        # Requirement 19.4: Asymmetric features produce >8 entries if they differ by >10%
        if len(self.features) < 8:
            raise ValueError(f"Must have at least 8 features, got {len(self.features)}")
        
        # Verify overall score is arithmetic mean (within rounding tolerance)
        feature_scores = [f.score for f in self.features]
        expected_mean = sum(feature_scores) / len(feature_scores)
        if abs(self.overall_score - expected_mean) > 0.15:  # Allow 0.15 tolerance for rounding
            raise ValueError(
                f"Overall score {self.overall_score} is not arithmetic mean "
                f"of feature scores (expected {expected_mean:.1f})"
            )
        
        # Requirement 7.2: Verify rounding to one decimal place
        # Check if overall_score has at most 1 decimal place
        if round(self.overall_score, 1) != self.overall_score:
            raise ValueError(
                f"Overall score must be rounded to 1 decimal place, got {self.overall_score}"
            )


@dataclass
class CelebrityMatchInternal:
    """
    Internal representation of celebrity match result.
    
    Requirements: 11.1-11.4 (celebrity matching with confidence thresholding)
    
    Represents the result of comparing user face embedding against celebrity
    database using cosine similarity. Matches are only returned if confidence
    meets or exceeds 0.55 threshold.
    
    Attributes:
        name: Celebrity name (None if no match above threshold)
        confidence: Cosine similarity score (0.0-1.0, None if no match)
        dataset_source: "Bollywood Celeb Dataset" or "Pins Face Recognition"
        thumbnail_path: Relative path to celebrity thumbnail image
        message: Optional message (e.g., "No close match found in our dataset")
    """
    name: Optional[str]
    confidence: Optional[float]
    dataset_source: Optional[str] = None
    thumbnail_path: Optional[str] = None
    message: Optional[str] = None
    top_matches: Optional[List[dict]] = None
    
    def __post_init__(self):
        """Validate celebrity match data."""
        if self.confidence is not None:
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError(f"Confidence must be in [0.0, 1.0], got {self.confidence}")
        
        # If name is provided, confidence should also be provided
        if self.name is not None and self.confidence is None:
            raise ValueError("Confidence must be provided when name is set")
        
        # If no match, message should be provided
        if self.name is None and self.message is None:
            self.message = "No close match found in our dataset"


@dataclass
class OverlayDataInternal:
    """
    Internal representation of overlay visualization data.
    
    Requirements: 9.1-9.5 (overlay generation with landmarks and golden ratio guides)
    
    Contains landmark coordinates and golden ratio guide positions for
    client-side rendering of geometric overlays on user photos.
    
    Attributes:
        landmarks_2d: 468 2D landmark coordinates as (x, y) pairs
        midline_x: X-coordinate of vertical midline (face symmetry axis)
        horizontal_thirds_y: List of 4 Y-coordinates for horizontal divisions
                           Values: [face_top, third_1, third_2, face_bottom]
                           Based on golden ratio divisions of face height
        phi_spiral_center: (x, y) tuple for phi spiral center (near nose bridge)
    """
    landmarks_2d: np.ndarray  # Shape: (468, 2)
    midline_x: float
    horizontal_thirds_y: List[float]  # Should have exactly 4 values
    phi_spiral_center: Tuple[float, float]  # (x, y)
    
    def __post_init__(self):
        """Validate overlay data."""
        if self.landmarks_2d.shape != (468, 2):
            raise ValueError(
                f"2D landmarks must have shape (468, 2), got {self.landmarks_2d.shape}"
            )
        if len(self.horizontal_thirds_y) != 4:
            raise ValueError(
                f"Must have exactly 4 horizontal thirds, got {len(self.horizontal_thirds_y)}"
            )
        
        # Validate horizontal thirds are in ascending order
        if not all(self.horizontal_thirds_y[i] < self.horizontal_thirds_y[i+1] 
                   for i in range(len(self.horizontal_thirds_y) - 1)):
            raise ValueError(
                f"Horizontal thirds must be in ascending order, got {self.horizontal_thirds_y}"
            )


@dataclass
class ProcessingMetadata:
    """
    Metadata for tracking processing flow (for logging, not API response).
    
    Requirements: 13.4 (privacy - log only metadata, not image data)
    
    Used for performance monitoring and debugging without storing any
    personally identifiable information or image data.
    
    Attributes:
        request_id: Unique identifier for this request (UUID)
        start_time: Unix timestamp when processing started
        component_times: Dict mapping component names to processing times (ms)
    """
    request_id: str
    start_time: float
    component_times: dict = field(default_factory=dict)
    
    def add_component_time(self, component: str, time_ms: int) -> None:
        """
        Add timing for a component.
        
        Args:
            component: Component name (e.g., "image_processor", "face_detector")
            time_ms: Processing time in milliseconds
        """
        self.component_times[component] = time_ms
    
    def get_total_time_ms(self, current_time: float) -> int:
        """
        Calculate total time elapsed in milliseconds.
        
        Args:
            current_time: Current Unix timestamp
        
        Returns:
            Total elapsed time in milliseconds
        """
        return int((current_time - self.start_time) * 1000)
    
    def get_component_summary(self) -> dict:
        """
        Get summary of component processing times.
        
        Returns:
            Dict with component names and times for logging
        """
        return {
            "request_id": self.request_id,
            "total_ms": self.get_total_time_ms(self.start_time),
            "components": self.component_times.copy()
        }
