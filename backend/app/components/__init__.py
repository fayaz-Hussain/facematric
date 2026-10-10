"""
Core processing components for facial analysis.

Components:
- ImageProcessor: Image normalization and validation
- FaceDetector: Face detection and landmark extraction
- SymmetryClassifier: Face shape classification
- GoldenRatioScorer: Facial proportion scoring
- PercentileEngine: Score-to-percentile conversion
- CelebrityMatcher: Celebrity look-alike matching
- OverlayGenerator: Visual overlay data generation
"""

from .image_processor import ImageProcessor, image_processor
from .face_detector import FaceDetector, get_face_detector
from .symmetry_classifier import SymmetryClassifier, symmetry_classifier
from .golden_ratio_scorer import GoldenRatioScorer, golden_ratio_scorer
from .percentile_engine import PercentileEngine, percentile_engine
from .celebrity_matcher import CelebrityMatcher, get_celebrity_matcher
from .overlay_generator import OverlayGenerator, overlay_generator

__all__ = [
    "ImageProcessor",
    "image_processor",
    "FaceDetector",
    "get_face_detector",
    "SymmetryClassifier",
    "symmetry_classifier",
    "GoldenRatioScorer",
    "golden_ratio_scorer",
    "PercentileEngine",
    "percentile_engine",
    "CelebrityMatcher",
    "get_celebrity_matcher",
    "OverlayGenerator",
    "overlay_generator",
]
