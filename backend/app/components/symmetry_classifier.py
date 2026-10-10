"""
Symmetry Classifier Component

Classifies face shape into one of six symmetry types based on geometric proportions.
"""

import numpy as np
from typing import Tuple

from app.models.internal import Landmarks, SymmetryType
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SymmetryClassifier:
    """
    Classifies face shape based on facial geometry.
    
    Classifications:
    - Oval: Length/width ≈ 1.5, rounded jawline
    - Rectangle: Length > width, angular jawline
    - Round: Length ≈ width, soft angles
    - Square: Length ≈ width, angular jawline
    - Heart: Forehead > chin width significantly
    - Diamond: Cheekbone width > forehead and chin
    """
    
    # MediaPipe landmark indices for key facial points
    FACE_TOP = 10          # Forehead top
    FACE_BOTTOM = 152      # Chin bottom
    FACE_LEFT = 234        # Left face contour (widest point)
    FACE_RIGHT = 454       # Right face contour (widest point)
    JAW_LEFT = 172         # Left jaw point (alternative index)
    JAW_RIGHT = 397        # Right jaw point (alternative index)
    JAW_LEFT_ALT = 427     # Left jaw angle (alternative index)
    JAW_RIGHT_ALT = 200    # Right jaw angle (alternative index)
    CHEEKBONE_LEFT = 234   # Left cheekbone (same as face width)
    CHEEKBONE_RIGHT = 454  # Right cheekbone
    FOREHEAD_LEFT = 21     # Left forehead point
    FOREHEAD_RIGHT = 251   # Right forehead point
    
    # Nose measurements (requirement 4.1)
    NOSTRIL_LEFT = 119     # Left nostril
    NOSTRIL_RIGHT = 348    # Right nostril
    
    # Eye measurements (requirement 4.1)
    # Note: Task specified "pupil centers" at landmarks 468 and 473
    # MediaPipe has 468 landmarks (0-467), so we compute pupil center as average of eye corners
    LEFT_EYE_OUTER = 33      # Left eye outer corner
    LEFT_EYE_INNER = 133     # Left eye inner corner
    RIGHT_EYE_OUTER = 263    # Right eye outer corner
    RIGHT_EYE_INNER = 362    # Right eye inner corner
    
    def classify(self, landmarks: Landmarks) -> SymmetryType:
        """
        Classify face symmetry type based on geometric proportions.
        
        Args:
            landmarks: 468 facial landmarks
        
        Returns:
            SymmetryType: Classified face shape
        """
        points = landmarks.points
        
        # Compute key dimensions
        face_length = self._euclidean_distance(
            points[self.FACE_TOP],
            points[self.FACE_BOTTOM]
        )
        
        face_width = self._euclidean_distance(
            points[self.FACE_LEFT],
            points[self.FACE_RIGHT]
        )
        
        jaw_width = self._euclidean_distance(
            points[self.JAW_LEFT],
            points[self.JAW_RIGHT]
        )
        
        cheekbone_width = self._euclidean_distance(
            points[self.CHEEKBONE_LEFT],
            points[self.CHEEKBONE_RIGHT]
        )
        
        forehead_width = self._euclidean_distance(
            points[self.FOREHEAD_LEFT],
            points[self.FOREHEAD_RIGHT]
        )
        
        # Compute ratios
        length_to_width_ratio = face_length / face_width if face_width > 0 else 0
        
        # Compute jawline angularity
        jawline_angle = self._compute_jawline_angle(points)
        
        # Log measurements for debugging
        logger.debug(
            f"Face measurements: length/width={length_to_width_ratio:.2f}, "
            f"jawline_angle={jawline_angle:.1f}°, "
            f"forehead_width={forehead_width:.1f}, jaw_width={jaw_width:.1f}, "
            f"cheekbone_width={cheekbone_width:.1f}"
        )
        
        # Classification logic based on decision tree
        symmetry_type = self._classify_by_ratios(
            length_to_width_ratio,
            jawline_angle,
            forehead_width,
            jaw_width,
            cheekbone_width
        )
        
        logger.info(f"Face classified as: {symmetry_type.value}")
        
        return symmetry_type
    
    def _classify_by_ratios(
        self,
        length_width_ratio: float,
        jawline_angle: float,
        forehead_width: float,
        jaw_width: float,
        cheekbone_width: float
    ) -> SymmetryType:
        """
        Apply decision tree to classify face shape.
        
        Decision tree based on empirical thresholds:
        - Oval: 1.4 <= ratio <= 1.6, rounded jaw (angle > 140°)
        - Rectangle: ratio > 1.6, angular jaw (angle < 130°)
        - Round: 0.9 <= ratio <= 1.1, rounded jaw (angle > 140°)
        - Square: 0.9 <= ratio <= 1.1, angular jaw (angle < 130°)
        - Heart: forehead significantly wider than jaw
        - Diamond: cheekbones wider than forehead and jaw
        """
        # Heart shape: Forehead notably wider than jaw
        if forehead_width > jaw_width * 1.2:
            return SymmetryType.HEART
        
        # Diamond shape: Cheekbones widest feature
        if cheekbone_width > max(forehead_width, jaw_width) * 1.1:
            return SymmetryType.DIAMOND
        
        # Oval shape: Moderate ratio, rounded jawline
        if 1.4 <= length_width_ratio <= 1.6 and jawline_angle > 140:
            return SymmetryType.OVAL
        
        # Rectangle shape: Long face, angular jawline
        if length_width_ratio > 1.6 and jawline_angle < 130:
            return SymmetryType.RECTANGLE
        
        # Round shape: Similar length/width, rounded jawline
        if 0.9 <= length_width_ratio <= 1.1 and jawline_angle > 140:
            return SymmetryType.ROUND
        
        # Square shape: Similar length/width, angular jawline
        if 0.9 <= length_width_ratio <= 1.1 and jawline_angle < 130:
            return SymmetryType.SQUARE
        
        # Default to oval if borderline
        return SymmetryType.OVAL
    
    def _euclidean_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Compute 2D Euclidean distance between two landmarks.
        
        Args:
            p1: First point [x, y, z]
            p2: Second point [x, y, z]
        
        Returns:
            Distance in pixels (2D, ignoring z)
        """
        return float(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))
    
    def _compute_jawline_angle(self, points: np.ndarray) -> float:
        """
        Compute angle at chin using three jawline points.
        
        Larger angle (closer to 180°) indicates rounder jawline.
        Smaller angle (closer to 90°) indicates more angular jawline.
        
        Args:
            points: All 468 landmarks
        
        Returns:
            Angle in degrees (90-180, where 180 = perfectly round)
        """
        # Use left jaw, chin, right jaw points
        left_jaw = points[self.JAW_LEFT][:2]    # Only x, y
        chin = points[self.FACE_BOTTOM][:2]
        right_jaw = points[self.JAW_RIGHT][:2]
        
        # Vectors from chin to jaw points
        vec1 = left_jaw - chin
        vec2 = right_jaw - chin
        
        # Compute angle between vectors
        cos_angle = np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8  # Avoid division by zero
        )
        
        # Clamp to valid range for arccos
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        angle_rad = np.arccos(cos_angle)
        angle_deg = float(np.degrees(angle_rad))
        
        return angle_deg
    
    # === Facial Dimension Measurements (Requirement 4.1) ===
    
    def measure_face_length(self, landmarks: Landmarks) -> float:
        """
        Measure face length: distance from top of head (landmark 10) to chin (landmark 152).
        
        Args:
            landmarks: 468 facial landmarks
        
        Returns:
            Distance in pixels
        """
        points = landmarks.points
        top = points[self.FACE_TOP]
        bottom = points[self.FACE_BOTTOM]
        return self._euclidean_distance(top, bottom)
    
    def measure_face_width(self, landmarks: Landmarks) -> float:
        """
        Measure face width: distance between left (234) and right (454) cheekbones.
        
        Args:
            landmarks: 468 facial landmarks
        
        Returns:
            Distance in pixels
        """
        points = landmarks.points
        left = points[self.FACE_LEFT]
        right = points[self.FACE_RIGHT]
        return self._euclidean_distance(left, right)
    
    def measure_jaw_width(self, landmarks: Landmarks) -> float:
        """
        Measure jaw width: distance between left (427) and right (200) jaw angles.
        
        Args:
            landmarks: 468 facial landmarks
        
        Returns:
            Distance in pixels
        """
        points = landmarks.points
        left = points[self.JAW_LEFT_ALT]
        right = points[self.JAW_RIGHT_ALT]
        return self._euclidean_distance(left, right)
    
    def measure_nose_width(self, landmarks: Landmarks) -> float:
        """
        Measure nose width: distance between left (119) and right (348) nostrils.
        
        Args:
            landmarks: 468 facial landmarks
        
        Returns:
            Distance in pixels
        """
        points = landmarks.points
        left = points[self.NOSTRIL_LEFT]
        right = points[self.NOSTRIL_RIGHT]
        return self._euclidean_distance(left, right)
    
    def measure_eye_distance(self, landmarks: Landmarks) -> float:
        """
        Measure eye distance: distance between left and right pupil centers.
        
        Computes pupil center as the midpoint between outer and inner eye corners
        for each eye (to approximate pupil location).
        
        Requirement 4.1: Measure facial dimensions for symmetry classification.
        
        Args:
            landmarks: 468 facial landmarks
        
        Returns:
            Distance in pixels between left and right pupil centers
        """
        points = landmarks.points
        
        # Compute left eye center as midpoint between outer and inner corners
        left_outer = points[self.LEFT_EYE_OUTER]
        left_inner = points[self.LEFT_EYE_INNER]
        left_pupil_center = (left_outer + left_inner) / 2
        
        # Compute right eye center as midpoint between outer and inner corners
        right_outer = points[self.RIGHT_EYE_OUTER]
        right_inner = points[self.RIGHT_EYE_INNER]
        right_pupil_center = (right_outer + right_inner) / 2
        
        return self._euclidean_distance(left_pupil_center, right_pupil_center)
    
    def get_all_facial_dimensions(self, landmarks: Landmarks) -> dict:
        """
        Get all five facial dimension measurements at once.
        
        Requirement 4.1: Measure facial dimensions for symmetry classification.
        
        Args:
            landmarks: 468 facial landmarks
        
        Returns:
            Dictionary with all measurements:
            {
                'face_length': float,  # Top of head to chin
                'face_width': float,   # Left cheekbone to right cheekbone
                'jaw_width': float,    # Left jaw angle to right jaw angle
                'nose_width': float,   # Left nostril to right nostril
                'eye_distance': float  # Left pupil to right pupil
            }
        """
        return {
            'face_length': self.measure_face_length(landmarks),
            'face_width': self.measure_face_width(landmarks),
            'jaw_width': self.measure_jaw_width(landmarks),
            'nose_width': self.measure_nose_width(landmarks),
            'eye_distance': self.measure_eye_distance(landmarks)
        }


# Global instance
symmetry_classifier = SymmetryClassifier()
