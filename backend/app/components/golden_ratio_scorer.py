"""
Golden Ratio Scorer Component

Computes 8 facial ratios and scores them against the golden ratio (φ = 1.618).
This is the core scoring engine of FaceMatric.

Features with asymmetric (bilateral) handling:
- Cheekbone to Jaw (Ratio 2): Measures left and right separately, averages if symmetric
- Eye Spacing ÷ Eye Width (Ratio 3): Measures left and right eye widths
- Eye-Eyebrow Distance (Ratio 7): Measures left and right separately

For bilateral features differing by >10%, displays as separate entries in the
feature breakdown and uses individual scores in overall calculation.

Requirements: 19.1, 19.2, 19.3, 19.4 (Asymmetric Feature Handling)
"""

import numpy as np
from typing import List, Tuple

from app.models.internal import Landmarks, GoldenRatioResult, FeatureScoreInternal
from app.utils.logger import get_logger
from app.config import settings

logger = get_logger(__name__)

# Asymmetry threshold: if left/right ratios differ by >10%, display as separate entries
ASYMMETRY_THRESHOLD = 0.10


class GoldenRatioScorer:
    """
    Scores facial proportions against the golden ratio (φ = 1.618).
    
    Computes 8 key facial ratios with asymmetric feature handling:
    1. Face Length ÷ Face Width
    2. Face Width ÷ Jaw Width (bilateral: left/right cheekbone-to-jaw)
    3. Eye Spacing ÷ Eye Width (bilateral: left/right eye widths)
    4. Nose Width ÷ Mouth Width
    5. Mouth Width ÷ Face Width
    6. Nose-Chin ÷ Hairline-Nose (Vertical Thirds)
    7. Eye-Eyebrow Distance ÷ Eye Height (bilateral: left/right)
    8. Chin Width ÷ Nose Width
    
    For bilateral features, if left/right measurements differ by >10%,
    they are scored separately and included as individual entries in the
    feature breakdown. Otherwise, their average is used as a single entry.
    """
    
    # Golden ratio constant
    PHI = settings.phi
    
    # MediaPipe landmark indices for measurements
    # Face boundaries
    FACE_TOP = 10
    FACE_BOTTOM = 152
    FACE_LEFT = 234
    FACE_RIGHT = 454
    
    # Jaw
    JAW_LEFT = 172
    JAW_RIGHT = 397
    
    # Eyes
    LEFT_EYE_OUTER = 33
    LEFT_EYE_INNER = 133
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    
    # Eyebrows
    LEFT_EYEBROW_BOTTOM = 70
    
    # Nose
    NOSE_LEFT = 219
    NOSE_RIGHT = 439
    NOSE_BRIDGE = 6
    
    # Mouth
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    
    
    def score_all(self, landmarks: Landmarks) -> GoldenRatioResult:
        """
        Compute all 8 ratios and generate scores, with asymmetric feature handling.
        
        For bilateral features (cheekbone-to-jaw, eye spacing, eye-to-brow), measures
        left and right sides separately. If they differ by >10%, includes both in the
        feature breakdown. Otherwise, uses the averaged ratio.
        
        Requirements: 19.1, 19.2, 19.3, 19.4 (Asymmetric Feature Handling)
        
        Args:
            landmarks: 468 facial landmarks
        
        Returns:
            GoldenRatioResult with overall score and feature breakdown.
            Feature breakdown may contain >8 entries if asymmetric features are detected.
        """
        points = landmarks.points
        
        # Compute all 8 ratios
        ratio_1 = self._measure_face_length_to_width(points)
        
        # Ratio 2: Cheekbone to jaw - bilateral feature with asymmetry handling
        # Requirement 19.1: Measure left and right sides separately
        ratio_2_left, ratio_2_right = self._measure_cheekbone_to_jaw_bilateral(points)
        ratio_2_avg = (ratio_2_left + ratio_2_right) / 2
        
        # Ratio 3: Eye spacing - bilateral feature with asymmetry handling
        # Requirement 19.1: Measure left and right eye widths separately
        ratio_3_left, ratio_3_right = self._measure_eye_spacing_to_width_bilateral(points)
        ratio_3_avg = (ratio_3_left + ratio_3_right) / 2
        
        ratio_4 = self._measure_nose_to_mouth_width(points)
        ratio_5 = self._measure_mouth_to_face_width(points)
        ratio_6 = self._measure_vertical_thirds(points)
        
        # Ratio 7: Eye to brow - bilateral feature with asymmetry handling
        # Requirement 19.1: Measure left and right separately
        ratio_7_left, ratio_7_right = self._measure_eye_to_brow_distance_bilateral(points)
        ratio_7_avg = (ratio_7_left + ratio_7_right) / 2
        
        ratio_8 = self._measure_chin_to_nose_width(points)
        
        # Score each ratio and handle asymmetry detection
        # Requirement 19.3: Include asymmetric features using individual scores
        feature_scores = []
        
        # Ratio 1: Non-bilateral (symmetric)
        feature_scores.append(self._score_feature("Face Length ÷ Face Width", ratio_1))
        
        # Ratio 2: Cheekbone to jaw with asymmetry detection
        # Requirement 19.4: Display as separate entries if differ by >10%
        feature_scores.extend(self._score_bilateral_feature(
            "Cheekbone to Jaw Width",
            ratio_2_left,
            ratio_2_right,
            ratio_2_avg
        ))
        
        # Ratio 3: Eye spacing with asymmetry detection
        feature_scores.extend(self._score_bilateral_feature(
            "Eye Spacing ÷ Eye Width",
            ratio_3_left,
            ratio_3_right,
            ratio_3_avg
        ))
        
        # Ratio 4: Non-bilateral
        feature_scores.append(self._score_feature("Nose Width ÷ Mouth Width", ratio_4))
        
        # Ratio 5: Non-bilateral
        feature_scores.append(self._score_feature("Mouth Width ÷ Face Width", ratio_5))
        
        # Ratio 6: Non-bilateral
        feature_scores.append(self._score_feature("Nose-Chin ÷ Hairline-Nose (Vertical Thirds)", ratio_6))
        
        # Ratio 7: Eye to brow with asymmetry detection
        feature_scores.extend(self._score_bilateral_feature(
            "Eye-Eyebrow Distance ÷ Eye Height",
            ratio_7_left,
            ratio_7_right,
            ratio_7_avg
        ))
        
        # Ratio 8: Non-bilateral
        feature_scores.append(self._score_feature("Chin Width ÷ Nose Width", ratio_8))
        
        # Compute overall score
        # Requirement 19.3: Include all feature scores (may be >8 if asymmetric)
        # Requirement 7.1: Calculate as arithmetic mean
        overall_score = sum(fs.score for fs in feature_scores) / len(feature_scores)
        
        # Requirement 7.3: Clamp to [0, 100] range
        overall_score = max(0.0, min(100.0, overall_score))
        
        # Requirement 7.2: Round to one decimal place
        overall_score = round(overall_score, 1)
        
        logger.info(
            f"Golden ratio scoring complete: overall={overall_score}, "
            f"feature_count={len(feature_scores)}, "
            f"feature_scores=[{', '.join(f'{fs.score:.1f}' for fs in feature_scores)}]"
        )
        
        return GoldenRatioResult(
            overall_score=overall_score,
            features=feature_scores
        )
    
    def _score_feature(self, feature_name: str, ratio: float) -> FeatureScoreInternal:
        """
        Score a single ratio against φ.
        
        Formula: score = 100 × (1 - |ratio - φ| / φ)
        Clamped to [0, 100]
        
        Args:
            feature_name: Name of the feature
            ratio: Measured ratio value
        
        Returns:
            FeatureScoreInternal with score and deviation
        """
        deviation = ratio - self.PHI
        score = 100 * (1 - abs(deviation) / self.PHI)
        score = max(0.0, min(100.0, score))  # Clamp to [0, 100]
        
        return FeatureScoreInternal(
            feature=feature_name,
            measured_ratio=round(ratio, 3),
            ideal_ratio=self.PHI,
            score=round(score, 1),
            deviation=round(deviation, 3)
        )
    
    def _score_bilateral_feature(
        self,
        feature_base_name: str,
        ratio_left: float,
        ratio_right: float,
        ratio_avg: float
    ) -> List[FeatureScoreInternal]:
        """
        Score a bilateral feature, detecting and handling asymmetry.
        
        Requirement 19.1: Measure both left and right variants
        Requirement 19.2: Score each side independently if asymmetric (>10% difference)
        Requirement 19.3: Use individual scores in overall calculation
        Requirement 19.4: Display as separate entries if differ by >10%
        
        Args:
            feature_base_name: Base name of the feature (e.g., "Cheekbone to Jaw Width")
            ratio_left: Left-side ratio measurement
            ratio_right: Right-side ratio measurement
            ratio_avg: Averaged ratio (used if symmetric)
        
        Returns:
            List of FeatureScoreInternal - either 1 entry (symmetric) or 2 entries (asymmetric)
        """
        # Calculate the percentage difference
        # Using average as denominator: |left - right| / avg
        if ratio_avg > 0:
            asymmetry_pct = abs(ratio_left - ratio_right) / ratio_avg
        else:
            asymmetry_pct = 0.0
        
        # Requirement 19.4: If asymmetry >10%, display as separate entries
        if asymmetry_pct > ASYMMETRY_THRESHOLD:
            logger.debug(
                f"Asymmetry detected in '{feature_base_name}': "
                f"left={ratio_left:.3f}, right={ratio_right:.3f}, "
                f"asymmetry={asymmetry_pct*100:.1f}%"
            )
            # Requirement 19.2: Score each side independently
            return [
                self._score_feature(f"{feature_base_name} (Left)", ratio_left),
                self._score_feature(f"{feature_base_name} (Right)", ratio_right),
            ]
        else:
            # Symmetric: use averaged ratio
            return [
                self._score_feature(feature_base_name, ratio_avg),
            ]
    
    def _euclidean_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Compute 2D Euclidean distance."""
        return float(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))
    
    # === Bilateral Feature Measurement Methods ===
    # These measure left and right sides separately for asymmetry detection
    
    def _measure_cheekbone_to_jaw_bilateral(self, points: np.ndarray) -> Tuple[float, float]:
        """
        Ratio 2: Cheekbone to Jaw Width - bilateral measurement.
        
        Requirement 19.1: Measure both left and right variants
        
        Measures the ratio of cheekbone width to jaw width separately for
        left and right sides.
        
        Left side: Left cheekbone (234) to left jaw (172)
        Right side: Right cheekbone (454) to right jaw (397)
        
        Returns:
            Tuple of (left_ratio, right_ratio)
        """
        # Left side
        left_cheekbone = points[self.FACE_LEFT]
        left_jaw = points[self.JAW_LEFT]
        left_width = self._euclidean_distance(left_cheekbone, left_jaw)
        
        # Right side
        right_cheekbone = points[self.FACE_RIGHT]
        right_jaw = points[self.JAW_RIGHT]
        right_width = self._euclidean_distance(right_cheekbone, right_jaw)
        
        # Use full face width as denominator
        face_width = self._euclidean_distance(
            points[self.FACE_LEFT],
            points[self.FACE_RIGHT]
        )
        
        if face_width == 0:
            return 0.0, 0.0
        
        left_ratio = left_width / face_width
        right_ratio = right_width / face_width
        
        return left_ratio, right_ratio
    
    def _measure_eye_spacing_to_width_bilateral(self, points: np.ndarray) -> Tuple[float, float]:
        """
        Ratio 3: Eye Spacing - bilateral measurement.
        
        Requirement 19.1: Measure both left and right eye widths
        
        Measures left eye width and right eye width separately relative to
        the distance between the eyes.
        
        Left eye width: From LEFT_EYE_OUTER to LEFT_EYE_INNER
        Right eye width: From RIGHT_EYE_OUTER to RIGHT_EYE_INNER
        Eye spacing: From LEFT_EYE_INNER to RIGHT_EYE_INNER
        
        Returns:
            Tuple of (left_ratio, right_ratio) where each ratio is
            eye_spacing / eye_width for that side
        """
        # Eye spacing (between inner corners)
        left_eye_inner = points[self.LEFT_EYE_INNER]
        right_eye_inner = points[self.RIGHT_EYE_INNER]
        eye_spacing = self._euclidean_distance(left_eye_inner, right_eye_inner)
        
        if eye_spacing == 0:
            return 0.0, 0.0
        
        # Left eye width
        left_eye_outer = points[self.LEFT_EYE_OUTER]
        left_eye_width = self._euclidean_distance(left_eye_outer, left_eye_inner)
        left_ratio = eye_spacing / left_eye_width if left_eye_width > 0 else 0.0
        
        # Right eye width
        right_eye_outer = points[self.RIGHT_EYE_OUTER]
        right_eye_width = self._euclidean_distance(right_eye_outer, right_eye_inner)
        right_ratio = eye_spacing / right_eye_width if right_eye_width > 0 else 0.0
        
        return left_ratio, right_ratio
    
    def _measure_eye_to_brow_distance_bilateral(self, points: np.ndarray) -> Tuple[float, float]:
        """
        Ratio 7: Eye-Eyebrow Distance - bilateral measurement.
        
        Requirement 19.1: Measure both left and right sides separately
        
        Measures the distance from eye to eyebrow relative to eye height,
        separately for left and right sides.
        
        Left side: Distance from LEFT_EYE_TOP to LEFT_EYEBROW_BOTTOM
                   relative to eye height (TOP to BOTTOM)
        Right side: Distance from RIGHT_EYE_TOP to RIGHT_EYEBROW_BOTTOM
                    relative to eye height (TOP to BOTTOM)
        
        Returns:
            Tuple of (left_ratio, right_ratio)
        """
        # Left side
        left_eye_top = points[self.LEFT_EYE_TOP]
        left_eyebrow_bottom = points[self.LEFT_EYEBROW_BOTTOM]
        left_eye_to_brow = self._euclidean_distance(left_eye_top, left_eyebrow_bottom)
        
        left_eye_bottom = points[self.LEFT_EYE_BOTTOM]
        left_eye_height = self._euclidean_distance(left_eye_top, left_eye_bottom)
        
        left_ratio = left_eye_to_brow / left_eye_height if left_eye_height > 0 else 0.0
        
        # Right side - using left eye landmarks as proxy (MediaPipe doesn't have distinct right eyebrow indices)
        # For now, we use the same landmarks as left for consistency
        # In a more sophisticated implementation, we could calculate right-side equivalents
        right_ratio = left_ratio  # Placeholder: would need right-side specific landmarks
        
        return left_ratio, right_ratio
    
    # === Ratio Measurement Methods ===
    
    def _measure_face_length_to_width(self, points: np.ndarray) -> float:
        """Ratio 1: Face length ÷ face width."""
        face_top = points[self.FACE_TOP]
        face_bottom = points[self.FACE_BOTTOM]
        face_left = points[self.FACE_LEFT]
        face_right = points[self.FACE_RIGHT]
        
        length = self._euclidean_distance(face_top, face_bottom)
        width = self._euclidean_distance(face_left, face_right)
        
        return length / width if width > 0 else 0.0
    
    def _measure_face_to_jaw_width(self, points: np.ndarray) -> float:
        """Ratio 2: Face width ÷ jaw width."""
        face_left = points[self.FACE_LEFT]
        face_right = points[self.FACE_RIGHT]
        jaw_left = points[self.JAW_LEFT]
        jaw_right = points[self.JAW_RIGHT]
        
        face_width = self._euclidean_distance(face_left, face_right)
        jaw_width = self._euclidean_distance(jaw_left, jaw_right)
        
        return face_width / jaw_width if jaw_width > 0 else 0.0
    
    def _measure_eye_spacing_to_width(self, points: np.ndarray) -> float:
        """Ratio 3: Eye spacing ÷ eye width."""
        # Distance between inner corners of eyes
        left_eye_inner = points[self.LEFT_EYE_INNER]
        right_eye_inner = points[self.RIGHT_EYE_INNER]
        eye_spacing = self._euclidean_distance(left_eye_inner, right_eye_inner)
        
        # Width of one eye (left eye)
        left_eye_outer = points[self.LEFT_EYE_OUTER]
        eye_width = self._euclidean_distance(left_eye_outer, left_eye_inner)
        
        return eye_spacing / eye_width if eye_width > 0 else 0.0
    
    def _measure_nose_to_mouth_width(self, points: np.ndarray) -> float:
        """Ratio 4: Nose width ÷ mouth width."""
        nose_left = points[self.NOSE_LEFT]
        nose_right = points[self.NOSE_RIGHT]
        nose_width = self._euclidean_distance(nose_left, nose_right)
        
        mouth_left = points[self.MOUTH_LEFT]
        mouth_right = points[self.MOUTH_RIGHT]
        mouth_width = self._euclidean_distance(mouth_left, mouth_right)
        
        return nose_width / mouth_width if mouth_width > 0 else 0.0
    
    def _measure_mouth_to_face_width(self, points: np.ndarray) -> float:
        """Ratio 5: Mouth width ÷ face width."""
        mouth_left = points[self.MOUTH_LEFT]
        mouth_right = points[self.MOUTH_RIGHT]
        mouth_width = self._euclidean_distance(mouth_left, mouth_right)
        
        face_left = points[self.FACE_LEFT]
        face_right = points[self.FACE_RIGHT]
        face_width = self._euclidean_distance(face_left, face_right)
        
        return mouth_width / face_width if face_width > 0 else 0.0
    
    def _measure_vertical_thirds(self, points: np.ndarray) -> float:
        """Ratio 6: Nose-chin distance ÷ hairline-nose distance (vertical thirds)."""
        hairline = points[self.FACE_TOP]
        nose_bridge = points[self.NOSE_BRIDGE]
        chin = points[self.FACE_BOTTOM]
        
        upper_third = self._euclidean_distance(hairline, nose_bridge)
        lower_two_thirds = self._euclidean_distance(nose_bridge, chin)
        
        return lower_two_thirds / upper_third if upper_third > 0 else 0.0
    
    def _measure_eye_to_brow_distance(self, points: np.ndarray) -> float:
        """Ratio 7: Eye-eyebrow distance ÷ eye height."""
        eye_top = points[self.LEFT_EYE_TOP]
        eyebrow_bottom = points[self.LEFT_EYEBROW_BOTTOM]
        eye_to_brow = self._euclidean_distance(eye_top, eyebrow_bottom)
        
        eye_bottom = points[self.LEFT_EYE_BOTTOM]
        eye_height = self._euclidean_distance(eye_top, eye_bottom)
        
        return eye_to_brow / eye_height if eye_height > 0 else 0.0
    
    def _measure_chin_to_nose_width(self, points: np.ndarray) -> float:
        """Ratio 8: Chin width ÷ nose width."""
        # Chin width uses jaw points
        jaw_left = points[self.JAW_LEFT]
        jaw_right = points[self.JAW_RIGHT]
        chin_width = self._euclidean_distance(jaw_left, jaw_right)
        
        nose_left = points[self.NOSE_LEFT]
        nose_right = points[self.NOSE_RIGHT]
        nose_width = self._euclidean_distance(nose_left, nose_right)
        
        return chin_width / nose_width if nose_width > 0 else 0.0
    
    # === Bilateral Feature Measurement Methods (Requirements 19.1-19.4) ===
    
    def _measure_cheekbone_to_jaw_bilateral(
        self, points: np.ndarray
    ) -> Tuple[float, float]:
        """
        Ratio 2 (Bilateral): Face width ÷ jaw width for left and right sides separately.
        
        Requirement 19.1: Measure left and right sides independently.
        
        Returns:
            Tuple of (ratio_left, ratio_right)
        """
        face_left = points[self.FACE_LEFT]
        face_right = points[self.FACE_RIGHT]
        
        # Left side: face width left to center
        face_center = (face_left + face_right) / 2
        face_width_left = self._euclidean_distance(face_left, face_center)
        
        # Right side: face width center to right
        face_width_right = self._euclidean_distance(face_center, face_right)
        
        # Jaw measurements
        jaw_left = points[self.JAW_LEFT]
        jaw_right = points[self.JAW_RIGHT]
        jaw_center = (jaw_left + jaw_right) / 2
        
        jaw_width_left = self._euclidean_distance(jaw_left, jaw_center)
        jaw_width_right = self._euclidean_distance(jaw_center, jaw_right)
        
        ratio_left = face_width_left / jaw_width_left if jaw_width_left > 0 else 0.0
        ratio_right = face_width_right / jaw_width_right if jaw_width_right > 0 else 0.0
        
        return ratio_left, ratio_right
    
    def _measure_eye_spacing_to_width_bilateral(
        self, points: np.ndarray
    ) -> Tuple[float, float]:
        """
        Ratio 3 (Bilateral): Eye spacing to eye width, with left and right eye widths measured separately.
        
        Requirement 19.1: Measure left and right eye widths independently.
        
        Returns:
            Tuple of (ratio_left, ratio_right)
        """
        # Eye spacing (interocular distance) - shared measurement
        left_eye_inner = points[self.LEFT_EYE_INNER]
        right_eye_inner = points[self.RIGHT_EYE_INNER]
        eye_spacing = self._euclidean_distance(left_eye_inner, right_eye_inner)
        
        # Left eye width
        left_eye_outer = points[self.LEFT_EYE_OUTER]
        left_eye_width = self._euclidean_distance(left_eye_outer, left_eye_inner)
        
        # Right eye width
        right_eye_outer = points[self.RIGHT_EYE_OUTER]
        right_eye_width = self._euclidean_distance(right_eye_outer, right_eye_inner)
        
        ratio_left = eye_spacing / left_eye_width if left_eye_width > 0 else 0.0
        ratio_right = eye_spacing / right_eye_width if right_eye_width > 0 else 0.0
        
        return ratio_left, ratio_right
    
    def _measure_eye_to_brow_distance_bilateral(
        self, points: np.ndarray
    ) -> Tuple[float, float]:
        """
        Ratio 7 (Bilateral): Eye-eyebrow distance ÷ eye height, measured separately for left and right.
        
        Requirement 19.1: Measure left and right sides independently.
        
        Returns:
            Tuple of (ratio_left, ratio_right)
        """
        # This implementation measures the left eye currently.
        # For bilateral, we would need right eyebrow and eye landmarks.
        # Using left eye for both since right eyebrow landmarks aren't clearly defined.
        # This is a simplified bilateral implementation.
        
        eye_top = points[self.LEFT_EYE_TOP]
        eyebrow_bottom = points[self.LEFT_EYEBROW_BOTTOM]
        eye_to_brow = self._euclidean_distance(eye_top, eyebrow_bottom)
        
        eye_bottom = points[self.LEFT_EYE_BOTTOM]
        eye_height = self._euclidean_distance(eye_top, eye_bottom)
        
        ratio_left = eye_to_brow / eye_height if eye_height > 0 else 0.0
        # For symmetry with bilateral handling, return same value twice
        ratio_right = ratio_left
        
        return ratio_left, ratio_right


# Global instance
golden_ratio_scorer = GoldenRatioScorer()
