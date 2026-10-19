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
    
    # Midline landmarks (FM-BUG-022)
    MIDLINE_TOP = 168      # Glabella / mid-nasal bridge
    MIDLINE_BOTTOM = 152   # Menton / chin tip
    
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
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374
    
    # Eyebrows (FM-BUG-007)
    LEFT_EYEBROW_BOTTOM = 70
    RIGHT_EYEBROW_BOTTOM = 300
    
    # Nose - using widest outer alar wing landmarks (129, 358)
    NOSE_LEFT = 129
    NOSE_RIGHT = 358
    NOSE_BRIDGE = 6
    
    # Mouth
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    
    # Chin landmarks (FM-BUG-026)
    LIP_BOTTOM = 17
    CHIN_TIP = 152
    FACE_OVAL = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
        400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
        54, 103, 67, 109
    ]
    
    def score_all(self, landmarks: Landmarks) -> GoldenRatioResult:
        """
        Compute all 8 ratios and generate scores, with equal-weighted 8-feature mean.
        
        For bilateral features (cheekbone-to-jaw, eye spacing, eye-to-brow), measures
        left and right sides separately. Their mean is used as the single feature value
        in the overall average, and left, right and asymmetry % are reported as extra
        metadata (FM-BUG-009). The feature breakdown has exactly 8 features.
        
        Args:
            landmarks: 468 facial landmarks
        
        Returns:
            GoldenRatioResult with overall score and exactly 8 feature breakdown items.
        """
        points = landmarks.points
        
        # Compute all 8 ratios
        ratio_1 = self._measure_face_length_to_width(points)
        
        # Ratio 2: Cheekbone to jaw - bilateral feature measured to midline (FM-BUG-022)
        ratio_2_left, ratio_2_right = self._measure_cheekbone_to_jaw_bilateral(points)
        ratio_2_avg = (ratio_2_left + ratio_2_right) / 2.0
        
        # Ratio 3: Eye spacing - bilateral feature
        ratio_3_left, ratio_3_right = self._measure_eye_spacing_to_width_bilateral(points)
        ratio_3_avg = (ratio_3_left + ratio_3_right) / 2.0
        
        ratio_4 = self._measure_nose_to_mouth_width(points)
        ratio_5 = self._measure_mouth_to_face_width(points)
        ratio_6 = self._measure_vertical_thirds(points)
        
        # Ratio 7: Eye to brow - bilateral feature with mirror landmarks (FM-BUG-007)
        ratio_7_left, ratio_7_right = self._measure_eye_to_brow_distance_bilateral(points)
        ratio_7_avg = (ratio_7_left + ratio_7_right) / 2.0
        
        # Ratio 8: Chin width to nose width (FM-BUG-026)
        ratio_8 = self._measure_chin_to_nose_width(points)
        
        # Build exactly 8 feature scores (FM-BUG-009)
        feature_scores = [
            self._score_feature("Face Length ÷ Face Width", ratio_1),
            self._score_bilateral_feature_single(
                "Cheekbone to Jaw Width",
                ratio_2_left,
                ratio_2_right,
                ratio_2_avg
            ),
            self._score_bilateral_feature_single(
                "Eye Spacing ÷ Eye Width",
                ratio_3_left,
                ratio_3_right,
                ratio_3_avg
            ),
            self._score_feature("Nose Width ÷ Mouth Width", ratio_4),
            self._score_feature("Mouth Width ÷ Face Width", ratio_5),
            self._score_feature("Nose-Chin ÷ Hairline-Nose (Vertical Thirds) - Approx. Hairline", ratio_6),
            self._score_bilateral_feature_single(
                "Eye-Eyebrow Distance ÷ Eye Height",
                ratio_7_left,
                ratio_7_right,
                ratio_7_avg
            ),
            self._score_feature("Chin Width ÷ Nose Width", ratio_8),
        ]
        
        # Compute overall score as equal-weighted arithmetic mean of exactly 8 features
        overall_score = sum(fs.score for fs in feature_scores) / 8.0
        overall_score = max(0.0, min(100.0, overall_score))
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
        """
        deviation = ratio - self.PHI
        score = 100.0 * (1.0 - abs(deviation) / self.PHI)
        score = max(0.0, min(100.0, score))
        
        return FeatureScoreInternal(
            feature=feature_name,
            measured_ratio=round(ratio, 3),
            ideal_ratio=self.PHI,
            score=round(score, 1),
            deviation=round(deviation, 3)
        )
    
    def _score_bilateral_feature_single(
        self,
        feature_base_name: str,
        ratio_left: float,
        ratio_right: float,
        ratio_avg: float
    ) -> FeatureScoreInternal:
        """
        Score a bilateral feature using the mean ratio, reporting left/right and asymmetry % as extra metadata.
        
        FM-BUG-009: Left and right are averaged into one feature value, ensuring the overall
        score is always the mean of exactly 8 features.
        """
        deviation = ratio_avg - self.PHI
        score = 100.0 * (1.0 - abs(deviation) / self.PHI)
        score = max(0.0, min(100.0, score))
        
        asymmetry_pct = (abs(ratio_left - ratio_right) / ratio_avg * 100.0) if ratio_avg > 0 else 0.0
        
        return FeatureScoreInternal(
            feature=feature_base_name,
            measured_ratio=round(ratio_avg, 3),
            ideal_ratio=self.PHI,
            score=round(score, 1),
            deviation=round(deviation, 3),
            ratio_left=round(ratio_left, 3),
            ratio_right=round(ratio_right, 3),
            asymmetry_pct=round(asymmetry_pct, 1)
        )
    
    def _score_bilateral_feature(
        self,
        feature_base_name: str,
        ratio_left: float,
        ratio_right: float,
        ratio_avg: float
    ) -> List[FeatureScoreInternal]:
        """
        Backward-compatible helper returning a list with the single averaged bilateral feature.
        """
        return [self._score_bilateral_feature_single(feature_base_name, ratio_left, ratio_right, ratio_avg)]
    
    def _euclidean_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """Compute 2D Euclidean distance."""
        return float(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))
    
    def _perpendicular_distance_to_line(
        self, point: np.ndarray, line_p1: np.ndarray, line_p2: np.ndarray
    ) -> float:
        """
        Compute 2D perpendicular distance from point (x0, y0) to line passing through line_p1 and line_p2.
        
        Formula: |(x2 - x1)(y1 - y0) - (x1 - x0)(y2 - y1)| / sqrt((x2 - x1)^2 + (y2 - y1)^2)
        """
        x0, y0 = float(point[0]), float(point[1])
        x1, y1 = float(line_p1[0]), float(line_p1[1])
        x2, y2 = float(line_p2[0]), float(line_p2[1])
        dx = x2 - x1
        dy = y2 - y1
        line_len = np.sqrt(dx * dx + dy * dy)
        if line_len == 0.0:
            return float(np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2))
        return float(abs(dx * (y1 - y0) - (x1 - x0) * dy) / line_len)
    
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
    
    def _measure_cheekbone_to_jaw_bilateral(self, points: np.ndarray) -> Tuple[float, float]:
        """
        Ratio 2 (Bilateral): Cheekbone width to jaw width measured independently relative to facial midline.
        
        FM-BUG-022: Midline is defined by the line passing through landmarks 168 (glabella) and 152 (menton).
        Left side: perpendicular distance of left cheek (234) to midline / left jaw (172) to midline.
        Right side: perpendicular distance of right cheek (454) to midline / right jaw (397) to midline.
        """
        p_top = points[self.MIDLINE_TOP][:2]
        p_bot = points[self.MIDLINE_BOTTOM][:2]
        
        d_cheek_left = self._perpendicular_distance_to_line(points[self.FACE_LEFT][:2], p_top, p_bot)
        d_jaw_left = self._perpendicular_distance_to_line(points[self.JAW_LEFT][:2], p_top, p_bot)
        
        d_cheek_right = self._perpendicular_distance_to_line(points[self.FACE_RIGHT][:2], p_top, p_bot)
        d_jaw_right = self._perpendicular_distance_to_line(points[self.JAW_RIGHT][:2], p_top, p_bot)
        
        ratio_left = d_cheek_left / d_jaw_left if d_jaw_left > 0 else 0.0
        ratio_right = d_cheek_right / d_jaw_right if d_jaw_right > 0 else 0.0
        
        return ratio_left, ratio_right
    
    def _measure_face_to_jaw_width(self, points: np.ndarray) -> float:
        """Ratio 2: Face width ÷ jaw width (mean of bilateral measurements)."""
        r_l, r_r = self._measure_cheekbone_to_jaw_bilateral(points)
        return (r_l + r_r) / 2.0
    
    def _measure_eye_spacing_to_width_bilateral(self, points: np.ndarray) -> Tuple[float, float]:
        """
        Ratio 3 (Bilateral): Eye spacing to eye width, with left and right eye widths measured separately.
        
        Left eye width: From LEFT_EYE_OUTER (33) to LEFT_EYE_INNER (133)
        Right eye width: From RIGHT_EYE_INNER (362) to RIGHT_EYE_OUTER (263)
        Eye spacing: From LEFT_EYE_INNER (133) to RIGHT_EYE_INNER (362)
        """
        left_eye_inner = points[self.LEFT_EYE_INNER]
        right_eye_inner = points[self.RIGHT_EYE_INNER]
        eye_spacing = self._euclidean_distance(left_eye_inner, right_eye_inner)
        
        left_eye_outer = points[self.LEFT_EYE_OUTER]
        left_eye_width = self._euclidean_distance(left_eye_outer, left_eye_inner)
        
        right_eye_outer = points[self.RIGHT_EYE_OUTER]
        right_eye_width = self._euclidean_distance(right_eye_outer, right_eye_inner)
        
        ratio_left = eye_spacing / left_eye_width if left_eye_width > 0 else 0.0
        ratio_right = eye_spacing / right_eye_width if right_eye_width > 0 else 0.0
        
        return ratio_left, ratio_right
    
    def _measure_eye_spacing_to_width(self, points: np.ndarray) -> float:
        """Ratio 3: Eye spacing ÷ eye width (mean of bilateral measurements)."""
        r_l, r_r = self._measure_eye_spacing_to_width_bilateral(points)
        return (r_l + r_r) / 2.0
    
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
        """
        Ratio 6: Lower third ÷ forehead-nose distance (vertical thirds).
        
        FM-BUG-016: MediaPipe landmark 10 approximates the anatomical trichion (hairline).
        """
        hairline = points[self.FACE_TOP]
        nose_bridge = points[self.NOSE_BRIDGE]
        chin = points[self.FACE_BOTTOM]
        
        upper_third = self._euclidean_distance(hairline, nose_bridge)
        lower_two_thirds = self._euclidean_distance(nose_bridge, chin)
        
        return lower_two_thirds / upper_third if upper_third > 0 else 0.0
    
    def _measure_eye_to_brow_distance_bilateral(self, points: np.ndarray) -> Tuple[float, float]:
        """
        Ratio 7 (Bilateral): Eye-eyebrow distance ÷ eye height measured independently for left and right.
        
        FM-BUG-007: Left side uses eyebrow 70, eye top 159, eye bottom 145.
        Right side uses mirror landmarks: eyebrow 300, eye top 386, eye bottom 374.
        """
        left_eye_top = points[self.LEFT_EYE_TOP]
        left_eyebrow_bottom = points[self.LEFT_EYEBROW_BOTTOM]
        left_eye_to_brow = self._euclidean_distance(left_eye_top, left_eyebrow_bottom)
        
        left_eye_bottom = points[self.LEFT_EYE_BOTTOM]
        left_eye_height = self._euclidean_distance(left_eye_top, left_eye_bottom)
        
        ratio_left = left_eye_to_brow / left_eye_height if left_eye_height > 0 else 0.0
        
        right_eye_top = points[self.RIGHT_EYE_TOP]
        right_eyebrow_bottom = points[self.RIGHT_EYEBROW_BOTTOM]
        right_eye_to_brow = self._euclidean_distance(right_eye_top, right_eyebrow_bottom)
        
        right_eye_bottom = points[self.RIGHT_EYE_BOTTOM]
        right_eye_height = self._euclidean_distance(right_eye_top, right_eye_bottom)
        
        ratio_right = right_eye_to_brow / right_eye_height if right_eye_height > 0 else 0.0
        
        return ratio_left, ratio_right
    
    def _measure_eye_to_brow_distance(self, points: np.ndarray) -> float:
        """Ratio 7: Eye-eyebrow distance ÷ eye height (mean of bilateral measurements)."""
        r_l, r_r = self._measure_eye_to_brow_distance_bilateral(points)
        return (r_l + r_r) / 2.0
    
    def _measure_chin_to_nose_width(self, points: np.ndarray) -> float:
        """
        Ratio 8: Chin width ÷ nose width.
        
        FM-BUG-026: Chin width is the horizontal distance across the chin at the height
        halfway between the lower lip (landmark 17) and the chin tip (landmark 152),
        taken from the left and right face-oval contour points nearest to that height.
        Nose width is the horizontal breadth across the outer alar wings (landmarks 129 to 358).
        """
        y_lip = points[self.LIP_BOTTOM][1]
        y_chin = points[self.CHIN_TIP][1]
        y_mid = (y_lip + y_chin) / 2.0
        x_mid = points[self.CHIN_TIP][0]
        
        left_contour = [idx for idx in self.FACE_OVAL if points[idx][0] < x_mid]
        right_contour = [idx for idx in self.FACE_OVAL if points[idx][0] >= x_mid]
        
        if not left_contour or not right_contour:
            return 0.0
            
        best_left = min(left_contour, key=lambda idx: abs(points[idx][1] - y_mid))
        best_right = min(right_contour, key=lambda idx: abs(points[idx][1] - y_mid))
        
        chin_width = abs(float(points[best_right][0] - points[best_left][0]))
        nose_width = self._euclidean_distance(points[self.NOSE_LEFT], points[self.NOSE_RIGHT])
        
        return chin_width / nose_width if nose_width > 0 else 0.0


# Global instance
golden_ratio_scorer = GoldenRatioScorer()
