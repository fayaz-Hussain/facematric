import math
import numpy as np
from typing import Dict, Tuple, List, Union

PHI_CONSTANT: float = 1.618

def score_ratio_oracle(ratio: float, phi: float = PHI_CONSTANT) -> float:
    """
    Independent oracle implementation of the Golden Ratio feature score formula:
    score = 100 * (1 - |ratio - phi| / phi)
    Clamped strictly to [0.0, 100.0].
    
    Returns 0.0 for invalid inputs (negative, non-finite, NaN).
    """
    if ratio is None or not math.isfinite(ratio) or ratio <= 0.0:
        return 0.0
    raw_score = 100.0 * (1.0 - abs(ratio - phi) / phi)
    return max(0.0, min(100.0, raw_score))

def euclidean_dist(p1: Union[np.ndarray, list, tuple], p2: Union[np.ndarray, list, tuple]) -> float:
    """Euclidean distance in 2D space: sqrt((x2 - x1)^2 + (y2 - y1)^2)"""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

class GoldenRatioOracle:
    """
    Independent reference implementation of facial ratio measurement and scoring.
    All calculations strictly use pixel-space 2D coordinates.
    """
    PHI = 1.618
    
    # 468 landmark indices according to FaceMetric specifications
    FACE_TOP = 10
    FACE_BOTTOM = 152
    FACE_LEFT = 234
    FACE_RIGHT = 454
    JAW_LEFT = 172
    JAW_RIGHT = 397
    LEFT_EYE_OUTER = 33
    LEFT_EYE_INNER = 133
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374
    LEFT_EYEBROW_BOTTOM = 70
    RIGHT_EYEBROW_BOTTOM = 300
    NOSE_LEFT = 219
    NOSE_RIGHT = 439
    NOSE_BRIDGE = 6
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291

    @classmethod
    def calculate_ratios(cls, points: np.ndarray) -> Dict[str, float]:
        """
        Calculates all 8 facial ratios from 2D points.
        Points array shape: (468, 2) or (468, 3) where only [:, :2] are used.
        """
        # Ratio 1: Face length / Face width
        len_face = euclidean_dist(points[cls.FACE_TOP], points[cls.FACE_BOTTOM])
        wid_face = euclidean_dist(points[cls.FACE_LEFT], points[cls.FACE_RIGHT])
        r1 = len_face / wid_face if wid_face > 0 else 0.0

        # Ratio 2: Face width / Jaw width
        wid_jaw = euclidean_dist(points[cls.JAW_LEFT], points[cls.JAW_RIGHT])
        r2 = wid_face / wid_jaw if wid_jaw > 0 else 0.0

        # Ratio 3: Eye spacing / Eye width
        eye_spacing = euclidean_dist(points[cls.LEFT_EYE_INNER], points[cls.RIGHT_EYE_INNER])
        left_eye_wid = euclidean_dist(points[cls.LEFT_EYE_OUTER], points[cls.LEFT_EYE_INNER])
        right_eye_wid = euclidean_dist(points[cls.RIGHT_EYE_INNER], points[cls.RIGHT_EYE_OUTER])
        avg_eye_wid = (left_eye_wid + right_eye_wid) / 2.0
        r3 = eye_spacing / avg_eye_wid if avg_eye_wid > 0 else 0.0

        # Ratio 4: Nose width / Mouth width
        wid_nose = euclidean_dist(points[cls.NOSE_LEFT], points[cls.NOSE_RIGHT])
        wid_mouth = euclidean_dist(points[cls.MOUTH_LEFT], points[cls.MOUTH_RIGHT])
        r4 = wid_nose / wid_mouth if wid_mouth > 0 else 0.0

        # Ratio 5: Mouth width / Face width
        r5 = wid_mouth / wid_face if wid_face > 0 else 0.0

        # Ratio 6: Vertical thirds (Lower 2/3 / Upper 1/3)
        lower_dist = euclidean_dist(points[cls.NOSE_BRIDGE], points[cls.FACE_BOTTOM])
        upper_dist = euclidean_dist(points[cls.FACE_TOP], points[cls.NOSE_BRIDGE])
        r6 = lower_dist / upper_dist if upper_dist > 0 else 0.0

        # Ratio 7: Eye-to-brow / Eye height
        # Left side
        dist_brow_left = euclidean_dist(points[cls.LEFT_EYEBROW_BOTTOM], points[cls.LEFT_EYE_TOP])
        height_eye_left = euclidean_dist(points[cls.LEFT_EYE_TOP], points[cls.LEFT_EYE_BOTTOM])
        r7_left = dist_brow_left / height_eye_left if height_eye_left > 0 else 0.0
        # Right side
        dist_brow_right = euclidean_dist(points[cls.RIGHT_EYEBROW_BOTTOM], points[cls.RIGHT_EYE_TOP])
        height_eye_right = euclidean_dist(points[cls.RIGHT_EYE_TOP], points[cls.RIGHT_EYE_BOTTOM])
        r7_right = dist_brow_right / height_eye_right if height_eye_right > 0 else 0.0
        r7 = (r7_left + r7_right) / 2.0

        # Ratio 8: Chin width / Nose width
        dist_chin = euclidean_dist(points[cls.JAW_LEFT], points[cls.JAW_RIGHT])  # Production uses JAW_LEFT to JAW_RIGHT
        r8 = dist_chin / wid_nose if wid_nose > 0 else 0.0

        return {
            "r1": r1,
            "r2": r2,
            "r3": r3,
            "r4": r4,
            "r5": r5,
            "r6": r6,
            "r7": r7,
            "r8": r8,
            "r7_left": r7_left,
            "r7_right": r7_right
        }

    @classmethod
    def calculate_production_ratios(cls, points: np.ndarray) -> Dict[str, float]:
        """
        Calculates all 8 facial ratios matching production GoldenRatioScorer exactly.
        Points array shape: (468, 2) or (468, 3).
        """
        # R1: Face Length / Face Width
        r1 = euclidean_dist(points[cls.FACE_TOP], points[cls.FACE_BOTTOM]) / euclidean_dist(points[cls.FACE_LEFT], points[cls.FACE_RIGHT])

        # R2: Bilateral Face width / Jaw width
        p_fl = points[cls.FACE_LEFT]
        p_fr = points[cls.FACE_RIGHT]
        p_fc = (p_fl + p_fr) / 2.0
        p_jl = points[cls.JAW_LEFT]
        p_jr = points[cls.JAW_RIGHT]
        p_jc = (p_jl + p_jr) / 2.0
        r2_l = euclidean_dist(p_fl, p_fc) / euclidean_dist(p_jl, p_jc)
        r2_r = euclidean_dist(p_fc, p_fr) / euclidean_dist(p_jc, p_jr)
        r2 = (r2_l + r2_r) / 2.0

        # R3: Bilateral Eye Spacing / Eye Width
        eye_spacing = euclidean_dist(points[cls.LEFT_EYE_INNER], points[cls.RIGHT_EYE_INNER])
        left_eye_w = euclidean_dist(points[cls.LEFT_EYE_OUTER], points[cls.LEFT_EYE_INNER])
        right_eye_w = euclidean_dist(points[cls.RIGHT_EYE_OUTER], points[cls.RIGHT_EYE_INNER])
        r3_l = eye_spacing / left_eye_w
        r3_r = eye_spacing / right_eye_w
        r3 = (r3_l + r3_r) / 2.0

        # R4: Nose Width / Mouth Width
        wid_nose = euclidean_dist(points[cls.NOSE_LEFT], points[cls.NOSE_RIGHT])
        wid_mouth = euclidean_dist(points[cls.MOUTH_LEFT], points[cls.MOUTH_RIGHT])
        r4 = wid_nose / wid_mouth

        # R5: Mouth Width / Face Width
        wid_face = euclidean_dist(points[cls.FACE_LEFT], points[cls.FACE_RIGHT])
        r5 = wid_mouth / wid_face

        # R6: Vertical Thirds (Nose-Chin / Hairline-Nose)
        r6 = euclidean_dist(points[cls.NOSE_BRIDGE], points[cls.FACE_BOTTOM]) / euclidean_dist(points[cls.FACE_TOP], points[cls.NOSE_BRIDGE])

        # R7: Eye-to-Brow / Eye Height (Left Eye used for both in production)
        dist_brow_l = euclidean_dist(points[cls.LEFT_EYE_TOP], points[cls.LEFT_EYEBROW_BOTTOM])
        h_eye_l = euclidean_dist(points[cls.LEFT_EYE_TOP], points[cls.LEFT_EYE_BOTTOM])
        r7_l = dist_brow_l / h_eye_l
        r7_r = r7_l  # Production hardcodes r7_right = r7_left
        r7 = (r7_l + r7_r) / 2.0

        # R8: Chin Width / Nose Width (using JAW_LEFT, JAW_RIGHT)
        dist_chin = euclidean_dist(points[cls.JAW_LEFT], points[cls.JAW_RIGHT])
        r8 = dist_chin / wid_nose

        return {
            "r1": r1, "r2": r2, "r3": r3, "r4": r4,
            "r5": r5, "r6": r6, "r7": r7, "r8": r8,
            "r2_left": r2_l, "r2_right": r2_r,
            "r3_left": r3_l, "r3_right": r3_r,
            "r7_left": r7_l, "r7_right": r7_r
        }

    @classmethod
    def calculate_overall_score_oracle(cls, feature_scores: List[float]) -> float:
        """
        Calculates the true arithmetic mean of the 8 feature scores.
        Each feature has exactly 1/8 (12.5%) weight.
        """
        assert len(feature_scores) == 8, f"Expected exactly 8 feature scores, got {len(feature_scores)}"
        mean_score = sum(feature_scores) / 8.0
        return round(max(0.0, min(100.0, mean_score)), 1)
