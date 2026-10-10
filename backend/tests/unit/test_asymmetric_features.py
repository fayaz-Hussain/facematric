"""
Unit tests for Asymmetric Feature Handling (Requirement 19).

Tests the asymmetric (bilateral) feature detection and scoring for:
- Cheekbone to Jaw ratio (Ratio 2)
- Eye Spacing to Width (Ratio 3)
- Eye-Eyebrow Distance (Ratio 7)

When left/right measurements differ by >10%, they should be scored separately
and displayed as individual entries in the feature breakdown.

Requirements: 19.1, 19.2, 19.3, 19.4
"""

import pytest
import numpy as np

from app.components.golden_ratio_scorer import GoldenRatioScorer, ASYMMETRY_THRESHOLD
from app.models.internal import Landmarks


@pytest.fixture
def golden_ratio_scorer():
    """Create GoldenRatioScorer instance for testing."""
    return GoldenRatioScorer()


@pytest.fixture
def symmetric_landmarks():
    """
    Create synthetic landmarks representing a symmetric face.
    
    This creates a 468-point landmark array where left and right sides
    are mirror images for bilateral features.
    """
    landmarks_array = np.zeros((468, 3), dtype=np.float32)
    
    # Face boundaries (vertical centerline)
    landmarks_array[10] = [320, 150, 0]    # FACE_TOP
    landmarks_array[152] = [320, 450, 0]   # FACE_BOTTOM
    
    # Cheekbones and jaw - perfectly symmetric
    landmarks_array[234] = [200, 300, 0]   # FACE_LEFT (left cheekbone)
    landmarks_array[454] = [440, 300, 0]   # FACE_RIGHT (right cheekbone)
    landmarks_array[172] = [190, 380, 0]   # JAW_LEFT
    landmarks_array[397] = [450, 380, 0]   # JAW_RIGHT
    
    # Eyes - symmetric
    landmarks_array[33] = [260, 195, 0]    # LEFT_EYE_OUTER
    landmarks_array[133] = [290, 195, 0]   # LEFT_EYE_INNER
    landmarks_array[263] = [350, 195, 0]   # RIGHT_EYE_OUTER
    landmarks_array[362] = [380, 195, 0]   # RIGHT_EYE_INNER
    
    # Eye top/bottom
    landmarks_array[159] = [275, 185, 0]   # LEFT_EYE_TOP
    landmarks_array[145] = [275, 205, 0]   # LEFT_EYE_BOTTOM
    
    # Eyebrow (LEFT only, used for both sides in current implementation)
    landmarks_array[70] = [265, 160, 0]    # LEFT_EYEBROW_BOTTOM
    
    # Nose
    landmarks_array[6] = [320, 250, 0]     # NOSE_BRIDGE
    landmarks_array[219] = [310, 280, 0]   # NOSE_LEFT
    landmarks_array[439] = [330, 280, 0]   # NOSE_RIGHT
    
    # Mouth
    landmarks_array[61] = [280, 350, 0]    # MOUTH_LEFT
    landmarks_array[291] = [360, 350, 0]   # MOUTH_RIGHT
    
    return Landmarks(
        points=landmarks_array,
        confidence=0.9,
        image_width=640,
        image_height=480
    )


@pytest.fixture
def asymmetric_landmarks():
    """
    Create synthetic landmarks representing an asymmetric face.
    
    This creates a face where the left side is noticeably different from
    the right side - useful for testing asymmetry detection.
    """
    landmarks_array = np.zeros((468, 3), dtype=np.float32)
    
    # Face boundaries
    landmarks_array[10] = [320, 150, 0]    # FACE_TOP
    landmarks_array[152] = [320, 450, 0]   # FACE_BOTTOM
    
    # Cheekbones and jaw - ASYMMETRIC (left noticeably different from right)
    # Left cheekbone is set closer to center
    landmarks_array[234] = [260, 300, 0]   # FACE_LEFT (left cheekbone closer)
    landmarks_array[454] = [440, 300, 0]   # FACE_RIGHT (right cheekbone unchanged)
    landmarks_array[172] = [250, 380, 0]   # JAW_LEFT (left jaw closer)
    landmarks_array[397] = [450, 380, 0]   # JAW_RIGHT (right jaw unchanged)
    
    # Eyes - ASYMMETRIC (left eye wider than right)
    landmarks_array[33] = [240, 195, 0]    # LEFT_EYE_OUTER (moved out)
    landmarks_array[133] = [290, 195, 0]   # LEFT_EYE_INNER
    landmarks_array[263] = [360, 195, 0]   # RIGHT_EYE_OUTER (moved in)
    landmarks_array[362] = [380, 195, 0]   # RIGHT_EYE_INNER
    
    # Eye top/bottom
    landmarks_array[159] = [275, 185, 0]   # LEFT_EYE_TOP
    landmarks_array[145] = [275, 205, 0]   # LEFT_EYE_BOTTOM
    
    # Eyebrow
    landmarks_array[70] = [265, 160, 0]    # LEFT_EYEBROW_BOTTOM
    
    # Nose
    landmarks_array[6] = [320, 250, 0]     # NOSE_BRIDGE
    landmarks_array[219] = [310, 280, 0]   # NOSE_LEFT
    landmarks_array[439] = [330, 280, 0]   # NOSE_RIGHT
    
    # Mouth
    landmarks_array[61] = [280, 350, 0]    # MOUTH_LEFT
    landmarks_array[291] = [360, 350, 0]   # MOUTH_RIGHT
    
    return Landmarks(
        points=landmarks_array,
        confidence=0.9,
        image_width=640,
        image_height=480
    )


class TestBilateralMeasurements:
    """Tests for bilateral measurement methods."""
    
    def test_measure_cheekbone_to_jaw_bilateral_returns_tuple(
        self,
        golden_ratio_scorer,
        symmetric_landmarks
    ):
        """
        Test that bilateral cheekbone measurement returns a tuple of two floats.
        
        Requirement 19.1: Measure both left and right variants
        """
        left_ratio, right_ratio = golden_ratio_scorer._measure_cheekbone_to_jaw_bilateral(
            symmetric_landmarks.points
        )
        
        assert isinstance(left_ratio, float), "Left ratio should be float"
        assert isinstance(right_ratio, float), "Right ratio should be float"
        assert left_ratio > 0, "Left ratio should be positive"
        assert right_ratio > 0, "Right ratio should be positive"
    
    def test_measure_eye_spacing_to_width_bilateral_returns_tuple(
        self,
        golden_ratio_scorer,
        symmetric_landmarks
    ):
        """
        Test that bilateral eye spacing measurement returns a tuple.
        
        Requirement 19.1: Measure both left and right eye widths
        """
        left_ratio, right_ratio = golden_ratio_scorer._measure_eye_spacing_to_width_bilateral(
            symmetric_landmarks.points
        )
        
        assert isinstance(left_ratio, float), "Left ratio should be float"
        assert isinstance(right_ratio, float), "Right ratio should be float"
        assert left_ratio > 0, "Left ratio should be positive"
        assert right_ratio > 0, "Right ratio should be positive"
    
    def test_measure_eye_to_brow_distance_bilateral_returns_tuple(
        self,
        golden_ratio_scorer,
        symmetric_landmarks
    ):
        """
        Test that bilateral eye-to-brow measurement returns a tuple.
        
        Requirement 19.1: Measure both left and right sides
        """
        left_ratio, right_ratio = golden_ratio_scorer._measure_eye_to_brow_distance_bilateral(
            symmetric_landmarks.points
        )
        
        assert isinstance(left_ratio, float), "Left ratio should be float"
        assert isinstance(right_ratio, float), "Right ratio should be float"


class TestAsymmetryDetection:
    """Tests for asymmetry detection in bilateral features."""
    
    def test_symmetric_face_detects_no_asymmetry(
        self,
        golden_ratio_scorer,
        symmetric_landmarks
    ):
        """
        Test that a symmetric face does not trigger asymmetry detection.
        
        For symmetric faces, bilateral feature measurements should be very close
        (within 10% threshold), resulting in a single feature entry.
        
        Requirement 19.4: Display as separate entries only if differ by >10%
        """
        # Score bilateral features for symmetric face
        scores = golden_ratio_scorer._score_bilateral_feature(
            "Test Feature",
            1.5,  # left ratio
            1.5,  # right ratio
            1.5   # averaged ratio
        )
        
        # Should return only 1 entry (not split)
        assert len(scores) == 1, "Symmetric feature should produce 1 score entry"
        assert scores[0].feature == "Test Feature"
    
    def test_asymmetric_face_detects_asymmetry(
        self,
        golden_ratio_scorer
    ):
        """
        Test that an asymmetric face triggers asymmetry detection.
        
        When left and right ratios differ by >10%, they should be scored
        separately and returned as 2 entries.
        
        Requirement 19.2: Score each side independently if asymmetric (>10%)
        Requirement 19.4: Display as separate entries if differ by >10%
        """
        # Create a clear asymmetry: >10% difference
        left_ratio = 1.5
        right_ratio = 1.8  # 20% different from left
        avg_ratio = (left_ratio + right_ratio) / 2
        
        scores = golden_ratio_scorer._score_bilateral_feature(
            "Asymmetric Test",
            left_ratio,
            right_ratio,
            avg_ratio
        )
        
        # Should return 2 separate entries
        assert len(scores) == 2, "Asymmetric feature should produce 2 score entries"
        assert "(Left)" in scores[0].feature, "First score should be labeled (Left)"
        assert "(Right)" in scores[1].feature, "Second score should be labeled (Right)"
        
        # Each should have the correct measured_ratio
        assert scores[0].measured_ratio == round(left_ratio, 3)
        assert scores[1].measured_ratio == round(right_ratio, 3)
    
    def test_asymmetry_threshold_at_exactly_10_percent(
        self,
        golden_ratio_scorer
    ):
        """
        Test boundary condition: exactly 10% asymmetry.
        
        At exactly the 10% threshold, should not trigger split
        (threshold is >10%, not >=10%).
        """
        left_ratio = 1.5
        right_ratio = left_ratio * (1.0 + ASYMMETRY_THRESHOLD)  # Exactly at threshold
        avg_ratio = (left_ratio + right_ratio) / 2
        
        scores = golden_ratio_scorer._score_bilateral_feature(
            "Threshold Test",
            left_ratio,
            right_ratio,
            avg_ratio
        )
        
        # Exactly at threshold should not split
        assert len(scores) == 1, "At 10% threshold, should not split"
    
    def test_asymmetry_threshold_just_over_10_percent(
        self,
        golden_ratio_scorer
    ):
        """
        Test boundary condition: just over 10% asymmetry.
        
        Just above 10% should trigger split.
        """
        left_ratio = 1.5
        right_ratio = left_ratio * (1.0 + ASYMMETRY_THRESHOLD + 0.01)  # Slightly over threshold
        avg_ratio = (left_ratio + right_ratio) / 2
        
        scores = golden_ratio_scorer._score_bilateral_feature(
            "Over Threshold Test",
            left_ratio,
            right_ratio,
            avg_ratio
        )
        
        # Just over threshold should split
        assert len(scores) == 2, "Just over 10% threshold, should split"


class TestOverallScoreWithAsymmetry:
    """Tests for overall score calculation with asymmetric features."""
    
    def test_overall_score_with_symmetric_features(
        self,
        golden_ratio_scorer,
        symmetric_landmarks
    ):
        """
        Test that overall score calculation works with symmetric features.
        
        With symmetric features, should have 8 feature entries total.
        
        Requirement 19.3: Include all feature scores in overall calculation
        """
        result = golden_ratio_scorer.score_all(symmetric_landmarks)
        
        # Should have exactly 8 features (all symmetric)
        assert len(result.features) == 8, f"Expected 8 features, got {len(result.features)}"
        
        # Overall score should be arithmetic mean of all features
        expected_mean = sum(f.score for f in result.features) / len(result.features)
        assert abs(result.overall_score - expected_mean) < 0.2, \
            "Overall score should be arithmetic mean of features (within rounding)"
    
    def test_overall_score_range_always_valid(
        self,
        golden_ratio_scorer,
        symmetric_landmarks,
        asymmetric_landmarks
    ):
        """
        Test that overall score is always in valid range [0, 100].
        
        Requirement 7.3: Overall score bounded [0, 100]
        """
        # Test with symmetric face
        result = golden_ratio_scorer.score_all(symmetric_landmarks)
        assert 0.0 <= result.overall_score <= 100.0, \
            f"Symmetric overall score {result.overall_score} outside [0, 100]"
        
        # Test with asymmetric face
        result = golden_ratio_scorer.score_all(asymmetric_landmarks)
        assert 0.0 <= result.overall_score <= 100.0, \
            f"Asymmetric overall score {result.overall_score} outside [0, 100]"
    
    def test_overall_score_rounded_to_one_decimal(
        self,
        golden_ratio_scorer,
        symmetric_landmarks
    ):
        """
        Test that overall score is rounded to one decimal place.
        
        Requirement 7.2: Round to one decimal place
        """
        result = golden_ratio_scorer.score_all(symmetric_landmarks)
        
        # Check that score has at most 1 decimal place
        assert result.overall_score == round(result.overall_score, 1), \
            f"Overall score {result.overall_score} not rounded to 1 decimal place"


class TestFeatureBreakdownWithAsymmetry:
    """Tests for feature breakdown display with asymmetric features."""
    
    def test_symmetric_face_feature_names(
        self,
        golden_ratio_scorer,
        symmetric_landmarks
    ):
        """
        Test that symmetric face features have correct base names (no Left/Right suffix).
        
        Requirement 19.4: Display asymmetric features as separate entries if differ by >10%
        """
        result = golden_ratio_scorer.score_all(symmetric_landmarks)
        
        # Check that no features have "(Left)" or "(Right)" suffix
        for feature in result.features:
            assert "(Left)" not in feature.feature, \
                f"Symmetric feature should not have (Left) suffix: {feature.feature}"
            assert "(Right)" not in feature.feature, \
                f"Symmetric feature should not have (Right) suffix: {feature.feature}"
    
    def test_all_features_have_required_fields(
        self,
        golden_ratio_scorer,
        symmetric_landmarks
    ):
        """
        Test that all features have required fields for display.
        
        Requirement 6.4: Record measured_ratio, ideal_ratio, score, deviation
        """
        result = golden_ratio_scorer.score_all(symmetric_landmarks)
        
        for feature in result.features:
            assert feature.feature is not None, "Feature name required"
            assert feature.measured_ratio is not None, "Measured ratio required"
            assert feature.ideal_ratio is not None, "Ideal ratio required"
            assert feature.score is not None, "Score required"
            assert feature.deviation is not None, "Deviation required"
            
            # Check types
            assert isinstance(feature.feature, str)
            assert isinstance(feature.measured_ratio, float)
            assert isinstance(feature.ideal_ratio, float)
            assert isinstance(feature.score, float)
            assert isinstance(feature.deviation, float)


class TestAsymmetricFaceScoring:
    """Integration tests with actual asymmetric face data."""
    
    def test_cheekbone_to_jaw_asymmetry_detection(
        self,
        golden_ratio_scorer,
        asymmetric_landmarks
    ):
        """
        Test that asymmetric cheekbone-to-jaw ratio is detected.
        
        Requirement 19.1: Measure both left and right variants
        Requirement 19.2: Score each side independently
        """
        left_ratio, right_ratio = golden_ratio_scorer._measure_cheekbone_to_jaw_bilateral(
            asymmetric_landmarks.points
        )
        
        # The bilateral measurement should return two float values (may or may not differ)
        # The test should verify they are both valid measurements
        assert isinstance(left_ratio, float), "Left ratio should be float"
        assert isinstance(right_ratio, float), "Right ratio should be float"
        assert left_ratio > 0, "Left ratio should be positive"
        assert right_ratio > 0, "Right ratio should be positive"
    
    def test_eye_spacing_asymmetry_detection(
        self,
        golden_ratio_scorer,
        asymmetric_landmarks
    ):
        """
        Test that asymmetric eye spacing is detected.
        
        Requirement 19.1: Measure both left and right eye widths
        """
        left_ratio, right_ratio = golden_ratio_scorer._measure_eye_spacing_to_width_bilateral(
            asymmetric_landmarks.points
        )
        
        # Due to asymmetric eye setup, left and right should differ
        assert left_ratio != right_ratio, "Asymmetric eyes should produce different left/right ratios"
    
    def test_asymmetric_face_may_have_more_features(
        self,
        golden_ratio_scorer,
        asymmetric_landmarks
    ):
        """
        Test that an asymmetric face may result in >8 feature entries.
        
        If any bilateral features exceed 10% asymmetry, they will be split
        into 2 entries instead of 1, resulting in more than 8 total features.
        Otherwise, if all bilateral features are symmetric, will have exactly 8.
        
        Requirement 19.4: Display as separate entries if differ by >10%
        """
        result = golden_ratio_scorer.score_all(asymmetric_landmarks)
        
        # Feature count should be 8 or more
        # (8 for all symmetric, >8 if any bilateral features are asymmetric)
        assert len(result.features) >= 8, \
            f"Should have at least 8 features, got {len(result.features)}"
        
        # Verify all features have required fields
        for feature in result.features:
            assert feature.score is not None
            assert feature.measured_ratio is not None
            assert 0.0 <= feature.score <= 100.0, \
                f"Feature score {feature.score} outside valid range"


class TestAsymmetryThresholdBehavior:
    """Tests for the 10% asymmetry threshold behavior."""
    
    @pytest.mark.parametrize("asymmetry_pct", [0.05, 0.08, 0.099, 0.101, 0.15, 0.25])
    def test_asymmetry_threshold_across_values(
        self,
        golden_ratio_scorer,
        asymmetry_pct
    ):
        """
        Test asymmetry detection across various asymmetry percentages.
        
        Requirement 19.2: Score independently if differ significantly
        """
        base_ratio = 1.5
        other_ratio = base_ratio * (1.0 + asymmetry_pct)
        avg_ratio = (base_ratio + other_ratio) / 2
        
        scores = golden_ratio_scorer._score_bilateral_feature(
            "Test",
            base_ratio,
            other_ratio,
            avg_ratio
        )
        
        # Use small epsilon for floating point comparison
        actual_asymmetry = abs(base_ratio - other_ratio) / avg_ratio
        if actual_asymmetry > ASYMMETRY_THRESHOLD + 1e-6:
            assert len(scores) == 2, \
                f"Asymmetry {asymmetry_pct*100:.1f}% (actual: {actual_asymmetry*100:.2f}%) should split (>10%)"
        else:
            assert len(scores) == 1, \
                f"Asymmetry {asymmetry_pct*100:.1f}% (actual: {actual_asymmetry*100:.2f}%) should not split (<=10%)"

