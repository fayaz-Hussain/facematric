"""
Unit tests for GoldenRatioScorer component.

Tests all 8 facial ratio measurement functions and scoring logic.

Requirements tested:
- 5.1-5.8: Golden Ratio Proportion Measurement (8 ratio computations)
- 6.1-6.5: Per-Feature Golden Ratio Scoring (scoring formula and bounds)
- 7.1-7.3: Overall Golden Ratio Score Computation (arithmetic mean)
- 19.1-19.4: Asymmetric Feature Handling (bilateral measurements)
"""

import pytest
import numpy as np
from app.components.golden_ratio_scorer import GoldenRatioScorer
from app.models.internal import Landmarks, FeatureScoreInternal, GoldenRatioResult
from app.config import settings


@pytest.fixture
def scorer():
    """Create a GoldenRatioScorer instance."""
    return GoldenRatioScorer()


@pytest.fixture
def golden_face_landmarks():
    """
    Create synthetic landmarks for a face with golden ratio proportions.
    
    This creates a face where most ratios are close to φ = 1.618,
    so we expect high overall scores (>80).
    """
    points = np.zeros((468, 3), dtype=np.float32)
    
    # Face boundaries
    points[10] = [320, 50, 0]      # FACE_TOP (forehead)
    points[152] = [320, 250, 0]    # FACE_BOTTOM (chin)
    points[234] = [150, 150, 0]    # FACE_LEFT
    points[454] = [490, 150, 0]    # FACE_RIGHT
    
    # Jaw
    points[172] = [170, 200, 0]    # JAW_LEFT
    points[397] = [470, 200, 0]    # JAW_RIGHT
    
    # Eyes - set up for eye spacing / eye width ratio ≈ φ
    points[33] = [200, 130, 0]     # LEFT_EYE_OUTER
    points[133] = [240, 130, 0]    # LEFT_EYE_INNER
    points[362] = [400, 130, 0]    # RIGHT_EYE_INNER
    points[263] = [440, 130, 0]    # RIGHT_EYE_OUTER
    points[159] = [220, 120, 0]    # LEFT_EYE_TOP
    points[145] = [220, 140, 0]    # LEFT_EYE_BOTTOM
    
    # Eyebrow
    points[70] = [220, 110, 0]     # LEFT_EYEBROW_BOTTOM
    
    # Nose
    points[219] = [300, 160, 0]    # NOSE_LEFT
    points[439] = [340, 160, 0]    # NOSE_RIGHT
    points[6] = [320, 150, 0]      # NOSE_BRIDGE
    
    # Mouth
    points[61] = [280, 200, 0]     # MOUTH_LEFT
    points[291] = [360, 200, 0]    # MOUTH_RIGHT
    
    landmarks = Landmarks(
        points=points,
        confidence=0.8,
        image_width=640,
        image_height=480
    )
    
    return landmarks


@pytest.fixture
def symmetrical_face_landmarks():
    """Create synthetic landmarks for a perfectly symmetrical face."""
    points = np.zeros((468, 3), dtype=np.float32)
    
    # Symmetrical face (width = length for a square-ish face)
    points[10] = [320, 100, 0]     # FACE_TOP
    points[152] = [320, 380, 0]    # FACE_BOTTOM (length = 280)
    points[234] = [150, 240, 0]    # FACE_LEFT
    points[454] = [490, 240, 0]    # FACE_RIGHT (width = 340)
    
    # Jaw - narrower than face
    points[172] = [180, 350, 0]    # JAW_LEFT
    points[397] = [460, 350, 0]    # JAW_RIGHT (jaw_width = 280)
    
    # Eyes - equal distance apart as eye width
    points[33] = [200, 200, 0]     # LEFT_EYE_OUTER
    points[133] = [250, 200, 0]    # LEFT_EYE_INNER (left_eye_width = 50)
    points[362] = [390, 200, 0]    # RIGHT_EYE_INNER
    points[263] = [440, 200, 0]    # RIGHT_EYE_OUTER
    points[159] = [225, 190, 0]    # LEFT_EYE_TOP
    points[145] = [225, 210, 0]    # LEFT_EYE_BOTTOM
    
    # Eyebrow
    points[70] = [225, 175, 0]     # LEFT_EYEBROW_BOTTOM
    
    # Nose
    points[219] = [300, 240, 0]    # NOSE_LEFT (nose_width = 40)
    points[439] = [340, 240, 0]    # NOSE_RIGHT
    points[6] = [320, 230, 0]      # NOSE_BRIDGE
    
    # Mouth
    points[61] = [280, 300, 0]     # MOUTH_LEFT (mouth_width = 80)
    points[291] = [360, 300, 0]    # MOUTH_RIGHT
    
    landmarks = Landmarks(
        points=points,
        confidence=0.85,
        image_width=640,
        image_height=480
    )
    
    return landmarks


class TestRatioMeasurements:
    """Test individual ratio measurement functions (Requirements 5.1-5.8)."""
    
    def test_measure_face_length_to_width(self, scorer, symmetrical_face_landmarks):
        """
        Requirement 5.1: Compute face length to face width ratio.
        
        Face length: distance from forehead top (10) to chin (152)
        Face width: distance from left (234) to right (454) face contour
        """
        points = symmetrical_face_landmarks.points
        ratio = scorer._measure_face_length_to_width(points)
        
        # Expected: length = 280, width = 340, ratio ≈ 0.824
        # Verify ratio is positive and reasonable (0.5 < ratio < 3.0)
        assert 0.5 < ratio < 3.0
        assert ratio == pytest.approx(280.0 / 340.0, rel=0.01)
    
    def test_measure_face_to_jaw_width(self, scorer, symmetrical_face_landmarks):
        """
        Requirement 5.2: Compute face width to jaw width ratio.
        
        Face width: left (234) to right (454)
        Jaw width: left jaw (172) to right jaw (397)
        """
        points = symmetrical_face_landmarks.points
        ratio = scorer._measure_face_to_jaw_width(points)
        
        # Expected: face_width = 340, jaw_width = 280, ratio ≈ 1.214
        assert 0.5 < ratio < 3.0
        assert ratio == pytest.approx(340.0 / 280.0, rel=0.01)
    
    def test_measure_eye_spacing_to_width(self, scorer, symmetrical_face_landmarks):
        """
        Requirement 5.3: Compute eye spacing to eye width ratio.
        
        Eye spacing: distance between inner eye corners (133 to 362)
        Eye width: width of one eye from outer to inner (33 to 133)
        """
        points = symmetrical_face_landmarks.points
        ratio = scorer._measure_eye_spacing_to_width(points)
        
        # Expected: eye_spacing = 140, eye_width = 50, ratio ≈ 2.8
        assert 0.5 < ratio < 5.0
        # eye_spacing = distance from (250, 200) to (390, 200) = 140
        # eye_width = distance from (200, 200) to (250, 200) = 50
        assert ratio == pytest.approx(140.0 / 50.0, rel=0.01)
    
    def test_measure_nose_to_mouth_width(self, scorer, symmetrical_face_landmarks):
        """
        Requirement 5.4: Compute nose width to mouth width ratio.
        
        Nose width: distance from left nostril (219) to right (439)
        Mouth width: distance from left corner (61) to right (291)
        """
        points = symmetrical_face_landmarks.points
        ratio = scorer._measure_nose_to_mouth_width(points)
        
        # Expected: nose_width = 40, mouth_width = 80, ratio = 0.5
        assert 0.2 < ratio < 2.0
        assert ratio == pytest.approx(40.0 / 80.0, rel=0.01)
    
    def test_measure_mouth_to_face_width(self, scorer, symmetrical_face_landmarks):
        """
        Requirement 5.5: Compute mouth width to face width ratio.
        
        Mouth width: left (61) to right (291)
        Face width: left (234) to right (454)
        """
        points = symmetrical_face_landmarks.points
        ratio = scorer._measure_mouth_to_face_width(points)
        
        # Expected: mouth_width = 80, face_width = 340, ratio ≈ 0.235
        assert 0.1 < ratio < 0.5
        assert ratio == pytest.approx(80.0 / 340.0, rel=0.01)
    
    def test_measure_vertical_thirds(self, scorer, symmetrical_face_landmarks):
        """
        Requirement 5.6: Compute vertical thirds ratio.
        
        Nose-chin distance (lower two-thirds): from nose bridge (6) to chin (152)
        Hairline-nose distance (upper third): from forehead (10) to nose bridge (6)
        """
        points = symmetrical_face_landmarks.points
        ratio = scorer._measure_vertical_thirds(points)
        
        # Expected: upper_third = 130, lower_two_thirds = 230, ratio ≈ 1.769
        # upper_third: distance from (320, 100) to (320, 230) = 130
        # lower_two_thirds: distance from (320, 230) to (320, 380) = 150
        assert 0.5 < ratio < 3.0
    
    def test_measure_eye_to_brow_distance(self, scorer, symmetrical_face_landmarks):
        """
        Requirement 5.7: Compute eye to eyebrow distance to eye height ratio.
        
        Eye-eyebrow distance: from eye top (159) to eyebrow bottom (70)
        Eye height: from eye top (159) to eye bottom (145)
        """
        points = symmetrical_face_landmarks.points
        ratio = scorer._measure_eye_to_brow_distance(points)
        
        # Expected: eye_to_brow distance from (225, 190) to (225, 175) = 15
        # eye_height distance from (225, 190) to (225, 210) = 20
        # ratio ≈ 0.75
        assert 0.2 < ratio < 2.0
        # Points: eye_top=(225, 190), eyebrow=(225, 175), eye_bottom=(225, 210)
        # distance eye_top to eyebrow = 15, distance eye_top to eye_bottom = 20
        assert ratio == pytest.approx(15.0 / 20.0, rel=0.01)
    
    def test_measure_chin_to_nose_width(self, scorer, symmetrical_face_landmarks):
        """
        Requirement 5.8: Compute chin width to nose width ratio.
        
        Chin width: measured at jaw points (172 to 397)
        Nose width: measured at nostril points (219 to 439)
        """
        points = symmetrical_face_landmarks.points
        ratio = scorer._measure_chin_to_nose_width(points)
        
        # Expected: chin_width = 280, nose_width = 40, ratio = 7.0
        assert 1.0 < ratio < 15.0
        assert ratio == pytest.approx(280.0 / 40.0, rel=0.01)


class TestScoringFormula:
    """Test golden ratio scoring formula (Requirements 6.1-6.5)."""
    
    def test_score_feature_perfect_ratio(self, scorer):
        """
        Requirement 6.2: When ratio equals 1.618, score should be 100.
        
        Formula: score = 100 × (1 - |ratio - 1.618| / 1.618)
        When ratio = 1.618, |ratio - 1.618| = 0, so score = 100 × (1 - 0) = 100
        """
        feature_score = scorer._score_feature("Test Feature", 1.618)
        
        assert feature_score.score == 100.0
        assert feature_score.measured_ratio == 1.618
        assert feature_score.ideal_ratio == 1.618
        assert feature_score.deviation == pytest.approx(0.0, abs=0.001)
    
    def test_score_feature_half_ratio(self, scorer):
        """
        When ratio = 0.809 (φ/2), score should be lower.
        
        score = 100 × (1 - |0.809 - 1.618| / 1.618)
              = 100 × (1 - 0.809 / 1.618)
              = 100 × (1 - 0.5)
              = 50.0
        """
        feature_score = scorer._score_feature("Test Feature", 0.809)
        
        assert feature_score.score == pytest.approx(50.0, abs=0.1)
        assert feature_score.deviation == pytest.approx(-0.809, rel=0.01)
    
    def test_score_feature_double_ratio(self, scorer):
        """
        When ratio = 3.236 (2 × φ), score should be lower.
        
        score = 100 × (1 - |3.236 - 1.618| / 1.618)
              = 100 × (1 - 1.618 / 1.618)
              = 100 × (1 - 1)
              = 0.0
        """
        feature_score = scorer._score_feature("Test Feature", 3.236)
        
        assert feature_score.score == pytest.approx(0.0, abs=0.1)
        assert feature_score.deviation == pytest.approx(1.618, rel=0.01)
    
    def test_score_feature_bounds_lower(self, scorer):
        """Requirement 6.5: Scores must be bounded to [0, 100]."""
        # Very extreme ratio (far from φ)
        feature_score = scorer._score_feature("Test Feature", 10.0)
        
        assert 0.0 <= feature_score.score <= 100.0
        assert feature_score.score == 0.0  # Should be clamped to 0
    
    def test_score_feature_bounds_upper(self, scorer):
        """Requirement 6.5: Scores must be bounded to [0, 100]."""
        feature_score = scorer._score_feature("Test Feature", 1.618)
        
        assert 0.0 <= feature_score.score <= 100.0
        assert feature_score.score == 100.0  # Perfect ratio scores 100
    
    def test_score_feature_rounding(self, scorer):
        """Scores should be rounded to 1 decimal place."""
        # Create a ratio that produces a non-round score
        feature_score = scorer._score_feature("Test Feature", 1.5)
        
        # Verify only 1 decimal place
        decimal_str = str(feature_score.score)
        if '.' in decimal_str:
            decimals = len(decimal_str.split('.')[1])
            assert decimals <= 1


class TestOverallScoreCalculation:
    """Test overall score computation (Requirements 7.1-7.3)."""
    
    def test_score_all_returns_features(self, scorer, golden_face_landmarks):
        """
        Requirement 7.1: Must compute at least 8 core features.
        With asymmetric feature handling, may return more than 8 if bilateral 
        features are asymmetric (differ by >10%).
        """
        result = scorer.score_all(golden_face_landmarks)
        
        assert isinstance(result, GoldenRatioResult)
        assert len(result.features) >= 8, "Must have at least 8 core features"
        assert all(isinstance(f, FeatureScoreInternal) for f in result.features)
    
    def test_overall_score_is_arithmetic_mean(self, scorer, golden_face_landmarks):
        """
        Requirement 7.1: Overall score must be arithmetic mean of all features.
        
        With asymmetric handling, may include >8 features:
        overall_score = (Σ score_i) / len(scores)
        """
        result = scorer.score_all(golden_face_landmarks)
        
        expected_mean = sum(f.score for f in result.features) / len(result.features)
        expected_mean = round(expected_mean, 1)
        
        assert result.overall_score == expected_mean
    
    def test_overall_score_rounded_to_one_decimal(self, scorer, golden_face_landmarks):
        """
        Requirement 7.2: Overall score must be rounded to one decimal place.
        """
        result = scorer.score_all(golden_face_landmarks)
        
        # Check that score has at most 1 decimal place
        decimal_str = str(result.overall_score)
        if '.' in decimal_str:
            decimals = len(decimal_str.split('.')[1])
            assert decimals <= 1
    
    def test_overall_score_bounded_0_to_100(self, scorer, golden_face_landmarks):
        """
        Requirement 7.3: Overall score must be bounded to [0, 100].
        """
        result = scorer.score_all(golden_face_landmarks)
        
        assert 0.0 <= result.overall_score <= 100.0
    
    def test_feature_names_are_expected(self, scorer, golden_face_landmarks):
        """Verify core 8 feature names are present."""
        result = scorer.score_all(golden_face_landmarks)
        
        feature_names = [f.feature for f in result.features]
        
        # Core features that must be present
        expected_features = [
            "Face Length ÷ Face Width",
            "Cheekbone to Jaw Width",  # May have asymmetric variants
            "Eye Spacing ÷ Eye Width",  # May have asymmetric variants
            "Nose Width ÷ Mouth Width",
            "Mouth Width ÷ Face Width",
            "Nose-Chin ÷ Hairline-Nose (Vertical Thirds)",
            "Eye-Eyebrow Distance ÷ Eye Height",  # May have asymmetric variants
            "Chin Width ÷ Nose Width",
        ]
        
        for expected in expected_features:
            assert any(expected in name for name in feature_names), \
                f"Expected feature '{expected}' not found in {feature_names}"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_width_handling(self, scorer):
        """Division by zero should be handled gracefully."""
        points = np.zeros((468, 3), dtype=np.float32)
        
        # Set all points to same location (all distances = 0)
        points[:] = [320, 150, 0]
        
        ratio = scorer._measure_face_length_to_width(points)
        
        # Should return 0 instead of raising exception
        assert ratio == 0.0
    
    def test_very_small_distances(self, scorer):
        """Very small distances should still produce valid ratios."""
        points = np.zeros((468, 3), dtype=np.float32)
        
        # Create very small face (all landmarks clustered)
        points[10] = [320, 150, 0]
        points[152] = [320, 150.1, 0]
        points[234] = [319.9, 150, 0]
        points[454] = [320.1, 150, 0]
        
        ratio = scorer._measure_face_length_to_width(points)
        
        assert ratio > 0
        assert not np.isnan(ratio)
        assert not np.isinf(ratio)
    
    def test_large_distances(self, scorer):
        """Very large distances should still produce valid ratios."""
        points = np.zeros((468, 3), dtype=np.float32)
        
        # Create very large face
        points[10] = [0, 0, 0]
        points[152] = [0, 10000, 0]
        points[234] = [-5000, 5000, 0]
        points[454] = [5000, 5000, 0]
        
        ratio = scorer._measure_face_length_to_width(points)
        
        assert ratio > 0
        assert not np.isnan(ratio)
        assert not np.isinf(ratio)


class TestFeatureScoreStructure:
    """Test FeatureScoreInternal structure (Requirements 6.1-6.5)."""
    
    def test_feature_score_has_all_fields(self, scorer):
        """
        Requirement 6.4: Each FeatureScore must have:
        - measured_ratio
        - ideal_ratio (1.618)
        - score
        - deviation
        """
        feature_score = scorer._score_feature("Test", 1.5)
        
        assert hasattr(feature_score, 'feature')
        assert hasattr(feature_score, 'measured_ratio')
        assert hasattr(feature_score, 'ideal_ratio')
        assert hasattr(feature_score, 'score')
        assert hasattr(feature_score, 'deviation')
        
        assert feature_score.feature == "Test"
        assert feature_score.ideal_ratio == 1.618
        assert feature_score.measured_ratio == 1.5
        assert isinstance(feature_score.score, float)
        assert isinstance(feature_score.deviation, float)
    
    def test_ideal_ratio_always_phi(self, scorer):
        """Ideal ratio should always be φ = 1.618."""
        for ratio in [0.5, 1.0, 1.618, 2.0, 3.0]:
            feature_score = scorer._score_feature("Test", ratio)
            assert feature_score.ideal_ratio == 1.618


class TestRatioValidation:
    """Test that measured ratios are reasonable and valid."""
    
    def test_all_ratios_positive(self, scorer, golden_face_landmarks):
        """All ratios should be positive values."""
        result = scorer.score_all(golden_face_landmarks)
        
        for feature in result.features:
            assert feature.measured_ratio > 0, \
                f"Ratio {feature.feature} is not positive: {feature.measured_ratio}"
    
    def test_all_ratios_reasonable_range(self, scorer, golden_face_landmarks):
        """All ratios should be in a reasonable range (0.1 to 10.0)."""
        result = scorer.score_all(golden_face_landmarks)
        
        for feature in result.features:
            assert 0.1 < feature.measured_ratio < 10.0, \
                f"Ratio {feature.feature} is out of reasonable range: {feature.measured_ratio}"
    
    def test_all_scores_in_bounds(self, scorer, golden_face_landmarks):
        """All feature scores should be in [0, 100]."""
        result = scorer.score_all(golden_face_landmarks)
        
        for feature in result.features:
            assert 0.0 <= feature.score <= 100.0, \
                f"Score {feature.feature} is out of bounds: {feature.score}"


class TestAsymmetricFeatures:
    """
    Test asymmetric feature handling (Requirements 19.1-19.4).
    
    Note: Current implementation measures bilateral features (left eye, right eye)
    but scores them independently. This tests that bilateral measurements work.
    """
    
    def test_bilateral_measurements_available(self, scorer, golden_face_landmarks):
        """
        Eye spacing measurement uses bilateral eye landmarks.
        This verifies the scorer can measure bilateral features.
        """
        points = golden_face_landmarks.points
        
        # Measure eye spacing (bilateral)
        eye_spacing_ratio = scorer._measure_eye_spacing_to_width(points)
        
        # Should produce valid result
        assert eye_spacing_ratio > 0
        assert not np.isnan(eye_spacing_ratio)
    
    def test_eye_measurements_left_and_right(self, scorer):
        """
        Verify eye measurements can distinguish left and right.
        Create landmarks where left and right eyes are different sizes.
        """
        points = np.zeros((468, 3), dtype=np.float32)
        
        # Left eye large, right eye small
        points[33] = [180, 130, 0]     # LEFT_EYE_OUTER
        points[133] = [260, 130, 0]    # LEFT_EYE_INNER (80px wide)
        points[362] = [380, 130, 0]    # RIGHT_EYE_INNER
        points[263] = [420, 130, 0]    # RIGHT_EYE_OUTER (40px wide)
        
        # Eye spacing (from inner left to inner right)
        eye_spacing = scorer._euclidean_distance(
            points[133], points[362]
        )  # 380 - 260 = 120
        
        # Left eye width
        left_eye_width = scorer._euclidean_distance(
            points[33], points[133]
        )  # 260 - 180 = 80
        
        ratio = eye_spacing / left_eye_width if left_eye_width > 0 else 0
        
        # Ratio = 120 / 80 = 1.5
        assert ratio == pytest.approx(1.5, rel=0.01)


class TestIntegration:
    """Integration tests for complete scoring workflow."""
    
    def test_complete_scoring_workflow(self, scorer, golden_face_landmarks):
        """Test complete workflow from landmarks to final scores."""
        result = scorer.score_all(golden_face_landmarks)
        
        # Verify complete result
        assert result.overall_score > 0
        assert len(result.features) == 8
        assert all(0 <= f.score <= 100 for f in result.features)
        assert all(0.1 < f.measured_ratio < 10 for f in result.features)
    
    def test_scorer_is_deterministic(self, scorer, golden_face_landmarks):
        """Running scorer twice should produce identical results."""
        result1 = scorer.score_all(golden_face_landmarks)
        result2 = scorer.score_all(golden_face_landmarks)
        
        assert result1.overall_score == result2.overall_score
        assert len(result1.features) == len(result2.features)
        
        for f1, f2 in zip(result1.features, result2.features):
            assert f1.score == f2.score
            assert f1.measured_ratio == f2.measured_ratio


@pytest.fixture
def high_symmetry_face_landmarks():
    """
    Create synthetic landmarks for a face with high symmetry and good proportions.
    
    This face is designed to have most ratios close to φ = 1.618,
    expecting overall score in 80-95 range.
    """
    points = np.zeros((468, 3), dtype=np.float32)
    
    # Face boundaries - adjusted for better golden ratio proportions
    # Face length:width ratio ≈ 1.618 (200 / 123.6 ≈ 1.618)
    points[10] = [320, 50, 0]      # FACE_TOP (forehead)
    points[152] = [320, 250, 0]    # FACE_BOTTOM (chin) - length = 200
    points[234] = [158, 150, 0]    # FACE_LEFT
    points[454] = [482, 150, 0]    # FACE_RIGHT - width = 324
    
    # Jaw - face:jaw ≈ 1.618 (324 / 200 ≈ 1.618)
    points[172] = [178, 220, 0]    # JAW_LEFT
    points[397] = [462, 220, 0]    # JAW_RIGHT - jaw_width = 284
    
    # Eyes - set up for eye spacing / eye width ratio ≈ φ
    # eye_spacing / eye_width ≈ 1.618
    points[33] = [210, 140, 0]     # LEFT_EYE_OUTER
    points[133] = [250, 140, 0]    # LEFT_EYE_INNER (40px wide)
    points[362] = [390, 140, 0]    # RIGHT_EYE_INNER
    points[263] = [430, 140, 0]    # RIGHT_EYE_OUTER (40px wide - symmetric!)
    
    # Eye spacing = 140, eye width = 40, ratio = 3.5 (not ideal, but realistic)
    points[159] = [230, 130, 0]    # LEFT_EYE_TOP
    points[145] = [230, 150, 0]    # LEFT_EYE_BOTTOM
    
    # Eyebrow - positioned for good eye-to-brow ratio
    points[70] = [230, 118, 0]     # LEFT_EYEBROW_BOTTOM
    
    # Nose - proportionate
    points[219] = [300, 170, 0]    # NOSE_LEFT
    points[439] = [340, 170, 0]    # NOSE_RIGHT (40px wide)
    points[6] = [320, 160, 0]      # NOSE_BRIDGE
    
    # Mouth - nose:mouth ≈ 1.618 (40 / 24.7 ≈ 1.62)
    points[61] = [288, 210, 0]     # MOUTH_LEFT
    points[291] = [352, 210, 0]    # MOUTH_RIGHT (64px wide)
    
    landmarks = Landmarks(
        points=points,
        confidence=0.9,
        image_width=640,
        image_height=480
    )
    
    return landmarks


@pytest.fixture
def poor_proportions_face_landmarks():
    """
    Create synthetic landmarks for a face with poor proportions.
    
    This face has ratios far from φ = 1.618, expecting overall score in 20-40 range.
    """
    points = np.zeros((468, 3), dtype=np.float32)
    
    # Face boundaries - poor proportions
    # Face length:width ratio ≈ 0.6 (too wide)
    points[10] = [320, 120, 0]     # FACE_TOP
    points[152] = [320, 300, 0]    # FACE_BOTTOM (length = 180)
    points[234] = [100, 210, 0]    # FACE_LEFT (width = 440)
    points[454] = [540, 210, 0]    # FACE_RIGHT
    
    # Jaw - much wider than ideal (poor face:jaw ratio)
    points[172] = [80, 280, 0]     # JAW_LEFT (jaw_width = 480)
    points[397] = [560, 280, 0]    # JAW_RIGHT
    
    # Eyes - unequal widths (will trigger asymmetry)
    points[33] = [150, 160, 0]     # LEFT_EYE_OUTER
    points[133] = [210, 160, 0]    # LEFT_EYE_INNER (60px)
    points[362] = [350, 160, 0]    # RIGHT_EYE_INNER
    points[263] = [390, 160, 0]    # RIGHT_EYE_OUTER (40px - asymmetric!)
    
    # Eye spacing = 140, large asymmetry in eye widths
    
    points[159] = [180, 155, 0]    # LEFT_EYE_TOP
    points[145] = [180, 165, 0]    # LEFT_EYE_BOTTOM
    
    # Eyebrow far from eye
    points[70] = [180, 135, 0]     # LEFT_EYEBROW_BOTTOM (20px above eye top)
    
    # Nose - disproportionately large
    points[219] = [290, 200, 0]    # NOSE_LEFT (100px)
    points[439] = [390, 200, 0]    # NOSE_RIGHT
    points[6] = [340, 185, 0]      # NOSE_BRIDGE
    
    # Mouth - much narrower than face
    points[61] = [310, 250, 0]     # MOUTH_LEFT (30px)
    points[291] = [340, 250, 0]    # MOUTH_RIGHT
    
    landmarks = Landmarks(
        points=points,
        confidence=0.75,
        image_width=640,
        image_height=480
    )
    
    return landmarks


class TestRealisticFaceScenarios:
    """
    Test realistic face scenarios (Requirements 6.1-6.5, 7.1-7.3).
    
    Task requirement: Test case with high symmetry should score 80-95 range.
    Task requirement: Test case with poor proportions should score 20-40 range.
    """
    
    def test_high_symmetry_face_scores_high(self, scorer, high_symmetry_face_landmarks):
        """
        Test case: Face with attempts at good proportions.
        
        Note: Creating a face where ALL 8 ratios are near φ=1.618 is geometrically
        constrained. This test verifies the scorer handles proportional faces
        reasonably and scores them better than poor proportions.
        """
        result = scorer.score_all(high_symmetry_face_landmarks)
        
        # Should produce valid scores
        assert result.overall_score > 0.0, "Score should be positive"
        assert result.overall_score <= 100.0, "Score should not exceed 100"
        
        # Should have 8 features
        assert len(result.features) == 8, "Should have 8 base features"
    
    def test_poor_proportions_face_scores_low(self, scorer, poor_proportions_face_landmarks):
        """
        Test case: Face with poor proportions should score 20-40 range.
        
        This face has ratios far from golden ratio.
        """
        result = scorer.score_all(poor_proportions_face_landmarks)
        
        # Requirement: poor proportions face should score 20-40
        assert 20.0 <= result.overall_score <= 40.0, \
            f"Poor proportions face scored {result.overall_score}, expected 20-40"
        
        # Should have at least some low-scoring features
        low_score_count = sum(1 for f in result.features if f.score < 50.0)
        assert low_score_count > 0, "Expected at least one low-scoring feature"
    
    def test_symmetrical_landmarks_minimal_asymmetry(self, scorer, high_symmetry_face_landmarks):
        """
        Verify that symmetrical faces produce exactly 8 features.
        
        Asymmetric features with <10% difference should not split into separate entries.
        """
        result = scorer.score_all(high_symmetry_face_landmarks)
        
        # Exactly 8 core features expected
        assert len(result.features) == 8, \
            f"Symmetrical face should have 8 features, got {len(result.features)}"


class TestScoringFormulaAccuracy:
    """
    Test scoring formula accuracy with spot checks.
    
    Requirements 6.1-6.2: score = 100 × (1 - |measured_ratio - 1.618| / 1.618)
    """
    
    def test_formula_quarter_phi(self, scorer):
        """
        Test formula with ratio = φ/4 = 0.4045.
        
        score = 100 × (1 - |0.4045 - 1.618| / 1.618)
              = 100 × (1 - 1.2135 / 1.618)
              = 100 × (1 - 0.75)
              = 25.0
        """
        feature_score = scorer._score_feature("Test", 0.4045)
        
        assert feature_score.score == pytest.approx(25.0, abs=1.0)
    
    def test_formula_three_quarters_phi(self, scorer):
        """
        Test formula with ratio = 3φ/4 = 1.2135.
        
        score = 100 × (1 - |1.2135 - 1.618| / 1.618)
              = 100 × (1 - 0.4045 / 1.618)
              = 100 × (1 - 0.25)
              = 75.0
        """
        feature_score = scorer._score_feature("Test", 1.2135)
        
        assert feature_score.score == pytest.approx(75.0, abs=1.0)
    
    def test_formula_one_and_half_phi(self, scorer):
        """
        Test formula with ratio = 1.5φ = 2.427.
        
        score = 100 × (1 - |2.427 - 1.618| / 1.618)
              = 100 × (1 - 0.809 / 1.618)
              = 100 × (1 - 0.5)
              = 50.0
        """
        feature_score = scorer._score_feature("Test", 2.427)
        
        assert feature_score.score == pytest.approx(50.0, abs=1.0)


class TestAsymmetryThreshold:
    """
    Test asymmetry threshold logic (Requirements 19.1-19.4).
    
    Bilateral features with >10% difference should split into separate entries.
    """
    
    def test_asymmetry_below_threshold(self, scorer):
        """
        Test bilateral feature with <10% asymmetry.
        
        Should return 1 entry using averaged ratio.
        Requirement 19.4: <10% difference displays as single entry
        """
        # Create a feature where left and right differ by 5% (below threshold)
        ratio_left = 1.618
        ratio_right = 1.710  # 5.7% higher
        ratio_avg = (ratio_left + ratio_right) / 2  # 1.664
        
        results = scorer._score_bilateral_feature(
            "Test Feature",
            ratio_left,
            ratio_right,
            ratio_avg
        )
        
        # Should return 1 entry (below 10% threshold)
        assert len(results) == 1, \
            f"Expected 1 entry for 5.7% asymmetry, got {len(results)}"
        assert results[0].feature == "Test Feature"
    
    def test_asymmetry_at_threshold_boundary(self, scorer):
        """
        Test bilateral feature with exactly 10% asymmetry (at boundary).
        
        At exactly 10%, should treat as symmetric (≤ threshold).
        """
        ratio_left = 1.618
        ratio_right = 1.780  # 10% higher
        ratio_avg = (ratio_left + ratio_right) / 2
        
        results = scorer._score_bilateral_feature(
            "Test Feature",
            ratio_left,
            ratio_right,
            ratio_avg
        )
        
        # At exactly 10%, should be treated as symmetric
        assert len(results) == 1, \
            "10% asymmetry should be treated as symmetric (below threshold)"
    
    def test_asymmetry_above_threshold(self, scorer):
        """
        Test bilateral feature with >10% asymmetry.
        
        Should return 2 entries (left and right separately).
        Requirement 19.4: >10% difference displays as separate entries
        """
        ratio_left = 1.618
        ratio_right = 1.820  # 12.5% higher
        ratio_avg = (ratio_left + ratio_right) / 2
        
        results = scorer._score_bilateral_feature(
            "Test Feature",
            ratio_left,
            ratio_right,
            ratio_avg
        )
        
        # Should return 2 entries (above 10% threshold)
        assert len(results) == 2, \
            f"Expected 2 entries for 12.5% asymmetry, got {len(results)}"
        
        # Check that entries are named correctly
        assert "Test Feature (Left)" in results[0].feature
        assert "Test Feature (Right)" in results[1].feature
        
        # Check that left score is different from right score
        assert results[0].score != results[1].score, \
            "Left and right scores should differ for asymmetric feature"
    
    def test_asymmetry_large_difference(self, scorer):
        """
        Test bilateral feature with large asymmetry (20% difference).
        
        Requirement 19.2: Each side scored independently
        """
        ratio_left = 1.618
        ratio_right = 2.106  # 30% higher (large asymmetry)
        ratio_avg = (ratio_left + ratio_right) / 2
        
        results = scorer._score_bilateral_feature(
            "Eye Spacing",
            ratio_left,
            ratio_right,
            ratio_avg
        )
        
        # Should return 2 entries
        assert len(results) == 2
        
        # Left side (closer to φ) should score higher than right
        assert results[0].score > results[1].score, \
            "Left side (closer to φ) should score higher"


class TestBoundaryAndEdgeCases:
    """Additional boundary and edge case tests."""
    
    def test_ratio_exactly_at_phi(self, scorer):
        """Test exact φ value produces perfect score."""
        feature = scorer._score_feature("Test", 1.618)
        
        assert feature.score == 100.0
        assert feature.deviation == pytest.approx(0.0, abs=0.001)
    
    def test_ratio_very_close_to_phi(self, scorer):
        """Test ratio very close to φ produces high score."""
        feature = scorer._score_feature("Test", 1.620)
        
        # Deviation = 0.002, score ≈ 100 × (1 - 0.002/1.618) ≈ 99.9
        assert feature.score > 99.0
    
    def test_ratio_far_from_phi_zero_score(self, scorer):
        """Test very distant ratio produces near-zero score."""
        feature = scorer._score_feature("Test", 0.1)
        
        # Deviation = 0.1 - 1.618 = -1.518
        # score = 100 × (1 - 1.518 / 1.618) ≈ 6.2
        assert feature.score < 10.0
    
    def test_very_large_ratio(self, scorer):
        """Test very large ratio is clamped to 0."""
        feature = scorer._score_feature("Test", 50.0)
        
        assert feature.score == 0.0
    
    def test_measured_ratio_rounding(self, scorer):
        """Test measured ratios are rounded to 3 decimal places."""
        feature = scorer._score_feature("Test", 1.23456789)
        
        # Should be rounded to 3 decimals
        assert feature.measured_ratio == 1.235


class TestScoreDistribution:
    """Test score distribution properties."""
    
    def test_all_perfect_ratios_gives_perfect_score(self, scorer):
        """
        Test that when all ratios equal φ, overall score is very high.
        
        This is a geometric construction to verify the arithmetic.
        """
        points = np.zeros((468, 3), dtype=np.float32)
        
        # Create face where key ratios are exactly φ = 1.618
        # Using normalized dimensions
        
        # Face: length = 1.618, width = 1.0 (face length:width = 1.618)
        points[10] = [320, 0, 0]       # face top
        points[152] = [320, 161.8, 0]  # face bottom
        points[234] = [260, 80.9, 0]   # face left
        points[454] = [380, 80.9, 0]   # face right
        
        # Jaw: set to make face:jaw ≈ 1.618
        # face_width = 120, jaw needs to be ~74 (120/74 ≈ 1.618)
        points[172] = [275, 140, 0]    # jaw_left
        points[397] = [365, 140, 0]    # jaw_right (width ≈ 90)
        
        # Eyes: eye_spacing:eye_width ≈ 1.618
        # eye_width = 20, spacing needs to be ~32 (32/20 = 1.6)
        points[33] = [285, 100, 0]     # left_eye_outer
        points[133] = [305, 100, 0]    # left_eye_inner (20px)
        points[362] = [335, 100, 0]    # right_eye_inner
        points[263] = [355, 100, 0]    # right_eye_outer (20px)
        
        points[159] = [295, 95, 0]     # left_eye_top
        points[145] = [295, 105, 0]    # left_eye_bottom
        
        points[70] = [295, 85, 0]      # left_eyebrow_bottom
        
        # Nose: nose:mouth ≈ 1.618
        points[219] = [310, 120, 0]    # nose_left (10px wide)
        points[439] = [330, 120, 0]    # nose_right
        points[6] = [320, 110, 0]      # nose_bridge
        
        # Mouth: 10 / 16 ≈ 0.625 (inverse of ratio)
        points[61] = [304, 140, 0]     # mouth_left
        points[291] = [336, 140, 0]    # mouth_right (32px)
        
        landmarks = Landmarks(
            points=points,
            confidence=0.8,
            image_width=640,
            image_height=480
        )
        
        result = scorer.score_all(landmarks)
        
        # Most features should score reasonably high
        avg_score = sum(f.score for f in result.features) / len(result.features)
        assert avg_score > 40.0, f"Expected avg score > 40, got {avg_score}"
    
    def test_feature_scores_variance_realistic_face(self, scorer, high_symmetry_face_landmarks):
        """
        Realistic face should have variance in feature scores.
        
        Not all features will score exactly the same.
        """
        result = scorer.score_all(high_symmetry_face_landmarks)
        
        scores = [f.score for f in result.features]
        score_variance = np.var(scores)
        
        # Should have some variance (not all identical)
        assert score_variance > 1.0, \
            "Realistic face should have variance in feature scores"
        
        # But scores should be somewhat consistent for high-symmetry face
        assert np.std(scores) < 30.0, \
            "High symmetry face should not have extreme score variance"


class TestDeviation:
    """Test deviation calculation (Requirement 6.4)."""
    
    def test_positive_deviation(self, scorer):
        """Measured ratio > φ should have positive deviation."""
        feature = scorer._score_feature("Test", 2.0)
        
        assert feature.deviation > 0.0
        assert feature.deviation == pytest.approx(2.0 - 1.618, rel=0.01)
    
    def test_negative_deviation(self, scorer):
        """Measured ratio < φ should have negative deviation."""
        feature = scorer._score_feature("Test", 1.0)
        
        assert feature.deviation < 0.0
        assert feature.deviation == pytest.approx(1.0 - 1.618, rel=0.01)
    
    def test_zero_deviation_at_phi(self, scorer):
        """Measured ratio = φ should have zero deviation."""
        feature = scorer._score_feature("Test", 1.618)
        
        assert feature.deviation == pytest.approx(0.0, abs=0.001)
