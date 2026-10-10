"""
Unit tests for Overall Score Calculation (Requirement 7).

Tests the overall score calculation as the arithmetic mean of 8 feature scores,
with proper rounding and clamping to [0, 100].

Requirement 7.1: Calculate overall score as arithmetic mean
Requirement 7.2: Round to one decimal place
Requirement 7.3: Ensure score is bounded [0, 100]
"""

import pytest
import numpy as np

from app.components.golden_ratio_scorer import GoldenRatioScorer
from app.models.internal import Landmarks, FeatureScoreInternal


@pytest.fixture
def golden_ratio_scorer():
    """Create GoldenRatioScorer instance for testing."""
    return GoldenRatioScorer()


@pytest.fixture
def perfect_landmarks():
    """
    Create synthetic landmarks for testing.
    
    These landmarks are designed to produce ideal ratios close to φ (1.618)
    for most features, resulting in high scores.
    """
    # Create 468 landmarks with z-coordinate = 0
    landmarks_array = np.zeros((468, 3), dtype=np.float32)
    
    # Face boundaries (proportioned for high scores)
    landmarks_array[10] = [320, 100, 0]    # FACE_TOP
    landmarks_array[152] = [320, 400, 0]   # FACE_BOTTOM (face_length = 300)
    landmarks_array[234] = [220, 250, 0]   # FACE_LEFT
    landmarks_array[454] = [420, 250, 0]   # FACE_RIGHT (face_width = 200)
    # Ratio: 300/200 = 1.5 (close to 1.618)
    
    # Jaw points (proportioned for high scores)
    landmarks_array[172] = [245, 350, 0]   # JAW_LEFT
    landmarks_array[397] = [395, 350, 0]   # JAW_RIGHT (jaw_width = 150)
    # Face width / Jaw width: 200/150 = 1.333 (suboptimal but valid)
    
    # Eyes
    landmarks_array[33] = [260, 200, 0]    # LEFT_EYE_OUTER
    landmarks_array[133] = [300, 200, 0]   # LEFT_EYE_INNER
    landmarks_array[263] = [340, 200, 0]   # RIGHT_EYE_OUTER
    landmarks_array[362] = [380, 200, 0]   # RIGHT_EYE_INNER
    # Eye spacing / Eye width: 80/40 = 2.0
    
    # Nose
    landmarks_array[219] = [300, 250, 0]   # NOSE_LEFT
    landmarks_array[439] = [340, 250, 0]   # NOSE_RIGHT (nose_width = 40)
    landmarks_array[6] = [320, 220, 0]     # NOSE_BRIDGE
    
    # Mouth
    landmarks_array[61] = [280, 320, 0]    # MOUTH_LEFT
    landmarks_array[291] = [360, 320, 0]   # MOUTH_RIGHT (mouth_width = 80)
    # Nose/Mouth: 40/80 = 0.5
    # Mouth/Face: 80/200 = 0.4
    
    # Eyebrow
    landmarks_array[159] = [275, 170, 0]   # LEFT_EYE_TOP
    landmarks_array[145] = [275, 210, 0]   # LEFT_EYE_BOTTOM
    landmarks_array[70] = [275, 140, 0]    # LEFT_EYEBROW_BOTTOM
    # Eye height: 40, Eye-brow: 30, Ratio: 0.75
    
    return Landmarks(
        points=landmarks_array,
        confidence=0.95,
        image_width=640,
        image_height=480
    )


class TestOverallScoreCalculation:
    """Tests for overall score calculation (Requirement 7)."""
    
    def test_overall_score_is_arithmetic_mean_of_eight_features(
        self, golden_ratio_scorer, perfect_landmarks
    ):
        """
        Requirement 7.1: Overall score must be arithmetic mean of 8 feature scores.
        
        The overall score should equal (sum of 8 feature scores) / 8.
        """
        result = golden_ratio_scorer.score_all(perfect_landmarks)
        
        # Calculate expected mean manually
        feature_scores = [f.score for f in result.features]
        expected_mean = sum(feature_scores) / len(feature_scores)
        
        # The result should match the expected mean (within rounding tolerance)
        assert abs(result.overall_score - expected_mean) < 0.15, \
            f"Overall score {result.overall_score} is not mean of features {feature_scores} (expected {expected_mean})"
    
    def test_overall_score_rounded_to_one_decimal_place(
        self, golden_ratio_scorer, perfect_landmarks
    ):
        """
        Requirement 7.2: Overall score must be rounded to one decimal place.
        
        The overall score should have at most 1 decimal place of precision.
        """
        result = golden_ratio_scorer.score_all(perfect_landmarks)
        
        # Check that rounding to 1 decimal place gives the same value
        rounded_once = round(result.overall_score, 1)
        assert result.overall_score == rounded_once, \
            f"Overall score {result.overall_score} is not rounded to 1 decimal place"
    
    def test_overall_score_bounded_between_0_and_100(
        self, golden_ratio_scorer, perfect_landmarks
    ):
        """
        Requirement 7.3: Overall score must be bounded [0, 100].
        
        Even if all features score very high or very low, the overall
        score should never exceed 100 or fall below 0.
        """
        result = golden_ratio_scorer.score_all(perfect_landmarks)
        
        assert 0.0 <= result.overall_score <= 100.0, \
            f"Overall score {result.overall_score} is outside [0, 100]"
    
    def test_exactly_eight_features_in_result(
        self, golden_ratio_scorer, perfect_landmarks
    ):
        """
        Requirement 5.1-5.8: Result must contain exactly 8 feature scores.
        
        The golden ratio scorer measures 8 ratios and should return exactly 8.
        """
        result = golden_ratio_scorer.score_all(perfect_landmarks)
        
        assert len(result.features) == 8, \
            f"Expected 8 features, got {len(result.features)}"
    
    def test_feature_breakdown_contains_all_required_fields(
        self, golden_ratio_scorer, perfect_landmarks
    ):
        """
        Requirement 6.4: Each feature must have measured_ratio, ideal_ratio, score, and deviation.
        
        Verify that each feature breakdown item contains all required fields.
        """
        result = golden_ratio_scorer.score_all(perfect_landmarks)
        
        for i, feature in enumerate(result.features):
            assert hasattr(feature, 'feature'), f"Feature {i} missing 'feature' field"
            assert hasattr(feature, 'measured_ratio'), f"Feature {i} missing 'measured_ratio' field"
            assert hasattr(feature, 'ideal_ratio'), f"Feature {i} missing 'ideal_ratio' field"
            assert hasattr(feature, 'score'), f"Feature {i} missing 'score' field"
            assert hasattr(feature, 'deviation'), f"Feature {i} missing 'deviation' field"
            
            # Verify ideal ratio is always φ (1.618)
            assert feature.ideal_ratio == golden_ratio_scorer.PHI, \
                f"Feature {i} ideal_ratio should be {golden_ratio_scorer.PHI}, got {feature.ideal_ratio}"
    
    def test_deterministic_scoring(
        self, golden_ratio_scorer, perfect_landmarks
    ):
        """
        Requirement 7.1 (determinism): Overall score must be reproducible.
        
        Running the scorer twice with the same landmarks should produce
        identical results (deterministic computation).
        """
        result1 = golden_ratio_scorer.score_all(perfect_landmarks)
        result2 = golden_ratio_scorer.score_all(perfect_landmarks)
        
        assert result1.overall_score == result2.overall_score, \
            f"Scores differ on identical input: {result1.overall_score} vs {result2.overall_score}"
        
        # All features should match too
        for i, (f1, f2) in enumerate(zip(result1.features, result2.features)):
            assert f1.score == f2.score, \
                f"Feature {i} score differs: {f1.score} vs {f2.score}"
    
    def test_feature_scores_are_bounded_0_to_100(
        self, golden_ratio_scorer, perfect_landmarks
    ):
        """
        Requirement 6.5: Each feature score must be bounded [0, 100].
        
        Before calculating the overall mean, all individual feature scores
        should already be clamped to [0, 100].
        """
        result = golden_ratio_scorer.score_all(perfect_landmarks)
        
        for i, feature in enumerate(result.features):
            assert 0.0 <= feature.score <= 100.0, \
                f"Feature {i} score {feature.score} is outside [0, 100]"
    
    def test_golden_ratio_scoring_formula(
        self, golden_ratio_scorer, perfect_landmarks
    ):
        """
        Requirement 6.1-6.3: Verify the golden ratio scoring formula.
        
        Formula: score = 100 × (1 - |ratio - φ| / φ)
        
        Test with a known ratio to verify the formula is applied correctly.
        """
        result = golden_ratio_scorer.score_all(perfect_landmarks)
        
        # Check first feature (Face Length ÷ Face Width)
        feature = result.features[0]
        
        # Manually calculate expected score
        phi = golden_ratio_scorer.PHI
        deviation = abs(feature.measured_ratio - phi)
        expected_score = 100 * (1 - deviation / phi)
        expected_score = max(0.0, min(100.0, expected_score))
        
        assert abs(feature.score - expected_score) < 0.1, \
            f"Feature score {feature.score} doesn't match formula result {expected_score}"
    
    def test_overall_score_mean_precision(
        self, golden_ratio_scorer
    ):
        """
        Test that the overall score is precisely the mean of features
        (within rounding tolerance).
        
        This ensures that no precision is lost in the calculation.
        """
        # Create test landmarks
        landmarks_array = np.zeros((468, 3), dtype=np.float32)
        
        # Set landmarks to create specific feature scores
        # Using simple ratios to verify the calculation
        landmarks_array[10] = [320, 100, 0]
        landmarks_array[152] = [320, 400, 0]
        landmarks_array[234] = [220, 250, 0]
        landmarks_array[454] = [420, 250, 0]
        landmarks_array[172] = [245, 350, 0]
        landmarks_array[397] = [395, 350, 0]
        landmarks_array[33] = [260, 200, 0]
        landmarks_array[133] = [300, 200, 0]
        landmarks_array[263] = [340, 200, 0]
        landmarks_array[362] = [380, 200, 0]
        landmarks_array[219] = [300, 250, 0]
        landmarks_array[439] = [340, 250, 0]
        landmarks_array[61] = [280, 320, 0]
        landmarks_array[291] = [360, 320, 0]
        landmarks_array[6] = [320, 220, 0]
        landmarks_array[159] = [275, 170, 0]
        landmarks_array[145] = [275, 210, 0]
        landmarks_array[70] = [275, 140, 0]
        
        landmarks = Landmarks(
            points=landmarks_array,
            confidence=0.95,
            image_width=640,
            image_height=480
        )
        
        result = golden_ratio_scorer.score_all(landmarks)
        
        # Calculate mean
        feature_scores = [f.score for f in result.features]
        calculated_mean = sum(feature_scores) / 8
        
        # The overall score should match (within small tolerance for rounding)
        assert abs(result.overall_score - round(calculated_mean, 1)) < 0.1, \
            f"Overall score {result.overall_score} differs from mean {calculated_mean}"
    
    def test_clamping_behavior_with_extreme_ratios(
        self, golden_ratio_scorer
    ):
        """
        Test that clamping works correctly when all features are either very good or very bad.
        """
        # Create landmarks with extreme dimensions
        landmarks_array = np.zeros((468, 3), dtype=np.float32)
        
        # Create very small face (all ratios will be extreme)
        landmarks_array[10] = [320, 319, 0]   # Very short face
        landmarks_array[152] = [320, 320, 0]
        landmarks_array[234] = [300, 320, 0]  # Very narrow face
        landmarks_array[454] = [340, 320, 0]
        landmarks_array[172] = [310, 320, 0]
        landmarks_array[397] = [330, 320, 0]
        landmarks_array[33] = [315, 320, 0]
        landmarks_array[133] = [325, 320, 0]
        landmarks_array[263] = [315, 320, 0]
        landmarks_array[362] = [325, 320, 0]
        landmarks_array[219] = [318, 320, 0]
        landmarks_array[439] = [322, 320, 0]
        landmarks_array[61] = [315, 320, 0]
        landmarks_array[291] = [325, 320, 0]
        landmarks_array[6] = [320, 320, 0]
        landmarks_array[159] = [320, 315, 0]
        landmarks_array[145] = [320, 325, 0]
        landmarks_array[70] = [320, 310, 0]
        
        landmarks = Landmarks(
            points=landmarks_array,
            confidence=0.95,
            image_width=640,
            image_height=480
        )
        
        result = golden_ratio_scorer.score_all(landmarks)
        
        # Even with extreme ratios, the score should be within [0, 100]
        assert 0.0 <= result.overall_score <= 100.0, \
            f"Overall score {result.overall_score} violated bounds with extreme landmarks"
    
    def test_all_eight_ratios_are_measured(
        self, golden_ratio_scorer, perfect_landmarks
    ):
        """
        Requirement 5.1-5.8: Verify all 8 ratios are measured and scored.
        
        The 8 ratios are:
        1. Face Length ÷ Face Width
        2. Face Width ÷ Jaw Width
        3. Eye Spacing ÷ Eye Width
        4. Nose Width ÷ Mouth Width
        5. Mouth Width ÷ Face Width
        6. Nose-Chin ÷ Hairline-Nose (Vertical Thirds)
        7. Eye-Eyebrow Distance ÷ Eye Height
        8. Chin Width ÷ Nose Width
        """
        result = golden_ratio_scorer.score_all(perfect_landmarks)
        
        # Note: With bilateral feature handling (Requirements 19.1-19.4),
        # Ratio 2 is now named "Cheekbone to Jaw Width" to reflect that it
        # measures cheekbone-to-jaw proportions bilaterally
        expected_feature_names = [
            "Face Length ÷ Face Width",
            "Cheekbone to Jaw Width",  # Bilateral feature
            "Eye Spacing ÷ Eye Width",  # Bilateral feature
            "Nose Width ÷ Mouth Width",
            "Mouth Width ÷ Face Width",
            "Nose-Chin ÷ Hairline-Nose (Vertical Thirds)",
            "Eye-Eyebrow Distance ÷ Eye Height",  # Bilateral feature
            "Chin Width ÷ Nose Width",
        ]
        
        actual_feature_names = [f.feature for f in result.features]
        
        assert actual_feature_names == expected_feature_names, \
            f"Feature names don't match.\nExpected: {expected_feature_names}\nActual: {actual_feature_names}"
    
    def test_score_determinism_with_numpy_seed(
        self, golden_ratio_scorer, perfect_landmarks
    ):
        """
        Test that golden ratio scoring is completely deterministic
        even across multiple calls and threads (no randomness involved).
        """
        scores = []
        for _ in range(5):
            result = golden_ratio_scorer.score_all(perfect_landmarks)
            scores.append(result.overall_score)
        
        # All scores should be identical
        assert all(s == scores[0] for s in scores), \
            f"Non-deterministic scoring: {scores}"


class TestOverallScoreValidation:
    """Tests for overall score validation through the model."""
    
    def test_golden_ratio_result_validates_overall_score(
        self, golden_ratio_scorer, perfect_landmarks
    ):
        """
        Test that GoldenRatioResult validates the overall score
        when constructed from scorer output.
        """
        result = golden_ratio_scorer.score_all(perfect_landmarks)
        
        # The result should pass validation in the model
        assert 0.0 <= result.overall_score <= 100.0
        assert result.overall_score == round(result.overall_score, 1)
        
        # The mean should be correct
        feature_scores = [f.score for f in result.features]
        expected_mean = sum(feature_scores) / len(feature_scores)
        assert abs(result.overall_score - expected_mean) < 0.15
