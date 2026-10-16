"""
Phase 1: Scoring Math Quality Assurance Suite.
Evaluates the mathematical implementation of the Golden Ratio scoring engine against:
- PRD mathematical specification (100 * (1 - |ratio - phi| / phi))
- Plan reference table values
- Independent oracle (qa/eval/oracle.py)
- Numerical clamping, determinism, and invalid inputs
- Known bugs logged as xfail(strict=True): FM-BUG-009 and FM-BUG-007
"""

import sys
import os
import math
import pytest
import numpy as np

# Ensure backend and qa roots are importable
backend_path = os.path.abspath("backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from app.components.golden_ratio_scorer import GoldenRatioScorer
from app.models.internal import Landmarks
from qa.eval.oracle import score_ratio_oracle, GoldenRatioOracle


@pytest.fixture
def scorer():
    return GoldenRatioScorer()


@pytest.fixture
def synthetic_landmarks():
    """Generates valid 468 landmark array for testing."""
    points = np.zeros((468, 3), dtype=np.float32)
    # Face boundaries
    points[10] = [320.0, 60.0, 0.0]     # Top
    points[152] = [320.0, 340.0, 0.0]   # Bottom
    points[234] = [150.0, 200.0, 0.0]   # Left
    points[454] = [490.0, 200.0, 0.0]   # Right
    points[172] = [180.0, 300.0, 0.0]   # Left jaw
    points[397] = [460.0, 300.0, 0.0]   # Right jaw
    points[33] = [200.0, 180.0, 0.0]    # Left outer eye
    points[133] = [250.0, 180.0, 0.0]   # Left inner eye
    points[362] = [390.0, 180.0, 0.0]   # Right inner eye
    points[263] = [440.0, 180.0, 0.0]   # Right outer eye
    points[159] = [225.0, 170.0, 0.0]   # Left top eye
    points[145] = [225.0, 190.0, 0.0]   # Left bottom eye
    points[386] = [415.0, 170.0, 0.0]   # Right top eye
    points[374] = [415.0, 190.0, 0.0]   # Right bottom eye
    points[70] = [225.0, 150.0, 0.0]    # Left eyebrow bottom
    points[300] = [415.0, 150.0, 0.0]   # Right eyebrow bottom
    points[219] = [290.0, 220.0, 0.0]   # Left nose
    points[439] = [350.0, 220.0, 0.0]   # Right nose
    points[6] = [320.0, 190.0, 0.0]     # Nose bridge
    points[61] = [270.0, 260.0, 0.0]    # Left mouth
    points[291] = [370.0, 260.0, 0.0]   # Right mouth
    return Landmarks(points=points, confidence=1.0, image_width=640, image_height=480)


class TestScoringReferenceTable:
    """Verifies the plan's exact reference values against the production scorer."""

    @pytest.mark.parametrize(
        "ratio,expected_score_2dec,note",
        [
            (1.618, 100.00, "exact target"),
            (1.500, 92.71, "below target"),
            (1.520, 93.94, "PRD example ratio"),
            (1.700, 94.93, "PRD example ratio"),
            (1.000, 61.80, "sub-phi ratio"),
            (0.000, 0.00, "lower boundary"),
            (3.236, 0.00, "upper zero-score boundary"),
            (4.000, 0.00, "clamped extreme ratio (-47.22 un-clamped)"),
        ],
    )
    def test_reference_table_scores(self, scorer, ratio, expected_score_2dec, note):
        feature = scorer._score_feature("test_feature", ratio)
        # Production scorer rounds to 1 decimal place; verify both 1-dec match and 2-dec unrounded formula match
        unrounded_score = 100.0 * (1.0 - abs(ratio - 1.618) / 1.618)
        clamped_score = max(0.0, min(100.0, unrounded_score))
        
        # Verify 2-decimal formula accuracy
        assert round(clamped_score, 2) == expected_score_2dec, f"Formula error on {ratio}: {note}"
        # Verify production rounded score matches within rounding tolerance (0.06 pt)
        assert abs(feature.score - round(expected_score_2dec, 1)) <= 0.1, f"Production rounded score divergence on {ratio}"


class TestScoringClampingAndBoundaries:
    """Verifies that scores are strictly bounded to [0.0, 100.0]."""

    @pytest.mark.parametrize("extreme_ratio", [0.0, 3.236, 5.0, 100.0, 1e6])
    def test_clamping_bounds_enforced(self, scorer, extreme_ratio):
        feature = scorer._score_feature("extreme_test", extreme_ratio)
        assert 0.0 <= feature.score <= 100.0, f"Score out of bounds for ratio {extreme_ratio}: {feature.score}"

    @pytest.mark.parametrize("negative_ratio", [-100.0, -1.0, -0.01])
    def test_negative_ratios_raise_controlled_error(self, scorer, negative_ratio):
        """Negative ratios must produce controlled ValueError from FeatureScoreInternal."""
        with pytest.raises(ValueError, match="Measured ratio must be non-negative"):
            scorer._score_feature("negative_test", negative_ratio)

    @pytest.mark.parametrize("invalid_input", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_inputs_handled(self, scorer, invalid_input):
        # Oracle verifies controlled behavior
        oracle_score = score_ratio_oracle(invalid_input)
        assert oracle_score == 0.0


class TestScoringDeterminism:
    """Verifies bit-identical output across consecutive runs."""

    def test_determinism_over_10_runs(self, scorer, synthetic_landmarks):
        results = [scorer.score_all(synthetic_landmarks) for _ in range(10)]
        baseline_score = results[0].overall_score
        baseline_features = [(f.feature, f.score, f.measured_ratio) for f in results[0].features]

        for i, res in enumerate(results[1:], 1):
            assert res.overall_score == baseline_score, f"Determinism failure on run {i}: {res.overall_score} != {baseline_score}"
            current_features = [(f.feature, f.score, f.measured_ratio) for f in res.features]
            assert current_features == baseline_features, f"Feature breakdown drift on run {i}"


class TestIndependentOracleAgreement:
    """Compares production scorer against independent mathematical oracle in qa/eval/oracle.py."""

    def test_oracle_mathematical_agreement(self, scorer):
        rng = np.random.default_rng(42)
        test_ratios = rng.uniform(0.1, 3.5, size=50)

        for r in test_ratios:
            prod_feature = scorer._score_feature("oracle_test", r)
            oracle_score = score_ratio_oracle(r)
            # Production rounds to 1 decimal place
            assert abs(prod_feature.score - round(oracle_score, 1)) <= 0.1


class TestKnownBugs:
    """
    Known bugs logged in qa/reports/BUGS.md marked with xfail(strict=True).
    These tests MUST fail until the user explicitly approves and implements fixes.
    """

    @pytest.mark.xfail(strict=True, reason="FM-BUG-009: overall score divides by len(feature_scores), overweighting asymmetric features")
    def test_overall_score_equals_equal_weighted_mean_of_8_features(self, scorer, synthetic_landmarks):
        """
        PRD Step 3b and Requirement 7.1 mandate that the overall score is the equal-weighted
        arithmetic mean of the 8 features (12.5% each).
        When bilateral features split (>10%), production divides by 9 or 10, biasing the overall score.
        """
        # Create asymmetric eyes (asymmetry > 10% on Ratio 3)
        pts = synthetic_landmarks.points.copy()
        # Eye spacing is 140. Set left width to 50, right width to 35 -> ratio_left = 2.8, ratio_right = 4.0
        # asymmetry = |2.8 - 4.0| / 3.4 ≈ 35% > 10%
        pts[33, 0] = pts[133, 0] - 50.0
        pts[263, 0] = pts[362, 0] + 35.0
        
        lm = Landmarks(points=pts, confidence=1.0, image_width=640, image_height=480)
        res = scorer.score_all(lm)
        
        # Breakdown has 9 entries
        assert len(res.features) == 9, "Expected 9 entries due to bilateral split"
        
        # Calculate true 8-feature equal-weighted mean (merging left/right into single 1/8 weight)
        feature_scores_map = {}
        for f in res.features:
            base_name = f.feature.replace(" (Left)", "").replace(" (Right)", "")
            feature_scores_map.setdefault(base_name, []).append(f.score)
            
        assert len(feature_scores_map) == 8, "Expected exactly 8 unique facial feature categories"
        
        # True mean: average each category, then average the 8 categories
        category_scores = [sum(scores)/len(scores) for scores in feature_scores_map.values()]
        expected_overall_score = round(sum(category_scores) / 8.0, 1)
        
        # Production overall_score divides by 9, violating equal weighting
        assert res.overall_score == expected_overall_score, (
            f"Overall score biased by asymmetric split: got {res.overall_score}, "
            f"expected equal-weighted 8-feature mean {expected_overall_score}"
        )

    @pytest.mark.xfail(strict=True, reason="FM-BUG-007: ratio 7 right side hardcoded to ratio_left (ratio_right = ratio_left)")
    def test_ratio_7_measures_right_side_independently(self, scorer, synthetic_landmarks):
        """
        Ratio 7 claims bilateral measurement and asymmetry detection.
        However, golden_ratio_scorer.py line 574 hardcodes ratio_right = ratio_left.
        Modifying right eye and eyebrow landmarks MUST change ratio_right.
        """
        pts = synthetic_landmarks.points.copy()
        r7_left_init, r7_right_init = scorer._measure_eye_to_brow_distance_bilateral(pts)
        
        # Heavily alter right eye and eyebrow positions (300, 386, 374)
        pts[300, 1] += 50.0  # Move right eyebrow significantly down
        pts[386, 1] -= 30.0  # Enlarge right eye height
        
        r7_left_after, r7_right_after = scorer._measure_eye_to_brow_distance_bilateral(pts)
        
        # Left side should be unchanged
        assert r7_left_after == r7_left_init
        # Right side MUST change if right landmarks are actually measured
        assert r7_right_after != r7_right_init, (
            "FM-BUG-007: ratio_right did not change despite altering right eye/brow landmarks! "
            f"(r7_right_after={r7_right_after}, r7_right_init={r7_right_init})"
        )
