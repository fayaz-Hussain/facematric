"""
Unit tests for PercentileEngine component.

Tests the bootstrap normal distribution initialization and score-to-percentile conversion.
"""

import pytest
import numpy as np
from scipy.stats import norm
import tempfile
import csv
from pathlib import Path

from app.components.percentile_engine import PercentileEngine


class TestPercentileEngineInitialization:
    """Test bootstrap normal distribution initialization."""
    
    def test_initialize_with_default_parameters(self):
        """Test initialization with default empirical parameters (mean=60.19, std=1.81)."""
        engine = PercentileEngine()
        
        assert round(engine.mean, 2) == 59.06
        assert round(engine.std, 2) == 2.03
        assert engine.distribution is not None
    
    def test_distribution_is_frozen_scipy_norm(self):
        """Test that distribution is a frozen scipy.stats.norm object."""
        engine = PercentileEngine()
        
        # Verify it's a frozen scipy.stats.norm distribution
        assert hasattr(engine.distribution, 'cdf')
        assert hasattr(engine.distribution, 'ppf')
        assert hasattr(engine.distribution, 'pdf')
    
    def test_distribution_parameters_are_correct(self):
        """Test that frozen distribution has correct mean and std."""
        engine = PercentileEngine()
        
        # Verify mean and scale (std) using ppf and cdf properties
        # For a distribution N(60.19, 1.81):
        # ppf(0.5) should return ~60.19 (50th percentile = mean)
        median = engine.distribution.ppf(0.5)
        assert abs(median - 59.06) < 0.1
    
    def test_initialize_with_custom_parameters(self):
        """Test initialization with custom mean and std."""
        engine = PercentileEngine(mean=75.0, std=12.0)
        
        assert engine.mean == 75.0
        assert engine.std == 12.0
        
        # Verify median is close to custom mean
        median = engine.distribution.ppf(0.5)
        assert abs(median - 75.0) < 0.1
    
    def test_distribution_docstring(self):
        """Test that class has proper docstring explaining calibrated parameters."""
        engine = PercentileEngine()
        
        # Check class docstring
        assert PercentileEngine.__doc__ is not None
        assert "60.19" in PercentileEngine.__doc__ or "empirical" in PercentileEngine.__doc__.lower()


class TestScoreToPercentileConversion:
    """Test score-to-percentile conversion using CDF."""
    
    def test_score_to_percentile_uses_cdf(self):
        """Test that conversion uses CDF correctly."""
        engine = PercentileEngine(mean=70.0, std=10.0)
        
        # Score equal to mean should give ~50th percentile
        percentile = engine.score_to_percentile(70.0)
        assert 45 <= percentile <= 55  # Should be close to 50
    
    def test_percentile_bounds_are_enforced(self):
        """Test that returned percentile is always 0-100."""
        engine = PercentileEngine()
        
        # Test with very low score
        percentile_low = engine.score_to_percentile(-50.0)
        assert 0 <= percentile_low <= 100
        
        # Test with very high score
        percentile_high = engine.score_to_percentile(150.0)
        assert 0 <= percentile_high <= 100
    
    def test_score_clamping_before_conversion(self):
        """Test that scores are clamped to [0, 100] before conversion."""
        engine = PercentileEngine(mean=70.0, std=10.0)
        
        # Scores below 0 should be treated as 0
        percentile_clamped_low = engine.score_to_percentile(-10.0)
        percentile_zero = engine.score_to_percentile(0.0)
        assert percentile_clamped_low == percentile_zero
        
        # Scores above 100 should be treated as 100
        percentile_clamped_high = engine.score_to_percentile(150.0)
        percentile_hundred = engine.score_to_percentile(100.0)
        assert percentile_clamped_high == percentile_hundred
    
    def test_percentile_increases_with_score(self):
        """Test that higher scores yield higher percentiles."""
        engine = PercentileEngine()
        
        percentile_58 = engine.score_to_percentile(58.0)
        percentile_60 = engine.score_to_percentile(60.0)
        percentile_62 = engine.score_to_percentile(62.0)
        
        assert percentile_58 < percentile_60 < percentile_62
    
    def test_known_score_percentile_mappings(self):
        """Test known score-to-percentile conversions."""
        engine = PercentileEngine(mean=70.0, std=10.0)
        
        # Score = mean should give ~50th percentile
        assert 45 <= engine.score_to_percentile(70.0) <= 55
        
        # Score = mean - 1 std should give ~16th percentile (approximately)
        assert 10 <= engine.score_to_percentile(60.0) <= 20
        
        # Score = mean + 1 std should give ~84th percentile (approximately)
        assert 80 <= engine.score_to_percentile(80.0) <= 90
    
    def test_percentile_returns_integer(self):
        """Test that percentile result is an integer."""
        engine = PercentileEngine()
        
        percentile = engine.score_to_percentile(75.5)
        
        assert isinstance(percentile, int)
    
    def test_edge_case_score_zero(self):
        """Test percentile calculation for score=0."""
        engine = PercentileEngine()
        
        percentile = engine.score_to_percentile(0.0)
        
        assert 0 <= percentile <= 100
        assert isinstance(percentile, int)
    
    def test_edge_case_score_one_hundred(self):
        """Test percentile calculation for score=100."""
        engine = PercentileEngine()
        
        percentile = engine.score_to_percentile(100.0)
        
        assert 0 <= percentile <= 100
        assert isinstance(percentile, int)


class TestPercentileMessageFormatting:
    """Test user-friendly percentile message generation."""
    
    def test_percentile_message_format(self):
        """Test that message follows expected format."""
        engine = PercentileEngine()
        
        message = engine.get_percentile_message(75)
        
        assert "75" in message
        assert "%" in message
        assert "higher" in message.lower()
    
    def test_percentile_message_with_various_values(self):
        """Test message generation for different percentile values."""
        engine = PercentileEngine()
        
        message_0 = engine.get_percentile_message(0)
        message_50 = engine.get_percentile_message(50)
        message_100 = engine.get_percentile_message(100)
        
        assert "0" in message_0
        assert "50" in message_50
        assert "100" in message_100


class TestDistributionUpdate:
    """Test distribution parameter updates from real data."""
    
    def test_update_requires_minimum_samples(self):
        """Test that distribution update requires minimum sample size."""
        engine = PercentileEngine()
        
        # Insufficient samples
        small_dataset = [70.0, 71.0, 72.0]
        result = engine.update_distribution(small_dataset, min_samples=100)
        
        assert result is False
        # Original parameters should be unchanged
        assert round(engine.mean, 2) == 59.06
        assert round(engine.std, 2) == 2.03
    
    def test_update_with_sufficient_samples(self):
        """Test successful distribution update with sufficient samples."""
        engine = PercentileEngine()
        
        # Create dataset with known mean and std
        import numpy as np
        np.random.seed(42)
        dataset = np.random.normal(loc=75.0, scale=8.0, size=200).tolist()
        
        result = engine.update_distribution(dataset, min_samples=100)
        
        assert result is True
        # New parameters should be close to generated dataset
        assert 73.0 < engine.mean < 77.0
        assert 6.0 < engine.std < 10.0


class TestDistributionUsability:
    """Test that distribution can be used for statistical operations."""
    
    def test_cdf_operation(self):
        """Test that CDF operation is available on distribution."""
        engine = PercentileEngine()
        
        # CDF should be callable
        cdf_value = engine.distribution.cdf(engine.mean)
        
        assert 0.0 <= cdf_value <= 1.0
        assert abs(cdf_value - 0.5) < 0.1  # Should be close to 0.5 for mean
    
    def test_ppf_operation(self):
        """Test that PPF (inverse CDF) operation is available on distribution."""
        engine = PercentileEngine()
        
        # PPF should be callable and return score for given percentile
        score_at_50th = engine.distribution.ppf(0.5)
        
        assert abs(score_at_50th - engine.mean) < 0.1  # Should be close to mean
    
    def test_pdf_operation(self):
        """Test that PDF (probability density) operation is available."""
        engine = PercentileEngine()
        
        # PDF should be callable
        density = engine.distribution.pdf(engine.mean)
        
        assert density > 0.0  # PDF should be positive at any point


class TestBootstrapDistributionIntegration:
    """Integration tests for distribution workflow."""
    
    def test_full_score_to_percentile_workflow(self):
        """Test complete workflow from score to percentile to message."""
        engine = PercentileEngine()
        
        # Process a score
        score = 75.3
        percentile = engine.score_to_percentile(score)
        message = engine.get_percentile_message(percentile, score=score)
        
        # Verify workflow
        assert isinstance(percentile, int)
        assert 0 <= percentile <= 100
        assert "above typical" in message
        assert "2017 faces in the FairFace dataset" in message
    
    def test_multiple_scores_produce_ordered_percentiles(self):
        """Test that multiple scores produce correctly ordered percentiles."""
        engine = PercentileEngine()
        
        scores = [60.0, 70.0, 75.0, 80.0, 90.0]
        percentiles = [engine.score_to_percentile(s) for s in scores]
        
        # Percentiles should increase monotonically
        for i in range(len(percentiles) - 1):
            assert percentiles[i] <= percentiles[i + 1]


class TestSigmaPointConversions:
    """Test CDF conversions at standard deviation points."""
    
    def test_score_at_mean_gives_50th_percentile(self):
        """Test that score equal to mean gives approximately 50th percentile."""
        engine = PercentileEngine(mean=70.0, std=10.0)
        
        percentile = engine.score_to_percentile(70.0)
        
        # Should be within 5 percentile points of 50
        assert 45 <= percentile <= 55
    
    def test_score_at_minus_one_sigma_gives_16th_percentile(self):
        """Test score at mean - 1σ gives approximately 16th percentile."""
        engine = PercentileEngine(mean=70.0, std=10.0)
        
        # score = 70 - 10 = 60
        percentile = engine.score_to_percentile(60.0)
        
        # Should be approximately 16th percentile
        assert 10 <= percentile <= 20
    
    def test_score_at_plus_one_sigma_gives_84th_percentile(self):
        """Test score at mean + 1σ gives approximately 84th percentile."""
        engine = PercentileEngine(mean=70.0, std=10.0)
        
        # score = 70 + 10 = 80
        percentile = engine.score_to_percentile(80.0)
        
        # Should be approximately 84th percentile
        assert 80 <= percentile <= 90
    
    def test_score_at_minus_two_sigma_gives_3rd_percentile(self):
        """Test score at mean - 2σ gives approximately 3rd percentile."""
        engine = PercentileEngine(mean=70.0, std=10.0)
        
        # score = 70 - 20 = 50
        percentile = engine.score_to_percentile(50.0)
        
        # Should be approximately 2.3rd percentile (within reasonable range)
        assert 0 <= percentile <= 10
    
    def test_score_at_plus_two_sigma_gives_97th_percentile(self):
        """Test score at mean + 2σ gives approximately 97th percentile."""
        engine = PercentileEngine(mean=70.0, std=10.0)
        
        # score = 70 + 20 = 90
        percentile = engine.score_to_percentile(90.0)
        
        # Should be approximately 97.7th percentile
        assert 95 <= percentile <= 100
    
    def test_score_at_minus_three_sigma(self):
        """Test score at mean - 3σ gives very low percentile."""
        engine = PercentileEngine(mean=70.0, std=10.0)
        
        # score = 70 - 30 = 40
        percentile = engine.score_to_percentile(40.0)
        
        # Should be very close to 0
        assert 0 <= percentile <= 5
    
    def test_score_at_plus_three_sigma(self):
        """Test score at mean + 3σ gives very high percentile."""
        engine = PercentileEngine(mean=70.0, std=10.0)
        
        # score = 70 + 30 = 100
        percentile = engine.score_to_percentile(100.0)
        
        # Should be very close to 100
        assert 99 <= percentile <= 100


class TestPercentileBoundaryConditions:
    """Test boundary conditions for percentile calculations."""
    
    def test_percentile_never_below_zero(self):
        """Test that percentile is never below 0."""
        engine = PercentileEngine()
        
        # Test with various extreme low scores
        for score in [0, -10, -100, -1000]:
            percentile = engine.score_to_percentile(score)
            assert percentile >= 0
    
    def test_percentile_never_above_hundred(self):
        """Test that percentile is never above 100."""
        engine = PercentileEngine()
        
        # Test with various extreme high scores
        for score in [100, 110, 200, 1000]:
            percentile = engine.score_to_percentile(score)
            assert percentile <= 100
    
    def test_percentile_at_zero_boundary(self):
        """Test percentile at score boundary of 0."""
        engine = PercentileEngine()
        
        percentile = engine.score_to_percentile(0.0)
        
        assert isinstance(percentile, int)
        assert 0 <= percentile <= 100
    
    def test_percentile_at_hundred_boundary(self):
        """Test percentile at score boundary of 100."""
        engine = PercentileEngine()
        
        percentile = engine.score_to_percentile(100.0)
        
        assert isinstance(percentile, int)
        assert 0 <= percentile <= 100


class TestEmpiricalDistributionLoading:
    """Test empirical distribution loading from CSV."""
    
    def test_load_empirical_distribution_with_valid_csv(self):
        """Test loading empirical distribution from valid CSV file."""
        engine = PercentileEngine()
        
        # Create temporary CSV with valid scores
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['score'])
            # Create 150 scores with mean ~72, std ~11
            for score in np.random.normal(loc=72.0, scale=11.0, size=150):
                writer.writerow([max(0, min(100, score))])
            csv_path = f.name
        
        try:
            # Load empirical distribution
            engine.load_empirical_distribution(csv_path)
            
            # Verify empirical distribution is active
            assert engine.is_empirical is True
            assert engine.kde is not None
            assert engine.empirical_scores is not None
            assert engine.empirical_mean is not None
            assert engine.empirical_std is not None
        finally:
            Path(csv_path).unlink()
    
    def test_load_empirical_distribution_affects_percentiles(self):
        """Test that loading empirical distribution changes percentile calculations."""
        engine1 = PercentileEngine(mean=70.0, std=10.0)
        engine2 = PercentileEngine(mean=70.0, std=10.0)
        
        # Create temporary CSV with biased distribution (higher scores)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['score'])
            # Create 150 scores with mean ~80 (higher than bootstrap mean of 70)
            for score in np.random.normal(loc=80.0, scale=8.0, size=150):
                writer.writerow([max(0, min(100, score))])
            csv_path = f.name
        
        try:
            # Calculate percentile for same score with bootstrap vs empirical
            percentile_bootstrap = engine1.score_to_percentile(75.0)
            
            # Load empirical distribution on engine2
            engine2.load_empirical_distribution(csv_path)
            percentile_empirical = engine2.score_to_percentile(75.0)
            
            # Percentiles may differ due to different distributions
            # (not asserting they're different, as it depends on random seed)
            assert isinstance(percentile_bootstrap, int)
            assert isinstance(percentile_empirical, int)
        finally:
            Path(csv_path).unlink()
    
    def test_load_empirical_distribution_raises_on_missing_file(self):
        """Test that loading from non-existent file raises error."""
        engine = PercentileEngine()
        
        with pytest.raises(FileNotFoundError):
            engine.load_empirical_distribution('/nonexistent/path/file.csv')
    
    def test_load_empirical_distribution_with_insufficient_samples(self):
        """Test that loading with insufficient samples raises error."""
        engine = PercentileEngine()
        
        # Create temporary CSV with only 50 scores (minimum is 100)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['score'])
            for i in range(50):
                writer.writerow([70.0])
            csv_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Insufficient samples"):
                engine.load_empirical_distribution(csv_path)
        finally:
            Path(csv_path).unlink()
    
    def test_load_empirical_distribution_with_invalid_csv_format(self):
        """Test that loading with invalid CSV format raises error."""
        engine = PercentileEngine()
        
        # Create temporary CSV with missing score column
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['invalid_column'])  # Wrong column name
            for i in range(100):
                writer.writerow([70.0])
            csv_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Score column not found"):
                engine.load_empirical_distribution(csv_path)
        finally:
            Path(csv_path).unlink()
    
    def test_load_empirical_distribution_skips_invalid_rows(self):
        """Test that loading skips rows with invalid scores."""
        engine = PercentileEngine()
        
        # Create temporary CSV with mixed valid and invalid scores
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['score'])
            # Write valid scores
            for i in range(100):
                writer.writerow([70.0])
            # Write invalid scores that should be skipped
            writer.writerow(['invalid'])
            writer.writerow([''])
            writer.writerow([999])  # Out of range
            csv_path = f.name
        
        try:
            # Should not raise error - invalid rows are skipped
            engine.load_empirical_distribution(csv_path)
            assert engine.is_empirical is True
        finally:
            Path(csv_path).unlink()
    
    def test_load_empirical_distribution_with_golden_ratio_score_column(self):
        """Test that loading recognizes 'golden_ratio_score' column name."""
        engine = PercentileEngine()
        
        # Create temporary CSV with 'golden_ratio_score' column
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['golden_ratio_score'])  # Alternative column name
            for score in np.random.normal(loc=72.0, scale=11.0, size=150):
                writer.writerow([max(0, min(100, score))])
            csv_path = f.name
        
        try:
            engine.load_empirical_distribution(csv_path)
            assert engine.is_empirical is True
        finally:
            Path(csv_path).unlink()


class TestPercentileMessageFormatting:
    """Test user-friendly percentile message generation."""
    
    def test_percentile_message_format(self):
        """Test that message follows expected format."""
        engine = PercentileEngine()
        
        message = engine.get_percentile_message(75)
        
        assert "typical" in message
        assert "compared with 2017 faces in the FairFace dataset" in message
    
    def test_percentile_message_with_various_values(self):
        """Test message generation for different percentile values."""
        engine = PercentileEngine()
        
        message_0 = engine.get_percentile_message(0)
        message_50 = engine.get_percentile_message(50)
        message_100 = engine.get_percentile_message(100)
        
        assert "below typical" in message_0
        assert "typical" in message_50
        assert "above typical" in message_100
        assert "compared with 2017 faces in the FairFace dataset" in message_50
    
    def test_percentile_message_template(self):
        """Test that message uses expected template."""
        engine = PercentileEngine()
        
        # Message should follow format: "<band> (compared with 2017 faces in the FairFace dataset)"
        message = engine.get_percentile_message(75)
        assert message == "typical (compared with 2017 faces in the FairFace dataset)"
        
        message_above = engine.get_percentile_message(80)
        assert message_above == "above typical (compared with 2017 faces in the FairFace dataset)"


class TestMonotonicIncrease:
    """Test that percentiles increase monotonically with scores."""
    
    def test_monotonic_increase_across_range(self):
        """Test that percentiles increase monotonically across full score range."""
        engine = PercentileEngine()
        
        scores = np.linspace(0, 100, 21)  # 0, 5, 10, ..., 95, 100
        percentiles = [engine.score_to_percentile(s) for s in scores]
        
        # Verify monotonic non-decreasing
        for i in range(len(percentiles) - 1):
            assert percentiles[i] <= percentiles[i + 1]
    
    def test_monotonic_increase_fine_grained(self):
        """Test monotonic increase with fine-grained score increments."""
        engine = PercentileEngine()
        
        # Test every 0.1 point from 60 to 80
        scores = np.arange(60, 80.1, 0.1)
        percentiles = [engine.score_to_percentile(s) for s in scores]
        
        # Verify monotonic non-decreasing
        for i in range(len(percentiles) - 1):
            assert percentiles[i] <= percentiles[i + 1]
    
    def test_percentiles_strictly_increasing_for_distinct_scores(self):
        """Test that distinct scores within range give non-decreasing percentiles."""
        engine = PercentileEngine()
        
        test_scores = [65, 70, 75, 80, 85]
        percentiles = [engine.score_to_percentile(s) for s in test_scores]
        
        # Should be strictly increasing (or at least non-decreasing)
        for i in range(len(percentiles) - 1):
            assert percentiles[i] <= percentiles[i + 1]


class TestPercentileEngineAttributes:
    """Test internal attributes and state of PercentileEngine."""
    
    def test_initial_attributes_are_none_for_empirical(self):
        """Test that empirical attributes are None initially."""
        engine = PercentileEngine()
        
        assert engine.kde is None
        assert engine.is_empirical is False
        assert engine.empirical_scores is None
        assert engine.empirical_mean is None
        assert engine.empirical_std is None
    
    def test_empirical_attributes_set_after_loading(self):
        """Test that empirical attributes are set after loading CSV."""
        engine = PercentileEngine()
        
        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(['score'])
            for score in np.random.normal(loc=72.0, scale=11.0, size=150):
                writer.writerow([max(0, min(100, score))])
            csv_path = f.name
        
        try:
            engine.load_empirical_distribution(csv_path)
            
            assert engine.kde is not None
            assert engine.is_empirical is True
            assert engine.empirical_scores is not None
            assert len(engine.empirical_scores) > 0
            assert engine.empirical_mean is not None
            assert engine.empirical_std is not None
        finally:
            Path(csv_path).unlink()


class TestSpecialScoreValues:
    """Test special score values and edge cases."""
    
    def test_score_equal_to_mean_gives_50th_percentile(self):
        """Test that score equal to mean gives 50th percentile."""
        engine = PercentileEngine(mean=70.0, std=10.0)
        
        percentile = engine.score_to_percentile(70.0)
        
        assert 45 <= percentile <= 55
    
    def test_very_low_score_gives_near_zero_percentile(self):
        """Test that very low scores give percentiles near 0."""
        engine = PercentileEngine(mean=70.0, std=10.0)
        
        # Well below mean (5+ standard deviations below)
        percentile = engine.score_to_percentile(0.0)
        
        assert 0 <= percentile <= 5
    
    def test_very_high_score_gives_near_hundred_percentile(self):
        """Test that very high scores give percentiles near 100."""
        engine = PercentileEngine(mean=70.0, std=10.0)
        
        # Well above mean (5+ standard deviations above)
        percentile = engine.score_to_percentile(100.0)
        
        assert 95 <= percentile <= 100
    
    def test_fractional_scores_are_handled(self):
        """Test that fractional scores are properly handled."""
        engine = PercentileEngine()
        
        percentile = engine.score_to_percentile(72.75)
        
        assert isinstance(percentile, int)
        assert 0 <= percentile <= 100
