"""
Unit tests for celebrity confidence thresholding (Task 8.4).

Tests the get_best_match() method that applies confidence thresholding
to ranked celebrities from task 8.3.

Requirements: 11.1, 11.2, 11.3, 11.4
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from app.components.celebrity_matcher import CelebrityMatcher


class TestGetBestMatch:
    """Test suite for get_best_match() method."""
    
    @pytest.fixture
    def matcher(self):
        """Create a celebrity matcher with mock database."""
        matcher = CelebrityMatcher(db_path="/nonexistent/path")
        
        # Mock celebrity database
        matcher.celebrity_embeddings = {
            "Priyanka_Chopra": {
                "embedding": [0.1, 0.2, 0.3],
                "dataset_source": "Bollywood Celeb Dataset",
                "thumbnail_path": "/thumbnails/priyanka.jpg"
            },
            "Emma_Stone": {
                "embedding": [0.4, 0.5, 0.6],
                "dataset_source": "Pins Face Recognition",
                "thumbnail_path": "/thumbnails/emma.jpg"
            },
            "Shah_Rukh_Khan": {
                "embedding": [0.7, 0.8, 0.9],
                "dataset_source": "Bollywood Celeb Dataset",
                "thumbnail_path": "/thumbnails/shah_rukh.jpg"
            }
        }
        matcher.is_available = True
        
        return matcher
    
    # Tests for scores above threshold (0.55)
    
    def test_above_threshold_0_75(self, matcher):
        """Test match with high confidence (0.75 > 0.55)."""
        ranked = [("Emma_Stone", 0.75), ("Priyanka_Chopra", 0.60)]
        
        result = matcher.get_best_match(ranked, threshold=0.55)
        
        assert result["name"] == "Emma Stone"
        assert result["confidence_score"] == 0.75
        assert result["dataset_source"] == "Pins Face Recognition"
        assert result["thumbnail_url"] == "/thumbnails/emma.jpg"
        assert "message" not in result
    
    def test_above_threshold_0_60(self, matcher):
        """Test match exactly at confidence boundary (0.60 > 0.55)."""
        ranked = [("Priyanka_Chopra", 0.60), ("Shah_Rukh_Khan", 0.50)]
        
        result = matcher.get_best_match(ranked, threshold=0.55)
        
        assert result["name"] == "Priyanka Chopra"
        assert result["confidence_score"] == 0.60
        assert result["dataset_source"] == "Bollywood Celeb Dataset"
        assert result["thumbnail_url"] == "/thumbnails/priyanka.jpg"
    
    def test_above_threshold_rounding(self, matcher):
        """Test confidence score is rounded to 2 decimal places."""
        ranked = [("Shah_Rukh_Khan", 0.756789)]
        
        result = matcher.get_best_match(ranked, threshold=0.55)
        
        assert result["confidence_score"] == 0.76  # Rounded to 2 decimals
        assert result["name"] == "Shah Rukh Khan"
    
    # Tests for scores below threshold (0.55)
    
    def test_below_threshold_0_50(self, matcher):
        """Test no match when confidence below threshold (0.50 < 0.55)."""
        ranked = [("Emma_Stone", 0.50)]
        
        result = matcher.get_best_match(ranked, threshold=0.55)
        
        assert result["name"] is None
        assert result["message"] == "No close match found in our dataset"
        assert "confidence_score" not in result
        assert "dataset_source" not in result
    
    def test_below_threshold_0_40(self, matcher):
        """Test no match with low confidence (0.40 < 0.55)."""
        ranked = [("Priyanka_Chopra", 0.40), ("Emma_Stone", 0.35)]
        
        result = matcher.get_best_match(ranked, threshold=0.55)
        
        assert result["name"] is None
        assert result["message"] == "No close match found in our dataset"
    
    def test_at_threshold_boundary_equal(self, matcher):
        """Test match when score equals threshold (0.55 == 0.55)."""
        ranked = [("Shah_Rukh_Khan", 0.55)]
        
        result = matcher.get_best_match(ranked, threshold=0.55)
        
        # Requirement 11.1: score >= threshold
        assert result["name"] == "Shah Rukh Khan"
        assert result["confidence_score"] == 0.55
    
    # Tests for empty list
    
    def test_empty_ranked_list(self, matcher):
        """Test with empty ranked celebrities list."""
        ranked = []
        
        result = matcher.get_best_match(ranked, threshold=0.55)
        
        assert result["name"] is None
        assert result["message"] == "No close match found in our dataset"
    
    # Tests for custom thresholds
    
    def test_custom_threshold_higher(self, matcher):
        """Test with custom threshold (0.70)."""
        ranked = [("Emma_Stone", 0.68)]
        
        result = matcher.get_best_match(ranked, threshold=0.70)
        
        # Score 0.68 is below threshold 0.70
        assert result["name"] is None
        assert result["message"] == "No close match found in our dataset"
    
    def test_custom_threshold_lower(self, matcher):
        """Test with custom threshold (0.40)."""
        ranked = [("Priyanka_Chopra", 0.50)]
        
        result = matcher.get_best_match(ranked, threshold=0.40)
        
        # Score 0.50 is above threshold 0.40
        assert result["name"] == "Priyanka Chopra"
        assert result["confidence_score"] == 0.50
    
    # Tests for name formatting
    
    def test_name_formatting_underscores_removed(self, matcher):
        """Test that underscores in names are replaced with spaces."""
        ranked = [("Shah_Rukh_Khan", 0.65)]
        
        result = matcher.get_best_match(ranked, threshold=0.55)
        
        assert result["name"] == "Shah Rukh Khan"
        assert "_" not in result["name"]
    
    # Tests for multiple celebrities in ranking
    
    def test_ignores_lower_ranked_celebrities(self, matcher):
        """Test that only highest-ranked celebrity is considered."""
        ranked = [
            ("Emma_Stone", 0.75),      # Highest
            ("Shah_Rukh_Khan", 0.80),  # Ignored - not highest
            ("Priyanka_Chopra", 0.60)  # Ignored - not highest
        ]
        
        result = matcher.get_best_match(ranked, threshold=0.55)
        
        # Should use first (highest-ranked) celebrity
        assert result["name"] == "Emma Stone"
        assert result["confidence_score"] == 0.75
    
    # Tests for threshold validation
    
    def test_threshold_validation_too_low(self, matcher):
        """Test that threshold must be >= 0.0."""
        ranked = [("Emma_Stone", 0.75)]
        
        with pytest.raises(ValueError, match="Threshold must be in"):
            matcher.get_best_match(ranked, threshold=-0.1)
    
    def test_threshold_validation_too_high(self, matcher):
        """Test that threshold must be <= 1.0."""
        ranked = [("Emma_Stone", 0.75)]
        
        with pytest.raises(ValueError, match="Threshold must be in"):
            matcher.get_best_match(ranked, threshold=1.1)
    
    def test_threshold_validation_min_valid(self, matcher):
        """Test threshold can be 0.0 (minimum valid)."""
        ranked = [("Emma_Stone", 0.01)]
        
        result = matcher.get_best_match(ranked, threshold=0.0)
        
        assert result["name"] == "Emma Stone"
    
    def test_threshold_validation_max_valid(self, matcher):
        """Test threshold can be 1.0 (maximum valid)."""
        ranked = [("Emma_Stone", 1.0)]
        
        result = matcher.get_best_match(ranked, threshold=1.0)
        
        assert result["name"] == "Emma Stone"
    
    # Tests for response structure
    
    def test_response_structure_on_match(self, matcher):
        """Test response structure when match found."""
        ranked = [("Priyanka_Chopra", 0.70)]
        
        result = matcher.get_best_match(ranked, threshold=0.55)
        
        # Verify all required keys present
        assert "name" in result
        assert "confidence_score" in result
        assert "dataset_source" in result
        assert "thumbnail_url" in result
        
        # Verify no extra unexpected keys
        expected_keys = {"name", "confidence_score", "dataset_source", "thumbnail_url"}
        assert set(result.keys()) == expected_keys
    
    def test_response_structure_no_match(self, matcher):
        """Test response structure when no match found."""
        ranked = [("Emma_Stone", 0.40)]
        
        result = matcher.get_best_match(ranked, threshold=0.55)
        
        # Verify required keys present
        assert "name" in result
        assert "message" in result
        assert result["name"] is None
        
        # Verify no match-specific keys present
        assert "confidence_score" not in result
        assert "dataset_source" not in result
        assert "thumbnail_url" not in result
    
    # Tests for edge cases
    
    def test_celebrity_not_in_database(self, matcher):
        """Test when ranked celebrity not in database."""
        # Celebrity name not in matcher.celebrity_embeddings
        ranked = [("Unknown_Celebrity", 0.75)]
        
        result = matcher.get_best_match(ranked, threshold=0.55)
        
        # Should still return match but with default values
        assert result["name"] == "Unknown Celebrity"
        assert result["confidence_score"] == 0.75
        assert result["dataset_source"] == "Unknown"
        assert result["thumbnail_url"] == ""
    
    def test_very_high_confidence(self, matcher):
        """Test with very high confidence score (1.0)."""
        ranked = [("Emma_Stone", 1.0)]
        
        result = matcher.get_best_match(ranked, threshold=0.55)
        
        assert result["name"] == "Emma Stone"
        assert result["confidence_score"] == 1.0
    
    def test_very_low_confidence(self, matcher):
        """Test with very low confidence score (0.0)."""
        ranked = [("Emma_Stone", 0.0)]
        
        result = matcher.get_best_match(ranked, threshold=0.55)
        
        assert result["name"] is None
        assert result["message"] == "No close match found in our dataset"
    
    def test_precision_rounding_down(self, matcher):
        """Test rounding down with precision loss."""
        ranked = [("Emma_Stone", 0.7544)]
        
        result = matcher.get_best_match(ranked, threshold=0.55)
        
        assert result["confidence_score"] == 0.75
    
    def test_precision_rounding_up(self, matcher):
        """Test rounding up with precision loss."""
        ranked = [("Emma_Stone", 0.7549)]
        
        result = matcher.get_best_match(ranked, threshold=0.55)
        
        assert result["confidence_score"] == 0.75
    
    # Tests for default threshold
    
    def test_default_threshold_applied(self, matcher):
        """Test that default threshold of 0.55 is applied when not specified."""
        ranked = [("Priyanka_Chopra", 0.55)]
        
        result = matcher.get_best_match(ranked)  # No threshold specified
        
        # Default threshold is 0.55, so 0.55 >= 0.55 should match
        assert result["name"] == "Priyanka Chopra"
    
    # Requirement-specific tests
    
    def test_requirement_11_1_highest_scoring_identified(self, matcher):
        """Req 11.1: Celebrity_Matcher identifies highest-scoring celebrity."""
        ranked = [
            ("Emma_Stone", 0.75),
            ("Priyanka_Chopra", 0.65),
            ("Shah_Rukh_Khan", 0.55)
        ]
        
        result = matcher.get_best_match(ranked, threshold=0.55)
        
        # Should identify highest (Emma Stone at 0.75)
        assert result["name"] == "Emma Stone"
        assert result["confidence_score"] == 0.75
    
    def test_requirement_11_2_default_threshold_0_55(self, matcher):
        """Req 11.2: Default threshold is 0.55."""
        # Call without specifying threshold
        ranked = [("Priyanka_Chopra", 0.55)]
        
        result = matcher.get_best_match(ranked)
        
        # Should accept 0.55 with default threshold (0.55)
        assert result["name"] == "Priyanka Chopra"
    
    def test_requirement_11_3_no_match_message(self, matcher):
        """Req 11.3: Return null name with message for scores < 0.55."""
        ranked = [("Emma_Stone", 0.50)]
        
        result = matcher.get_best_match(ranked, threshold=0.55)
        
        assert result["name"] is None
        assert result["message"] == "No close match found in our dataset"
    
    def test_requirement_11_4_confidence_rounded_to_2_decimals(self, matcher):
        """Req 11.4: Confidence rounded to 2 decimal places."""
        ranked = [("Emma_Stone", 0.123456789)]
        
        result = matcher.get_best_match(ranked, threshold=0.0)
        
        # Should be rounded to 2 decimals
        assert result["confidence_score"] == 0.12
        # Verify it's actually rounded (not truncated)
        assert isinstance(result["confidence_score"], float)
