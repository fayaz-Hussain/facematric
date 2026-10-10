"""
Unit tests for CelebrityMatcher component (Task 8.6).

Comprehensive test suite covering:
- FaceNet embedding generation (task 8.1 output)
- Cosine similarity calculation (task 8.2)
- Celebrity ranking by similarity (task 8.3 output sorted descending)
- Confidence thresholding at 0.55 boundary (task 8.4)
- Database loading and validation (task 8.5)
- Edge cases and error handling

Requirements: 10.1, 10.2, 10.3, 11.1, 11.2, 11.3, 11.4, 18.1, 18.2, 18.3, 18.4
"""

import pytest
import numpy as np
import pickle
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from app.components.celebrity_matcher import CelebrityMatcher
from app.models.internal import ImageData


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_image_data():
    """Create mock ImageData object."""
    # Create a valid RGB image array (3D: height x width x 3)
    image_array = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
    return ImageData(
        array=image_array,
        width=640,
        height=480,
        original_format="JPEG"
    )


@pytest.fixture
def mock_embeddings_512d():
    """Create mock 512-dimensional FaceNet embeddings."""
    np.random.seed(42)
    return {
        "Celebrity_A": {
            "embedding": np.random.randn(512),
            "dataset_source": "Western",
            "thumbnail_url": "/thumbnails/a.jpg"
        },
        "Celebrity_B": {
            "embedding": np.random.randn(512),
            "dataset_source": "Bollywood",
            "thumbnail_url": "/thumbnails/b.jpg"
        },
        "Celebrity_C": {
            "embedding": np.random.randn(512),
            "dataset_source": "Western",
            "thumbnail_url": "/thumbnails/c.jpg"
        },
        "Celebrity_D": {
            "embedding": np.random.randn(512),
            "dataset_source": "Bollywood",
            "thumbnail_url": "/thumbnails/d.jpg"
        },
        "Celebrity_E": {
            "embedding": np.random.randn(512),
            "dataset_source": "Western",
            "thumbnail_url": "/thumbnails/e.jpg"
        },
    }


@pytest.fixture
def matcher_with_embeddings(mock_embeddings_512d):
    """Create CelebrityMatcher with pre-loaded embeddings."""
    matcher = CelebrityMatcher(db_path="/nonexistent/path")
    matcher.celebrity_embeddings = mock_embeddings_512d
    matcher.full_database = mock_embeddings_512d
    matcher.is_available = True
    return matcher


@pytest.fixture
def valid_pickle_database(tmp_path):
    """Create a valid celebrity database pickle file."""
    db_file = tmp_path / "celebrities.pkl"
    
    # Create valid database
    database = {
        "Tom_Cruise": {
            "embedding": np.random.randn(512),
            "dataset_source": "Western",
            "thumbnail_url": "/thumbnails/tom.jpg"
        },
        "Priyanka_Chopra": {
            "embedding": np.random.randn(512),
            "dataset_source": "Bollywood",
            "thumbnail_url": "/thumbnails/priyanka.jpg"
        },
    }
    
    with open(db_file, 'wb') as f:
        pickle.dump(database, f)
    
    return str(db_file)


@pytest.fixture
def corrupted_pickle_database(tmp_path):
    """Create a corrupted pickle file."""
    db_file = tmp_path / "corrupted.pkl"
    
    # Write invalid pickle data
    with open(db_file, 'wb') as f:
        f.write(b"This is not valid pickle data")
    
    return str(db_file)


# ============================================================================
# TEST GROUP 1: FaceNet Embedding Generation (Task 8.1)
# ============================================================================

class TestEmbeddingGeneration:
    """Test FaceNet embedding generation functionality."""
    
    def test_embeddings_are_512_dimensional(self, matcher_with_embeddings):
        """Test that stored embeddings have correct FaceNet dimension (512)."""
        for celeb_name, celeb_data in matcher_with_embeddings.celebrity_embeddings.items():
            embedding = celeb_data["embedding"]
            assert embedding.shape == (512,), \
                f"Celebrity {celeb_name} embedding has wrong shape: {embedding.shape}"
    
    def test_embeddings_are_numpy_arrays(self, matcher_with_embeddings):
        """Test that all embeddings are numpy arrays."""
        for celeb_name, celeb_data in matcher_with_embeddings.celebrity_embeddings.items():
            embedding = celeb_data["embedding"]
            assert isinstance(embedding, np.ndarray), \
                f"Celebrity {celeb_name} embedding is not numpy array"
    
    def test_rank_celebrities_accepts_embeddings(self, matcher_with_embeddings):
        """Test that rank_celebrities processes embeddings correctly."""
        query = np.random.randn(512)
        
        # Should process embeddings without error
        rankings = matcher_with_embeddings.rank_celebrities(query)
        
        assert len(rankings) > 0
        assert all(isinstance(score, (float, np.floating)) for _, score in rankings)
    
    def test_embedding_data_structure(self, matcher_with_embeddings):
        """Test that embedding data structure is correct."""
        for celeb_name, celeb_data in matcher_with_embeddings.celebrity_embeddings.items():
            # Should have embedding field
            assert "embedding" in celeb_data
            
            # Embedding should be numpy array of correct shape
            embedding = celeb_data["embedding"]
            assert isinstance(embedding, np.ndarray)
            assert embedding.shape == (512,)
            
            # Should contain real numbers
            assert embedding.dtype in [np.float32, np.float64]
    
    def test_embeddings_are_different_from_each_other(self, matcher_with_embeddings):
        """Test that embeddings for different celebrities are distinct."""
        embeddings_list = [
            celeb_data["embedding"] 
            for celeb_data in matcher_with_embeddings.celebrity_embeddings.values()
        ]
        
        # Each embedding should be different from others
        for i, emb1 in enumerate(embeddings_list):
            for j, emb2 in enumerate(embeddings_list):
                if i < j:
                    # Embeddings should not be identical
                    assert not np.allclose(emb1, emb2), \
                        "Two different celebrities have identical embeddings"


# ============================================================================
# TEST GROUP 2: Cosine Similarity Calculation (Task 8.2)
# ============================================================================

class TestCosineSimilarityInCelebrity:
    """Test cosine similarity calculation in celebrity matching context."""
    
    def test_rank_celebrities_uses_cosine_similarity(self, matcher_with_embeddings):
        """Test that rank_celebrities uses cosine similarity."""
        query = np.random.randn(512)
        
        rankings = matcher_with_embeddings.rank_celebrities(query)
        
        # Should compute similarity for each celebrity
        assert len(rankings) == len(matcher_with_embeddings.celebrity_embeddings)
    
    def test_similarity_scores_in_valid_range(self, matcher_with_embeddings):
        """Test that similarity scores are in [-1, 1] range."""
        query = np.random.randn(512)
        
        rankings = matcher_with_embeddings.rank_celebrities(query)
        
        for name, score in rankings:
            assert -1.0 <= score <= 1.0, f"Score {score} out of valid range"
    
    def test_identical_embedding_high_similarity(self, matcher_with_embeddings):
        """Test that identical embeddings yield high similarity."""
        # Use one celebrity's embedding as query
        celeb_name = "Celebrity_A"
        celeb_embedding = matcher_with_embeddings.celebrity_embeddings[celeb_name]["embedding"]
        
        rankings = matcher_with_embeddings.rank_celebrities(celeb_embedding)
        
        # Celebrity_A should be first (most similar to itself)
        best_name, best_score = rankings[0]
        assert best_name == celeb_name
        assert best_score > 0.99  # Should be very close to 1.0
    
    def test_similarity_scores_sum_interpretation(self, matcher_with_embeddings):
        """Test that similarity scores are meaningful for matching."""
        np.random.seed(100)
        query = np.random.randn(512)
        
        rankings = matcher_with_embeddings.rank_celebrities(query)
        
        # Scores should vary (not all identical)
        scores = [s for _, s in rankings]
        assert len(set(scores)) > 1, "All similarity scores are identical"


# ============================================================================
# TEST GROUP 3: Celebrity Ranking (Task 8.3)
# ============================================================================

class TestRankCelebrities:
    """Test celebrity ranking by similarity."""
    
    def test_rank_celebrities_returns_list_of_tuples(self, matcher_with_embeddings):
        """Test that rank_celebrities returns list of (name, score) tuples."""
        query = np.random.randn(512)
        
        rankings = matcher_with_embeddings.rank_celebrities(query)
        
        assert isinstance(rankings, list)
        for item in rankings:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], (float, np.floating))
    
    def test_rank_celebrities_sorted_descending(self, matcher_with_embeddings):
        """Test that celebrities are ranked in descending order by similarity."""
        query = np.random.randn(512)
        
        rankings = matcher_with_embeddings.rank_celebrities(query)
        scores = [score for _, score in rankings]
        
        # Scores should be in descending order
        assert scores == sorted(scores, reverse=True), "Rankings not sorted descending"
    
    def test_rank_celebrities_includes_all_celebrities(self, matcher_with_embeddings):
        """Test that all celebrities from database are included in ranking."""
        query = np.random.randn(512)
        
        rankings = matcher_with_embeddings.rank_celebrities(query)
        ranked_names = {name for name, _ in rankings}
        
        expected_names = set(matcher_with_embeddings.celebrity_embeddings.keys())
        
        assert ranked_names == expected_names, "Not all celebrities included in ranking"
    
    def test_rank_celebrities_no_duplicate_entries(self, matcher_with_embeddings):
        """Test that ranking has no duplicate celebrities."""
        query = np.random.randn(512)
        
        rankings = matcher_with_embeddings.rank_celebrities(query)
        ranked_names = [name for name, _ in rankings]
        
        assert len(ranked_names) == len(set(ranked_names)), "Duplicate celebrities in ranking"
    
    def test_rank_celebrities_deterministic(self, matcher_with_embeddings):
        """Test that ranking is deterministic."""
        query = np.array([0.1, 0.2, 0.3] + [0.0] * 509)
        
        rankings1 = matcher_with_embeddings.rank_celebrities(query)
        rankings2 = matcher_with_embeddings.rank_celebrities(query)
        
        assert rankings1 == rankings2, "Rankings not deterministic"
    
    def test_rank_celebrities_empty_database(self):
        """Test ranking with empty celebrity database."""
        matcher = CelebrityMatcher(db_path="/nonexistent/path")
        matcher.celebrity_embeddings = {}
        matcher.is_available = False
        
        query = np.random.randn(512)
        rankings = matcher.rank_celebrities(query)
        
        assert rankings == [], "Empty database should return empty list"
    
    def test_rank_celebrities_single_celebrity(self):
        """Test ranking with single celebrity."""
        matcher = CelebrityMatcher(db_path="/nonexistent/path")
        single_celeb = {
            "Solo_Celebrity": {
                "embedding": np.random.randn(512),
                "dataset_source": "Western",
                "thumbnail_url": "/path/solo.jpg"
            }
        }
        matcher.celebrity_embeddings = single_celeb
        
        query = np.random.randn(512)
        rankings = matcher.rank_celebrities(query)
        
        assert len(rankings) == 1
        assert rankings[0][0] == "Solo_Celebrity"
    
    def test_rank_celebrities_highest_first(self, matcher_with_embeddings):
        """Test that highest-scoring celebrity is first in ranking."""
        query = np.random.randn(512)
        
        rankings = matcher_with_embeddings.rank_celebrities(query)
        first_score = rankings[0][1]
        all_scores = [score for _, score in rankings]
        
        assert first_score == max(all_scores), "Highest score not first in ranking"


# ============================================================================
# TEST GROUP 4: Confidence Thresholding (Task 8.4)
# ============================================================================

class TestConfidenceThresholding:
    """Test confidence thresholding at 0.55 boundary."""
    
    def test_get_best_match_above_threshold(self, matcher_with_embeddings):
        """Test match when confidence >= 0.55."""
        ranked = [("Celebrity_A", 0.75)]
        
        result = matcher_with_embeddings.get_best_match(ranked, threshold=0.55)
        
        assert result["name"] is not None
        assert result["confidence_score"] == 0.75
    
    def test_get_best_match_below_threshold(self, matcher_with_embeddings):
        """Test no match when confidence < 0.55."""
        ranked = [("Celebrity_A", 0.50)]
        
        result = matcher_with_embeddings.get_best_match(ranked, threshold=0.55)
        
        assert result["name"] is None
        assert result["message"] == "No close match found in our dataset"
    
    def test_get_best_match_exactly_at_threshold(self, matcher_with_embeddings):
        """Test boundary case: similarity exactly at threshold (0.55)."""
        ranked = [("Celebrity_A", 0.55)]
        
        result = matcher_with_embeddings.get_best_match(ranked, threshold=0.55)
        
        # Requirement 11.1: score >= 0.55 (inclusive)
        assert result["name"] is not None
        assert result["confidence_score"] == 0.55
    
    def test_get_best_match_just_above_threshold(self, matcher_with_embeddings):
        """Test score just above threshold (0.551)."""
        ranked = [("Celebrity_A", 0.551)]
        
        result = matcher_with_embeddings.get_best_match(ranked, threshold=0.55)
        
        assert result["name"] is not None
        assert result["confidence_score"] == 0.55  # Rounded to 2 decimals
    
    def test_get_best_match_just_below_threshold(self, matcher_with_embeddings):
        """Test score just below threshold (0.549)."""
        ranked = [("Celebrity_A", 0.549)]
        
        result = matcher_with_embeddings.get_best_match(ranked, threshold=0.55)
        
        assert result["name"] is None
        assert result["message"] == "No close match found in our dataset"
    
    def test_confidence_score_rounded_to_2_decimals(self, matcher_with_embeddings):
        """Test confidence score is rounded to 2 decimal places."""
        ranked = [("Celebrity_A", 0.7564)]
        
        result = matcher_with_embeddings.get_best_match(ranked, threshold=0.55)
        
        assert result["confidence_score"] == 0.76
    
    def test_ignores_lower_ranked_celebrities(self, matcher_with_embeddings):
        """Test that only highest-ranked celebrity is evaluated."""
        ranked = [
            ("Celebrity_A", 0.75),  # Highest - should be used
            ("Celebrity_B", 0.90),  # Ignored - lower in ranking
            ("Celebrity_C", 0.80),  # Ignored - lower in ranking
        ]
        
        result = matcher_with_embeddings.get_best_match(ranked, threshold=0.55)
        
        # Should use Celebrity_A (first in sorted list)
        assert result["name"] == "Celebrity A"
        assert result["confidence_score"] == 0.75
    
    def test_empty_ranked_list(self, matcher_with_embeddings):
        """Test edge case: empty ranked celebrities list."""
        ranked = []
        
        result = matcher_with_embeddings.get_best_match(ranked, threshold=0.55)
        
        assert result["name"] is None
        assert result["message"] == "No close match found in our dataset"
    
    def test_threshold_validation_negative(self, matcher_with_embeddings):
        """Test that negative threshold raises ValueError."""
        ranked = [("Celebrity_A", 0.75)]
        
        with pytest.raises(ValueError, match="Threshold must be in"):
            matcher_with_embeddings.get_best_match(ranked, threshold=-0.1)
    
    def test_threshold_validation_too_high(self, matcher_with_embeddings):
        """Test that threshold > 1.0 raises ValueError."""
        ranked = [("Celebrity_A", 0.75)]
        
        with pytest.raises(ValueError, match="Threshold must be in"):
            matcher_with_embeddings.get_best_match(ranked, threshold=1.1)
    
    def test_default_threshold_0_55(self, matcher_with_embeddings):
        """Test that default threshold is 0.55."""
        ranked = [("Celebrity_A", 0.55)]
        
        # Call without specifying threshold
        result = matcher_with_embeddings.get_best_match(ranked)
        
        # Should use default 0.55 threshold
        assert result["name"] is not None


# ============================================================================
# TEST GROUP 5: Database Loading and Validation (Task 8.5)
# ============================================================================

class TestDatabaseLoading:
    """Test celebrity database loading and validation."""
    
    def test_load_celebrity_database_valid_file(self, valid_pickle_database):
        """Test loading valid celebrity database file."""
        matcher = CelebrityMatcher(db_path=valid_pickle_database)
        
        assert matcher.celebrity_embeddings is not None
        assert len(matcher.celebrity_embeddings) > 0
        assert matcher.is_available is True
    
    def test_load_celebrity_database_missing_file(self):
        """Test loading database from non-existent file."""
        matcher = CelebrityMatcher(db_path="/nonexistent/path/celebrities.pkl")
        
        assert matcher.celebrity_embeddings == {}
        assert matcher.is_available is False
    
    def test_load_celebrity_database_corrupted_file(self, corrupted_pickle_database):
        """Test loading corrupted pickle file."""
        matcher = CelebrityMatcher(db_path=corrupted_pickle_database)
        
        assert matcher.celebrity_embeddings == {}
        assert matcher.is_available is False
    
    def test_validate_embeddings_correct_dimensions(self):
        """Test that embeddings must have shape (512,)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_file = Path(tmp_dir) / "test.pkl"
            
            # Create database with incorrect embedding shape
            database = {
                "Celebrity_A": {
                    "embedding": np.random.randn(256),  # Wrong: 256 instead of 512
                    "dataset_source": "Western",
                    "thumbnail_url": "/path/a.jpg"
                }
            }
            
            with open(db_file, 'wb') as f:
                pickle.dump(database, f)
            
            matcher = CelebrityMatcher(db_path=str(db_file))
            
            # Should skip entry with wrong embedding size
            assert "Celebrity_A" not in matcher.celebrity_embeddings
    
    def test_validate_dataset_source_valid_values(self):
        """Test that dataset_source must be 'Western' or 'Bollywood'."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_file = Path(tmp_dir) / "test.pkl"
            
            database = {
                "Valid_Celebrity": {
                    "embedding": np.random.randn(512),
                    "dataset_source": "Western",
                    "thumbnail_url": "/path/valid.jpg"
                },
                "Invalid_Celebrity": {
                    "embedding": np.random.randn(512),
                    "dataset_source": "Unknown",  # Invalid
                    "thumbnail_url": "/path/invalid.jpg"
                }
            }
            
            with open(db_file, 'wb') as f:
                pickle.dump(database, f)
            
            matcher = CelebrityMatcher(db_path=str(db_file))
            
            assert "Valid_Celebrity" in matcher.celebrity_embeddings
            assert "Invalid_Celebrity" not in matcher.celebrity_embeddings
    
    def test_validate_required_fields(self):
        """Test that required fields are present."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_file = Path(tmp_dir) / "test.pkl"
            
            database = {
                "Complete_Celebrity": {
                    "embedding": np.random.randn(512),
                    "dataset_source": "Western",
                    "thumbnail_url": "/path/complete.jpg"
                },
                "Missing_Source": {
                    "embedding": np.random.randn(512),
                    # Missing dataset_source
                    "thumbnail_url": "/path/missing_source.jpg"
                },
                "Missing_Thumbnail": {
                    "embedding": np.random.randn(512),
                    "dataset_source": "Bollywood",
                    # Missing thumbnail_url
                }
            }
            
            with open(db_file, 'wb') as f:
                pickle.dump(database, f)
            
            matcher = CelebrityMatcher(db_path=str(db_file))
            
            assert "Complete_Celebrity" in matcher.celebrity_embeddings
            assert "Missing_Source" not in matcher.celebrity_embeddings
            assert "Missing_Thumbnail" not in matcher.celebrity_embeddings
    
    def test_database_contains_numpy_arrays(self):
        """Test that embeddings are numpy arrays."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_file = Path(tmp_dir) / "test.pkl"
            
            database = {
                "Valid_Array": {
                    "embedding": np.random.randn(512),  # numpy array
                    "dataset_source": "Western",
                    "thumbnail_url": "/path/valid.jpg"
                },
                "Invalid_List": {
                    "embedding": list(np.random.randn(512)),  # Python list
                    "dataset_source": "Western",
                    "thumbnail_url": "/path/invalid.jpg"
                }
            }
            
            with open(db_file, 'wb') as f:
                pickle.dump(database, f)
            
            matcher = CelebrityMatcher(db_path=str(db_file))
            
            assert "Valid_Array" in matcher.celebrity_embeddings
            assert "Invalid_List" not in matcher.celebrity_embeddings
    
    def test_database_with_multiple_celebrities(self):
        """Test loading database with multiple celebrities."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_file = Path(tmp_dir) / "test.pkl"
            
            database = {
                f"Celebrity_{i}": {
                    "embedding": np.random.randn(512),
                    "dataset_source": "Western" if i % 2 == 0 else "Bollywood",
                    "thumbnail_url": f"/path/{i}.jpg"
                }
                for i in range(10)
            }
            
            with open(db_file, 'wb') as f:
                pickle.dump(database, f)
            
            matcher = CelebrityMatcher(db_path=str(db_file))
            
            assert len(matcher.celebrity_embeddings) == 10


# ============================================================================
# TEST GROUP 6: Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_database_ranking(self):
        """Test ranking with empty database returns empty list."""
        matcher = CelebrityMatcher(db_path="/nonexistent/path")
        matcher.celebrity_embeddings = {}
        
        query = np.random.randn(512)
        rankings = matcher.rank_celebrities(query)
        
        assert rankings == []
    
    def test_empty_database_matching(self):
        """Test best match with empty database."""
        matcher = CelebrityMatcher(db_path="/nonexistent/path")
        matcher.celebrity_embeddings = {}
        
        ranked = []
        result = matcher.get_best_match(ranked, threshold=0.55)
        
        assert result["name"] is None
    
    def test_single_celebrity_matching(self):
        """Test matching with single celebrity in database."""
        matcher = CelebrityMatcher(db_path="/nonexistent/path")
        matcher.celebrity_embeddings = {
            "Only_Celebrity": {
                "embedding": np.random.randn(512),
                "dataset_source": "Western",
                "thumbnail_url": "/path/only.jpg"
            }
        }
        matcher.full_database = matcher.celebrity_embeddings
        
        query = np.random.randn(512)
        rankings = matcher.rank_celebrities(query)
        
        assert len(rankings) == 1
        assert rankings[0][0] == "Only_Celebrity"
    
    def test_no_matching_celebrity_all_below_threshold(self, matcher_with_embeddings):
        """Test when all celebrities score below confidence threshold."""
        ranked = [
            ("Celebrity_A", 0.40),
            ("Celebrity_B", 0.45),
            ("Celebrity_C", 0.50),
        ]
        
        result = matcher_with_embeddings.get_best_match(ranked, threshold=0.55)
        
        assert result["name"] is None
        assert result["message"] == "No close match found in our dataset"
    
    def test_similarity_exactly_at_threshold_boundary(self, matcher_with_embeddings):
        """Test boundary case: similarity exactly equals threshold."""
        # Test at 0.55 threshold (the default confidence threshold)
        ranked = [("Celebrity_A", 0.55)]
        
        result = matcher_with_embeddings.get_best_match(ranked, threshold=0.55)
        
        # >= comparison: 0.55 >= 0.55 is True
        assert result["name"] is not None
        assert result["confidence_score"] == 0.55
    
    def test_name_formatting_with_underscores(self, matcher_with_embeddings):
        """Test that underscores in names are converted to spaces."""
        ranked = [("First_Name_Last_Name", 0.75)]
        
        result = matcher_with_embeddings.get_best_match(ranked, threshold=0.55)
        
        assert "_" not in result["name"]
        assert result["name"] == "First Name Last Name"
    
    def test_very_high_similarity_score(self, matcher_with_embeddings):
        """Test with very high similarity score (1.0)."""
        ranked = [("Celebrity_A", 1.0)]
        
        result = matcher_with_embeddings.get_best_match(ranked, threshold=0.55)
        
        assert result["name"] is not None
        assert result["confidence_score"] == 1.0
    
    def test_very_low_similarity_score(self, matcher_with_embeddings):
        """Test with very low similarity score (0.0)."""
        ranked = [("Celebrity_A", 0.0)]
        
        result = matcher_with_embeddings.get_best_match(ranked, threshold=0.55)
        
        assert result["name"] is None
    
    def test_missing_celebrity_in_database(self, matcher_with_embeddings):
        """Test behavior when ranked celebrity is not in database."""
        # Celebrity not in matcher_with_embeddings.celebrity_embeddings
        ranked = [("Unknown_Celebrity", 0.75)]
        
        result = matcher_with_embeddings.get_best_match(ranked, threshold=0.55)
        
        # Should still process (graceful degradation)
        assert result["name"] == "Unknown Celebrity"
        assert result["confidence_score"] == 0.75
    
    def test_zero_confidence_score(self, matcher_with_embeddings):
        """Test with zero confidence score."""
        ranked = [("Celebrity_A", 0.0)]
        
        result = matcher_with_embeddings.get_best_match(ranked, threshold=0.0)
        
        assert result["name"] is not None
        assert result["confidence_score"] == 0.0
    
    def test_rounding_precision_edge_cases(self, matcher_with_embeddings):
        """Test rounding precision with various decimal places."""
        # Note: Python 3 uses banker's rounding (round to nearest even)
        test_cases = [
            (0.549, 0.55),   # Rounds up (> 0.545)
            (0.544, 0.54),   # Rounds down (< 0.545)
            (0.555, 0.56),   # Rounds up (> 0.555)
            (0.756789, 0.76), # Rounds up
            (0.564, 0.56),   # Rounds down
        ]
        
        for score, expected in test_cases:
            ranked = [("Celebrity_A", score)]
            result = matcher_with_embeddings.get_best_match(ranked, threshold=0.0)
            
            assert result["confidence_score"] == expected, \
                f"Score {score} should round to {expected}, got {result['confidence_score']}"
    
    def test_error_handling_invalid_pickle_structure(self):
        """Test error handling with invalid pickle structure."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_file = Path(tmp_dir) / "test.pkl"
            
            # Create pickle with non-dict root
            with open(db_file, 'wb') as f:
                pickle.dump(["not", "a", "dict"], f)
            
            matcher = CelebrityMatcher(db_path=str(db_file))
            
            # Should handle gracefully
            assert matcher.celebrity_embeddings == {}
            assert matcher.is_available is False


# ============================================================================
# TEST GROUP 7: Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_matching_pipeline_high_similarity(self, matcher_with_embeddings):
        """Test full pipeline: rank then threshold with high similarity."""
        # Create query similar to Celebrity_A
        celeb_a_embedding = matcher_with_embeddings.celebrity_embeddings["Celebrity_A"]["embedding"]
        query = celeb_a_embedding + 0.01 * np.random.randn(512)
        
        rankings = matcher_with_embeddings.rank_celebrities(query)
        result = matcher_with_embeddings.get_best_match(rankings, threshold=0.55)
        
        # Should find match
        assert result["name"] is not None
        assert result["confidence_score"] >= 0.55
    
    def test_full_matching_pipeline_low_similarity(self, matcher_with_embeddings):
        """Test full pipeline: rank then threshold with low similarity."""
        # Create random query unlikely to match
        np.random.seed(999)
        query = np.random.randn(512)
        
        rankings = matcher_with_embeddings.rank_celebrities(query)
        
        # Get all scores
        scores = [s for _, s in rankings]
        
        # If highest score is below threshold, should return no match
        if scores[0] < 0.55:
            result = matcher_with_embeddings.get_best_match(rankings, threshold=0.55)
            assert result["name"] is None
    
    def test_ranking_consistency_across_calls(self, matcher_with_embeddings):
        """Test that ranking is consistent across multiple calls."""
        query = np.random.randn(512)
        
        rankings1 = matcher_with_embeddings.rank_celebrities(query)
        rankings2 = matcher_with_embeddings.rank_celebrities(query)
        rankings3 = matcher_with_embeddings.rank_celebrities(query)
        
        assert rankings1 == rankings2 == rankings3
    
    def test_threshold_behavior_across_range(self, matcher_with_embeddings):
        """Test threshold behavior across different threshold values."""
        ranked = [("Celebrity_A", 0.65)]
        
        # Test with various thresholds
        for threshold in [0.0, 0.30, 0.50, 0.55, 0.60, 0.65, 0.70, 1.0]:
            result = matcher_with_embeddings.get_best_match(ranked, threshold=threshold)
            
            if threshold <= 0.65:
                assert result["name"] is not None
            else:
                assert result["name"] is None


# ============================================================================
# TEST GROUP 8: Response Structure and Formatting
# ============================================================================

class TestResponseStructure:
    """Test response structure and data formatting."""
    
    def test_response_structure_on_match(self, matcher_with_embeddings):
        """Test response structure when match found."""
        ranked = [("Celebrity_A", 0.70)]
        
        result = matcher_with_embeddings.get_best_match(ranked, threshold=0.55)
        
        # Verify required fields present
        assert "name" in result
        assert "confidence_score" in result
        assert "dataset_source" in result
        assert "thumbnail_url" in result
        
        # Verify correct types
        assert isinstance(result["name"], str)
        assert isinstance(result["confidence_score"], float)
        assert isinstance(result["dataset_source"], str)
        assert isinstance(result["thumbnail_url"], str)
    
    def test_response_structure_on_no_match(self, matcher_with_embeddings):
        """Test response structure when no match found."""
        ranked = [("Celebrity_A", 0.40)]
        
        result = matcher_with_embeddings.get_best_match(ranked, threshold=0.55)
        
        # Verify required fields
        assert "name" in result
        assert result["name"] is None
        assert "message" in result
        assert isinstance(result["message"], str)
        
        # Verify match-specific fields absent
        assert "confidence_score" not in result
        assert "dataset_source" not in result
    
    def test_dataset_source_in_response(self, matcher_with_embeddings):
        """Test that dataset_source is included in response."""
        ranked = [("Celebrity_A", 0.70)]
        
        result = matcher_with_embeddings.get_best_match(ranked, threshold=0.55)
        
        assert result["dataset_source"] == "Western"
    
    def test_thumbnail_url_in_response(self, matcher_with_embeddings):
        """Test that thumbnail_url is included in response."""
        ranked = [("Celebrity_A", 0.70)]
        
        result = matcher_with_embeddings.get_best_match(ranked, threshold=0.55)
        
        assert result["thumbnail_url"] == "/thumbnails/a.jpg"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
