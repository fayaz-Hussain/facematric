"""
Unit tests for CelebrityMatcher.rank_celebrities method (Task 8.3)

Tests the celebrity database comparison and ranking functionality.
Validates:
- Similarity score computation for each celebrity
- Proper sorting in descending order (highest first)
- Correct return format: [(name, score), ...]
- Edge cases: empty database, single celebrity
"""

import pytest
import numpy as np
from app.components.celebrity_matcher import CelebrityMatcher


@pytest.fixture
def sample_embedding():
    """Generate a sample 128-dimensional FaceNet embedding."""
    return np.random.rand(128)


@pytest.fixture
def sample_celebrity_database():
    """
    Create a sample celebrity database with 5 celebrities.
    
    Database structure:
    {
        "celebrity_name": {
            "embedding": [...],  # 128-dim vector
            "dataset_source": "Western" or "Bollywood",
            "thumbnail_path": "path/to/image.jpg"
        },
        ...
    }
    """
    # Create deterministic embeddings for testing
    embeddings = {
        "Celebrity_A": {
            "embedding": np.array([1.0] + [0.0] * 127),  # [1, 0, 0, ...]
            "dataset_source": "Western",
            "thumbnail_path": "/path/to/a.jpg"
        },
        "Celebrity_B": {
            "embedding": np.array([0.9] + [0.1] * 127),  # Very similar to A
            "dataset_source": "Western",
            "thumbnail_path": "/path/to/b.jpg"
        },
        "Celebrity_C": {
            "embedding": np.array([0.5] + [0.5] * 127),  # Moderately similar
            "dataset_source": "Bollywood",
            "thumbnail_path": "/path/to/c.jpg"
        },
        "Celebrity_D": {
            "embedding": np.array([0.1] + [0.9] * 127),  # Less similar
            "dataset_source": "Bollywood",
            "thumbnail_path": "/path/to/d.jpg"
        },
        "Celebrity_E": {
            "embedding": np.array([0.0] + [1.0] * 127),  # Least similar
            "dataset_source": "Western",
            "thumbnail_path": "/path/to/e.jpg"
        },
    }
    return embeddings


@pytest.fixture
def matcher_with_sample_db(sample_celebrity_database, monkeypatch):
    """Create a CelebrityMatcher with a mocked celebrity database."""
    # Create matcher without loading from file
    matcher = CelebrityMatcher()
    
    # Inject sample database
    matcher.celebrity_embeddings = sample_celebrity_database
    matcher.is_available = True
    
    return matcher


class TestRankCelebrities:
    """Test suite for rank_celebrities method."""
    
    def test_rank_celebrities_basic(self, matcher_with_sample_db, sample_embedding):
        """Test basic ranking with sample database."""
        rankings = matcher_with_sample_db.rank_celebrities(sample_embedding)
        
        # Should return all celebrities
        assert len(rankings) == 5
        
        # Each item should be (name, score) tuple
        for item in rankings:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)  # name
            assert isinstance(item[1], (float, np.floating))  # score
    
    def test_rank_celebrities_descending_order(self, matcher_with_sample_db, sample_embedding):
        """Test that celebrities are ranked in descending order by similarity."""
        rankings = matcher_with_sample_db.rank_celebrities(sample_embedding)
        
        # Extract scores
        scores = [score for _, score in rankings]
        
        # Check that scores are in descending order
        assert scores == sorted(scores, reverse=True)
    
    def test_rank_celebrities_similarity_range(self, matcher_with_sample_db, sample_embedding):
        """Test that similarity scores are in valid range [-1, 1]."""
        rankings = matcher_with_sample_db.rank_celebrities(sample_embedding)
        
        for _, score in rankings:
            assert -1.0 <= score <= 1.0, f"Score {score} out of [-1, 1] range"
    
    def test_rank_celebrities_highest_first(self, matcher_with_sample_db):
        """Test that highest similarity score is first in ranking."""
        # Create query embedding similar to Celebrity_A
        query = np.array([1.0] + [0.0] * 127)
        
        rankings = matcher_with_sample_db.rank_celebrities(query)
        
        # First should be Celebrity_A (most similar)
        best_name, best_score = rankings[0]
        assert best_name == "Celebrity_A"
        assert best_score > 0.99  # Should be very close to 1.0
    
    def test_rank_celebrities_multiple_same_embedding(self, matcher_with_sample_db):
        """Test ranking with embedding identical to one celebrity."""
        # Use Celebrity_B's embedding as query
        celeb_b_embedding = matcher_with_sample_db.celebrity_embeddings["Celebrity_B"]["embedding"]
        
        rankings = matcher_with_sample_db.rank_celebrities(celeb_b_embedding)
        
        # First should be Celebrity_B with very high score
        best_name, best_score = rankings[0]
        assert best_name == "Celebrity_B"
        assert best_score > 0.99
    
    def test_rank_celebrities_returns_all_celebrities(self, matcher_with_sample_db, sample_embedding):
        """Test that all celebrities are included in ranking."""
        rankings = matcher_with_sample_db.rank_celebrities(sample_embedding)
        ranked_names = {name for name, _ in rankings}
        
        expected_names = {
            "Celebrity_A", "Celebrity_B", "Celebrity_C", "Celebrity_D", "Celebrity_E"
        }
        
        assert ranked_names == expected_names
    
    def test_rank_celebrities_empty_database(self, matcher_with_sample_db):
        """Test edge case: empty celebrity database returns empty list."""
        # Clear database
        matcher_with_sample_db.celebrity_embeddings = {}
        
        embedding = np.random.rand(128)
        rankings = matcher_with_sample_db.rank_celebrities(embedding)
        
        assert rankings == []
        assert len(rankings) == 0
    
    def test_rank_celebrities_none_database(self, matcher_with_sample_db):
        """Test edge case: None database returns empty list."""
        # Set database to None
        matcher_with_sample_db.celebrity_embeddings = None
        
        embedding = np.random.rand(128)
        rankings = matcher_with_sample_db.rank_celebrities(embedding)
        
        assert rankings == []
    
    def test_rank_celebrities_single_celebrity(self, matcher_with_sample_db):
        """Test edge case: single celebrity returns [(name, score)]."""
        # Keep only one celebrity
        single_celeb = {"Celebrity_A": matcher_with_sample_db.celebrity_embeddings["Celebrity_A"]}
        
        embedding = np.random.rand(128)
        rankings = matcher_with_sample_db.rank_celebrities(embedding)
        
        # Should be called with default database, not our single_celeb
        # Let's test by using the override parameter
        rankings = matcher_with_sample_db.rank_celebrities(embedding, single_celeb)
        
        assert len(rankings) == 1
        assert rankings[0][0] == "Celebrity_A"
    
    def test_rank_celebrities_override_database(self, matcher_with_sample_db, sample_embedding):
        """Test that override_database parameter works."""
        # Create a small override database
        override_db = {
            "Custom_Celebrity": {
                "embedding": sample_embedding,
                "dataset_source": "Custom",
                "thumbnail_path": "/custom/path.jpg"
            }
        }
        
        rankings = matcher_with_sample_db.rank_celebrities(
            sample_embedding,
            override_db
        )
        
        # Should use override database
        assert len(rankings) == 1
        assert rankings[0][0] == "Custom_Celebrity"
        # Should have very high similarity (identical embedding)
        assert rankings[0][1] > 0.99
    
    def test_rank_celebrities_deterministic(self, matcher_with_sample_db):
        """Test that ranking is deterministic (same input → same output)."""
        embedding = np.array([0.5] * 128)
        
        rankings1 = matcher_with_sample_db.rank_celebrities(embedding)
        rankings2 = matcher_with_sample_db.rank_celebrities(embedding)
        
        # Should produce identical results
        assert rankings1 == rankings2
    
    def test_rank_celebrities_format(self, matcher_with_sample_db, sample_embedding):
        """Test return format is exactly [(name, score), ...]."""
        rankings = matcher_with_sample_db.rank_celebrities(sample_embedding)
        
        # Should be a list
        assert isinstance(rankings, list)
        
        # Each element should be tuple
        for item in rankings:
            assert isinstance(item, tuple)
            assert len(item) == 2
            
            name, score = item
            assert isinstance(name, str)
            assert not name.startswith("_")  # Names should be readable
            assert isinstance(score, (float, np.floating))


class TestRankCelebritiesWithRealismEmbeddings:
    """Test rank_celebrities with realistic embedding data."""
    
    def test_rank_celebrities_with_normalized_embeddings(self):
        """Test ranking with L2-normalized embeddings (realistic for FaceNet)."""
        # Create normalized embeddings (unit vectors)
        embeddings = {}
        for i in range(3):
            # Random vector
            vec = np.random.randn(128)
            # Normalize to unit length
            vec = vec / np.linalg.norm(vec)
            embeddings[f"Celebrity_{i}"] = {
                "embedding": vec,
                "dataset_source": "Western",
                "thumbnail_path": f"/path/{i}.jpg"
            }
        
        matcher = CelebrityMatcher()
        matcher.celebrity_embeddings = embeddings
        
        # Query embedding
        query = np.random.randn(128)
        query = query / np.linalg.norm(query)
        
        rankings = matcher.rank_celebrities(query)
        
        # All similarities should be in [-1, 1] (typically [0, 1] for normalized)
        for name, score in rankings:
            assert -1.0 <= score <= 1.0
        
        # Should be sorted descending
        scores = [s for _, s in rankings]
        assert scores == sorted(scores, reverse=True)
    
    def test_rank_celebrities_orthogonal_embeddings(self):
        """Test ranking with orthogonal embeddings (zero similarity)."""
        # Create orthogonal embeddings
        embeddings = {
            "Celebrity_1": {
                "embedding": np.array([1.0] + [0.0] * 127),
                "dataset_source": "Western",
                "thumbnail_path": "/path/1.jpg"
            },
            "Celebrity_2": {
                "embedding": np.array([0.0, 1.0] + [0.0] * 126),
                "dataset_source": "Western",
                "thumbnail_path": "/path/2.jpg"
            }
        }
        
        matcher = CelebrityMatcher()
        matcher.celebrity_embeddings = embeddings
        
        # Query that is orthogonal to Celebrity_1
        query = np.array([0.0, 1.0] + [0.0] * 126)
        
        rankings = matcher.rank_celebrities(query)
        
        # Celebrity_2 should rank higher (more similar)
        assert rankings[0][0] == "Celebrity_2"


class TestRankCelebritiesIntegration:
    """Integration tests for rank_celebrities with _find_best_match."""
    
    def test_rank_celebrities_finds_best_match(self, matcher_with_sample_db):
        """Test that ranking is used correctly by _find_best_match."""
        # Create embedding similar to Celebrity_A
        query = np.array([1.0] + [0.0] * 127)
        
        rankings = matcher_with_sample_db.rank_celebrities(query)
        best_name, best_score = rankings[0]
        
        # _find_best_match should use this ranking
        match = matcher_with_sample_db._find_best_match(query)
        
        # Best match should correspond to best ranking
        if best_score >= matcher_with_sample_db.confidence_threshold:
            assert match.name == best_name.replace("_", " ")
            assert match.confidence == round(best_score, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
