"""
Tests for cosine similarity calculation.

Tests the calculate_cosine_similarity function with various embedding pairs,
including edge cases and real-world scenarios.
"""

import pytest
import numpy as np
from app.utils.similarity import calculate_cosine_similarity


class TestCosineSimlarityBasics:
    """Test basic cosine similarity calculations."""
    
    def test_identical_embeddings_returns_one(self):
        """Test that identical embeddings yield similarity ≈ 1.0"""
        embedding = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        similarity = calculate_cosine_similarity(embedding, embedding)
        
        assert abs(similarity - 1.0) < 1e-10
    
    def test_orthogonal_embeddings_returns_zero(self):
        """Test that orthogonal embeddings yield similarity ≈ 0.0"""
        # Standard basis vectors are orthogonal
        e1 = np.array([1.0, 0.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0, 0.0])
        
        similarity = calculate_cosine_similarity(e1, e2)
        
        assert abs(similarity) < 1e-10
    
    def test_opposite_embeddings_returns_negative_one(self):
        """Test that opposite embeddings yield similarity ≈ -1.0"""
        e1 = np.array([1.0, 2.0, 3.0])
        e2 = np.array([-1.0, -2.0, -3.0])
        
        similarity = calculate_cosine_similarity(e1, e2)
        
        assert abs(similarity - (-1.0)) < 1e-10
    
    def test_normalized_vectors_scaled_embeddings(self):
        """Test that scaling vectors doesn't change cosine similarity."""
        e1 = np.array([1.0, 2.0, 3.0, 4.0])
        e2 = np.array([1.5, 3.0, 4.5, 6.0])  # Scaled by 1.5
        
        similarity = calculate_cosine_similarity(e1, e2)
        
        # Should be 1.0 since e2 = 1.5 * e1 (same direction)
        assert abs(similarity - 1.0) < 1e-10


class TestCosineSimlarityEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_vector_returns_zero(self):
        """Test that zero vector has zero similarity with any vector."""
        zero = np.array([0.0, 0.0, 0.0])
        non_zero = np.array([1.0, 2.0, 3.0])
        
        similarity = calculate_cosine_similarity(zero, non_zero)
        
        assert similarity == 0.0
    
    def test_both_zero_vectors_returns_zero(self):
        """Test that zero vectors have zero similarity with each other."""
        zero1 = np.array([0.0, 0.0, 0.0, 0.0])
        zero2 = np.array([0.0, 0.0, 0.0, 0.0])
        
        similarity = calculate_cosine_similarity(zero1, zero2)
        
        assert similarity == 0.0
    
    def test_single_element_vectors(self):
        """Test with 1-dimensional embeddings."""
        e1 = np.array([5.0])
        e2 = np.array([3.0])
        
        similarity = calculate_cosine_similarity(e1, e2)
        
        # Both positive, same direction → similarity = 1.0
        assert abs(similarity - 1.0) < 1e-10
    
    def test_single_element_opposite_signs(self):
        """Test with 1-dimensional embeddings with opposite signs."""
        e1 = np.array([5.0])
        e2 = np.array([-3.0])
        
        similarity = calculate_cosine_similarity(e1, e2)
        
        # Opposite direction → similarity = -1.0
        assert abs(similarity - (-1.0)) < 1e-10
    
    def test_very_high_dimensional_embeddings(self):
        """Test with high-dimensional embeddings (FaceNet: 512-dim)."""
        # Simulate 512-dimensional FaceNet embeddings
        np.random.seed(42)
        e1 = np.random.randn(512)
        e2 = e1 + 0.01 * np.random.randn(512)  # Slightly perturbed
        
        similarity = calculate_cosine_similarity(e1, e2)
        
        # Should be close to 1.0 but not exactly 1.0
        assert 0.9 < similarity <= 1.0


class TestCosineSimlarityInputValidation:
    """Test input validation and error handling."""
    
    def test_non_numpy_array_input_raises_error(self):
        """Test that non-numpy arrays raise ValueError."""
        e1 = [1.0, 2.0, 3.0]  # Python list
        e2 = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="must be a numpy array"):
            calculate_cosine_similarity(e1, e2)
    
    def test_both_non_numpy_arrays_raises_error(self):
        """Test that both inputs being lists raises ValueError."""
        e1 = [1.0, 2.0, 3.0]
        e2 = [1.0, 2.0, 3.0]
        
        with pytest.raises(ValueError):
            calculate_cosine_similarity(e1, e2)
    
    def test_2d_array_raises_error(self):
        """Test that 2D arrays raise ValueError."""
        e1 = np.array([[1.0, 2.0, 3.0]])  # 2D: shape (1, 3)
        e2 = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError, match="must be a 1D array"):
            calculate_cosine_similarity(e1, e2)
    
    def test_mismatched_lengths_raises_error(self):
        """Test that embeddings of different lengths raise ValueError."""
        e1 = np.array([1.0, 2.0, 3.0])
        e2 = np.array([1.0, 2.0, 3.0, 4.0])
        
        with pytest.raises(ValueError, match="must have same length"):
            calculate_cosine_similarity(e1, e2)
    
    def test_complex_numbers_handled(self):
        """Test that complex embeddings work (if supported by numpy)."""
        e1 = np.array([1.0 + 0.0j, 2.0 + 0.0j])
        e2 = np.array([1.0 + 0.0j, 2.0 + 0.0j])
        
        # Should work since numpy.dot handles complex numbers
        similarity = calculate_cosine_similarity(e1, e2)
        
        # Result should be real and approximately 1.0
        assert np.isreal(similarity)


class TestCosineSimlarityRealWorldScenarios:
    """Test with realistic embedding scenarios."""
    
    def test_similar_face_embeddings(self):
        """Test with embeddings representing similar faces."""
        np.random.seed(42)
        
        # Create base embedding
        base_embedding = np.random.randn(512)
        
        # Create slightly perturbed version (same person, different photo)
        perturbed = base_embedding + 0.05 * np.random.randn(512)
        
        similarity = calculate_cosine_similarity(base_embedding, perturbed)
        
        # Should be quite high but not perfect
        assert 0.95 < similarity < 1.0
    
    def test_diverse_embeddings(self):
        """Test with completely different embeddings."""
        np.random.seed(42)
        
        e1 = np.random.randn(512)
        
        np.random.seed(43)  # Different seed
        e2 = np.random.randn(512)
        
        similarity = calculate_cosine_similarity(e1, e2)
        
        # Should be closer to 0 (independent random vectors)
        assert abs(similarity) < 0.3
    
    def test_celebrity_match_scenario(self):
        """Test scenario: user embedding vs celebrity database embedding."""
        # Simulate user embedding
        user_embedding = np.random.randn(512)
        
        # Simulate celebrity matching the user
        celebrity_embedding = user_embedding + 0.02 * np.random.randn(512)
        
        similarity = calculate_cosine_similarity(user_embedding, celebrity_embedding)
        
        # Should exceed typical confidence threshold (0.55)
        assert similarity > 0.95
    
    def test_non_matching_celebrity_scenario(self):
        """Test scenario: user embedding vs non-matching celebrity."""
        np.random.seed(42)
        user_embedding = np.random.randn(512)
        
        np.random.seed(100)  # Very different seed
        non_match_embedding = np.random.randn(512)
        
        similarity = calculate_cosine_similarity(user_embedding, non_match_embedding)
        
        # Should be below typical confidence threshold (0.55)
        assert similarity < 0.55


class TestCosineSimlarityNumericalProperties:
    """Test mathematical and numerical properties."""
    
    def test_commutative_property(self):
        """Test that similarity(a, b) == similarity(b, a)."""
        e1 = np.array([1.0, 2.0, 3.0, 4.0])
        e2 = np.array([5.0, 6.0, 7.0, 8.0])
        
        sim_1_2 = calculate_cosine_similarity(e1, e2)
        sim_2_1 = calculate_cosine_similarity(e2, e1)
        
        assert abs(sim_1_2 - sim_2_1) < 1e-10
    
    def test_symmetry_property(self):
        """Test numerical stability with various magnitudes."""
        e1 = np.array([1e-6, 2e-6, 3e-6])
        e2 = np.array([1e6, 2e6, 3e6])
        
        similarity = calculate_cosine_similarity(e1, e2)
        
        # Should still be 1.0 (same direction, different magnitude)
        assert abs(similarity - 1.0) < 1e-8
    
    def test_result_in_valid_range(self):
        """Test that result is always in [-1, 1]."""
        np.random.seed(42)
        
        for _ in range(100):
            e1 = np.random.randn(np.random.randint(1, 100))
            e2 = np.random.randn(len(e1))
            
            similarity = calculate_cosine_similarity(e1, e2)
            
            assert -1.0 <= similarity <= 1.0
    
    def test_large_magnitude_vectors(self):
        """Test with very large magnitude vectors."""
        e1 = np.array([1e10, 2e10, 3e10])
        e2 = np.array([1e10, 2e10, 3e10])
        
        similarity = calculate_cosine_similarity(e1, e2)
        
        assert abs(similarity - 1.0) < 1e-8
    
    def test_small_magnitude_vectors(self):
        """Test with very small magnitude vectors."""
        e1 = np.array([1e-10, 2e-10, 3e-10])
        e2 = np.array([1e-10, 2e-10, 3e-10])
        
        similarity = calculate_cosine_similarity(e1, e2)
        
        assert abs(similarity - 1.0) < 1e-8


class TestCosineSimlarityDocumentedExamples:
    """Test examples from docstring."""
    
    def test_docstring_example_identical(self):
        """Test docstring example: identical embeddings."""
        embedding = np.array([1.0, 2.0, 3.0])
        result = calculate_cosine_similarity(embedding, embedding)
        
        assert result == 1.0
    
    def test_docstring_example_orthogonal(self):
        """Test docstring example: orthogonal embeddings."""
        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0])
        result = calculate_cosine_similarity(e1, e2)
        
        assert result == 0.0
    
    def test_docstring_example_diverse(self):
        """Test docstring example: diverse embeddings."""
        e1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        e2 = np.array([0.12, 0.19, 0.31, 0.39, 0.51])
        sim = calculate_cosine_similarity(e1, e2)
        
        assert 0.99 < sim < 1.0


class TestCosineSimlarityFaceNetContext:
    """Test in FaceNet-specific context (512-dimensional embeddings)."""
    
    def test_facenet_512_dim_identical(self):
        """Test 512-dim embeddings (FaceNet standard)."""
        np.random.seed(42)
        embedding = np.random.randn(512)
        
        similarity = calculate_cosine_similarity(embedding, embedding)
        
        assert abs(similarity - 1.0) < 1e-10
    
    def test_facenet_512_dim_strong_match(self):
        """Test 512-dim: embeddings with high similarity."""
        np.random.seed(42)
        e1 = np.random.randn(512)
        e2 = 0.95 * e1 + 0.05 * np.random.randn(512)
        
        similarity = calculate_cosine_similarity(e1, e2)
        
        # Should indicate a strong match (> confidence threshold 0.55)
        assert similarity > 0.90
    
    def test_facenet_512_dim_weak_match(self):
        """Test 512-dim: embeddings with low similarity."""
        np.random.seed(42)
        e1 = np.random.randn(512)
        
        np.random.seed(100)
        e2 = np.random.randn(512)
        
        similarity = calculate_cosine_similarity(e1, e2)
        
        # Should indicate weak match (< confidence threshold 0.55)
        assert similarity < 0.55
    
    def test_facenet_confidence_threshold_boundary(self):
        """Test embeddings at confidence threshold boundary (0.55)."""
        np.random.seed(42)
        
        # Create embedding pair with specific similarity
        e1 = np.array([1.0] * 512)
        # Create e2 to have approximately 0.55 similarity with e1
        e2 = 0.55 * e1 + np.sqrt(1 - 0.55**2) * np.random.randn(512)
        
        similarity = calculate_cosine_similarity(e1, e2)
        
        # Should be close to threshold
        assert 0.50 < similarity < 0.60
