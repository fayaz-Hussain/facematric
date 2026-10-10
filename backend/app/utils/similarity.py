"""
Similarity Calculation Utilities

Provides functions for computing similarity metrics between embeddings.
"""

import numpy as np
from typing import Tuple


def calculate_cosine_similarity(
    embedding1: np.ndarray,
    embedding2: np.ndarray
) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Cosine similarity is computed as: sim = (u · v) / (||u|| × ||v||)
    where u and v are the embedding vectors.
    
    The result is in the range [-1, 1], where:
    - 1.0 = identical embeddings (parallel vectors)
    - 0.0 = orthogonal embeddings (perpendicular vectors)
    - -1.0 = opposite embeddings (anti-parallel vectors)
    
    For face embeddings from normalized models like FaceNet, the result
    typically falls in [0, 1] range where higher values indicate greater similarity.
    
    Args:
        embedding1: First embedding vector (1D numpy array)
        embedding2: Second embedding vector (1D numpy array)
    
    Returns:
        float: Cosine similarity score in range [-1, 1]
    
    Raises:
        ValueError: If embeddings are not 1D numpy arrays
        ValueError: If embeddings have different lengths
    
    Examples:
        >>> import numpy as np
        >>> from app.utils.similarity import calculate_cosine_similarity
        
        # Identical embeddings
        >>> embedding = np.array([1.0, 2.0, 3.0])
        >>> calculate_cosine_similarity(embedding, embedding)
        1.0
        
        # Orthogonal embeddings (dot product = 0)
        >>> e1 = np.array([1.0, 0.0, 0.0])
        >>> e2 = np.array([0.0, 1.0, 0.0])
        >>> calculate_cosine_similarity(e1, e2)
        0.0
        
        # Diverse embeddings (typical face similarity)
        >>> e1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        >>> e2 = np.array([0.12, 0.19, 0.31, 0.39, 0.51])
        >>> sim = calculate_cosine_similarity(e1, e2)
        >>> 0.99 < sim < 1.0  # Very similar
        True
    """
    # Validate input types and dimensions
    if not isinstance(embedding1, np.ndarray):
        raise ValueError(
            f"embedding1 must be a numpy array, got {type(embedding1).__name__}"
        )
    if not isinstance(embedding2, np.ndarray):
        raise ValueError(
            f"embedding2 must be a numpy array, got {type(embedding2).__name__}"
        )
    
    if embedding1.ndim != 1:
        raise ValueError(
            f"embedding1 must be a 1D array, got shape {embedding1.shape}"
        )
    if embedding2.ndim != 1:
        raise ValueError(
            f"embedding2 must be a 1D array, got shape {embedding2.shape}"
        )
    
    if embedding1.shape[0] != embedding2.shape[0]:
        raise ValueError(
            f"embeddings must have same length: "
            f"{embedding1.shape[0]} vs {embedding2.shape[0]}"
        )
    
    # Compute dot product
    dot_product = np.dot(embedding1, embedding2)
    
    # Compute norms (Euclidean length of vectors)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    # Handle zero-norm case (zero vectors)
    if norm1 == 0.0 or norm2 == 0.0:
        # By convention, similarity between zero vector and any vector is 0
        return 0.0
    
    # Compute cosine similarity
    similarity = dot_product / (norm1 * norm2)
    
    # Ensure result is in valid range (numerical errors might cause slight overflow)
    similarity = float(np.clip(similarity, -1.0, 1.0))
    
    return similarity
