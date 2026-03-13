import numpy as np
import pytest
from common.src.utils import get_embedding, calculate_cosine_similarity, aggregate_scores, _embedding_model, _tokenizer

MODEL_LOADED = (_embedding_model is not None) and (_tokenizer is not None)

print("\n--- Running Tests for embedding_utils ---")
print(f"Real embedding model loaded: {MODEL_LOADED}")


def test_get_embedding_return_type_and_dimension():
    """Test if get_embedding returns a numpy array and has expected dimension."""
    test_text = "This is a test sentence."
    embedding = get_embedding(test_text)
    assert isinstance(embedding, np.ndarray)
    if MODEL_LOADED:
        # E5-large-v2 embeddings have a dimension of 1024
        assert embedding.shape == (1024,)
    else:
        # Mock embedding is a single float
        assert embedding.shape == (1,)

def test_get_embedding_consistency():
    """Test if identical texts produce identical embeddings."""
    text1 = "The quick brown fox jumps over the lazy dog."
    text2 = "The quick brown fox jumps over the lazy dog."
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    assert np.allclose(emb1, emb2, atol=1e-6) # Use allclose for float comparisons

def test_calculate_cosine_similarity_identical_texts():
    """Test cosine similarity for identical texts."""
    text1 = "This is an absolutely identical text."
    text2 = "This is an absolutely identical text."
    similarity = calculate_cosine_similarity(text1, text2)
    assert similarity >= 0.99 # Should be very close to 1.0 for real models, exactly 1.0 for mock

def test_calculate_cosine_similarity_similar_texts():
    """Test cosine similarity for similar texts (expected high similarity)."""
    text1 = "A cat sat on the mat."
    text2 = "On the mat, a feline was sitting."
    similarity = calculate_cosine_similarity(text1, text2)
    if MODEL_LOADED:
        assert similarity > 0.8 # Expect reasonably high similarity for real models
    else:
        assert similarity >= 0.8 # Should be greater than baseline 0.1

def test_calculate_cosine_similarity_dissimilar_texts():
    """Test cosine similarity for dissimilar texts (expected low similarity)."""
    text1 = "Programming is a fascinating subject."
    text2 = "A delicious apple pie recipe."
    similarity = calculate_cosine_similarity(text1, text2)
    if MODEL_LOADED:
        assert similarity < 0.78 # Expect low similarity for real models
    else:
        # For mock fallback, common words heuristic applies
        assert similarity < 0.78

def test_calculate_cosine_similarity_empty_texts():
    """Test cosine similarity with empty texts (should handle gracefully)."""
    text1 = ""
    text2 = "Some text."
    similarity = calculate_cosine_similarity(text1, text2)
    assert similarity < 0.78

def test_aggregate_scores_mean_method():
    """Test aggregation with 'mean' method."""
    scores = [0.8, 0.9, 0.75, 0.95]
    assert aggregate_scores(scores, 'mean') == pytest.approx(0.85)

def test_aggregate_scores_max_method():
    """Test aggregation with 'max' method."""
    scores = [0.8, 0.9, 0.75, 0.95]
    assert aggregate_scores(scores, 'max') == 0.95

def test_aggregate_scores_empty_list():
    """Test aggregation with an empty list of scores."""
    scores = []
    assert aggregate_scores(scores, 'mean') == 0.0
    assert aggregate_scores(scores, 'max') == 0.0

def test_aggregate_scores_invalid_method():
    """Test aggregation with an invalid method (should raise ValueError)."""
    scores = [0.5, 0.6]
    with pytest.raises(ValueError, match="Unknown aggregation method: invalid_method. Choose 'mean' or 'max'"):
        aggregate_scores(scores, 'invalid_method')

print("\n--- All tests completed ---")
