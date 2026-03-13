import numpy as np
from typing import List, Optional
from transformers import AutoModel, AutoTokenizer
from dotenv import load_dotenv
import torch

load_dotenv()

_embedding_model: Optional[AutoModel] = None
_tokenizer: Optional[AutoTokenizer] = None

try:
    _embedding_model = AutoModel.from_pretrained("intfloat/e5-large-v2")
    _tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-large-v2")

    # Move model to GPU if available
    if torch.cuda.is_available():
        _embedding_model.to('cuda')
        print("Embedding model moved to GPU.")

    print("Embedding model (intfloat/e5-large-v2) loaded successfully using transformers.AutoModel and AutoTokenizer.")
except Exception as e:
    print(f"Error loading embedding model with transformers.AutoModel: {e}. Falling back to mock embeddings.")


def get_embedding(text: str) -> np.ndarray:
    """
    Function to get embeddings of a text using the loaded E5 model.
    Falls back to a mock hash if the model failed to load.

    Args:
        text (str): The input text to embed.

    Returns:
        np.ndarray: A NumPy array representing the embedding of the text.
    """
    if _embedding_model and _tokenizer:
        inputs = _tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')

        # Move input tensors to the same device as the model (GPU if available)
        if _embedding_model.device.type == 'cuda':
            inputs = {k: v.to(_embedding_model.device) for k, v in inputs.items()}

        with torch.no_grad():  # Disable gradient calculations for speed and memory efficiency
            model_output = _embedding_model(**inputs)
            embeddings = model_output.last_hidden_state  # Get hidden states
            attention_mask = inputs['attention_mask']

            # Average pooling of token embeddings, masking out padded tokens.
            embeddings = (embeddings * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1).unsqueeze(
                -1)

            return embeddings.cpu().numpy().flatten()
    else:
        # Fallback to a deterministic mock "embedding" if model not loaded
        return np.array([hash(text) % 1000]).astype(float)  # Ensure float type for mock embedding


def calculate_cosine_similarity(text1: str, text2: str) -> float:
    """
    Calculates cosine similarity between two texts using E5 embeddings.
    Falls back to a heuristic if the embedding model is not loaded.

    Args:
        text1 (str): The first text.
        text2 (str): The second text.

    Returns:
        float: The cosine similarity between the two texts.
    """
    if _embedding_model and _tokenizer:  # Ensure tokenizer is also loaded
        emb1 = get_embedding(text1)
        emb2 = get_embedding(text2)

        # Ensure embeddings are NumPy arrays for np.dot and np.linalg.norm
        if not isinstance(emb1, np.ndarray):
            emb1 = np.array(emb1)
        if not isinstance(emb2, np.ndarray):
            emb2 = np.array(emb2)

        # Handle zero vectors to avoid division by zero
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0  # If one of the vectors is zero, similarity is 0

        similarity = np.dot(emb1, emb2) / (norm1 * norm2)
        return float(similarity)
    else:
        # Fallback to a heuristic if model not loaded
        if text1 == text2:
            return 1.0
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        common_words_count = len(words1.intersection(words2))
        all_words_count = len(words1.union(words2))
        return 0.1 + 0.9 * (common_words_count / all_words_count)


def aggregate_scores(scores: List[float], method: str = 'mean') -> float:
    """
    Aggregates a list of numerical scores (e.g., intrinsic strengths).

    Args:
        scores (List[float]): A list of scores to aggregate.
        method (str): The aggregation method ('mean' or 'max'). Defaults to 'mean'.

    Returns:
        float: The aggregated score.

    Raises:
        ValueError: If an unknown aggregation method is provided.
    """
    if not scores:
        return 0.0
    if method == 'mean':
        return float(sum(scores) / len(scores))
    elif method == 'max':
        return float(max(scores))
    else:
        raise ValueError(f"Unknown aggregation method: {method}. Choose 'mean' or 'max'")