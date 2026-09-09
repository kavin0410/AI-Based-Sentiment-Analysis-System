"""Feature Extraction Module for AI Sentiment Intelligence.

Handles TF-IDF vectorization for converting text data
into numerical features for machine learning models.
"""

from pathlib import Path
from typing import Optional, Tuple, Any
import joblib
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

# Optional import of configuration constants
try:
    from src.config import MAX_FEATURES, TFIDF_VECTORIZER_PATH
except ImportError:
    try:
        from config import MAX_FEATURES, TFIDF_VECTORIZER_PATH
    except ImportError:
        MAX_FEATURES = 5000
        TFIDF_VECTORIZER_PATH = Path("models/tfidf_vectorizer.joblib")


def create_tfidf_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    """Create and return a configured TF-IDF vectorizer.

    Args:
        max_features: Maximum number of top features to extract by term frequency.

    Returns:
        Configured TfidfVectorizer instance.
    """
    # TODO: Create and return configured TF-IDF vectorizer
    raise NotImplementedError("To be implemented in Stage 2.")


def fit_transform_tfidf(
    texts: pd.Series, vectorizer: Optional[TfidfVectorizer] = None
) -> Tuple[spmatrix, TfidfVectorizer]:
    """Fit TF-IDF vectorizer on text data and transform to feature matrix.

    Args:
        texts: Pandas Series containing preprocessed text strings.
        vectorizer: Optional unfitted or existing vectorizer. If None, creates a new one.

    Returns:
        Tuple of (features_matrix, fitted_vectorizer).
    """
    # TODO: Fit and transform text data, return (features_matrix, fitted_vectorizer)
    raise NotImplementedError("To be implemented in Stage 2.")


def transform_tfidf(texts: pd.Series, vectorizer: TfidfVectorizer) -> spmatrix:
    """Transform new text using an already-fitted TF-IDF vectorizer.

    Args:
        texts: Pandas Series containing preprocessed text strings.
        vectorizer: Pre-fitted TfidfVectorizer instance.

    Returns:
        Sparse feature matrix representing transformed texts.
    """
    # TODO: Transform new text using already-fitted vectorizer
    raise NotImplementedError("To be implemented in Stage 2.")


def save_vectorizer(vectorizer: TfidfVectorizer, filepath: Path) -> None:
    """Save fitted TF-IDF vectorizer to disk using joblib.

    Args:
        vectorizer: Fitted TfidfVectorizer instance.
        filepath: Destination file path for saving.
    """
    # TODO: Save fitted vectorizer using joblib
    raise NotImplementedError("To be implemented in Stage 2.")


def load_vectorizer(filepath: Path) -> TfidfVectorizer:
    """Load saved TF-IDF vectorizer from disk.

    Args:
        filepath: Path to the serialized vectorizer file.

    Returns:
        Loaded TfidfVectorizer instance.
    """
    # TODO: Load saved vectorizer
    raise NotImplementedError("To be implemented in Stage 2.")
