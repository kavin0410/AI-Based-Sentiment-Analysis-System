"""Feature Extraction Module for AI Sentiment Intelligence.

Handles TF-IDF vectorization for converting preprocessed text data
into numerical feature representations for machine learning models.

Viva Note:
    TF-IDF (Term Frequency - Inverse Document Frequency) reflects how important
    a word is to a document in a collection. By using n-gram ranges (1, 2), we
    capture both single words and two-word phrases (e.g., 'not good', 'highly recommend'),
    which is critical for preserving sentiment context. Sublinear term-frequency scaling
    (sublinear_tf=True) replaces tf with 1 + log(tf), mitigating the bias of repetitively
    used words.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

# Import configuration constants with fallback
try:
    from config import (
        MAX_FEATURES,
        MIN_DF,
        MODEL_PATH,
        NGRAM_RANGE,
        SUBLINEAR_TF,
        TFIDF_VECTORIZER_FILE,
    )
except ImportError:
    try:
        from src.config import (
            MAX_FEATURES,
            MIN_DF,
            MODEL_PATH,
            NGRAM_RANGE,
            SUBLINEAR_TF,
            TFIDF_VECTORIZER_FILE,
        )
    except ImportError:
        MAX_FEATURES = 5000
        NGRAM_RANGE = (1, 2)
        SUBLINEAR_TF = True
        MIN_DF = 1
        MODEL_PATH = Path("models")
        TFIDF_VECTORIZER_FILE = MODEL_PATH / "tfidf_vectorizer.pkl"


def create_tfidf_vectorizer(
    max_features: int = MAX_FEATURES,
    ngram_range: Tuple[int, int] = NGRAM_RANGE,
    sublinear_tf: bool = SUBLINEAR_TF,
    min_df: int = MIN_DF,
) -> TfidfVectorizer:
    """Create and return a configured TfidfVectorizer instance.

    Args:
        max_features: Maximum number of top vocabulary features ordered by term frequency.
        ngram_range: Lower and upper boundary of range of n-values for different n-grams.
                     (1, 1) = unigrams only, (1, 2) = unigrams and bigrams.
        sublinear_tf: Apply sublinear tf scaling, replacing tf with 1 + log(tf).
        min_df: Minimum document frequency threshold for terms.

    Returns:
        Configured, unfitted TfidfVectorizer.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
        dtype=np.float64,
    )
    return vectorizer


def fit_transform_tfidf(
    texts: Union[pd.Series, List[str]],
    vectorizer: Optional[TfidfVectorizer] = None,
    max_features: int = MAX_FEATURES,
    ngram_range: Tuple[int, int] = NGRAM_RANGE,
) -> Tuple[spmatrix, TfidfVectorizer]:
    """Fit TF-IDF vectorizer on training text data and transform to sparse feature matrix.

    IMPORTANT FOR PREVENTING DATA LEAKAGE:
        This function MUST be called ONLY on training data (X_train).
        Never fit the vectorizer on the test dataset or full unsplit dataset.

    Args:
        texts: Training text corpus (pd.Series or list of strings).
        vectorizer: Optional pre-configured vectorizer. If None, a new one is created.
        max_features: Used only if vectorizer is None.
        ngram_range: Used only if vectorizer is None.

    Returns:
        Tuple of (features_matrix, fitted_vectorizer).
    """
    if vectorizer is None:
        vectorizer = create_tfidf_vectorizer(
            max_features=max_features, ngram_range=ngram_range
        )

    if isinstance(texts, pd.Series):
        text_list = texts.fillna("").astype(str).tolist()
    else:
        text_list = [str(t) if t is not None else "" for t in texts]

    features_matrix = vectorizer.fit_transform(text_list)
    return features_matrix, vectorizer


def transform_tfidf(
    texts: Union[pd.Series, List[str], str], vectorizer: TfidfVectorizer
) -> spmatrix:
    """Transform text using an already-fitted TF-IDF vectorizer.

    Used for transforming test data (X_test) and inference inputs without data leakage.

    Args:
        texts: Preprocessed text string(s) or pd.Series.
        vectorizer: Pre-fitted TfidfVectorizer instance.

    Returns:
        Sparse feature matrix representing transformed texts.

    Raises:
        ValueError: If vectorizer is not fitted.
    """
    if not hasattr(vectorizer, "vocabulary_"):
        raise ValueError("The provided TfidfVectorizer is not fitted yet. Fit before transforming.")

    if isinstance(texts, str):
        text_list = [texts]
    elif isinstance(texts, pd.Series):
        text_list = texts.fillna("").astype(str).tolist()
    else:
        text_list = [str(t) if t is not None else "" for t in texts]

    return vectorizer.transform(text_list)


def get_feature_info(
    vectorizer: TfidfVectorizer, X_matrix: Optional[spmatrix] = None
) -> Dict[str, Any]:
    """Extract vocabulary information, feature count, and top TF-IDF weights.

    Args:
        vectorizer: Fitted TfidfVectorizer instance.
        X_matrix: Optional sparse matrix to compute average TF-IDF weights per feature.

    Returns:
        Dictionary containing vocabulary size, sample feature names, and top features.
    """
    if not hasattr(vectorizer, "vocabulary_"):
        raise ValueError("Vectorizer must be fitted to retrieve feature info.")

    feature_names = vectorizer.get_feature_names_out()
    vocab_size = len(feature_names)

    info: Dict[str, Any] = {
        "vocabulary_size": vocab_size,
        "sample_features": feature_names[:20].tolist() if vocab_size >= 20 else feature_names.tolist(),
        "ngram_range": vectorizer.ngram_range,
        "max_features": vectorizer.max_features,
        "sublinear_tf": vectorizer.sublinear_tf,
    }

    if X_matrix is not None and X_matrix.shape[0] > 0:
        mean_weights = np.asarray(X_matrix.mean(axis=0)).flatten()
        top_indices = mean_weights.argsort()[::-1][:15]
        top_features = [
            {"term": feature_names[idx], "mean_tfidf": round(float(mean_weights[idx]), 4)}
            for idx in top_indices
        ]
        info["top_features_by_weight"] = top_features

    return info


def save_vectorizer(
    vectorizer: TfidfVectorizer, filepath: Optional[Union[Path, str]] = None
) -> Path:
    """Serialize and save fitted TF-IDF vectorizer to disk using joblib.

    Args:
        vectorizer: Fitted TfidfVectorizer instance.
        filepath: Destination file path. Defaults to TFIDF_VECTORIZER_FILE.

    Returns:
        Path to the saved vectorizer artifact.
    """
    if filepath is None:
        filepath = TFIDF_VECTORIZER_FILE

    target_path = Path(filepath)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, target_path)
    print(f"[SUCCESS] TF-IDF Vectorizer successfully saved to: {target_path}")
    return target_path


def load_vectorizer(filepath: Optional[Union[Path, str]] = None) -> TfidfVectorizer:
    """Load serialized TF-IDF vectorizer from disk.

    Args:
        filepath: Path to the saved vectorizer file. Defaults to TFIDF_VECTORIZER_FILE.

    Returns:
        Loaded TfidfVectorizer instance.

    Raises:
        FileNotFoundError: If the vectorizer file does not exist.
    """
    if filepath is None:
        filepath = TFIDF_VECTORIZER_FILE

    target_path = Path(filepath)
    if not target_path.exists():
        raise FileNotFoundError(
            f"TF-IDF vectorizer artifact not found at: {target_path.resolve()}.\n"
            f"Please run the feature engineering stage first to generate and save the vectorizer."
        )

    vectorizer: TfidfVectorizer = joblib.load(target_path)
    print(f"[SUCCESS] TF-IDF Vectorizer loaded successfully from: {target_path}")
    return vectorizer
