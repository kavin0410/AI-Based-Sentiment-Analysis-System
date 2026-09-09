"""NLP Preprocessing Module for AI Sentiment Intelligence.

Handles text preprocessing including tokenization, stopword removal,
and lemmatization for sentiment analysis.
"""

from pathlib import Path
from typing import List, Optional
import pandas as pd

# Optional import of configuration constants
try:
    from src.config import NEGATION_WORDS
except ImportError:
    try:
        from config import NEGATION_WORDS
    except ImportError:
        NEGATION_WORDS = {"not", "no", "never", "neither", "nor"}


def tokenize_text(text: str) -> list[str]:
    """Tokenize input text into individual word tokens.

    Args:
        text: Raw input text string.

    Returns:
        List of extracted string tokens.
    """
    # TODO: Implement tokenization using NLTK word_tokenize
    raise NotImplementedError("To be implemented in Stage 2.")


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove stopwords from token list while preserving negation words.

    Negation words to preserve: not, no, never, neither, nor, etc.

    Args:
        tokens: List of word tokens.

    Returns:
        List of filtered tokens without non-negation stopwords.
    """
    # TODO: Remove stopwords but PRESERVE negation words
    # Negation words to preserve: not, no, never, neither, nor, etc.
    raise NotImplementedError("To be implemented in Stage 2.")


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Reduce tokens to their base lemma using morphological analysis.

    Args:
        tokens: List of word tokens.

    Returns:
        List of lemmatized tokens.
    """
    # TODO: Implement lemmatization using NLTK WordNetLemmatizer
    raise NotImplementedError("To be implemented in Stage 2.")


def preprocess_text(text: str) -> str:
    """Run full NLP preprocessing pipeline on single text string.

    Pipeline: tokenize -> remove stopwords (preserving negations) -> lemmatize -> rejoin.

    Args:
        text: Raw input text string.

    Returns:
        Cleaned, normalized, and preprocessed text string.
    """
    # TODO: Full preprocessing pipeline: tokenize -> remove stopwords -> lemmatize -> rejoin
    raise NotImplementedError("To be implemented in Stage 2.")


def preprocess_dataset(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """Apply full preprocessing pipeline across a DataFrame text column.

    Args:
        df: Input DataFrame containing text data.
        text_column: Name of the column containing raw text. Defaults to 'text'.

    Returns:
        DataFrame with an added or updated preprocessed text column.
    """
    # TODO: Apply preprocess_text to entire DataFrame column
    raise NotImplementedError("To be implemented in Stage 2.")
