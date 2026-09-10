"""Data Cleaning Module for AI Sentiment Intelligence.

Provides reusable functions for text-level and dataset-level cleaning,
safely handling missing values, duplicates, URLs, HTML tags, emails,
and unwanted special characters while strictly preserving sentiment-bearing words.
"""

import html
from pathlib import Path
import re
from typing import Any, List, Optional

import pandas as pd

# Import project configuration constants
try:
    from config import SENTIMENT_COLUMN, TEXT_COLUMN
except ImportError:
    try:
        from src.config import SENTIMENT_COLUMN, TEXT_COLUMN
    except ImportError:
        TEXT_COLUMN = "text"
        SENTIMENT_COLUMN = "sentiment"


# ==============================================================================
# Text-Level Cleaning Functions
# ==============================================================================

def safe_str(text: Any) -> str:
    """Safely convert any input (None, float, NaN, int) to a string.

    Args:
        text: Any arbitrary input value.

    Returns:
        String representation; empty string if input was None or NaN.
    """
    if text is None or pd.isna(text):
        return ""
    return str(text)


def to_lowercase(text: str) -> str:
    """Convert text to lowercase for vocabulary normalization.

    Viva Note:
        Lowercasing ensures that words like 'Awesome', 'AWESOME', and 'awesome'
        map to the exact same feature index, reducing vocabulary sparsity.
        It preserves all words, including crucial negations (e.g., 'NOT' -> 'not').

    Args:
        text: Input string.

    Returns:
        Lowercased string.
    """
    return safe_str(text).lower()


# Backward compatibility alias
normalize_text = to_lowercase


def remove_urls(text: str) -> str:
    """Strip website URLs and domain links from text.

    Viva Note:
        URLs (e.g., https://example.com) introduce high-frequency noise tokens
        with zero sentiment polarity. Removing them prevents vocabulary bloat.

    Args:
        text: Input text string.

    Returns:
        Text with URLs removed.
    """
    if not isinstance(text, str):
        text = safe_str(text)
    # Matches http://, https://, www., ftp://, and raw domain patterns
    pattern = re.compile(r"(?:https?://|www\.|ftp://)\S+", re.IGNORECASE)
    return pattern.sub("", text)


def remove_html_tags(text: str) -> str:
    """Remove HTML markup tags and decode common HTML entities.

    Viva Note:
        Web-scraped feedback or reviews often contain artifacts like '<br>',
        '<div>', '&amp;', and '&quot;'. Decoding and stripping them restores
        the underlying human text.

    Args:
        text: Input text string.

    Returns:
        Text with HTML tags and entities cleaned.
    """
    if not isinstance(text, str):
        text = safe_str(text)
    # 1. Decode HTML entities (e.g. &amp; -> &)
    text = html.unescape(text)
    # 2. Strip HTML tags like <p>, </p>, <br />
    tag_pattern = re.compile(r"<[^>]+>")
    return tag_pattern.sub(" ", text)


def remove_emails(text: str) -> str:
    """Remove email addresses from text to preserve privacy and reduce noise.

    Args:
        text: Input text string.

    Returns:
        Text without email addresses.
    """
    if not isinstance(text, str):
        text = safe_str(text)
    pattern = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
    return pattern.sub("", text)


def remove_special_characters(text: str, preserve_apostrophes: bool = False) -> str:
    """Remove unwanted punctuation and special characters.

    Args:
        text: Input text string.
        preserve_apostrophes: If True, retains apostrophes (') to allow
            subsequent contraction expansion (e.g., don't). If False,
            removes all non-alphabetic characters.

    Returns:
        Cleaned text containing letters and whitespace.
    """
    if not isinstance(text, str):
        text = safe_str(text)

    if preserve_apostrophes:
        # Keep letters, standard/curly apostrophes, and spaces
        pattern = re.compile(r"[^a-zA-Z\s'’‘]")
    else:
        # Keep letters and spaces only
        pattern = re.compile(r"[^a-zA-Z\s]")

    return pattern.sub(" ", text)


def remove_extra_whitespace(text: str) -> str:
    """Collapse consecutive whitespaces, tabs, and newlines into a single space.

    Args:
        text: Input text string.

    Returns:
        Trimmed text string with single-space word separation.
    """
    if not isinstance(text, str):
        text = safe_str(text)
    return re.sub(r"\s+", " ", text).strip()


def clean_text_basic(
    text: str,
    remove_punct: bool = False,
    preserve_apostrophes: bool = True,
) -> str:
    """Apply foundational text cleaning steps in sequence.

    Sequential Operations:
        1. Safe string conversion
        2. Lowercase normalization
        3. Remove URLs
        4. Remove HTML tags & entities
        5. Remove Email addresses
        6. Remove special characters (optional)
        7. Normalize extra whitespace

    Args:
        text: Input text string.
        remove_punct: Whether to strip punctuation and special characters.
        preserve_apostrophes: Whether to keep apostrophes for contraction expansion.

    Returns:
        Cleaned text string.
    """
    text = safe_str(text)
    text = to_lowercase(text)
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_emails(text)
    if remove_punct:
        text = remove_special_characters(text, preserve_apostrophes=preserve_apostrophes)
    text = remove_extra_whitespace(text)
    return text


def clean_text(text: str) -> str:
    """Foundational text cleaning pipeline (Stage 1 compatible).

    Steps:
        1. Lowercase normalization
        2. Remove URLs
        3. Remove HTML tags
        4. Normalize whitespace

    Args:
        text: Input text string.

    Returns:
        Cleaned text string.
    """
    return clean_text_basic(text, remove_punct=False)


# ==============================================================================
# Dataset-Level Cleaning Functions
# ==============================================================================

def remove_missing_values(
    df: pd.DataFrame, columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """Remove DataFrame rows that have NaN / null values in key columns.

    Args:
        df: Input DataFrame.
        columns: List of columns to inspect. Defaults to ['text', 'sentiment'].

    Returns:
        Cleaned DataFrame with null rows dropped.
    """
    if columns is None:
        columns = [TEXT_COLUMN, SENTIMENT_COLUMN]

    existing_columns = [col for col in columns if col in df.columns]
    initial_len = len(df)
    cleaned_df = df.dropna(subset=existing_columns).copy()
    removed_count = initial_len - len(cleaned_df)

    if removed_count > 0:
        print(f"[CLEANING] Removed {removed_count} row(s) with missing values in {existing_columns}.")
    return cleaned_df


def remove_duplicates(
    df: pd.DataFrame, subset: Optional[List[str]] = None
) -> pd.DataFrame:
    """Remove duplicate records from the DataFrame.

    Viva Note:
        Duplicate rows bias machine learning classifiers towards over-represented
        samples and can cause data leakage between train and test splits.

    Args:
        df: Input DataFrame.
        subset: Columns to consider for detecting duplicates. Defaults to ['text'].

    Returns:
        Deduplicated DataFrame.
    """
    if subset is None:
        subset = [TEXT_COLUMN]

    existing_subset = [col for col in subset if col in df.columns]
    if not existing_subset:
        return df

    initial_len = len(df)
    cleaned_df = df.drop_duplicates(subset=existing_subset).copy()
    removed_count = initial_len - len(cleaned_df)

    if removed_count > 0:
        print(f"[CLEANING] Removed {removed_count} duplicate row(s) based on {existing_subset}.")
    return cleaned_df


def clean_dataset(
    df: pd.DataFrame,
    text_column: Optional[str] = None,
    sentiment_column: Optional[str] = None,
) -> pd.DataFrame:
    """Execute the dataset-level cleaning pipeline.

    Pipeline Steps:
        1. Remove rows with null/missing text or sentiment
        2. Remove duplicate text rows
        3. Apply clean_text_basic to text column
        4. Remove any rows that became empty after cleaning

    Args:
        df: Raw input DataFrame.
        text_column: Column containing text.
        sentiment_column: Column containing sentiment labels.

    Returns:
        Cleaned DataFrame.
    """
    if text_column is None:
        text_column = TEXT_COLUMN
    if sentiment_column is None:
        sentiment_column = SENTIMENT_COLUMN

    print("-" * 50)
    print("DATASET CLEANING PIPELINE")
    print("-" * 50)

    initial_len = len(df)

    # 1. Remove missing values
    df_clean = remove_missing_values(df, columns=[text_column, sentiment_column])

    # 2. Remove duplicates
    df_clean = remove_duplicates(df_clean, subset=[text_column])

    # 3. Apply basic text cleaning
    if text_column in df_clean.columns:
        print(f"[CLEANING] Applying basic text cleaning to column '{text_column}'...")
        df_clean[text_column] = df_clean[text_column].apply(clean_text_basic)

        # 4. Remove rows that became empty
        empty_mask = (df_clean[text_column] == "") | (df_clean[text_column].isna())
        empty_count = int(empty_mask.sum())
        if empty_count > 0:
            df_clean = df_clean[~empty_mask].copy()
            print(f"[CLEANING] Removed {empty_count} row(s) that became empty after text cleaning.")

    final_len = len(df_clean)
    print(f"[SUMMARY] Original rows: {initial_len} | Final rows: {final_len} | Removed: {initial_len - final_len}")
    print("-" * 50)

    return df_clean


if __name__ == "__main__":
    sample = "Check this out: https://example.com/test! Contact: user@mail.com. <b>AWESOME</b> app...   loved it!"
    print("Sample Before:")
    print(sample)
    print("\nSample After clean_text_basic:")
    print(clean_text_basic(sample))
