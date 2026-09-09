"""Data Cleaning Module for AI Sentiment Intelligence.

Provides reusable functions for cleaning and normalizing
text data for sentiment analysis.
"""

import pandas as pd
import re

# Import project configuration constants
try:
    from config import TEXT_COLUMN, SENTIMENT_COLUMN
except ImportError:
    try:
        from src.config import TEXT_COLUMN, SENTIMENT_COLUMN
    except ImportError:
        TEXT_COLUMN = 'text'
        SENTIMENT_COLUMN = 'sentiment'


def remove_missing_values(df: pd.DataFrame, columns: list[str] = None) -> pd.DataFrame:
    """Remove rows with missing values in specified columns.
    
    Args:
        df: Input DataFrame.
        columns: List of columns to check for missing values.
        
    Returns:
        Cleaned DataFrame.
    """
    if columns is None:
        columns = [TEXT_COLUMN, SENTIMENT_COLUMN]
        
    # Only use columns that exist in the dataframe
    existing_columns = [col for col in columns if col in df.columns]
    
    initial_len = len(df)
    cleaned_df = df.dropna(subset=existing_columns).copy()
    removed_count = initial_len - len(cleaned_df)
    
    print(f"Removed {removed_count} rows with missing values in {existing_columns}.")
    return cleaned_df


def remove_duplicates(df: pd.DataFrame, subset: list[str] = None) -> pd.DataFrame:
    """Remove duplicate rows.
    
    Args:
        df: Input DataFrame.
        subset: Columns to consider for identifying duplicates.
        
    Returns:
        Cleaned DataFrame.
    """
    if subset is None:
        subset = [TEXT_COLUMN]
        
    # Only use columns that exist in the dataframe
    existing_subset = [col for col in subset if col in df.columns]
    
    if not existing_subset:
        return df
        
    initial_len = len(df)
    cleaned_df = df.drop_duplicates(subset=existing_subset).copy()
    removed_count = initial_len - len(cleaned_df)
    
    print(f"Removed {removed_count} duplicate rows based on {existing_subset}.")
    return cleaned_df


def remove_urls(text: str) -> str:
    """Remove URLs from text.
    
    Args:
        text: Input text string.
        
    Returns:
        Text string without URLs.
    """
    if not isinstance(text, str):
        return str(text)
    # Remove http, https, www patterns
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub('', text)


def remove_html_tags(text: str) -> str:
    """Remove HTML tags from text.
    
    Args:
        text: Input text string.
        
    Returns:
        Text string without HTML tags.
    """
    if not isinstance(text, str):
        return str(text)
    pattern = re.compile(r'<.*?>')
    return pattern.sub('', text)


def remove_extra_whitespace(text: str) -> str:
    """Collapse multiple spaces and strip leading/trailing whitespace.
    
    Args:
        text: Input text string.
        
    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str):
        return str(text)
    # Replace multiple whitespaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_text(text: str) -> str:
    """Convert text to lowercase.
    
    Note: PRESERVE negation words (not, no, never, neither, nor, nobody, 
    nothing, nowhere, hardly, scarcely, barely). This function simply 
    lowercases the text, preserving all words.
    
    Args:
        text: Input text string.
        
    Returns:
        Lowercased text string.
    """
    if not isinstance(text, str):
        return str(text)
    return text.lower()


def clean_text(text: str) -> str:
    """Apply all text cleaning steps in sequence.
    
    Steps:
    1. normalize_text
    2. remove_urls
    3. remove_html_tags
    4. remove_extra_whitespace
    
    Args:
        text: Input text string.
        
    Returns:
        Fully cleaned text string.
    """
    if not isinstance(text, str):
        text = str(text)
        
    text = normalize_text(text)
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_extra_whitespace(text)
    
    return text


def clean_dataset(df: pd.DataFrame, text_column: str = None, sentiment_column: str = None) -> pd.DataFrame:
    """Apply the full cleaning pipeline to a DataFrame.
    
    Steps:
    1. Remove missing values
    2. Remove duplicates
    3. Apply clean_text to the text column
    4. Remove any rows that became empty after cleaning
    
    Args:
        df: Input DataFrame.
        text_column: Name of the text column.
        sentiment_column: Name of the sentiment column.
        
    Returns:
        Cleaned DataFrame.
    """
    if text_column is None:
        text_column = TEXT_COLUMN
    if sentiment_column is None:
        sentiment_column = SENTIMENT_COLUMN
        
    print("-" * 40)
    print("DATA CLEANING PIPELINE")
    print("-" * 40)
    
    initial_len = len(df)
    
    # 1. Remove missing values
    df_clean = remove_missing_values(df, columns=[text_column, sentiment_column])
    
    # 2. Remove duplicates
    df_clean = remove_duplicates(df_clean, subset=[text_column])
    
    # 3. Apply clean_text
    if text_column in df_clean.columns:
        print(f"Applying text cleaning steps to '{text_column}' column...")
        df_clean[text_column] = df_clean[text_column].apply(clean_text)
        
        # 4. Remove empty text rows
        empty_mask = (df_clean[text_column] == '') | (df_clean[text_column].isna())
        empty_count = empty_mask.sum()
        if empty_count > 0:
            df_clean = df_clean[~empty_mask].copy()
            print(f"Removed {empty_count} rows that became empty after cleaning.")
            
    final_len = len(df_clean)
    print("-" * 40)
    print(f"Cleaning complete. Original rows: {initial_len}, Final rows: {final_len}")
    print(f"Total rows removed: {initial_len - final_len}")
    print("-" * 40)
    
    return df_clean
