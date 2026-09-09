"""Data Loading Module for AI Sentiment Intelligence.

Provides reusable functions for loading, validating, and inspecting
sentiment analysis datasets.
"""

import pandas as pd
from pathlib import Path
import sys

# Import project configuration constants
try:
    from config import TEXT_COLUMN, SENTIMENT_COLUMN, SUPPORTED_LABELS
except ImportError:
    try:
        from src.config import TEXT_COLUMN, SENTIMENT_COLUMN, SUPPORTED_LABELS
    except ImportError:
        TEXT_COLUMN = 'text'
        SENTIMENT_COLUMN = 'sentiment'
        SUPPORTED_LABELS = ['Positive', 'Negative', 'Neutral']


def load_dataset(filepath: Path) -> pd.DataFrame:
    """Load a CSV dataset.
    
    Args:
        filepath: Path to the CSV file.
        
    Returns:
        Loaded DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded successfully from {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        print("Please ensure the data file exists at the correct location.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)


def validate_columns(df: pd.DataFrame, required_columns: list[str] = None) -> bool:
    """Validate that required columns exist in the DataFrame.
    
    Args:
        df: Input DataFrame.
        required_columns: List of required column names.
        
    Returns:
        True if valid, False otherwise.
    """
    if required_columns is None:
        required_columns = [TEXT_COLUMN, SENTIMENT_COLUMN]
        
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return False
        
    return True


def validate_labels(df: pd.DataFrame, sentiment_column: str = None, valid_labels: list[str] = None) -> pd.DataFrame:
    """Filter the DataFrame to only include valid sentiment labels.
    
    Args:
        df: Input DataFrame.
        sentiment_column: Name of the sentiment column.
        valid_labels: List of valid label strings.
        
    Returns:
        Filtered DataFrame.
    """
    if sentiment_column is None:
        sentiment_column = SENTIMENT_COLUMN
    if valid_labels is None:
        valid_labels = SUPPORTED_LABELS
        
    if sentiment_column not in df.columns:
        print(f"Error: Sentiment column '{sentiment_column}' not found.")
        return df
        
    invalid_mask = ~df[sentiment_column].isin(valid_labels)
    invalid_count = invalid_mask.sum()
    
    if invalid_count > 0:
        invalid_labels = df.loc[invalid_mask, sentiment_column].unique()
        print(f"Warning: Found {invalid_count} rows with unexpected labels: {invalid_labels}")
        print(f"Filtering dataset to keep only valid labels: {valid_labels}")
        df = df[~invalid_mask].copy()
        
    return df


def report_dataset_info(df: pd.DataFrame) -> dict:
    """Report detailed information about the dataset.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Dictionary containing dataset statistics.
    """
    print("-" * 40)
    print("DATASET INFORMATION")
    print("-" * 40)
    
    shape = df.shape
    print(f"Shape: {shape[0]} rows, {shape[1]} columns")
    
    print("\nColumns and Data Types:")
    for col, dtype in zip(df.columns, df.dtypes):
        print(f"  - {col}: {dtype}")
        
    missing = df.isnull().sum().to_dict()
    print("\nMissing Values:")
    for col, count in missing.items():
        print(f"  - {col}: {count}")
        
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate Rows: {duplicates}")
    
    sentiment_dist = {}
    if SENTIMENT_COLUMN in df.columns:
        print(f"\nSentiment Distribution ({SENTIMENT_COLUMN}):")
        sentiment_dist = df[SENTIMENT_COLUMN].value_counts().to_dict()
        for label, count in sentiment_dist.items():
            print(f"  - {label}: {count}")
    print("-" * 40)
    
    return {
        'shape': shape,
        'columns': list(df.columns),
        'dtypes': {str(k): str(v) for k, v in df.dtypes.items()},
        'missing_values': missing,
        'duplicate_rows': duplicates,
        'sentiment_distribution': sentiment_dist
    }


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Get a quick summary of the dataset.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Dictionary with basic dataset statistics.
    """
    sentiment_dist = {}
    if SENTIMENT_COLUMN in df.columns:
        sentiment_dist = df[SENTIMENT_COLUMN].value_counts().to_dict()
        
    return {
        'shape': df.shape,
        'missing_count': int(df.isnull().sum().sum()),
        'duplicate_count': int(df.duplicated().sum()),
        'label_distribution': sentiment_dist
    }
