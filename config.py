"""AI Sentiment Intelligence - Central Configuration.

This module defines directory paths, model hyperparameters, dataset
column names, label definitions, and file locations used across the entire project.
It also provides utility functions to ensure required directories exist.
"""

from pathlib import Path
from typing import List, Set

# ==============================================================================
# Base Directory
# ==============================================================================
# Resolved relative to this config file's location
BASE_DIR: Path = Path(__file__).resolve().parent

# ==============================================================================
# Project Directory Paths
# ==============================================================================
# Data directories
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_PATH: Path = DATA_DIR / "raw"
PROCESSED_DATA_PATH: Path = DATA_DIR / "processed"

# Model and artifact directories
MODEL_PATH: Path = BASE_DIR / "models"
VECTORIZER_PATH: Path = BASE_DIR / "models"  # Directory for TF-IDF vectorizer pickle

# Evaluation & visualization output directory
RESULTS_PATH: Path = BASE_DIR / "results"

# Documentation and screenshots
DOCS_PATH: Path = BASE_DIR / "docs"
SCREENSHOTS_PATH: Path = BASE_DIR / "screenshots"

# ==============================================================================
# Dataset Specifications & File Locations
# ==============================================================================
# Standard column names expected across datasets
TEXT_COLUMN: str = "text"
SENTIMENT_COLUMN: str = "sentiment"
CLEAN_TEXT_COLUMN: str = "clean_text"

# Target sentiment labels supported by the system
SUPPORTED_LABELS: List[str] = ["Positive", "Negative", "Neutral"]

# Default dataset files
DEFAULT_RAW_FILE: Path = RAW_DATA_PATH / "demo_dataset.csv"
SAMPLE_DATA_FILE: Path = RAW_DATA_PATH / "sample_data.csv"
CLEANED_DATA_FILE: Path = PROCESSED_DATA_PATH / "cleaned_dataset.csv"

# ==============================================================================
# NLP Preprocessing Settings
# ==============================================================================
# Sentiment-critical negation and modifier words that must NOT be stripped during stopword removal
PRESERVED_WORDS: Set[str] = {
    "not", "no", "never", "neither", "nor", "nothing",
    "nowhere", "hardly", "scarcely", "barely", "without", "against"
}

# ==============================================================================
# Results & Report File Paths
# ==============================================================================
DATA_QUALITY_REPORT_JSON: Path = RESULTS_PATH / "data_quality_report.json"
DATA_QUALITY_REPORT_TXT: Path = RESULTS_PATH / "data_quality_report.txt"
SENTIMENT_DIST_PLOT: Path = RESULTS_PATH / "sentiment_distribution.png"
TEXT_LENGTH_PLOT: Path = RESULTS_PATH / "text_length_distribution.png"
WORD_FREQ_PLOT: Path = RESULTS_PATH / "word_frequency_analysis.png"
WORD_FREQ_JSON: Path = RESULTS_PATH / "word_frequencies.json"

# ==============================================================================
# Machine Learning Hyperparameters & Settings (Stage 3+)
# ==============================================================================
# Seed for reproducibility across data splits and model training
RANDOM_STATE: int = 42

# Train/Test split proportion
TEST_SIZE: float = 0.2

# Maximum number of features extracted by TF-IDF Vectorizer
MAX_FEATURES: int = 5000


# ==============================================================================
# Directory Management Helper
# ==============================================================================
def ensure_directories() -> None:
    """Create all required project directories if they do not already exist.

    Ensures the existence of:
        - data/raw/
        - data/processed/
        - models/
        - results/
        - docs/
        - screenshots/
    """
    directories = [
        RAW_DATA_PATH,
        PROCESSED_DATA_PATH,
        MODEL_PATH,
        RESULTS_PATH,
        DOCS_PATH,
        SCREENSHOTS_PATH,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print("[INFO] All project directories verified/created successfully.")
    print(f"[INFO] Base Directory: {BASE_DIR}")
