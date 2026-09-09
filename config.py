"""AI Sentiment Intelligence - Central Configuration.

This module defines directory paths, model hyperparameters, dataset
column names, and label definitions used across the entire project.
It also provides utility functions to ensure required directories exist.
"""

from pathlib import Path
from typing import List

# ==============================================================================
# Base Directory
# ==============================================================================
# Resolved relative to this config file's location
BASE_DIR: Path = Path(__file__).resolve().parent

# ==============================================================================
# Project Paths
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

# ==============================================================================
# Dataset Specifications
# ==============================================================================
# Standard column names expected across datasets
TEXT_COLUMN: str = "text"
SENTIMENT_COLUMN: str = "sentiment"

# Target sentiment labels supported by the system
SUPPORTED_LABELS: List[str] = ["Positive", "Negative", "Neutral"]

# ==============================================================================
# Machine Learning Hyperparameters & Settings
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
    """
    directories = [
        RAW_DATA_PATH,
        PROCESSED_DATA_PATH,
        MODEL_PATH,
        RESULTS_PATH,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print("[INFO] All project directories verified/created successfully.")
    print(f"[INFO] Base Directory: {BASE_DIR}")
