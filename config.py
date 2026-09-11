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
# Model Artifact File Paths (Stage 3)
# ==============================================================================
LOGISTIC_REGRESSION_FILE: Path = MODEL_PATH / "logistic_regression.pkl"
NAIVE_BAYES_FILE: Path = MODEL_PATH / "naive_bayes.pkl"
LINEAR_SVM_FILE: Path = MODEL_PATH / "linear_svm.pkl"
TFIDF_VECTORIZER_FILE: Path = MODEL_PATH / "tfidf_vectorizer.pkl"
MODEL_METADATA_FILE: Path = MODEL_PATH / "model_metadata.json"

# ==============================================================================
# Machine Learning Hyperparameters & Settings (Stage 3+)
# ==============================================================================
# Seed for reproducibility across data splits and model training
RANDOM_STATE: int = 42

# Train/Test split proportion
TEST_SIZE: float = 0.2

# TF-IDF Vectorizer Hyperparameters
MAX_FEATURES: int = 5000
NGRAM_RANGE: tuple = (1, 2)
SUBLINEAR_TF: bool = True
MIN_DF: int = 1

# ==============================================================================
# Stage 4: Evaluation Artifact File Paths
# ==============================================================================
# Classification reports
CLASSIFICATION_REPORTS_DIR: Path = RESULTS_PATH / "classification_reports"

# Model comparison
MODEL_COMPARISON_CSV: Path = RESULTS_PATH / "model_comparison.csv"
BEST_MODEL_JSON: Path = RESULTS_PATH / "best_model.json"

# Confusion matrices
CONFUSION_MATRICES_DIR: Path = RESULTS_PATH / "confusion_matrices"
CONFUSION_MATRICES_JSON: Path = RESULTS_PATH / "confusion_matrices" / "confusion_matrices.json"

# Performance visualizations
MODEL_PERFORMANCE_PLOT: Path = RESULTS_PATH / "model_performance_comparison.png"
BEST_MODEL_CM_PLOT: Path = RESULTS_PATH / "best_model_confusion_matrix.png"

# Per-class and error analysis
PER_CLASS_METRICS_CSV: Path = RESULTS_PATH / "per_class_metrics.csv"
ERROR_ANALYSIS_CSV: Path = RESULTS_PATH / "error_analysis.csv"

# Evaluation results
EVALUATION_RESULTS_JSON: Path = RESULTS_PATH / "evaluation_results.json"
EVALUATION_RESULTS_CSV: Path = RESULTS_PATH / "evaluation_results.csv"

# Confusion matrix label ordering (consistent across all evaluations)
CM_LABEL_ORDER: List[str] = ["Negative", "Neutral", "Positive"]

# ==============================================================================
# Stage 5: Prediction Engine Configuration
# ==============================================================================
# Maximum input text length in characters
MAX_INPUT_LENGTH: int = 5000

# Session prediction history limit
PREDICTION_HISTORY_LIMIT: int = 50


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
        CLASSIFICATION_REPORTS_DIR,
        CONFUSION_MATRICES_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print("[INFO] All project directories verified/created successfully.")
    print(f"[INFO] Base Directory: {BASE_DIR}")
