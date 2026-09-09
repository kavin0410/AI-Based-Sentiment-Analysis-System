"""Model Evaluation Module for AI Sentiment Intelligence.

Handles evaluation of trained models including accuracy, precision,
recall, F1-score, confusion matrix, and error analysis.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Optional import of configuration constants
try:
    from src.config import METRICS, REPORTS_DIR
except ImportError:
    try:
        from config import METRICS, REPORTS_DIR
    except ImportError:
        METRICS = ["accuracy", "precision", "recall", "f1_score"]
        REPORTS_DIR = Path("reports")


def evaluate_model(
    model: Any, X_test: Any, y_test: Any, model_name: str = "Model"
) -> dict[str, Any]:
    """Evaluate a trained model on test data across standard classification metrics.

    Args:
        model: Trained classifier with predict method.
        X_test: Test feature matrix.
        y_test: True test labels.
        model_name: Name identifier for the model. Defaults to 'Model'.

    Returns:
        Dictionary containing metric values (accuracy, precision, recall, f1_score).
    """
    # TODO: Calculate accuracy, precision, recall, F1-score
    # Return dict with all metrics
    raise NotImplementedError("To be implemented in Stage 2.")


def compare_models(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Compare evaluation results from multiple models in a consolidated tabular format.

    Args:
        results: Dictionary mapping model names to evaluation metric dictionaries.

    Returns:
        Pandas DataFrame comparing models across metrics.
    """
    # TODO: Compare multiple models, return DataFrame with metrics
    raise NotImplementedError("To be implemented in Stage 2.")


def plot_confusion_matrix(
    model: Any,
    X_test: Any,
    y_test: Any,
    labels: list[str],
    model_name: str,
) -> None:
    """Plot and save confusion matrix heatmap using seaborn.

    Args:
        model: Trained classifier.
        X_test: Test feature matrix.
        y_test: True test labels.
        labels: Class label names for axes display.
        model_name: Name identifier of the model for title and filename.
    """
    # TODO: Plot confusion matrix using seaborn heatmap
    raise NotImplementedError("To be implemented in Stage 2.")


def error_analysis(
    model: Any, X_test: Any, y_test: Any, texts: pd.Series
) -> pd.DataFrame:
    """Identify misclassified samples and perform qualitative error analysis.

    Args:
        model: Trained classifier.
        X_test: Test feature matrix.
        y_test: True test labels.
        texts: Original text strings corresponding to test set rows.

    Returns:
        DataFrame containing misclassified texts, true labels, and predicted labels.
    """
    # TODO: Identify misclassified samples and return DataFrame
    raise NotImplementedError("To be implemented in Stage 2.")


def select_best_model(
    results: dict[str, dict[str, Any]], metric: str = "f1_score"
) -> str:
    """Select the best performing model name based on a specified evaluation metric.

    Args:
        results: Dictionary mapping model names to metric dictionaries.
        metric: Metric to prioritize for selection (e.g. 'f1_score', 'accuracy').

    Returns:
        Name of the best-performing model.
    """
    # TODO: Select best model based on specified metric
    raise NotImplementedError("To be implemented in Stage 2.")
