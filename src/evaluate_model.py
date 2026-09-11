"""Model Evaluation Module for AI Sentiment Intelligence (Stage 4).

Provides reusable functions for computing classification metrics,
generating confusion matrices, creating comparison tables, performing
error analysis, and automatically selecting the best-performing model.

All evaluation results are derived from actual model predictions on
the held-out test dataset. No values are fabricated or hardcoded.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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

# Import configuration constants with fallback
try:
    from config import (
        BEST_MODEL_CM_PLOT,
        BEST_MODEL_JSON,
        CLASSIFICATION_REPORTS_DIR,
        CM_LABEL_ORDER,
        CONFUSION_MATRICES_DIR,
        CONFUSION_MATRICES_JSON,
        ERROR_ANALYSIS_CSV,
        EVALUATION_RESULTS_CSV,
        EVALUATION_RESULTS_JSON,
        MODEL_COMPARISON_CSV,
        MODEL_PERFORMANCE_PLOT,
        PER_CLASS_METRICS_CSV,
        RESULTS_PATH,
        SUPPORTED_LABELS,
    )
except ImportError:
    try:
        from src.config import (
            BEST_MODEL_CM_PLOT,
            BEST_MODEL_JSON,
            CLASSIFICATION_REPORTS_DIR,
            CM_LABEL_ORDER,
            CONFUSION_MATRICES_DIR,
            CONFUSION_MATRICES_JSON,
            ERROR_ANALYSIS_CSV,
            EVALUATION_RESULTS_CSV,
            EVALUATION_RESULTS_JSON,
            MODEL_COMPARISON_CSV,
            MODEL_PERFORMANCE_PLOT,
            PER_CLASS_METRICS_CSV,
            RESULTS_PATH,
            SUPPORTED_LABELS,
        )
    except ImportError:
        RESULTS_PATH = Path("results")
        CLASSIFICATION_REPORTS_DIR = RESULTS_PATH / "classification_reports"
        CONFUSION_MATRICES_DIR = RESULTS_PATH / "confusion_matrices"
        MODEL_COMPARISON_CSV = RESULTS_PATH / "model_comparison.csv"
        BEST_MODEL_JSON = RESULTS_PATH / "best_model.json"
        CONFUSION_MATRICES_JSON = CONFUSION_MATRICES_DIR / "confusion_matrices.json"
        MODEL_PERFORMANCE_PLOT = RESULTS_PATH / "model_performance_comparison.png"
        BEST_MODEL_CM_PLOT = RESULTS_PATH / "best_model_confusion_matrix.png"
        PER_CLASS_METRICS_CSV = RESULTS_PATH / "per_class_metrics.csv"
        ERROR_ANALYSIS_CSV = RESULTS_PATH / "error_analysis.csv"
        EVALUATION_RESULTS_JSON = RESULTS_PATH / "evaluation_results.json"
        EVALUATION_RESULTS_CSV = RESULTS_PATH / "evaluation_results.csv"
        SUPPORTED_LABELS = ["Positive", "Negative", "Neutral"]
        CM_LABEL_ORDER = ["Negative", "Neutral", "Positive"]


# ==============================================================================
# Metric Calculation Functions
# ==============================================================================

def calculate_model_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "weighted",
) -> Dict[str, float]:
    """Calculate accuracy, precision, recall, and F1-score for a single model.

    Args:
        y_true: Ground-truth labels from the test set.
        y_pred: Predicted labels from the trained model.
        average: Averaging strategy for multiclass ('weighted', 'macro', 'micro').

    Returns:
        Dictionary with keys: accuracy, precision, recall, f1_score.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }


def calculate_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Calculate precision, recall, and F1-score for each individual sentiment class.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Ordered list of class labels.

    Returns:
        Nested dictionary: {class_label: {precision, recall, f1_score, support}}.
    """
    if labels is None:
        labels = CM_LABEL_ORDER

    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    per_class: Dict[str, Dict[str, float]] = {}
    for label in labels:
        if label in report:
            per_class[label] = {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1_score": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
    return per_class


def generate_classification_report_text(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
) -> str:
    """Generate a formatted classification report string.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Ordered class labels.

    Returns:
        Formatted classification report string.
    """
    if labels is None:
        labels = CM_LABEL_ORDER
    return classification_report(y_true, y_pred, labels=labels, zero_division=0)


def generate_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
) -> np.ndarray:
    """Compute confusion matrix with consistent label ordering.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Ordered class labels (rows=actual, cols=predicted).

    Returns:
        Confusion matrix as a 2D numpy array.
    """
    if labels is None:
        labels = CM_LABEL_ORDER
    return confusion_matrix(y_true, y_pred, labels=labels)


# ==============================================================================
# Multi-Model Evaluation
# ==============================================================================

def evaluate_all_models(
    models_predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    labels: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate all models and collect comprehensive metrics.

    Args:
        models_predictions: {model_name: y_pred_array} dictionary.
        y_true: Ground-truth test labels.
        labels: Class label ordering.

    Returns:
        Nested dict: {model_name: {metrics, per_class, confusion_matrix, report}}.
    """
    if labels is None:
        labels = CM_LABEL_ORDER

    results: Dict[str, Dict[str, Any]] = {}
    for name, y_pred in models_predictions.items():
        metrics = calculate_model_metrics(y_true, y_pred)
        per_class = calculate_per_class_metrics(y_true, y_pred, labels)
        cm = generate_confusion_matrix(y_true, y_pred, labels)
        report = generate_classification_report_text(y_true, y_pred, labels)

        results[name] = {
            "metrics": metrics,
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }
    return results


# ==============================================================================
# Best Model Selection
# ==============================================================================

def select_best_model(
    evaluation_results: Dict[str, Dict[str, Any]],
    primary_metric: str = "f1_score",
    secondary_metric: str = "accuracy",
) -> Dict[str, Any]:
    """Automatically select the best-performing model based on metrics.

    Primary selection criterion: weighted F1-score.
    Tie-breaker: accuracy.

    Args:
        evaluation_results: Complete evaluation results from evaluate_all_models().
        primary_metric: Primary ranking metric key.
        secondary_metric: Tie-breaking metric key.

    Returns:
        Dictionary describing the best model and its metrics.
    """
    ranked = []
    for name, result in evaluation_results.items():
        m = result["metrics"]
        ranked.append({
            "model_name": name,
            primary_metric: m[primary_metric],
            secondary_metric: m[secondary_metric],
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1_score": m["f1_score"],
        })

    ranked.sort(key=lambda x: (x[primary_metric], x[secondary_metric]), reverse=True)
    best = ranked[0]
    best["selection_metric"] = primary_metric
    best["rank"] = 1
    return best


# ==============================================================================
# Model Comparison Table
# ==============================================================================

def create_comparison_table(
    evaluation_results: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """Build a DataFrame comparing all models side-by-side.

    Args:
        evaluation_results: Complete evaluation results.

    Returns:
        DataFrame with columns: Model, Accuracy, Precision, Recall, F1 Score.
    """
    rows = []
    for name, result in evaluation_results.items():
        m = result["metrics"]
        rows.append({
            "Model": name,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1 Score": m["f1_score"],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("F1 Score", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "Rank"
    return df


# ==============================================================================
# Error Analysis
# ==============================================================================

def perform_error_analysis(
    y_true: pd.Series,
    y_pred: np.ndarray,
    texts: pd.Series,
    clean_texts: pd.Series,
    model_name: str,
) -> pd.DataFrame:
    """Identify misclassified test samples for a single model.

    Args:
        y_true: Ground-truth labels (pandas Series preserving index).
        y_pred: Predicted labels array.
        texts: Original raw text (aligned with y_true index).
        clean_texts: Preprocessed text (aligned with y_true index).
        model_name: Name of the model being analyzed.

    Returns:
        DataFrame of misclassified records with columns:
        text, clean_text, actual_sentiment, predicted_sentiment, model.
    """
    y_true_arr = y_true.values
    y_pred_arr = np.asarray(y_pred)
    mask = y_true_arr != y_pred_arr

    errors_df = pd.DataFrame({
        "text": texts.values[mask],
        "clean_text": clean_texts.values[mask],
        "actual_sentiment": y_true_arr[mask],
        "predicted_sentiment": y_pred_arr[mask],
        "model": model_name,
    })
    return errors_df


def perform_all_error_analyses(
    y_true: pd.Series,
    models_predictions: Dict[str, np.ndarray],
    texts: pd.Series,
    clean_texts: pd.Series,
) -> pd.DataFrame:
    """Perform error analysis across all models and concatenate results.

    Args:
        y_true: Ground-truth labels.
        models_predictions: {model_name: y_pred_array}.
        texts: Original raw text.
        clean_texts: Preprocessed clean text.

    Returns:
        Combined DataFrame of all misclassified records across all models.
    """
    all_errors = []
    for name, y_pred in models_predictions.items():
        errors = perform_error_analysis(y_true, y_pred, texts, clean_texts, name)
        all_errors.append(errors)
    return pd.concat(all_errors, ignore_index=True) if all_errors else pd.DataFrame()


def summarize_errors(error_df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a structured summary of misclassification patterns.

    Args:
        error_df: DataFrame from perform_all_error_analyses.

    Returns:
        Dictionary with error counts, rates, and common confusion pairs.
    """
    if error_df.empty:
        return {"total_errors": 0, "models": {}}

    summary: Dict[str, Any] = {
        "total_errors": len(error_df),
    }

    # Per-model breakdown
    model_summaries = {}
    for model_name in error_df["model"].unique():
        model_errors = error_df[error_df["model"] == model_name]
        confusion_pairs = (
            model_errors.groupby(["actual_sentiment", "predicted_sentiment"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        pairs_list = [
            {"actual": row["actual_sentiment"], "predicted": row["predicted_sentiment"], "count": int(row["count"])}
            for _, row in confusion_pairs.iterrows()
        ]
        model_summaries[model_name] = {
            "error_count": len(model_errors),
            "confusion_pairs": pairs_list,
        }
    summary["models"] = model_summaries
    return summary


# ==============================================================================
# Visualization Functions
# ==============================================================================

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    model_name: str,
    filepath: Optional[Union[Path, str]] = None,
    figsize: Tuple[int, int] = (7, 6),
    highlight_best: bool = False,
) -> Path:
    """Generate and save a presentation-ready confusion matrix heatmap.

    Args:
        cm: Confusion matrix array (rows=actual, cols=predicted).
        labels: Ordered class labels.
        model_name: Name of the model (used in title).
        filepath: Output file path. Auto-generated if None.
        figsize: Figure dimensions.
        highlight_best: If True, uses a distinct color palette for the best model.

    Returns:
        Path to the saved image file.
    """
    if filepath is None:
        safe_name = model_name.lower().replace(" ", "_")
        filepath = CONFUSION_MATRICES_DIR / f"{safe_name}_confusion_matrix.png"

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    cmap = "YlOrRd" if highlight_best else "Blues"

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=1,
        linecolor="white",
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 14, "weight": "bold"},
    )

    title_prefix = " Best Model  " if highlight_best else ""
    ax.set_title(f"{title_prefix}{model_name}\nConfusion Matrix", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("Actual Label", fontsize=12, labelpad=10)
    ax.tick_params(axis="both", labelsize=11)

    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Confusion matrix saved: {filepath}")
    return filepath


def plot_model_performance_comparison(
    comparison_df: pd.DataFrame,
    filepath: Optional[Union[Path, str]] = None,
) -> Path:
    """Generate a grouped bar chart comparing all models across key metrics.

    Args:
        comparison_df: DataFrame from create_comparison_table().
        filepath: Output file path.

    Returns:
        Path to the saved image file.
    """
    if filepath is None:
        filepath = MODEL_PERFORMANCE_PLOT

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    models = comparison_df["Model"].tolist()
    x = np.arange(len(metrics))
    width = 0.22

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for i, model in enumerate(models):
        row = comparison_df[comparison_df["Model"] == model].iloc[0]
        values = [row[m] * 100 for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=model, color=colors[i % len(colors)], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Metric", fontsize=12, labelpad=10)
    ax.set_ylabel("Score (%)", fontsize=12, labelpad=10)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Performance comparison chart saved: {filepath}")
    return filepath


# ==============================================================================
# Save Functions
# ==============================================================================

def save_classification_reports(
    evaluation_results: Dict[str, Dict[str, Any]],
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """Save classification reports as individual text files and a combined JSON.

    Args:
        evaluation_results: Complete evaluation results.
        output_dir: Target directory.

    Returns:
        List of created file paths.
    """
    if output_dir is None:
        output_dir = CLASSIFICATION_REPORTS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created_files: List[Path] = []
    all_reports: Dict[str, str] = {}

    name_map = {
        "Logistic Regression": "logistic_regression",
        "Multinomial Naive Bayes": "naive_bayes",
        "Linear SVM": "linear_svm",
    }

    for model_name, result in evaluation_results.items():
        report_text = result["classification_report"]
        safe_name = name_map.get(model_name, model_name.lower().replace(" ", "_"))

        # Save text report
        txt_path = output_dir / f"{safe_name}_report.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Classification Report: {model_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(report_text)
        created_files.append(txt_path)
        all_reports[model_name] = report_text

    # Save combined JSON
    json_path = output_dir / "classification_reports.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2)
    created_files.append(json_path)

    print(f"[SUCCESS] Classification reports saved to: {output_dir}")
    return created_files


def save_confusion_matrices_json(
    evaluation_results: Dict[str, Dict[str, Any]],
    filepath: Optional[Path] = None,
) -> Path:
    """Save numerical confusion matrices as JSON.

    Args:
        evaluation_results: Complete evaluation results.
        filepath: Output JSON file path.

    Returns:
        Path to saved JSON file.
    """
    if filepath is None:
        filepath = CONFUSION_MATRICES_JSON
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    cm_data = {}
    for name, result in evaluation_results.items():
        cm_data[name] = {
            "matrix": result["confusion_matrix"],
            "labels": CM_LABEL_ORDER,
        }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(cm_data, f, indent=2)
    print(f"[SUCCESS] Confusion matrices JSON saved to: {filepath}")
    return filepath


def save_comparison_csv(
    comparison_df: pd.DataFrame,
    filepath: Optional[Path] = None,
) -> Path:
    """Save model comparison table as CSV.

    Args:
        comparison_df: DataFrame from create_comparison_table().
        filepath: Output CSV path.

    Returns:
        Path to saved CSV.
    """
    if filepath is None:
        filepath = MODEL_COMPARISON_CSV
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(filepath)
    print(f"[SUCCESS] Model comparison CSV saved to: {filepath}")
    return filepath


def save_best_model_json(
    best_model: Dict[str, Any],
    filepath: Optional[Path] = None,
) -> Path:
    """Save best model selection result as JSON.

    Args:
        best_model: Dictionary from select_best_model().
        filepath: Output JSON path.

    Returns:
        Path to saved JSON.
    """
    if filepath is None:
        filepath = BEST_MODEL_JSON
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(best_model, f, indent=2)
    print(f"[SUCCESS] Best model JSON saved to: {filepath}")
    return filepath


def save_per_class_metrics_csv(
    evaluation_results: Dict[str, Dict[str, Any]],
    filepath: Optional[Path] = None,
) -> Path:
    """Save per-class metrics for all models as CSV.

    Args:
        evaluation_results: Complete evaluation results.
        filepath: Output CSV path.

    Returns:
        Path to saved CSV.
    """
    if filepath is None:
        filepath = PER_CLASS_METRICS_CSV
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name, result in evaluation_results.items():
        for class_label, metrics in result["per_class"].items():
            rows.append({
                "Model": model_name,
                "Class": class_label,
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1 Score": metrics["f1_score"],
                "Support": metrics["support"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"[SUCCESS] Per-class metrics CSV saved to: {filepath}")
    return filepath


def save_error_analysis_csv(
    error_df: pd.DataFrame,
    filepath: Optional[Path] = None,
) -> Path:
    """Save error analysis results as CSV.

    Args:
        error_df: DataFrame from perform_all_error_analyses().
        filepath: Output CSV path.

    Returns:
        Path to saved CSV.
    """
    if filepath is None:
        filepath = ERROR_ANALYSIS_CSV
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    error_df.to_csv(filepath, index=False)
    print(f"[SUCCESS] Error analysis CSV saved to: {filepath}")
    return filepath


def save_evaluation_results_json(
    evaluation_results: Dict[str, Dict[str, Any]],
    best_model: Dict[str, Any],
    test_samples: int,
    filepath: Optional[Path] = None,
) -> Path:
    """Save comprehensive evaluation results as JSON.

    Args:
        evaluation_results: Complete evaluation results.
        best_model: Best model selection result.
        test_samples: Number of test samples evaluated.
        filepath: Output JSON path.

    Returns:
        Path to saved JSON.
    """
    if filepath is None:
        filepath = EVALUATION_RESULTS_JSON
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "evaluation_date": datetime.now().isoformat(),
        "test_samples": test_samples,
        "class_labels": CM_LABEL_ORDER,
        "models": {},
        "best_model": best_model,
    }

    for name, result in evaluation_results.items():
        output["models"][name] = {
            "metrics": result["metrics"],
            "per_class": result["per_class"],
        }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"[SUCCESS] Evaluation results JSON saved to: {filepath}")
    return filepath


def save_evaluation_results_csv(
    evaluation_results: Dict[str, Dict[str, Any]],
    test_samples: int,
    filepath: Optional[Path] = None,
) -> Path:
    """Save evaluation results as a flat CSV.

    Args:
        evaluation_results: Complete evaluation results.
        test_samples: Number of test samples.
        filepath: Output CSV path.

    Returns:
        Path to saved CSV.
    """
    if filepath is None:
        filepath = EVALUATION_RESULTS_CSV
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, result in evaluation_results.items():
        m = result["metrics"]
        rows.append({
            "model": name,
            "accuracy": m["accuracy"],
            "precision_weighted": m["precision"],
            "recall_weighted": m["recall"],
            "f1_weighted": m["f1_score"],
            "test_samples": test_samples,
        })

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"[SUCCESS] Evaluation results CSV saved to: {filepath}")
    return filepath
