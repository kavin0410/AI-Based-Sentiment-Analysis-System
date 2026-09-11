"""Unit tests for Stage 4: Model Evaluation & Performance Intelligence.

Tests the evaluation functions in src/evaluate_model.py including:
- Metric calculation (accuracy, precision, recall, F1)
- Per-class metrics
- Confusion matrix generation
- Classification report generation
- Multi-model evaluation and comparison
- Best model selection
- Error analysis

All tests use synthetic data to validate function correctness.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
import numpy as np
import pandas as pd

from src.evaluate_model import (
    calculate_model_metrics,
    calculate_per_class_metrics,
    generate_classification_report_text,
    generate_confusion_matrix,
    evaluate_all_models,
    select_best_model,
    create_comparison_table,
    perform_error_analysis,
    perform_all_error_analyses,
    summarize_errors,
)


# ==============================================================================
# Common Test Data
# ==============================================================================
Y_TRUE = np.array([
    "Negative", "Neutral", "Positive", "Negative", "Neutral", "Positive",
    "Negative", "Neutral", "Positive", "Negative", "Neutral", "Positive",
])

Y_PRED_PERFECT = Y_TRUE.copy()

Y_PRED_PARTIAL = np.array([
    "Negative", "Neutral", "Positive", "Positive", "Neutral", "Positive",
    "Negative", "Positive", "Positive", "Negative", "Neutral", "Negative",
])

Y_PRED_ALL_WRONG = np.array([
    "Positive", "Negative", "Neutral", "Positive", "Negative", "Neutral",
    "Positive", "Negative", "Neutral", "Positive", "Negative", "Neutral",
])

CM_LABEL_ORDER = ["Negative", "Neutral", "Positive"]


# ==============================================================================
# Test Classes
# ==============================================================================

class TestMetricCalculation(unittest.TestCase):
    """Tests for overall metric calculation (accuracy, precision, recall, F1)."""

    def test_perfect_predictions(self):
        """All predictions match labels — accuracy should be 1.0."""
        metrics = calculate_model_metrics(Y_TRUE, Y_PRED_PERFECT)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_all_wrong_predictions(self):
        """No predictions match — accuracy should be 0.0."""
        metrics = calculate_model_metrics(Y_TRUE, Y_PRED_ALL_WRONG)
        self.assertEqual(metrics["accuracy"], 0.0)

    def test_partial_accuracy(self):
        """Mixed predictions — accuracy should be between 0 and 1."""
        metrics = calculate_model_metrics(Y_TRUE, Y_PRED_PARTIAL)
        self.assertGreater(metrics["accuracy"], 0.0)
        self.assertLess(metrics["accuracy"], 1.0)

    def test_metrics_keys(self):
        """Returned dict must contain accuracy, precision, recall, f1_score."""
        metrics = calculate_model_metrics(Y_TRUE, Y_PRED_PARTIAL)
        for key in ["accuracy", "precision", "recall", "f1_score"]:
            self.assertIn(key, metrics, f"Missing key: {key}")

    def test_metrics_range(self):
        """All metric values must be between 0.0 and 1.0."""
        metrics = calculate_model_metrics(Y_TRUE, Y_PRED_PARTIAL)
        for key in ["accuracy", "precision", "recall", "f1_score"]:
            self.assertGreaterEqual(metrics[key], 0.0, f"{key} below 0")
            self.assertLessEqual(metrics[key], 1.0, f"{key} above 1")


class TestPerClassMetrics(unittest.TestCase):
    """Tests for per-class (per-sentiment) metric calculation."""

    def test_per_class_keys(self):
        """Each label in CM_LABEL_ORDER must be present as a key."""
        metrics = calculate_per_class_metrics(Y_TRUE, Y_PRED_PARTIAL, CM_LABEL_ORDER)
        for label in CM_LABEL_ORDER:
            self.assertIn(label, metrics, f"Missing class: {label}")

    def test_per_class_metric_keys(self):
        """Each class must have precision, recall, f1_score, support."""
        metrics = calculate_per_class_metrics(Y_TRUE, Y_PRED_PARTIAL, CM_LABEL_ORDER)
        for label in CM_LABEL_ORDER:
            for key in ["precision", "recall", "f1_score", "support"]:
                self.assertIn(key, metrics[label], f"Missing {key} for {label}")

    def test_support_counts(self):
        """Total support across all classes must equal total test samples."""
        metrics = calculate_per_class_metrics(Y_TRUE, Y_PRED_PARTIAL, CM_LABEL_ORDER)
        total_support = sum(metrics[label]["support"] for label in CM_LABEL_ORDER)
        self.assertEqual(total_support, len(Y_TRUE))


class TestConfusionMatrix(unittest.TestCase):
    """Tests for confusion matrix generation and structure."""

    def test_confusion_matrix_shape(self):
        """Confusion matrix should be 3×3 for 3 sentiment classes."""
        cm = generate_confusion_matrix(Y_TRUE, Y_PRED_PARTIAL, CM_LABEL_ORDER)
        self.assertEqual(cm.shape, (3, 3))

    def test_confusion_matrix_label_order(self):
        """Matrix dimensions must match the number of labels."""
        cm = generate_confusion_matrix(Y_TRUE, Y_PRED_PARTIAL, CM_LABEL_ORDER)
        self.assertEqual(cm.shape[0], len(CM_LABEL_ORDER))
        self.assertEqual(cm.shape[1], len(CM_LABEL_ORDER))

    def test_perfect_cm_diagonal(self):
        """Perfect predictions: all values on diagonal, zero off-diagonal."""
        cm = generate_confusion_matrix(Y_TRUE, Y_PRED_PERFECT, CM_LABEL_ORDER)
        self.assertEqual(np.sum(np.diag(cm)), len(Y_TRUE))
        self.assertEqual(np.sum(cm) - np.sum(np.diag(cm)), 0)

    def test_confusion_matrix_sum(self):
        """Total CM sum should equal total number of samples."""
        cm = generate_confusion_matrix(Y_TRUE, Y_PRED_PARTIAL, CM_LABEL_ORDER)
        self.assertEqual(np.sum(cm), len(Y_TRUE))


class TestClassificationReport(unittest.TestCase):
    """Tests for classification report text generation."""

    def test_report_is_string(self):
        """Classification report must be a string."""
        report = generate_classification_report_text(Y_TRUE, Y_PRED_PARTIAL, CM_LABEL_ORDER)
        self.assertIsInstance(report, str)

    def test_report_contains_labels(self):
        """Report text must mention all sentiment labels."""
        report = generate_classification_report_text(Y_TRUE, Y_PRED_PARTIAL, CM_LABEL_ORDER)
        for label in CM_LABEL_ORDER:
            self.assertIn(label, report)

    def test_report_contains_metrics(self):
        """Report text must mention precision, recall, f1-score."""
        report = generate_classification_report_text(Y_TRUE, Y_PRED_PARTIAL, CM_LABEL_ORDER)
        self.assertIn("precision", report)
        self.assertIn("recall", report)
        self.assertIn("f1-score", report)


class TestModelComparison(unittest.TestCase):
    """Tests for multi-model evaluation and comparison table."""

    def setUp(self):
        self.predictions = {
            "Model A": Y_PRED_PERFECT,
            "Model B": Y_PRED_PARTIAL,
            "Model C": Y_PRED_ALL_WRONG,
        }

    def test_evaluate_all_models_keys(self):
        """All model names must be present in evaluation results."""
        results = evaluate_all_models(self.predictions, Y_TRUE, CM_LABEL_ORDER)
        for model_name in self.predictions:
            self.assertIn(model_name, results)

    def test_comparison_table_columns(self):
        """Comparison DataFrame must have Model, Accuracy, Precision, Recall, F1 Score."""
        results = evaluate_all_models(self.predictions, Y_TRUE, CM_LABEL_ORDER)
        df = create_comparison_table(results)
        for col in ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]:
            self.assertIn(col, df.columns)

    def test_comparison_table_sorted(self):
        """Comparison table must be sorted by F1 Score in descending order."""
        results = evaluate_all_models(self.predictions, Y_TRUE, CM_LABEL_ORDER)
        df = create_comparison_table(results)
        f1_scores = df["F1 Score"].tolist()
        self.assertEqual(f1_scores, sorted(f1_scores, reverse=True))

    def test_comparison_table_rows(self):
        """Number of rows must equal number of models."""
        results = evaluate_all_models(self.predictions, Y_TRUE, CM_LABEL_ORDER)
        df = create_comparison_table(results)
        self.assertEqual(len(df), len(self.predictions))


class TestBestModelSelection(unittest.TestCase):
    """Tests for automatic best-model selection logic."""

    def setUp(self):
        self.predictions = {
            "Model A": Y_PRED_PERFECT,
            "Model B": Y_PRED_PARTIAL,
        }
        self.results = evaluate_all_models(self.predictions, Y_TRUE, CM_LABEL_ORDER)

    def test_best_model_has_name(self):
        """Result must include 'model_name' key."""
        best = select_best_model(self.results)
        self.assertIn("model_name", best)

    def test_best_model_highest_f1(self):
        """Best model should be the one with the highest F1 score."""
        best = select_best_model(self.results)
        self.assertEqual(best["model_name"], "Model A")

    def test_best_model_tiebreak(self):
        """When F1 is tied, model with higher accuracy should win."""
        # Manually construct tied results
        tied_results = {
            "Model X": {
                "metrics": {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1_score": 0.8},
                "per_class": {},
                "confusion_matrix": [[0]],
                "classification_report": "",
            },
            "Model Y": {
                "metrics": {"accuracy": 0.8, "precision": 0.8, "recall": 0.8, "f1_score": 0.8},
                "per_class": {},
                "confusion_matrix": [[0]],
                "classification_report": "",
            },
        }
        best = select_best_model(tied_results)
        self.assertEqual(best["model_name"], "Model X")

    def test_best_model_has_selection_metric(self):
        """Result must include selection_metric = 'f1_score'."""
        best = select_best_model(self.results)
        self.assertEqual(best["selection_metric"], "f1_score")


class TestErrorAnalysis(unittest.TestCase):
    """Tests for misclassification error analysis."""

    def setUp(self):
        self.y_true = pd.Series(Y_TRUE)
        self.texts = pd.Series([f"Original text {i}" for i in range(12)])
        self.clean_texts = pd.Series([f"clean text {i}" for i in range(12)])

    def test_error_analysis_columns(self):
        """Error DataFrame must have text, clean_text, actual, predicted, model."""
        errors = perform_error_analysis(
            y_true=self.y_true,
            y_pred=Y_PRED_PARTIAL,
            texts=self.texts,
            clean_texts=self.clean_texts,
            model_name="Test Model",
        )
        for col in ["text", "clean_text", "actual_sentiment", "predicted_sentiment", "model"]:
            self.assertIn(col, errors.columns)

    def test_error_analysis_count(self):
        """Number of errors should match actual misclassification count."""
        errors = perform_error_analysis(
            y_true=self.y_true,
            y_pred=Y_PRED_PARTIAL,
            texts=self.texts,
            clean_texts=self.clean_texts,
            model_name="Test Model",
        )
        expected = int(np.sum(Y_TRUE != Y_PRED_PARTIAL))
        self.assertEqual(len(errors), expected)

    def test_no_errors_when_perfect(self):
        """Perfect predictions should yield an empty DataFrame."""
        errors = perform_error_analysis(
            y_true=self.y_true,
            y_pred=Y_PRED_PERFECT,
            texts=self.texts,
            clean_texts=self.clean_texts,
            model_name="Perfect Model",
        )
        self.assertEqual(len(errors), 0)

    def test_error_summary_structure(self):
        """summarize_errors must return dict with 'total_errors' and 'models'."""
        predictions = {
            "Model A": Y_PRED_PERFECT,
            "Model B": Y_PRED_PARTIAL,
        }
        all_errors = perform_all_error_analyses(
            y_true=self.y_true,
            models_predictions=predictions,
            texts=self.texts,
            clean_texts=self.clean_texts,
        )
        summary = summarize_errors(all_errors)
        self.assertIn("total_errors", summary)
        self.assertIn("models", summary)


if __name__ == "__main__":
    unittest.main()
