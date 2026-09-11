"""Unit tests for Stage 5: Real-Time Sentiment Prediction Engine.

Tests:
- Model and vectorizer artifact loading
- Dynamic best-model resolution
- Input text validation (empty, whitespace, max length)
- Preprocessing and TF-IDF transformation consistency
- Sentiment prediction pipeline
- Probability vs Decision Score handling
- Session prediction history management and CSV export
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import unittest
import pandas as pd

from src.model_loader import (
    load_best_model,
    load_best_model_metadata,
    load_model,
    load_vectorizer,
    validate_model_artifacts,
)
from src.predict import (
    PredictionHistoryManager,
    format_prediction_result,
    predict_batch,
    predict_sentiment,
    validate_input_text,
)


class TestModelLoader(unittest.TestCase):
    """Tests for model loading and artifact validation functions."""

    def test_load_best_model_metadata(self):
        """Verify loading best model metadata JSON."""
        meta = load_best_model_metadata()
        self.assertIn("model_name", meta)
        self.assertIn("f1_score", meta)
        self.assertIn("accuracy", meta)

    def test_load_best_model(self):
        """Verify dynamic loading of best model and name."""
        model, model_name, meta = load_best_model()
        self.assertIsNotNone(model)
        self.assertIsInstance(model_name, str)
        self.assertTrue(hasattr(model, "predict"))

    def test_load_vectorizer(self):
        """Verify TF-IDF vectorizer artifact loads properly."""
        vec = load_vectorizer()
        self.assertIsNotNone(vec)
        self.assertTrue(hasattr(vec, "transform"))
        self.assertGreater(len(vec.get_feature_names_out()), 0)

    def test_validate_model_artifacts(self):
        """Verify workspace artifact validation report."""
        report = validate_model_artifacts()
        self.assertTrue(report["valid"], f"Artifact validation failed: {report.get('errors')}")
        self.assertIsNotNone(report["best_model_name"])
        self.assertGreater(report["vectorizer_vocab_size"], 0)


class TestInputValidation(unittest.TestCase):
    """Tests for user text input validation rules."""

    def test_empty_input(self):
        """Empty string should return invalid with error message."""
        valid, msg = validate_input_text("")
        self.assertFalse(valid)
        self.assertIn("enter some text", msg)

    def test_whitespace_input(self):
        """Whitespace-only string should return invalid."""
        valid, msg = validate_input_text("   \n\t  ")
        self.assertFalse(valid)
        self.assertIn("enter some text", msg)

    def test_none_input(self):
        """None input should return invalid."""
        valid, msg = validate_input_text(None)
        self.assertFalse(valid)

    def test_long_input(self):
        """Text exceeding max_length should return invalid with informative message."""
        long_str = "a" * 5001
        valid, msg = validate_input_text(long_str, max_length=5000)
        self.assertFalse(valid)
        self.assertIn("exceeds maximum allowed length", msg)

    def test_valid_input(self):
        """Normal text should return valid."""
        valid, msg = validate_input_text("This product is amazing!")
        self.assertTrue(valid)
        self.assertIsNone(msg)


class TestRealTimePrediction(unittest.TestCase):
    """Tests for prediction execution and output structure."""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.model_name, cls.meta = load_best_model()
        cls.vectorizer = load_vectorizer()

    def test_predict_sentiment_structure(self):
        """Prediction output must contain all required keys."""
        res = predict_sentiment("Great customer service!", model=self.model, vectorizer=self.vectorizer)
        self.assertTrue(res["valid"])
        self.assertIn("sentiment", res)
        self.assertIn(res["sentiment"], ["Positive", "Negative", "Neutral"])
        self.assertIn("model_name", res)
        self.assertIn("score", res)
        self.assertIn("score_type", res)
        self.assertIn("processed_text", res)
        self.assertIn("timestamp", res)

    def test_predict_invalid_input_handling(self):
        """Invalid input should gracefully return valid=False dictionary."""
        res = predict_sentiment("", model=self.model, vectorizer=self.vectorizer)
        self.assertFalse(res["valid"])
        self.assertIn("error", res)

    def test_positive_text_prediction(self):
        """Test prediction on strongly positive text."""
        res = predict_sentiment("I absolutely love this amazing product!", model=self.model, vectorizer=self.vectorizer)
        self.assertTrue(res["valid"])

    def test_negative_text_prediction(self):
        """Test prediction on strongly negative text."""
        res = predict_sentiment("This is terrible and complete waste of money.", model=self.model, vectorizer=self.vectorizer)
        self.assertTrue(res["valid"])

    def test_negation_preservation_in_processed_text(self):
        """Verify negation words like 'not' are retained in preprocessed output."""
        res = predict_sentiment("This product is not good at all.", model=self.model, vectorizer=self.vectorizer)
        self.assertTrue(res["valid"])
        self.assertIn("not", res["processed_text"].split())

    def test_predict_batch(self):
        """Batch prediction function returns list of valid prediction dicts."""
        texts = ["Loved it!", "Hated it.", "It was average."]
        batch_res = predict_batch(texts, model=self.model, vectorizer=self.vectorizer)
        self.assertEqual(len(batch_res), 3)
        for item in batch_res:
            self.assertTrue(item["valid"])

    def test_format_prediction_result(self):
        """Formatted display text contains sentiment and score."""
        res = predict_sentiment("Excellent quality!", model=self.model, vectorizer=self.vectorizer)
        formatted = format_prediction_result(res)
        self.assertIn("Sentiment", formatted)
        self.assertIn(res["sentiment"], formatted)


class TestProbabilityAndDecisionScore(unittest.TestCase):
    """Tests for score and confidence classification types."""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.model_name, _ = load_best_model()
        cls.vectorizer = load_vectorizer()

    def test_score_type_validity(self):
        """Score type must be 'probability' or 'decision_score'."""
        res = predict_sentiment("Testing score output", model=self.model, vectorizer=self.vectorizer)
        self.assertIn(res["score_type"], ["probability", "decision_score"])

        if res["score_type"] == "probability":
            self.assertGreaterEqual(res["score"], 0.0)
            self.assertLessEqual(res["score"], 1.0)


class TestPredictionHistory(unittest.TestCase):
    """Tests for in-memory session prediction history manager."""

    def setUp(self):
        self.manager = PredictionHistoryManager(max_limit=5)

    def test_add_and_retrieve_history(self):
        """Added predictions should be stored in history."""
        res = {
            "valid": True,
            "text": "Great app!",
            "processed_text": "great app",
            "sentiment": "Positive",
            "model_name": "TestModel",
            "score": 0.95,
            "score_type": "probability",
            "timestamp": "2026-09-11T12:00:00",
        }
        self.manager.add_prediction(res)
        history = self.manager.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["text"], "Great app!")

    def test_history_limit_capping(self):
        """History should not exceed max_limit."""
        for i in range(10):
            res = {
                "valid": True,
                "text": f"Sample {i}",
                "processed_text": f"sample {i}",
                "sentiment": "Positive",
                "model_name": "TestModel",
                "score": 0.9,
                "score_type": "probability",
                "timestamp": f"2026-09-11T12:00:0{i}",
            }
            self.manager.add_prediction(res)

        history = self.manager.get_history()
        self.assertEqual(len(history), 5)
        self.assertEqual(history[-1]["text"], "Sample 9")

    def test_clear_history(self):
        """Clear should remove all entries."""
        res = {"valid": True, "text": "Test", "sentiment": "Neutral", "score": 0.5}
        self.manager.add_prediction(res)
        self.assertEqual(len(self.manager.get_history()), 1)
        self.manager.clear()
        self.assertEqual(len(self.manager.get_history()), 0)

    def test_export_csv(self):
        """Exporting history should return CSV formatted string."""
        res = {"valid": True, "text": "Hello world", "sentiment": "Neutral", "score": 0.5, "score_type": "probability"}
        self.manager.add_prediction(res)
        csv_str = self.manager.to_csv()
        self.assertIn("timestamp,text,processed_text", csv_str)
        self.assertIn("Hello world", csv_str)


if __name__ == "__main__":
    unittest.main()
