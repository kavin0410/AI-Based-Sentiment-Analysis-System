"""Unit tests for Stage 6: Sentiment Intelligence Experience.

Tests:
- Workspace health validation report
- Overview metrics and dataset health disclaimers
- Dynamic Best Model card data extraction
- Model Lab selection logic verification
- Session prediction history integration with Overview page
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import unittest
import pandas as pd

import config
from src.model_loader import (
    load_best_model,
    load_best_model_metadata,
    validate_model_artifacts,
)
from src.predict import PredictionHistoryManager, predict_sentiment


class TestStage6Experience(unittest.TestCase):
    """Tests for Stage 6 UI experience, metadata extraction, and disclaimers."""

    def test_best_model_card_metadata(self):
        """Verify best model card dynamically reads results/best_model.json."""
        self.assertTrue(config.BEST_MODEL_JSON.exists(), "best_model.json must exist")
        with open(config.BEST_MODEL_JSON, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.assertIn("model_name", meta)
        self.assertIn("f1_score", meta)
        self.assertIn("accuracy", meta)
        self.assertIn("selection_metric", meta)

    def test_dataset_health_metrics(self):
        """Verify cleaned dataset contains 60 records and balanced classes."""
        self.assertTrue(config.CLEANED_DATA_FILE.exists())
        df = pd.read_csv(config.CLEANED_DATA_FILE)
        self.assertEqual(len(df), 60)
        counts = df["sentiment"].value_counts().to_dict()
        self.assertEqual(counts.get("Positive"), 20)
        self.assertEqual(counts.get("Negative"), 20)
        self.assertEqual(counts.get("Neutral"), 20)

    def test_system_status_validation(self):
        """Verify system status report is valid and online."""
        report = validate_model_artifacts()
        self.assertTrue(report["valid"])
        self.assertIn(report["best_model_name"], ["Logistic Regression", "Multinomial Naive Bayes", "Linear SVM"])
        self.assertGreater(report["vectorizer_vocab_size"], 0)

    def test_hero_sentiment_analyzer_integration(self):
        """Verify quick hero sentiment prediction produces valid response."""
        res = predict_sentiment("Great product! Really enjoyed using it.")
        self.assertTrue(res["valid"])
        self.assertIn(res["sentiment"], ["Positive", "Negative", "Neutral"])
        self.assertIn("score", res)

    def test_session_history_tracking(self):
        """Verify session prediction history increments properly."""
        history_mgr = PredictionHistoryManager(max_limit=10)
        self.assertEqual(len(history_mgr.get_history()), 0)

        res = predict_sentiment("Testing session prediction tracking")
        history_mgr.add_prediction(res)
        self.assertEqual(len(history_mgr.get_history()), 1)


if __name__ == "__main__":
    unittest.main()
