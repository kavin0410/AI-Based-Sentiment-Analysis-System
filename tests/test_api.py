"""API Endpoint Integration Tests for SentimentLab on FastAPI / Vercel.

Validates all production REST endpoints:
- GET  /api/health
- POST /api/predict (Positive, Negative, Neutral, invalid inputs)
- POST /api/batch-predict
- POST /api/compare
- POST /api/nlp-explain
- GET  /api/models
- GET  /api/insights
- GET  /api/dataset
- GET  /api/reports
"""

from pathlib import Path
import sys
import unittest

# Ensure base dir is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi.testclient import TestClient
from api.index import app


class TestSentimentLabAPI(unittest.TestCase):
    """Integration test suite for SentimentLab FastAPI backend."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_check(self):
        """Test GET /api/health returns 200 and expected status fields."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertEqual(data["app_name"], "SentimentLab")
        self.assertEqual(data["nlp"], "ready")
        self.assertEqual(data["vectorizer"], "loaded")
        self.assertIn("best_model", data)
        self.assertGreaterEqual(data["models"], 1)

    def test_predict_positive(self):
        """Test POST /api/predict on a positive review."""
        payload = {"text": "I absolutely love this product! It is amazing and works wonderfully."}
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("sentiment", data)
        self.assertIn("confidence", data)
        self.assertIn("probabilities", data)
        self.assertIn("model", data)
        self.assertIn("processing_time_ms", data)
        self.assertIsInstance(data["top_keywords"], list)

    def test_predict_negative(self):
        """Test POST /api/predict on a negative review."""
        payload = {"text": "Terrible customer service. The device broke on day one and refund was refused."}
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("sentiment", data)
        self.assertIn(data["sentiment"], ["Positive", "Negative", "Neutral"])

    def test_predict_neutral(self):
        """Test POST /api/predict on a neutral factual statement."""
        payload = {"text": "The package arrived on Wednesday with two cables and instructions."}
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("sentiment", data)

    def test_predict_empty_text_error(self):
        """Test POST /api/predict with empty string returns 422 validation error."""
        payload = {"text": ""}
        response = self.client.post("/api/predict", json=payload)
        self.assertEqual(response.status_code, 422)

    def test_batch_predict(self):
        """Test POST /api/batch-predict on a list of texts."""
        payload = {
            "texts": [
                "Excellent service and high quality item.",
                "Worst experience ever, defective and horrible.",
                "Package delivered at 3pm today.",
            ]
        }
        response = self.client.post("/api/batch-predict", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total"], 3)
        self.assertIn("summary", data)
        self.assertIn("results", data)
        self.assertEqual(len(data["results"]), 3)

    def test_compare_models(self):
        """Test POST /api/compare returns predictions for all 3 models."""
        payload = {"text": "The application UI is modern and works reliably."}
        response = self.client.post("/api/compare", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("models", data)
        self.assertIn("Logistic Regression", data["models"])
        self.assertIn("Multinomial Naive Bayes", data["models"])
        self.assertIn("Linear SVM", data["models"])

    def test_nlp_explain(self):
        """Test POST /api/nlp-explain provides step-by-step breakdown."""
        payload = {"text": "The service was not slow and I loved the experience."}
        response = self.client.post("/api/nlp-explain", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("raw_text", data)
        self.assertIn("cleaned_text", data)
        self.assertIn("tokens", data)
        self.assertIn("stopwords_removed", data)
        self.assertIn("lemmas", data)
        self.assertIn("tfidf_features", data)

    def test_get_models_info(self):
        """Test GET /api/models returns model registry and leaderboard."""
        response = self.client.get("/api/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("leaderboard", data)
        self.assertIn("per_class_metrics", data)
        self.assertIn("confusion_matrices", data)
        self.assertIn("best_model", data)

    def test_get_insights(self):
        """Test GET /api/insights returns corpus metrics and balance."""
        response = self.client.get("/api/insights")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_records"], 60)
        self.assertEqual(data["train_records"], 48)
        self.assertEqual(data["test_records"], 12)
        self.assertIn("class_distribution", data)

    def test_get_dataset(self):
        """Test GET /api/dataset returns dataset samples."""
        response = self.client.get("/api/dataset")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_count"], 60)
        self.assertIn("raw_samples", data)
        self.assertIn("cleaned_samples", data)

    def test_get_reports(self):
        """Test GET /api/reports returns evaluation and error analysis data."""
        response = self.client.get("/api/reports")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("evaluation_results", data)
        self.assertIn("error_analysis", data)


if __name__ == "__main__":
    unittest.main()
