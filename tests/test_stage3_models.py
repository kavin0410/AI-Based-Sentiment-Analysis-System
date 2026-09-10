"""Stage 3 Unit Tests for AI Sentiment Intelligence.

Tests TF-IDF feature engineering, model training, prediction,
serialization, metadata persistence, and data leakage prevention.
"""

import json
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
    CLEANED_DATA_FILE,
    LINEAR_SVM_FILE,
    LOGISTIC_REGRESSION_FILE,
    MODEL_METADATA_FILE,
    MODEL_PATH,
    NAIVE_BAYES_FILE,
    RANDOM_STATE,
    SUPPORTED_LABELS,
    TEST_SIZE,
    TFIDF_VECTORIZER_FILE,
)
from src.data_loader import load_dataset, split_dataset
from src.feature_extraction import (
    create_tfidf_vectorizer,
    fit_transform_tfidf,
    get_feature_info,
    load_vectorizer,
    save_vectorizer,
    transform_tfidf,
)
from src.train_model import (
    load_model,
    load_model_metadata,
    predict_all_models,
    predict_model,
    save_all_models,
    save_model,
    save_model_metadata,
    train_all_models,
    train_linear_svm,
    train_logistic_regression,
    train_naive_bayes,
)


class TestTfidfFeatureExtraction(unittest.TestCase):
    """Test suite for TF-IDF feature extraction module."""

    def setUp(self):
        self.sample_texts = pd.Series([
            "great excellent fantastic performance",
            "terrible horrible worst experience ever",
            "average standard regular product delivery",
            "not good disappointed completely",
        ])
        self.test_text = pd.Series(["great product not terrible"])

    def test_create_tfidf_vectorizer(self):
        """Verify TF-IDF vectorizer is initialized with expected hyperparameters."""
        vec = create_tfidf_vectorizer(max_features=100, ngram_range=(1, 2), sublinear_tf=True)
        self.assertEqual(vec.max_features, 100)
        self.assertEqual(vec.ngram_range, (1, 2))
        self.assertTrue(vec.sublinear_tf)

    def test_fit_transform_tfidf(self):
        """Verify vectorizer fits and transforms text into a sparse matrix."""
        X_matrix, vec = fit_transform_tfidf(self.sample_texts)
        self.assertTrue(isinstance(X_matrix, spmatrix))
        self.assertEqual(X_matrix.shape[0], len(self.sample_texts))
        self.assertTrue(hasattr(vec, "vocabulary_"))
        self.assertGreater(len(vec.vocabulary_), 0)

    def test_transform_tfidf(self):
        """Verify transform operates on new text using an already fitted vectorizer."""
        _, vec = fit_transform_tfidf(self.sample_texts)
        test_matrix = transform_tfidf(self.test_text, vec)
        self.assertTrue(isinstance(test_matrix, spmatrix))
        self.assertEqual(test_matrix.shape[0], len(self.test_text))
        self.assertEqual(test_matrix.shape[1], len(vec.get_feature_names_out()))

    def test_unfitted_transform_raises_error(self):
        """Verify transforming with an unfitted vectorizer raises a ValueError."""
        unfitted_vec = create_tfidf_vectorizer()
        with self.assertRaises(ValueError):
            transform_tfidf(self.test_text, unfitted_vec)

    def test_get_feature_info(self):
        """Verify feature info extraction returns vocabulary statistics."""
        X_matrix, vec = fit_transform_tfidf(self.sample_texts)
        info = get_feature_info(vec, X_matrix)
        self.assertIn("vocabulary_size", info)
        self.assertIn("sample_features", info)
        self.assertIn("top_features_by_weight", info)
        self.assertEqual(info["vocabulary_size"], len(vec.vocabulary_))


class TestTrainTestSplit(unittest.TestCase):
    """Test suite for dataset splitting and data leakage prevention."""

    def setUp(self):
        # Create a mock balanced dataset
        self.df = pd.DataFrame({
            "clean_text": [f"sample text sentence {i}" for i in range(30)],
            "sentiment": ["Positive"] * 10 + ["Negative"] * 10 + ["Neutral"] * 10,
        })

    def test_stratified_split_proportions(self):
        """Verify split respects 80/20 ratio and preserves class stratification."""
        X_train, X_test, y_train, y_test = split_dataset(
            self.df,
            text_column="clean_text",
            sentiment_column="sentiment",
            test_size=0.2,
            random_state=42,
            stratify=True,
        )
        self.assertEqual(len(X_train), 24)
        self.assertEqual(len(X_test), 6)
        # Verify stratified class distribution in test set
        test_counts = y_test.value_counts().to_dict()
        self.assertEqual(test_counts.get("Positive", 0), 2)
        self.assertEqual(test_counts.get("Negative", 0), 2)
        self.assertEqual(test_counts.get("Neutral", 0), 2)

    def test_split_reproducibility(self):
        """Verify identical splits are produced with identical random_state."""
        X_tr1, X_te1, y_tr1, y_te1 = split_dataset(self.df, random_state=42)
        X_tr2, X_te2, y_tr2, y_te2 = split_dataset(self.df, random_state=42)
        pd.testing.assert_series_equal(X_tr1, X_tr2)
        pd.testing.assert_series_equal(y_tr1, y_tr2)

    def test_data_leakage_strict_isolation(self):
        """Verify vectorizer fit on train does NOT include vocabulary unique to test set."""
        train_corpus = pd.Series(["apple banana orange", "grape kiwi lemon"])
        test_corpus = pd.Series(["watermelon papaya mango"])  # words completely absent in train

        _, vec = fit_transform_tfidf(train_corpus)
        vocab = set(vec.get_feature_names_out())

        # Assert no test-only words leaked into vocabulary
        self.assertNotIn("watermelon", vocab)
        self.assertNotIn("papaya", vocab)
        self.assertNotIn("mango", vocab)


class TestModelTraining(unittest.TestCase):
    """Test suite for model training, prediction, and registry."""

    def setUp(self):
        texts = pd.Series([
            "great excellent love wonderful awesome",
            "superb quality delightful impressive amazing",
            "terrible horrible awful poor bad",
            "disaster broke defective useless worst",
            "normal regular standard average okay",
            "routine common neutral moderate typical",
        ])
        labels = pd.Series([
            "Positive", "Positive",
            "Negative", "Negative",
            "Neutral", "Neutral",
        ])
        self.X_matrix, self.vec = fit_transform_tfidf(texts)
        self.y = labels

    def test_train_logistic_regression(self):
        """Verify Logistic Regression trains and predicts valid classes."""
        model = train_logistic_regression(self.X_matrix, self.y, random_state=42)
        self.assertIsInstance(model, LogisticRegression)
        preds = predict_model(model, self.X_matrix)
        self.assertEqual(len(preds), len(self.y))
        for p in preds:
            self.assertIn(p, SUPPORTED_LABELS)

    def test_train_naive_bayes(self):
        """Verify Multinomial Naive Bayes trains and predicts valid classes."""
        model = train_naive_bayes(self.X_matrix, self.y)
        self.assertIsInstance(model, MultinomialNB)
        preds = predict_model(model, self.X_matrix)
        self.assertEqual(len(preds), len(self.y))
        for p in preds:
            self.assertIn(p, SUPPORTED_LABELS)

    def test_train_linear_svm(self):
        """Verify Linear SVM trains and predicts valid classes."""
        model = train_linear_svm(self.X_matrix, self.y, random_state=42)
        self.assertIsInstance(model, LinearSVC)
        preds = predict_model(model, self.X_matrix)
        self.assertEqual(len(preds), len(self.y))
        for p in preds:
            self.assertIn(p, SUPPORTED_LABELS)

    def test_train_all_models(self):
        """Verify train_all_models returns the complete model registry."""
        models = train_all_models(self.X_matrix, self.y, random_state=42)
        self.assertIn("Logistic Regression", models)
        self.assertIn("Multinomial Naive Bayes", models)
        self.assertIn("Linear SVM", models)

        preds_dict = predict_all_models(models, self.X_matrix)
        self.assertEqual(len(preds_dict), 3)
        for name, preds in preds_dict.items():
            self.assertEqual(len(preds), len(self.y))


class TestModelPersistence(unittest.TestCase):
    """Test suite for saving, loading, and persistence verification."""

    def test_saved_artifacts_exist(self):
        """Verify all Stage 3 model artifacts and vectorizer exist on disk."""
        self.assertTrue(LOGISTIC_REGRESSION_FILE.exists(), f"Missing: {LOGISTIC_REGRESSION_FILE}")
        self.assertTrue(NAIVE_BAYES_FILE.exists(), f"Missing: {NAIVE_BAYES_FILE}")
        self.assertTrue(LINEAR_SVM_FILE.exists(), f"Missing: {LINEAR_SVM_FILE}")
        self.assertTrue(TFIDF_VECTORIZER_FILE.exists(), f"Missing: {TFIDF_VECTORIZER_FILE}")
        self.assertTrue(MODEL_METADATA_FILE.exists(), f"Missing: {MODEL_METADATA_FILE}")

    def test_load_and_predict_with_saved_artifacts(self):
        """Verify that loaded models and vectorizer produce consistent predictions."""
        vec = load_vectorizer(TFIDF_VECTORIZER_FILE)
        lr = load_model(LOGISTIC_REGRESSION_FILE)
        nb = load_model(NAIVE_BAYES_FILE)
        svm = load_model(LINEAR_SVM_FILE)

        sample = ["great amazing product service"]
        feats = transform_tfidf(sample, vec)

        p_lr = lr.predict(feats)[0]
        p_nb = nb.predict(feats)[0]
        p_svm = svm.predict(feats)[0]

        self.assertIn(p_lr, SUPPORTED_LABELS)
        self.assertIn(p_nb, SUPPORTED_LABELS)
        self.assertIn(p_svm, SUPPORTED_LABELS)

    def test_metadata_integrity(self):
        """Verify metadata contains required non-fabricated fields."""
        metadata = load_model_metadata(MODEL_METADATA_FILE)
        self.assertEqual(metadata["project"], "AI Sentiment Intelligence")
        self.assertEqual(metadata["total_samples"], 60)
        self.assertEqual(metadata["train_samples"], 48)
        self.assertEqual(metadata["test_samples"], 12)
        self.assertEqual(metadata["random_state"], 42)
        self.assertTrue(metadata["data_leakage_check_passed"])
        self.assertIn("models_trained", metadata)
        self.assertEqual(len(metadata["models_trained"]), 3)


if __name__ == "__main__":
    unittest.main()
