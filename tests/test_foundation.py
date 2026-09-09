"""Foundation Tests for AI Sentiment Intelligence.

Verifies that the project foundation is correctly set up:
- Module imports work
- Configuration paths are valid
- Basic cleaning functions work
- Data loading and column validation work
"""

import os
from pathlib import Path
import sys
import unittest

# Add parent directory to path so imports resolve correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestImports(unittest.TestCase):
    """Test module imports across the project structure."""

    def test_import_config(self) -> None:
        """Test importing the config module."""
        import config
        self.assertIsNotNone(config)

    def test_import_data_loader(self) -> None:
        """Test importing src.data_loader."""
        import src.data_loader
        self.assertIsNotNone(src.data_loader)

    def test_import_data_cleaning(self) -> None:
        """Test importing src.data_cleaning."""
        import src.data_cleaning
        self.assertIsNotNone(src.data_cleaning)

    def test_import_preprocessing(self) -> None:
        """Test importing src.preprocessing."""
        import src.preprocessing
        self.assertIsNotNone(src.preprocessing)

    def test_import_feature_extraction(self) -> None:
        """Test importing src.feature_extraction."""
        import src.feature_extraction
        self.assertIsNotNone(src.feature_extraction)

    def test_import_train_model(self) -> None:
        """Test importing src.train_model."""
        import src.train_model
        self.assertIsNotNone(src.train_model)

    def test_import_evaluate_model(self) -> None:
        """Test importing src.evaluate_model."""
        import src.evaluate_model
        self.assertIsNotNone(src.evaluate_model)

    def test_import_predict(self) -> None:
        """Test importing src.predict."""
        import src.predict
        self.assertIsNotNone(src.predict)


class TestConfiguration(unittest.TestCase):
    """Test project configuration constants and path setup."""

    def test_raw_data_path_exists(self) -> None:
        """Check that RAW_DATA_PATH is a Path object."""
        import config
        self.assertIsInstance(config.RAW_DATA_PATH, Path)

    def test_processed_data_path_exists(self) -> None:
        """Check that PROCESSED_DATA_PATH is a Path object."""
        import config
        self.assertIsInstance(config.PROCESSED_DATA_PATH, Path)

    def test_model_path_exists(self) -> None:
        """Check that MODEL_PATH is a Path object."""
        import config
        self.assertIsInstance(config.MODEL_PATH, Path)

    def test_results_path_exists(self) -> None:
        """Check that RESULTS_PATH is a Path object."""
        import config
        self.assertIsInstance(config.RESULTS_PATH, Path)

    def test_random_state_is_int(self) -> None:
        """Check that RANDOM_STATE is an integer."""
        import config
        self.assertIsInstance(config.RANDOM_STATE, int)

    def test_supported_labels(self) -> None:
        """Check that SUPPORTED_LABELS contains exactly ['Positive', 'Negative', 'Neutral']."""
        import config
        self.assertEqual(list(config.SUPPORTED_LABELS), ["Positive", "Negative", "Neutral"])

    def test_ensure_directories(self) -> None:
        """Call ensure_directories() and verify required directories are created."""
        import config
        config.ensure_directories()
        self.assertTrue(config.RAW_DATA_PATH.exists(), "RAW_DATA_PATH does not exist")
        self.assertTrue(config.PROCESSED_DATA_PATH.exists(), "PROCESSED_DATA_PATH does not exist")
        self.assertTrue(config.MODEL_PATH.exists(), "MODEL_PATH does not exist")
        self.assertTrue(config.RESULTS_PATH.exists(), "RESULTS_PATH does not exist")


class TestDataCleaning(unittest.TestCase):
    """Test text cleaning and normalization functions."""

    def test_remove_urls(self) -> None:
        """Test that URLs are removed from text."""
        from src.data_cleaning import remove_urls
        text_with_urls = "Visit https://example.com/demo and http://test.org for info."
        cleaned = remove_urls(text_with_urls)
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("http://", cleaned)
        self.assertNotIn("example.com", cleaned)

    def test_remove_html_tags(self) -> None:
        """Test that HTML tags are removed."""
        from src.data_cleaning import remove_html_tags
        text_with_html = "<div><p>This product is <b>awesome</b>!</p></div>"
        cleaned = remove_html_tags(text_with_html)
        self.assertNotIn("<p>", cleaned)
        self.assertNotIn("</p>", cleaned)
        self.assertNotIn("<b>", cleaned)
        self.assertNotIn("</b>", cleaned)
        self.assertIn("awesome", cleaned)

    def test_remove_extra_whitespace(self) -> None:
        """Test that extra whitespace is collapsed."""
        from src.data_cleaning import remove_extra_whitespace
        text_with_spaces = "   Multiple   spaces    between    words.   "
        cleaned = remove_extra_whitespace(text_with_spaces)
        self.assertEqual(cleaned, "Multiple spaces between words.")

    def test_normalize_text(self) -> None:
        """Test that text is lowercased."""
        from src.data_cleaning import normalize_text
        text = "Sentiment Analysis With PYTHON"
        self.assertEqual(normalize_text(text), "sentiment analysis with python")

    def test_clean_text(self) -> None:
        """Test the full clean_text pipeline."""
        from src.data_cleaning import clean_text
        raw_text = "  <p>Check OUT https://github.com/sentiment-analysis for AWESOME updates!  </p>  "
        cleaned = clean_text(raw_text)
        self.assertNotIn("<p>", cleaned)
        self.assertNotIn("https://", cleaned)
        self.assertEqual(cleaned, "check out for awesome updates!")

    def test_preserve_negation(self) -> None:
        """Test that negation words (not, no, never) are preserved after cleaning."""
        from src.data_cleaning import clean_text
        text = "This is not good, never buying again, no satisfaction."
        cleaned = clean_text(text)
        tokens = cleaned.split()
        self.assertIn("not", tokens)
        self.assertIn("never", tokens)
        self.assertIn("no", tokens)


class TestDataLoader(unittest.TestCase):
    """Test data loader operations and schema validation."""

    def test_load_nonexistent_file(self) -> None:
        """Test that loading a nonexistent file raises an appropriate error or returns None."""
        from pathlib import Path
        from src.data_loader import load_dataset
        nonexistent = Path("data/raw/nonexistent_test_file_9999.csv")
        with self.assertRaises((SystemExit, FileNotFoundError)):
            load_dataset(nonexistent)

    def test_validate_columns_valid(self) -> None:
        """Create a small DataFrame with 'text' and 'sentiment' columns, verify validate_columns returns True."""
        import pandas as pd
        from src.data_loader import validate_columns
        df_valid = pd.DataFrame({
            "text": ["Excellent device!", "Broken on arrival."],
            "sentiment": ["Positive", "Negative"]
        })
        self.assertTrue(validate_columns(df_valid))

    def test_validate_columns_invalid(self) -> None:
        """Create DataFrame without required columns, verify returns False."""
        import pandas as pd
        from src.data_loader import validate_columns
        df_invalid = pd.DataFrame({
            "review": ["Excellent device!", "Broken on arrival."],
            "rating": [5, 1]
        })
        self.assertFalse(validate_columns(df_invalid))


if __name__ == "__main__":
    unittest.main()
