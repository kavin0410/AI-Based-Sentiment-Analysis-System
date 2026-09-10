"""Stage 2 NLP Preprocessing & Data Engineering Unit Tests.

Validates:
- Contraction expansion (don't -> do not, couldn't -> could not)
- Lowercase normalization
- URL, HTML tag, and Email removal
- Whitespace normalization
- Controlled stopword filtering (strict preservation of negation terms)
- Tokenization
- WordNet Lemmatization
- Complete preprocessing pipeline
- Dataset validation & error handling
"""

import os
from pathlib import Path
import sys
import unittest

import pandas as pd

# Add base directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import CLEANED_DATA_FILE, SUPPORTED_LABELS
from src.data_cleaning import (
    clean_text_basic,
    remove_emails,
    remove_extra_whitespace,
    remove_html_tags,
    remove_special_characters,
    remove_urls,
    safe_str,
    to_lowercase,
)
from src.data_loader import (
    calculate_text_statistics,
    get_default_dataset_path,
    load_dataset,
    validate_dataset,
)
from src.preprocessing import (
    analyze_word_frequencies,
    expand_contractions,
    get_controlled_stopwords,
    lemmatize_tokens,
    preprocess_dataset,
    preprocess_text,
    remove_stopwords,
    tokenize_text,
)


class TestDataCleaningStage2(unittest.TestCase):
    """Test text cleaning functions introduced or enhanced in Stage 2."""

    def test_safe_str(self) -> None:
        """Test safe string conversion for None, NaN, float, and int."""
        self.assertEqual(safe_str(None), "")
        self.assertEqual(safe_str(float("nan")), "")
        self.assertEqual(safe_str(123), "123")
        self.assertEqual(safe_str("hello"), "hello")

    def test_to_lowercase(self) -> None:
        """Test lowercase conversion preserving letters."""
        self.assertEqual(to_lowercase("AWESOME Product NOT GOOD"), "awesome product not good")

    def test_remove_urls(self) -> None:
        """Test removing http, https, www, and ftp links."""
        text = "Visit https://google.com/test and www.example.org or ftp://ftp.site.com."
        cleaned = remove_urls(text)
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("www.example.org", cleaned)
        self.assertNotIn("ftp://", cleaned)

    def test_remove_html_tags_and_entities(self) -> None:
        """Test removing HTML tags and unescaping HTML entities."""
        text = "<h1>Great!</h1> &amp; lovely &quot;service&quot; <br />"
        cleaned = remove_html_tags(text)
        self.assertNotIn("<h1>", cleaned)
        self.assertNotIn("<br />", cleaned)
        self.assertIn("Great!", cleaned)
        self.assertIn("&", cleaned)

    def test_remove_emails(self) -> None:
        """Test removing email addresses."""
        text = "Contact support@example.com or sales.dept@company.co.uk today."
        cleaned = remove_emails(text)
        self.assertNotIn("support@example.com", cleaned)
        self.assertNotIn("sales.dept@company.co.uk", cleaned)

    def test_remove_special_characters(self) -> None:
        """Test removing punctuation and numbers while keeping letters."""
        text = "Price: $99.99! Order #1234; rating: 100%."
        cleaned = remove_special_characters(text, preserve_apostrophes=False)
        self.assertNotIn("$", cleaned)
        self.assertNotIn("99", cleaned)
        self.assertNotIn("100", cleaned)
        self.assertIn("Price", cleaned)
        self.assertIn("Order", cleaned)

    def test_normalize_whitespace(self) -> None:
        """Test stripping and collapsing consecutive spaces/tabs/newlines."""
        text = "   hello \t\t  world  \n\n  again   "
        self.assertEqual(remove_extra_whitespace(text), "hello world again")


class TestContractionHandling(unittest.TestCase):
    """Test contraction expansion dictionary and regex matching."""

    def test_common_contractions(self) -> None:
        """Test expansion of common negative and auxiliary contractions."""
        self.assertEqual(expand_contractions("don't"), "do not")
        self.assertEqual(expand_contractions("doesn't"), "does not")
        self.assertEqual(expand_contractions("didn't"), "did not")
        self.assertEqual(expand_contractions("can't"), "cannot")
        self.assertEqual(expand_contractions("couldn't"), "could not")
        self.assertEqual(expand_contractions("won't"), "will not")
        self.assertEqual(expand_contractions("wouldn't"), "would not")
        self.assertEqual(expand_contractions("wasn't"), "was not")
        self.assertEqual(expand_contractions("isn't"), "is not")

    def test_curly_smart_apostrophes(self) -> None:
        """Test handling of smart/curly apostrophes (’ and ‘)."""
        self.assertEqual(expand_contractions("don’t"), "do not")
        self.assertEqual(expand_contractions("wasn’t"), "was not")
        self.assertEqual(expand_contractions("it’s"), "it is")

    def test_contractions_in_sentence(self) -> None:
        """Test contraction expansion inside a full sentence."""
        sentence = "I didn't like it and wouldn't buy it again; it's bad."
        expanded = expand_contractions(sentence)
        self.assertIn("did not", expanded)
        self.assertIn("would not", expanded)
        self.assertIn("it is", expanded)


class TestNLPPreprocessing(unittest.TestCase):
    """Test tokenization, controlled stopwords, lemmatization, and pipeline."""

    def test_tokenize_text(self) -> None:
        """Test tokenization producing lowercase alphabetic tokens."""
        text = "Hello world! This is a test."
        tokens = tokenize_text(text)
        self.assertIn("hello", tokens)
        self.assertIn("world", tokens)
        self.assertNotIn("!", tokens)

    def test_controlled_stopwords_preserve_negations(self) -> None:
        """Verify that sentiment-critical negation terms are strictly preserved."""
        stops = get_controlled_stopwords(exclude_negations=True)
        # Standard stopwords must be in the stopword set
        self.assertIn("the", stops)
        self.assertIn("is", stops)
        self.assertIn("at", stops)
        # Negation words must NOT be in the stopword set
        self.assertNotIn("not", stops)
        self.assertNotIn("no", stops)
        self.assertNotIn("never", stops)
        self.assertNotIn("neither", stops)
        self.assertNotIn("nor", stops)
        self.assertNotIn("nothing", stops)

    def test_remove_stopwords(self) -> None:
        """Test stopword filtering preserves 'not' and 'never'."""
        tokens = ["this", "movie", "is", "not", "good", "never", "watch"]
        filtered = remove_stopwords(tokens)
        self.assertIn("not", filtered)
        self.assertIn("never", filtered)
        self.assertIn("good", filtered)
        self.assertNotIn("this", filtered)
        self.assertNotIn("is", filtered)

    def test_lemmatize_tokens(self) -> None:
        """Test WordNet lemmatization on verbs and plurals."""
        tokens = ["loved", "loving", "loves", "issues"]
        lemmas = lemmatize_tokens(tokens)
        self.assertEqual(lemmas[0], "love")
        self.assertEqual(lemmas[1], "love")
        self.assertEqual(lemmas[2], "love")
        self.assertEqual(lemmas[3], "issue")

    def test_full_preprocess_text(self) -> None:
        """Test complete pipeline execution from raw text to clean string."""
        raw = "I don't recommend this product! It's NOT good and broke immediately: https://test.com"
        cleaned = preprocess_text(raw)
        # Verify negations survive
        self.assertIn("not", cleaned)
        # Verify URL is gone
        self.assertNotIn("https", cleaned)
        # Verify contraction expanded and cleaned
        self.assertNotIn("don't", cleaned)
        self.assertNotIn("it's", cleaned)
        self.assertIn("recommend", cleaned)
        self.assertIn("good", cleaned)

    def test_preprocess_empty_text(self) -> None:
        """Test preprocessing gracefully handles empty or whitespace input."""
        self.assertEqual(preprocess_text(""), "")
        self.assertEqual(preprocess_text("    "), "")
        self.assertEqual(preprocess_text(None), "")

    def test_preprocess_dataset(self) -> None:
        """Test preprocess_dataset adds clean_text column and preserves raw text."""
        df = pd.DataFrame({
            "text": ["Great laptop!", "Terrible service, won't return."],
            "sentiment": ["Positive", "Negative"]
        })
        processed_df = preprocess_dataset(df, text_column="text", new_column="clean_text")
        self.assertIn("clean_text", processed_df.columns)
        self.assertIn("text", processed_df.columns)
        self.assertEqual(len(processed_df), 2)
        self.assertEqual(processed_df.iloc[0]["text"], "Great laptop!")
        self.assertIn("great", processed_df.iloc[0]["clean_text"])
        self.assertIn("not", processed_df.iloc[1]["clean_text"])


class TestDatasetValidationAndAnalysis(unittest.TestCase):
    """Test dataset validation, text statistics, and quality reports."""

    def test_validate_dataset_valid(self) -> None:
        """Test validate_dataset on well-formed data."""
        df = pd.DataFrame({
            "text": ["Positive feedback.", "Negative feedback.", "Neutral statement."],
            "sentiment": ["Positive", "Negative", "Neutral"]
        })
        summary = validate_dataset(df)
        self.assertTrue(summary["is_valid"])
        self.assertEqual(summary["missing_text"], 0)
        self.assertEqual(summary["missing_sentiment"], 0)
        self.assertEqual(summary["duplicate_rows"], 0)
        self.assertEqual(summary["class_distribution"], {"Positive": 1, "Negative": 1, "Neutral": 1})

    def test_validate_dataset_with_issues(self) -> None:
        """Test validate_dataset detects missing values, empty strings, and invalid labels."""
        df = pd.DataFrame({
            "text": ["Good item", None, "   ", "Good item"],
            "sentiment": ["Positive", "Negative", "UnknownLabel", "Positive"]
        })
        summary = validate_dataset(df)
        self.assertEqual(summary["missing_text"], 1)
        self.assertEqual(summary["empty_text_rows"], 1)
        self.assertEqual(summary["duplicate_rows"], 1)
        self.assertEqual(summary["invalid_label_rows"], 1)
        self.assertGreater(len(summary["issues"]), 0)

    def test_calculate_text_statistics(self) -> None:
        """Test calculation of character and word statistics."""
        df = pd.DataFrame({
            "text": ["One two three", "Four five six seven eight"],
            "sentiment": ["Positive", "Negative"]
        })
        stats = calculate_text_statistics(df, text_column="text")
        self.assertEqual(stats["total_records"], 2)
        self.assertEqual(stats["word_stats"]["min"], 3)
        self.assertEqual(stats["word_stats"]["max"], 5)
        self.assertEqual(stats["word_stats"]["mean"], 4.0)

    def test_word_frequency_analysis(self) -> None:
        """Test word frequency counting per sentiment class."""
        df = pd.DataFrame({
            "clean_text": ["fast delivery fast service", "slow delivery broken item", "package arrive Tuesday"],
            "sentiment": ["Positive", "Negative", "Neutral"]
        })
        freqs = analyze_word_frequencies(df, text_column="clean_text", sentiment_column="sentiment")
        self.assertIn("Positive", freqs)
        self.assertEqual(freqs["Positive"].get("fast"), 2)
        self.assertIn("Negative", freqs)
        self.assertEqual(freqs["Negative"].get("broken"), 1)

    def test_cleaned_dataset_file_integrity(self) -> None:
        """Test that data/processed/cleaned_dataset.csv exists and has expected columns."""
        if CLEANED_DATA_FILE.exists():
            df = pd.read_csv(CLEANED_DATA_FILE)
            self.assertIn("text", df.columns)
            self.assertIn("sentiment", df.columns)
            self.assertIn("clean_text", df.columns)
            self.assertGreater(len(df), 0)


if __name__ == "__main__":
    unittest.main()
