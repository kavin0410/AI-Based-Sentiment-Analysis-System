"""NLP Preprocessing Module for AI Sentiment Intelligence.

Handles text preprocessing including contraction expansion, normalization,
tokenization, controlled stopword removal (with strict preservation of negation
and sentiment-bearing terms), and WordNet lemmatization.
"""

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set

import pandas as pd

# Attempt to load NLTK components
try:
    import nltk
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Import project configuration constants
try:
    from config import CLEAN_TEXT_COLUMN, PRESERVED_WORDS, SENTIMENT_COLUMN, TEXT_COLUMN
except ImportError:
    try:
        from src.config import CLEAN_TEXT_COLUMN, PRESERVED_WORDS, SENTIMENT_COLUMN, TEXT_COLUMN
    except ImportError:
        TEXT_COLUMN = "text"
        SENTIMENT_COLUMN = "sentiment"
        CLEAN_TEXT_COLUMN = "clean_text"
        PRESERVED_WORDS = {
            "not", "no", "never", "neither", "nor", "nothing",
            "nowhere", "hardly", "scarcely", "barely", "without", "against"
        }

# Import base cleaning functions
try:
    from src.data_cleaning import (
        remove_emails,
        remove_extra_whitespace,
        remove_html_tags,
        remove_special_characters,
        remove_urls,
        safe_str,
        to_lowercase,
    )
except ImportError:
    from data_cleaning import (
        remove_emails,
        remove_extra_whitespace,
        remove_html_tags,
        remove_special_characters,
        remove_urls,
        safe_str,
        to_lowercase,
    )


# ==============================================================================
# Contraction Handling
# ==============================================================================
# Comprehensive mapping of common English contractions to their expanded forms.
# Preserving contractions like "don't" -> "do not" is vital so the negation "not"
# survives downstream stopword filtering.
CONTRACTION_MAP: Dict[str, str] = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "y'all": "you all",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}

# Compile regex pattern for fast contraction replacement
CONTRACTION_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in CONTRACTION_MAP.keys()) + r")\b",
    flags=re.IGNORECASE,
)


def expand_contractions(text: str) -> str:
    """Expand English contractions into their full grammatical equivalents.

    Viva Note:
        Contraction expansion converts 'don't' into 'do not' and 'wasn't' into 'was not'.
        Without this step, punctuation removal would produce 'dont' and 'wasnt', which
        are often either discarded as non-dictionary words or fail to register as negations.

    Args:
        text: Input string.

    Returns:
        Expanded text string.
    """
    if not isinstance(text, str):
        text = safe_str(text)

    # Normalize curly / smart apostrophes to standard ASCII apostrophe
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")

    def _replace(match: re.Match) -> str:
        matched_str = match.group(0).lower()
        return CONTRACTION_MAP.get(matched_str, match.group(0))

    return CONTRACTION_PATTERN.sub(_replace, text)


# ==============================================================================
# Tokenization
# ==============================================================================

def tokenize_text(text: str) -> List[str]:
    """Tokenize input text into a list of word tokens.

    Uses NLTK's word_tokenize when available, with a fast regex fallback
    (\\b[a-zA-Z]+\\b) ensuring portability.

    Args:
        text: Input text string.

    Returns:
        List of lowercase alphabetical tokens.
    """
    if not isinstance(text, str):
        text = safe_str(text)

    if not text.strip():
        return []

    if NLTK_AVAILABLE:
        try:
            tokens = word_tokenize(text)
            # Filter to alphabetic tokens
            return [t.lower() for t in tokens if t.isalpha()]
        except Exception:
            pass

    # Regex fallback: extracts word tokens directly
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


# ==============================================================================
# Controlled Stopword Strategy
# ==============================================================================

_STOPWORDS_CACHE: Optional[Set[str]] = None


def get_controlled_stopwords(exclude_negations: bool = True) -> Set[str]:
    """Build a controlled set of stopwords that preserves sentiment-critical tokens.

    Academic Viva Rationale:
        Standard stopword lists blindly discard words like 'not', 'no', and 'never'.
        In sentiment analysis, this leads to fatal inversion errors:
            'This phone is NOT good' -> 'phone good' (classified as Positive!)
        By subtracting negation words from the stopword set, we retain the critical
        signals that invert polarities.

    Args:
        exclude_negations: If True, preserves negation and modifier words.

    Returns:
        Set of lowercase stopword strings.
    """
    global _STOPWORDS_CACHE
    if _STOPWORDS_CACHE is not None and exclude_negations:
        return _STOPWORDS_CACHE

    base_stops: Set[str] = set()
    if NLTK_AVAILABLE:
        try:
            base_stops = set(stopwords.words("english"))
        except Exception:
            pass

    if not base_stops:
        # Fallback basic English stopword list
        base_stops = {
            "a", "about", "above", "after", "again", "all", "am", "an", "and",
            "any", "are", "as", "at", "be", "because", "been", "before", "being",
            "below", "between", "both", "but", "by", "could", "did", "do", "does",
            "doing", "down", "during", "each", "few", "for", "from", "further",
            "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
            "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it",
            "its", "itself", "just", "me", "more", "most", "my", "myself", "of",
            "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves",
            "out", "over", "own", "same", "she", "should", "so", "some", "such",
            "than", "that", "the", "their", "theirs", "them", "themselves", "then",
            "there", "these", "they", "this", "those", "through", "to", "too",
            "under", "until", "up", "very", "was", "we", "were", "what", "when",
            "where", "which", "while", "who", "whom", "why", "with", "you", "your",
            "yours", "yourself", "yourselves",
        }

    if exclude_negations:
        # Subtract sentiment-critical words so they survive filtering
        controlled = base_stops - PRESERVED_WORDS
        _STOPWORDS_CACHE = controlled
        return controlled

    return base_stops


def remove_stopwords(
    tokens: List[str], stopwords_set: Optional[Set[str]] = None
) -> List[str]:
    """Filter out non-sentiment stopwords from a list of tokens.

    Args:
        tokens: List of string tokens.
        stopwords_set: Custom stopword set. Defaults to get_controlled_stopwords().

    Returns:
        Filtered list of tokens with negations preserved.
    """
    if stopwords_set is None:
        stopwords_set = get_controlled_stopwords(exclude_negations=True)

    return [t for t in tokens if t not in stopwords_set]


# ==============================================================================
# Lemmatization
# ==============================================================================

_LEMMATIZER: Optional[Any] = None


def get_lemmatizer() -> Optional[Any]:
    """Retrieve or initialize the WordNetLemmatizer instance."""
    global _LEMMATIZER
    if _LEMMATIZER is None and NLTK_AVAILABLE:
        try:
            _LEMMATIZER = WordNetLemmatizer()
        except Exception:
            _LEMMATIZER = None
    return _LEMMATIZER


def lemmatize_tokens(
    tokens: List[str], lemmatizer: Optional[Any] = None
) -> List[str]:
    """Reduce tokens to their root dictionary lemma using WordNet morphological analysis.

    Viva Note:
        Lemmatization collapses inflectional variants (e.g., 'loved', 'loving',
        'loves' -> 'love') into a single base form. Unlike aggressive stemming
        (which produces non-words like 'lov'), lemmatization maintains valid
        vocabulary words, leading to cleaner TF-IDF feature matrices.

    Args:
        tokens: List of word tokens.
        lemmatizer: Optional custom lemmatizer instance.

    Returns:
        List of lemmatized word tokens.
    """
    if lemmatizer is None:
        lemmatizer = get_lemmatizer()

    if lemmatizer is None:
        return tokens

    try:
        # Primary verb lemmatization handles common action/sentiment inflections (loved -> love)
        lemmatized: List[str] = []
        for token in tokens:
            v_lemma = lemmatizer.lemmatize(token, pos="v")
            if v_lemma != token:
                lemmatized.append(v_lemma)
            else:
                # Default noun lemmatization handles plurals (e.g. issues -> issue)
                lemmatized.append(lemmatizer.lemmatize(token, pos="n"))
        return lemmatized
    except Exception:
        return tokens


# ==============================================================================
# Full NLP Preprocessing Pipeline
# ==============================================================================

def preprocess_text(
    text: str,
    expand_contractions_flag: bool = True,
    remove_stops: bool = True,
    lemmatize: bool = True,
) -> str:
    """Run the complete NLP text preprocessing pipeline on a single text string.

    Sequential Pipeline:
        1. Safe string conversion
        2. Expand English contractions (e.g., don't -> do not)
        3. Lowercase normalization
        4. Remove URLs
        5. Remove HTML tags and entities
        6. Remove email addresses
        7. Remove special characters (retain letters and word spacing)
        8. Normalize extra whitespace
        9. Tokenize into individual words
        10. Controlled stopword removal (strictly preserving negation words)
        11. Lemmatization (optional, enabled by default)
        12. Rejoin into clean space-separated string suitable for TF-IDF

    Args:
        text: Raw input text string.
        expand_contractions_flag: Whether to expand contractions.
        remove_stops: Whether to filter controlled stopwords.
        lemmatize: Whether to apply WordNet lemmatization.

    Returns:
        Fully cleaned and preprocessed text string.
    """
    # 1. Convert to string safely
    text = safe_str(text)
    if not text.strip():
        return ""

    # 2. Contraction expansion (must precede character cleaning so ' stays intact)
    if expand_contractions_flag:
        text = expand_contractions(text)

    # 3. Lowercase normalization
    text = to_lowercase(text)

    # 4. Remove URLs
    text = remove_urls(text)

    # 5. Remove HTML tags and entities
    text = remove_html_tags(text)

    # 6. Remove Emails
    text = remove_emails(text)

    # 7. Remove non-alphabetic special characters and digits
    text = remove_special_characters(text, preserve_apostrophes=False)

    # 8. Normalize whitespace
    text = remove_extra_whitespace(text)

    # 9. Tokenize
    tokens = tokenize_text(text)
    if not tokens:
        return ""

    # 10. Controlled stopword removal
    if remove_stops:
        tokens = remove_stopwords(tokens)

    # 11. Lemmatization
    if lemmatize:
        tokens = lemmatize_tokens(tokens)

    # 12. Rejoin into space-separated string for TF-IDF compatibility
    return " ".join(tokens)


def preprocess_dataset(
    df: pd.DataFrame,
    text_column: str = "text",
    new_column: str = "clean_text",
    remove_empty_cleaned: bool = False,
) -> pd.DataFrame:
    """Apply the complete NLP preprocessing pipeline across a DataFrame text column.

    Viva Note:
        The raw dataset text and sentiment labels are strictly preserved in their
        original columns. The cleaned text is added to a new column ('clean_text'),
        ensuring full traceability and auditability.

    Args:
        df: Input DataFrame containing raw text.
        text_column: Column name containing raw text. Defaults to 'text'.
        new_column: Target column name for preprocessed text. Defaults to 'clean_text'.
        remove_empty_cleaned: If True, filters out rows where preprocessed text is empty.

    Returns:
        DataFrame with preprocessed text column added.
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame. Available: {list(df.columns)}")

    print(f"[PREPROCESSING] Running NLP pipeline on {len(df)} records in column '{text_column}'...")
    df_out = df.copy()
    df_out[new_column] = df_out[text_column].apply(preprocess_text)

    if remove_empty_cleaned:
        empty_mask = df_out[new_column].str.strip() == ""
        empty_count = int(empty_mask.sum())
        if empty_count > 0:
            df_out = df_out[~empty_mask].copy()
            print(f"[PREPROCESSING] Filtered {empty_count} row(s) with empty preprocessed text.")

    print(f"[PREPROCESSING] Completed. Preprocessed column: '{new_column}'")
    return df_out


def analyze_word_frequencies(
    df: pd.DataFrame,
    text_column: str = "clean_text",
    sentiment_column: str = "sentiment",
    top_n: int = 10,
    save_json_path: Optional[Path] = None,
) -> Dict[str, Dict[str, int]]:
    """Calculate the most frequent words for each sentiment class.

    Viva Note:
        Word frequency analysis provides interpretability before modeling.
        It reveals distinctive vocabulary signals across classes:
        e.g., 'exceptional', 'great', 'love' in Positive vs. 'damage', 'slow',
        'terrible' in Negative.

    Args:
        df: DataFrame containing cleaned text and sentiment columns.
        text_column: Preprocessed text column name.
        sentiment_column: Sentiment label column name.
        top_n: Number of top words to extract per class.
        save_json_path: Optional path to save JSON output. Defaults to results/word_frequencies.json.

    Returns:
        Nested dictionary: {sentiment_class: {word: frequency_count}}.
    """
    from collections import Counter
    import json

    if text_column not in df.columns or sentiment_column not in df.columns:
        raise ValueError(
            f"Required columns ('{text_column}', '{sentiment_column}') not found. "
            f"Available: {list(df.columns)}"
        )

    if save_json_path is None:
        try:
            from config import WORD_FREQ_JSON
            save_json_path = WORD_FREQ_JSON
        except ImportError:
            save_json_path = Path("results/word_frequencies.json")

    results: Dict[str, Dict[str, int]] = {}

    for label in df[sentiment_column].dropna().unique():
        subset = df[df[sentiment_column] == label][text_column].dropna().astype(str)
        all_words: List[str] = []
        for text in subset:
            all_words.extend(text.split())

        counter = Counter(all_words)
        results[str(label)] = dict(counter.most_common(top_n))

    if save_json_path:
        save_json_path = Path(save_json_path)
        save_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"[INFO] Word frequencies saved to: {save_json_path}")

    return results


def plot_word_frequency_analysis(
    freq_data: Dict[str, Dict[str, int]],
    output_path: Optional[Path] = None,
) -> Path:
    """Generate and save publication-quality bar charts showing top words per sentiment.

    Args:
        freq_data: Dictionary returned by analyze_word_frequencies().
        output_path: Destination path for PNG chart. Defaults to results/word_frequency_analysis.png.

    Returns:
        Path to the saved image file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if output_path is None:
        try:
            from config import WORD_FREQ_PLOT
            output_path = WORD_FREQ_PLOT
        except ImportError:
            output_path = Path("results/word_frequency_analysis.png")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    classes = [c for c in ["Positive", "Neutral", "Negative"] if c in freq_data]
    if not classes:
        classes = list(freq_data.keys())

    num_classes = len(classes)
    if num_classes == 0:
        print("[WARNING] No frequency data to plot.")
        return output_path

    fig, axes = plt.subplots(1, num_classes, figsize=(5.5 * num_classes, 5.5), sharey=False)
    if num_classes == 1:
        axes = [axes]

    palette_map = {
        "Positive": "Greens_r",
        "Neutral": "Blues_r",
        "Negative": "Reds_r",
    }

    for idx, (label, ax) in enumerate(zip(classes, axes)):
        word_counts = freq_data[label]
        if not word_counts:
            ax.text(0.5, 0.5, "No words found", ha="center", va="center")
            ax.set_title(f"{label} Top Words", fontsize=12, fontweight="bold")
            continue

        words = list(word_counts.keys())
        counts = list(word_counts.values())

        # Reverse order so top word appears at the top of the horizontal bar chart
        words_rev = words[::-1]
        counts_rev = counts[::-1]

        cmap_name = palette_map.get(label, "viridis")
        colors = sns.color_palette(cmap_name, len(words_rev))

        bars = ax.barh(words_rev, counts_rev, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_title(f"Top Words: {label}", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Frequency Count", fontsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        # Annotate counts at the end of each bar
        for bar in bars:
            width = bar.get_width()
            ax.annotate(
                f"{int(width)}",
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(3, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Word frequency plot saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    test_sentences = [
        "I don't like this product at all, it's NOT good!",
        "Customer support was quick, friendly, and resolved my issue in minutes.",
        "Check this out: https://example.com/test! Contact: user@mail.com. <b>AWESOME</b> app... loved it!",
    ]
    print("NLP Preprocessing Pipeline Test:")
    print("=" * 60)
    for s in test_sentences:
        print(f"Original: {s}")
        print(f"Cleaned:  {preprocess_text(s)}")
        print("-" * 60)
