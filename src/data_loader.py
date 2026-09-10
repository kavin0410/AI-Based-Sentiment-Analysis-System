"""Data Loading Module for AI Sentiment Intelligence.

Provides reusable functions for loading, validating, inspecting,
and generating quality reports for sentiment analysis datasets.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import pandas as pd

# Import project configuration constants
try:
    from config import (
        DATA_QUALITY_REPORT_JSON,
        DATA_QUALITY_REPORT_TXT,
        DEFAULT_RAW_FILE,
        RAW_DATA_PATH,
        SAMPLE_DATA_FILE,
        SENTIMENT_COLUMN,
        SUPPORTED_LABELS,
        TEXT_COLUMN,
    )
except ImportError:
    try:
        from src.config import (
            DATA_QUALITY_REPORT_JSON,
            DATA_QUALITY_REPORT_TXT,
            DEFAULT_RAW_FILE,
            RAW_DATA_PATH,
            SAMPLE_DATA_FILE,
            SENTIMENT_COLUMN,
            SUPPORTED_LABELS,
            TEXT_COLUMN,
        )
    except ImportError:
        RAW_DATA_PATH = Path("data/raw")
        DEFAULT_RAW_FILE = RAW_DATA_PATH / "demo_dataset.csv"
        SAMPLE_DATA_FILE = RAW_DATA_PATH / "sample_data.csv"
        TEXT_COLUMN = "text"
        SENTIMENT_COLUMN = "sentiment"
        SUPPORTED_LABELS = ["Positive", "Negative", "Neutral"]
        DATA_QUALITY_REPORT_JSON = Path("results/data_quality_report.json")
        DATA_QUALITY_REPORT_TXT = Path("results/data_quality_report.txt")


def get_default_dataset_path() -> Path:
    """Resolve the default dataset file path based on availability.

    Search priority:
        1. data/raw/dataset.csv (User-provided full dataset)
        2. data/raw/demo_dataset.csv (60-record Stage 2 benchmark/demo dataset)
        3. data/raw/sample_data.csv (18-record Stage 1 test dataset)

    Returns:
        Path to the first available dataset file.
    """
    user_dataset = RAW_DATA_PATH / "dataset.csv"
    if user_dataset.exists():
        return user_dataset
    if DEFAULT_RAW_FILE.exists():
        return DEFAULT_RAW_FILE
    if SAMPLE_DATA_FILE.exists():
        return SAMPLE_DATA_FILE
    return user_dataset


def load_dataset(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load a CSV dataset into a pandas DataFrame.

    Args:
        filepath: Path to the CSV file. If None, uses get_default_dataset_path().

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If file is empty or corrupted.
    """
    if filepath is None:
        filepath = get_default_dataset_path()

    filepath = Path(filepath)

    if not filepath.exists():
        error_msg = (
            f"\n[ERROR] Dataset file not found at: {filepath.resolve()}\n"
            f"Please place your CSV dataset containing '{TEXT_COLUMN}' and "
            f"'{SENTIMENT_COLUMN}' columns into '{RAW_DATA_PATH.resolve()}'.\n"
            f"Expected sentiment classes: {SUPPORTED_LABELS}"
        )
        print(error_msg, file=sys.stderr)
        raise FileNotFoundError(f"The file {filepath} was not found.")

    try:
        df = pd.read_csv(filepath)
        print(f"[SUCCESS] Dataset loaded successfully from: {filepath} ({len(df)} records)")
        return df
    except pd.errors.EmptyDataError:
        error_msg = f"[ERROR] The CSV file at {filepath} is completely empty."
        print(error_msg, file=sys.stderr)
        raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"[ERROR] Unexpected error reading {filepath}: {e}"
        print(error_msg, file=sys.stderr)
        raise


def validate_columns(
    df: pd.DataFrame, required_columns: Optional[List[str]] = None
) -> bool:
    """Validate that required columns exist in the DataFrame.

    Args:
        df: Input DataFrame.
        required_columns: List of column names that must be present.

    Returns:
        True if all required columns exist, False otherwise.
    """
    if required_columns is None:
        required_columns = [TEXT_COLUMN, SENTIMENT_COLUMN]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return False

    return True


def validate_labels(
    df: pd.DataFrame,
    sentiment_column: Optional[str] = None,
    valid_labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Filter the DataFrame to only retain records with supported sentiment labels.

    Args:
        df: Input DataFrame.
        sentiment_column: Name of the sentiment column.
        valid_labels: List of supported label strings.

    Returns:
        Filtered DataFrame containing only valid label rows.
    """
    if sentiment_column is None:
        sentiment_column = SENTIMENT_COLUMN
    if valid_labels is None:
        valid_labels = SUPPORTED_LABELS

    if sentiment_column not in df.columns:
        print(f"Error: Sentiment column '{sentiment_column}' not found in DataFrame.")
        return df

    invalid_mask = ~df[sentiment_column].isin(valid_labels)
    invalid_count = int(invalid_mask.sum())

    if invalid_count > 0:
        invalid_labels = df.loc[invalid_mask, sentiment_column].unique().tolist()
        print(
            f"[WARNING] Found {invalid_count} row(s) with unsupported labels: {invalid_labels}"
        )
        print(f"Filtering dataset to retain only supported labels: {valid_labels}")
        df = df[~invalid_mask].copy()

    return df


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform comprehensive data quality and integrity checks on the dataset.

    Checks:
        - Total row and column counts
        - Presence of required columns ('text', 'sentiment')
        - Missing values in text and sentiment
        - Empty or whitespace-only text values
        - Duplicate rows
        - Unsupported sentiment labels
        - Class balance / distribution
        - Character and word length extremes (shortest and longest text)

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary with structured validation metrics.
    """
    summary: Dict[str, Any] = {
        "is_valid": True,
        "rows": len(df),
        "columns": list(df.columns),
        "missing_text": 0,
        "missing_sentiment": 0,
        "empty_text_rows": 0,
        "duplicate_rows": 0,
        "invalid_label_rows": 0,
        "class_distribution": {},
        "text_length_stats": {},
        "issues": [],
    }

    # 1. Check required columns
    cols_ok = validate_columns(df)
    if not cols_ok:
        summary["is_valid"] = False
        summary["issues"].append("Missing required columns ('text', 'sentiment')")
        return summary

    # 2. Missing values
    summary["missing_text"] = int(df[TEXT_COLUMN].isnull().sum())
    summary["missing_sentiment"] = int(df[SENTIMENT_COLUMN].isnull().sum())
    if summary["missing_text"] > 0 or summary["missing_sentiment"] > 0:
        summary["issues"].append(
            f"Missing values found (text: {summary['missing_text']}, "
            f"sentiment: {summary['missing_sentiment']})"
        )

    # 3. Empty or whitespace-only text
    non_null_text = df[TEXT_COLUMN].dropna().astype(str)
    empty_mask = non_null_text.str.strip() == ""
    summary["empty_text_rows"] = int(empty_mask.sum())
    if summary["empty_text_rows"] > 0:
        summary["issues"].append(
            f"Found {summary['empty_text_rows']} empty or whitespace-only text rows"
        )

    # 4. Duplicate rows
    summary["duplicate_rows"] = int(df.duplicated(subset=[TEXT_COLUMN]).sum())
    if summary["duplicate_rows"] > 0:
        summary["issues"].append(
            f"Found {summary['duplicate_rows']} duplicate text rows"
        )

    # 5. Invalid labels
    valid_labels_set = set(SUPPORTED_LABELS)
    actual_labels = df[SENTIMENT_COLUMN].dropna().unique().tolist()
    invalid_labels = [lbl for lbl in actual_labels if lbl not in valid_labels_set]
    if invalid_labels:
        invalid_count = int((~df[SENTIMENT_COLUMN].isin(valid_labels_set)).sum())
        summary["invalid_label_rows"] = invalid_count
        summary["issues"].append(
            f"Found {invalid_count} row(s) with invalid labels: {invalid_labels}"
        )

    # 6. Class distribution
    summary["class_distribution"] = df[SENTIMENT_COLUMN].value_counts().to_dict()

    # 7. Text length statistics
    if len(non_null_text) > 0:
        char_lengths = non_null_text.str.len()
        word_counts = non_null_text.str.split().str.len()
        summary["text_length_stats"] = {
            "min_chars": int(char_lengths.min()),
            "max_chars": int(char_lengths.max()),
            "avg_chars": round(float(char_lengths.mean()), 2),
            "min_words": int(word_counts.min()),
            "max_words": int(word_counts.max()),
            "avg_words": round(float(word_counts.mean()), 2),
        }

    return summary


def format_validation_summary(summary: Dict[str, Any]) -> str:
    """Format the validation summary dictionary into a clean readable string.

    Args:
        summary: Dictionary returned by validate_dataset().

    Returns:
        Formatted multi-line report string.
    """
    lines = [
        "=" * 50,
        "DATASET VALIDATION SUMMARY",
        "=" * 50,
        f"Status: {'VALID' if summary['is_valid'] else 'INVALID'}",
        f"Dataset Shape: {summary['rows']} rows, {len(summary['columns'])} columns",
        f"Columns: {summary['columns']}",
        "",
        "Missing Values:",
        f"  Text: {summary['missing_text']}",
        f"  Sentiment: {summary['missing_sentiment']}",
        "",
        f"Duplicate Rows: {summary['duplicate_rows']}",
        f"Empty / Whitespace-only Strings: {summary['empty_text_rows']}",
        f"Invalid Sentiment Label Rows: {summary['invalid_label_rows']}",
        "",
        "Sentiment Classes:",
    ]
    for label in SUPPORTED_LABELS:
        count = summary["class_distribution"].get(label, 0)
        lines.append(f"  {label}: {count}")

    if summary.get("text_length_stats"):
        stats = summary["text_length_stats"]
        lines.extend([
            "",
            "Text Length Statistics:",
            f"  Char Length: min={stats['min_chars']}, avg={stats['avg_chars']}, max={stats['max_chars']}",
            f"  Word Count:  min={stats['min_words']}, avg={stats['avg_words']}, max={stats['max_words']}",
        ])

    if summary["issues"]:
        lines.extend(["", "Issues Detected:"])
        for issue in summary["issues"]:
            lines.append(f"  - {issue}")
    else:
        lines.extend(["", "Quality Check: No data integrity issues detected."])

    lines.append("=" * 50)
    return "\n".join(lines)


def report_dataset_info(df: pd.DataFrame) -> Dict[str, Any]:
    """Report detailed information about the dataset.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary containing dataset statistics.
    """
    summary = validate_dataset(df)
    print(format_validation_summary(summary))

    missing = df.isnull().sum().to_dict()
    sentiment_dist = {}
    if SENTIMENT_COLUMN in df.columns:
        sentiment_dist = df[SENTIMENT_COLUMN].value_counts().to_dict()

    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
        "missing_values": missing,
        "duplicate_rows": int(df.duplicated().sum()),
        "sentiment_distribution": sentiment_dist,
        "validation": summary,
    }


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Get a quick summary of the dataset for UI and programmatic consumers.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary with basic dataset statistics.
    """
    sentiment_dist = {}
    if SENTIMENT_COLUMN in df.columns:
        sentiment_dist = df[SENTIMENT_COLUMN].value_counts().to_dict()

    return {
        "shape": df.shape,
        "missing_count": int(df.isnull().sum().sum()),
        "duplicate_count": int(df.duplicated(subset=[TEXT_COLUMN] if TEXT_COLUMN in df.columns else None).sum()),
        "label_distribution": sentiment_dist,
    }


def generate_data_quality_report(
    df: pd.DataFrame,
    output_json_path: Optional[Path] = None,
    output_txt_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate and persist a detailed data quality report.

    Args:
        df: Input DataFrame.
        output_json_path: Optional path for JSON output.
        output_txt_path: Optional path for text output.

    Returns:
        Dictionary containing the full report.
    """
    if output_json_path is None:
        output_json_path = DATA_QUALITY_REPORT_JSON
    if output_txt_path is None:
        output_txt_path = DATA_QUALITY_REPORT_TXT

    output_json_path = Path(output_json_path)
    output_txt_path = Path(output_txt_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)

    summary = validate_dataset(df)
    report_text = format_validation_summary(summary)

    # Save JSON report
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    # Save Text report
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"[INFO] Data quality report saved to:\n  - {output_json_path}\n  - {output_txt_path}")
    return summary


def calculate_text_statistics(
    df: pd.DataFrame, text_column: str = "text"
) -> Dict[str, Any]:
    """Calculate character and word count statistics for the dataset.

    Args:
        df: Input DataFrame.
        text_column: Name of the text column to analyze.

    Returns:
        Dictionary containing character and word count statistics.
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")

    text_series = df[text_column].dropna().astype(str)
    char_lengths = text_series.str.len()
    word_counts = text_series.str.split().str.len()

    stats: Dict[str, Any] = {
        "total_records": len(text_series),
        "character_stats": {
            "min": int(char_lengths.min()) if len(char_lengths) > 0 else 0,
            "max": int(char_lengths.max()) if len(char_lengths) > 0 else 0,
            "mean": round(float(char_lengths.mean()), 2) if len(char_lengths) > 0 else 0.0,
            "median": float(char_lengths.median()) if len(char_lengths) > 0 else 0.0,
        },
        "word_stats": {
            "min": int(word_counts.min()) if len(word_counts) > 0 else 0,
            "max": int(word_counts.max()) if len(word_counts) > 0 else 0,
            "mean": round(float(word_counts.mean()), 2) if len(word_counts) > 0 else 0.0,
            "median": float(word_counts.median()) if len(word_counts) > 0 else 0.0,
        },
        "class_breakdown": {},
    }

    if SENTIMENT_COLUMN in df.columns:
        for label in df[SENTIMENT_COLUMN].dropna().unique():
            class_subset = df[df[SENTIMENT_COLUMN] == label][text_column].dropna().astype(str)
            sub_words = class_subset.str.split().str.len()
            sub_chars = class_subset.str.len()
            stats["class_breakdown"][str(label)] = {
                "count": len(class_subset),
                "avg_words": round(float(sub_words.mean()), 2) if len(sub_words) > 0 else 0.0,
                "avg_chars": round(float(sub_chars.mean()), 2) if len(sub_chars) > 0 else 0.0,
            }

    return stats


def plot_sentiment_distribution(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    sentiment_column: Optional[str] = None,
) -> Path:
    """Generate and save a publication-quality sentiment class distribution visualization.

    Viva Note:
        Visualizing class balance is essential before model training to detect
        class imbalance, which could lead to majority-class bias and misleading
        accuracy metrics.

    Args:
        df: Input DataFrame.
        output_path: Destination path for PNG chart. Defaults to results/sentiment_distribution.png.
        sentiment_column: Name of sentiment label column.

    Returns:
        Path to the saved image file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if sentiment_column is None:
        sentiment_column = SENTIMENT_COLUMN
    if output_path is None:
        try:
            from config import SENTIMENT_DIST_PLOT
            output_path = SENTIMENT_DIST_PLOT
        except ImportError:
            output_path = Path("results/sentiment_distribution.png")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if sentiment_column not in df.columns:
        raise ValueError(f"Column '{sentiment_column}' not found in DataFrame.")

    counts = df[sentiment_column].value_counts()
    # Reindex to standard order if present
    ordered_labels = [lbl for lbl in SUPPORTED_LABELS if lbl in counts.index]
    counts = counts.reindex(ordered_labels)

    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(13, 5))
    palette = sns.color_palette("muted", len(counts))

    # 1. Bar Chart
    bars = ax_bar.bar(counts.index, counts.values, color=palette, edgecolor="black", linewidth=0.8)
    ax_bar.set_title("Sentiment Class Distribution", fontsize=13, fontweight="bold", pad=12)
    ax_bar.set_xlabel("Sentiment Class", fontsize=11)
    ax_bar.set_ylabel("Number of Samples", fontsize=11)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate count values on bars
    for bar in bars:
        height = bar.get_height()
        ax_bar.annotate(
            f"{int(height)}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # 2. Donut / Pie Chart
    wedges, texts, autotexts = ax_pie.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=palette,
        wedgeprops={"edgecolor": "white", "linewidth": 2, "antialiased": True},
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")
    ax_pie.set_title("Sentiment Class Proportions", fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Sentiment distribution plot saved to: {output_path}")
    return output_path


def plot_text_length_distribution(
    df: pd.DataFrame,
    text_column: str = "text",
    sentiment_column: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Generate and save text length (word count & character count) distribution charts.

    Viva Note:
        Analyzing text length distributions reveals whether reviews are predominantly
        short phrases or long paragraphs, informing max_features and n-gram choices
        for TF-IDF feature extraction.

    Args:
        df: Input DataFrame.
        text_column: Text column to analyze.
        sentiment_column: Optional sentiment column for stratified grouping.
        output_path: Output file path. Defaults to results/text_length_distribution.png.

    Returns:
        Path to the saved image file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if sentiment_column is None:
        sentiment_column = SENTIMENT_COLUMN
    if output_path is None:
        try:
            from config import TEXT_LENGTH_PLOT
            output_path = TEXT_LENGTH_PLOT
        except ImportError:
            output_path = Path("results/text_length_distribution.png")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_copy = df[[text_column]].dropna().copy()
    df_copy["word_count"] = df_copy[text_column].astype(str).str.split().str.len()
    df_copy["char_count"] = df_copy[text_column].astype(str).str.len()

    if sentiment_column in df.columns:
        df_copy[sentiment_column] = df[sentiment_column]

    fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Word count histogram with KDE
    sns.histplot(
        data=df_copy,
        x="word_count",
        hue=sentiment_column if sentiment_column in df.columns else None,
        kde=True,
        ax=ax_hist,
        palette="muted" if sentiment_column in df.columns else None,
        bins=15,
    )
    ax_hist.set_title("Word Count Distribution by Sentiment", fontsize=13, fontweight="bold", pad=12)
    ax_hist.set_xlabel("Words per Sample", fontsize=11)
    ax_hist.set_ylabel("Frequency", fontsize=11)
    ax_hist.grid(axis="y", linestyle="--", alpha=0.5)

    # 2. Boxplot across sentiment classes
    if sentiment_column in df.columns:
        ordered_labels = [lbl for lbl in SUPPORTED_LABELS if lbl in df_copy[sentiment_column].unique()]
        sns.boxplot(
            data=df_copy,
            x=sentiment_column,
            y="word_count",
            hue=sentiment_column,
            order=ordered_labels,
            palette="muted",
            legend=False,
            ax=ax_box,
            boxprops=dict(alpha=0.8),
        )
        ax_box.set_title("Word Count Comparison across Sentiments", fontsize=13, fontweight="bold", pad=12)
        ax_box.set_xlabel("Sentiment Class", fontsize=11)
        ax_box.set_ylabel("Word Count", fontsize=11)
        ax_box.grid(axis="y", linestyle="--", alpha=0.5)
    else:
        sns.boxplot(data=df_copy, y="word_count", ax=ax_box, color="skyblue")
        ax_box.set_title("Word Count Boxplot", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Text length distribution plot saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    filepath = get_default_dataset_path()
    print(f"Loading default dataset from: {filepath}")
    data = load_dataset(filepath)
    report_dataset_info(data)
    generate_data_quality_report(data)
    plot_sentiment_distribution(data)
    plot_text_length_distribution(data)
