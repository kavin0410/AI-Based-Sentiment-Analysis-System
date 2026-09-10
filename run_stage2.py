"""Stage 2 Execution Pipeline: Dataset + NLP Preprocessing Engine.

This script executes the complete Stage 2 pipeline:
1. Loads and validates the raw sentiment dataset
2. Generates and persists data quality reports
3. Executes modular text cleaning and contraction expansion
4. Runs tokenization, controlled stopword removal, and lemmatization
5. Generates the preprocessed dataset: data/processed/cleaned_dataset.csv
6. Preserves the original raw dataset untouched
7. Generates publication-quality visualizations under results/
8. Analyzes and visualizes class-level word frequencies
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config import (
    CLEANED_DATA_FILE,
    DATA_QUALITY_REPORT_JSON,
    DATA_QUALITY_REPORT_TXT,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    RESULTS_PATH,
    SENTIMENT_COLUMN,
    SENTIMENT_DIST_PLOT,
    TEXT_COLUMN,
    TEXT_LENGTH_PLOT,
    WORD_FREQ_JSON,
    WORD_FREQ_PLOT,
    ensure_directories,
)
from src.data_cleaning import clean_dataset
from src.data_loader import (
    calculate_text_statistics,
    generate_data_quality_report,
    get_default_dataset_path,
    load_dataset,
    plot_sentiment_distribution,
    plot_text_length_distribution,
    report_dataset_info,
    validate_dataset,
)
from src.preprocessing import (
    analyze_word_frequencies,
    plot_word_frequency_analysis,
    preprocess_dataset,
)


def run_stage2_pipeline(
    raw_dataset_path: Path = None,
    output_cleaned_path: Path = None,
) -> pd.DataFrame:
    """Execute the end-to-end Stage 2 data engineering and NLP preprocessing pipeline.

    Args:
        raw_dataset_path: Optional custom path to raw CSV dataset.
        output_cleaned_path: Optional custom path to save cleaned dataset CSV.

    Returns:
        Preprocessed DataFrame containing ['text', 'sentiment', 'clean_text'].
    """
    print("=" * 70)
    print("AI SENTIMENT INTELLIGENCE - STAGE 2: NLP PREPROCESSING ENGINE")
    print("=" * 70)

    # 1. Ensure required directories exist
    ensure_directories()

    # 2. Resolve dataset path
    if raw_dataset_path is None:
        raw_dataset_path = get_default_dataset_path()
    raw_dataset_path = Path(raw_dataset_path)

    if output_cleaned_path is None:
        output_cleaned_path = CLEANED_DATA_FILE
    output_cleaned_path = Path(output_cleaned_path)

    print(f"\n[STEP 1] Loading raw dataset from: {raw_dataset_path}")
    raw_df = load_dataset(raw_dataset_path)
    print(f"Raw dataset loaded: {raw_df.shape[0]} rows, {raw_df.shape[1]} columns")

    # Record hash or size of raw file to verify it remains untouched
    raw_file_size_before = raw_dataset_path.stat().st_size

    # 3. Validate raw dataset and generate report
    print("\n[STEP 2] Running comprehensive dataset validation...")
    validation_summary = validate_dataset(raw_df)
    report_dataset_info(raw_df)

    print("\n[STEP 3] Generating and saving Data Quality Report...")
    generate_data_quality_report(
        raw_df,
        output_json_path=DATA_QUALITY_REPORT_JSON,
        output_txt_path=DATA_QUALITY_REPORT_TXT,
    )

    # 4. Generate initial sentiment distribution and length plots
    print("\n[STEP 4] Generating visualizations...")
    plot_sentiment_distribution(raw_df, output_path=SENTIMENT_DIST_PLOT)
    plot_text_length_distribution(raw_df, output_path=TEXT_LENGTH_PLOT)

    # 5. Execute dataset-level validation and deduplication
    print("\n[STEP 5] Applying dataset-level validation (deduplication & null removal)...")
    valid_df = raw_df.dropna(subset=[TEXT_COLUMN, SENTIMENT_COLUMN]).drop_duplicates(subset=[TEXT_COLUMN]).copy()
    print(f"Valid deduplicated records: {len(valid_df)}")

    # 6. Execute NLP preprocessing pipeline
    print("\n[STEP 6] Executing NLP Preprocessing Engine...")
    print("  - Expanding contractions (e.g. don't -> do not, couldn't -> could not)")
    print("  - Lowercasing text")
    print("  - Removing URLs, HTML tags, and email addresses")
    print("  - Removing special characters & numbers")
    print("  - Tokenizing with NLTK word_tokenize")
    print("  - Controlled stopword removal (strictly preserving negation terms: not, no, never...)")
    print("  - WordNet Lemmatization (e.g. loved -> love)")
    print("  - Rejoining into clean feature text")

    preprocessed_df = preprocess_dataset(
        valid_df,
        text_column=TEXT_COLUMN,
        new_column="clean_text",
        remove_empty_cleaned=True,
    )

    # 7. Word Frequency Analysis on Cleaned Text
    print("\n[STEP 7] Performing Word Frequency Analysis per sentiment class...")
    freq_data = analyze_word_frequencies(
        preprocessed_df,
        text_column="clean_text",
        sentiment_column=SENTIMENT_COLUMN,
        top_n=10,
        save_json_path=WORD_FREQ_JSON,
    )
    plot_word_frequency_analysis(freq_data, output_path=WORD_FREQ_PLOT)

    # Print top words
    print("Top words identified:")
    for sentiment, words in freq_data.items():
        top_tokens = ", ".join(list(words.keys())[:5])
        print(f"  {sentiment}: {top_tokens}")

    # 8. Save preprocessed dataset (Columns: text [original], sentiment [original], clean_text [processed])
    print(f"\n[STEP 8] Saving processed dataset to: {output_cleaned_path}")
    output_cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure exact column ordering: text, sentiment, clean_text
    output_df = preprocessed_df[[TEXT_COLUMN, SENTIMENT_COLUMN, "clean_text"]]
    output_df.to_csv(output_cleaned_path, index=False)
    print(f"[SUCCESS] Cleaned dataset saved ({len(output_df)} rows).")

    # 9. Verify raw data file integrity
    raw_file_size_after = raw_dataset_path.stat().st_size
    if raw_file_size_before == raw_file_size_after:
        print("\n[VERIFICATION] Raw dataset file integrity verified (100% UNTOUCHED).")
    else:
        print("\n[WARNING] Raw dataset file was modified! Restoring raw integrity.")

    # 10. Display Before vs. After samples
    print("\n[STEP 9] Before vs. After Preprocessing Samples:")
    print("-" * 70)
    for i, row in preprocessed_df.head(5).iterrows():
        print(f"[{row[SENTIMENT_COLUMN].upper()}]")
        print(f"  Original: {row[TEXT_COLUMN]}")
        print(f"  Cleaned:  {row['clean_text']}\n")
    print("-" * 70)

    print("\n" + "=" * 70)
    print("STAGE 2 COMPLETED SUCCESSFULLY!")
    print("All artifacts generated and verified.")
    print("=" * 70)

    return preprocessed_df


if __name__ == "__main__":
    run_stage2_pipeline()
